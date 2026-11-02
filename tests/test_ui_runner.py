"""Bounded physical checks of the actual UI preparation/execution path."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('pyogrio')
pytest.importorskip('shapely')
pytest.importorskip('pyproj')
pytest.importorskip('matplotlib')

from floodwarp.inputs import read_terrain, with_active_mask
from floodwarp.model import advance, create_model, diagnostics, load_checkpoint
from floodwarp.sampling import sample_targets
from floodwarp.ui_runner import (RunConfig, boundary_preset, create_run_model, execute_run,
    polygon_mask, prepare_run, read_hyetograph)


@pytest.fixture
def inputs(tmp_path):
    source = Path(__file__).parents[1]/'src/floodwarp/ui_assets/example'
    # From an installed wheel, use its own packaged inputs instead.
    if not source.exists():
        import floodwarp
        source = Path(floodwarp.__file__).parent/'ui_assets/example'
    rain = tmp_path/'rain.csv'
    rain.write_text('time_h,rainfall_mm_h\n0,360\n0.0002777777777777778,0\n')
    return RunConfig(dem=str(source/'dem.tif'), rainfall_csv=str(rain),
        output_parent=str(tmp_path), vertical_datum='local', duration_s=2., interval_s=5.,
        device='cpu', boundary='closed')


def write_field(*, path, config, values, shift=0., crs='EPSG:2193', nodata=None):
    from affine import Affine
    with rasterio.open(config.dem) as source:
        profile = source.profile.copy()
        profile.update(dtype='float64', nodata=nodata, crs=crs,
            transform=source.transform @ Affine.translation(xoff=shift, yoff=0.))
        with rasterio.open(path, 'w', **profile) as output:
            output.write(np.asarray(values, dtype=np.float64), 1)


@pytest.mark.parametrize('duration', [2., 22., 23., 20.])
def test_runner_direct_api_and_exact_final_time(inputs, tmp_path, duration):
    config = replace(inputs, duration_s=duration)
    prepared = prepare_run(config=config)
    output = tmp_path/'result'
    result = execute_run(config=config, output=output)
    # Construct independently through the ordinary API, with identical resolved inputs.
    direct = create_model(terrain=prepared.terrain, solver_method='fv1', manning_n=config.manning_n,
        infiltration=config.infiltration_mm_h/3.6e6, rainfall_series=(prepared.times_s, prepared.rates_m_s),
        boundary_qx=prepared.qx, boundary_qy=prepared.qy, device='cpu')
    direct.solver.set_rainfall_spatial_multiplier(multiplier=prepared.multiplier)
    for target in sample_targets(duration=duration, interval=config.interval_s):
        advance(model=direct, end_time=float(target))
    assert result['actual_final_time_s'] == duration
    samples = json.loads((output/'diagnostics.json').read_text())
    assert samples[-1]['time_s'] == duration
    assert abs(samples[-1]['residual_m3']) < 1.e-11
    with rasterio.open(output/'final_depth.tif') as actual, rasterio.open(config.dem) as dem:
        assert actual.crs == dem.crs and actual.transform == dem.transform
        assert actual.units == ('m',)
        assert actual.tags()['quantity'] == 'final_water_depth'
        np.testing.assert_array_equal(actual=actual.read_masks(1)[::-1] > 0, desired=prepared.terrain.active)
        np.testing.assert_array_equal(actual=actual.read(1)[::-1][prepared.terrain.active],
            desired=direct.solver.get_water_depth()[prepared.terrain.active])
    with rasterio.open(output/'maximum_depth.tif') as maximum:
        np.testing.assert_array_equal(actual=maximum.read(1)[::-1][prepared.terrain.active],
            desired=direct.solver.get_max_water_depth()[prepared.terrain.active])
    fresh = create_run_model(prepared=prepared)
    load_checkpoint(model=fresh, path=output/'checkpoint.npz')
    np.testing.assert_array_equal(actual=fresh.solver.get_water_depth(), desired=direct.solver.get_water_depth())
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'completed'
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['accepted_steps'] == direct.solver.steps
    assert metrics['actual_time_s'] == duration
    assert metrics['dt_mean_s'] == pytest.approx(duration/direct.solver.steps)
    assert 0. < metrics['dt_min_s'] <= metrics['dt_mean_s'] <= metrics['dt_max_s']
    assert metrics['hydraulic_seconds'] <= metrics['advance_wall_seconds'] <= metrics['worker_wall_seconds']
    assert status['metrics']['accepted_steps'] == metrics['accepted_steps']
    assert metrics['complete']
    assert (output/'runtime.csv').is_file()


def test_uniform_equals_ones_and_spatial_mass(inputs, tmp_path):
    terrain = read_terrain(path=Path(inputs.dem), vertical_datum='local')
    shape = terrain.active.shape
    ones = tmp_path/'ones.tif'
    write_field(path=ones, config=inputs, values=np.ones(shape))
    uniform = create_run_model(prepared=prepare_run(config=inputs))
    scaled = create_run_model(prepared=prepare_run(config=replace(inputs, distribution='map', multiplier=str(ones))))
    varying = np.broadcast_to(np.linspace(0., 2., shape[1]), shape).copy()
    variable_path = tmp_path/'varying.tif'
    write_field(path=variable_path, config=inputs, values=varying)
    variable_prepared = prepare_run(config=replace(inputs, distribution='map', multiplier=str(variable_path)))
    spatial = create_run_model(prepared=variable_prepared)
    for model in (uniform, scaled, spatial):
        advance(model=model, end_time=2.)
    np.testing.assert_array_equal(actual=uniform.solver.get_water_depth(), desired=scaled.solver.get_water_depth())
    assert not np.array_equal(uniform.solver.get_water_depth(), spatial.solver.get_water_depth())
    expected_rain = .5 * .0001 * terrain.dx * terrain.dy * varying[::-1][terrain.active].sum()
    assert diagnostics(model=spatial)['rain'] == pytest.approx(expected_rain, abs=1.e-12)
    assert abs(diagnostics(model=spatial)['residual_m3']) < 1.e-11
    assert variable_prepared.summary['multiplier_range'] == [0., 2.]


def test_domain_intersection_and_enclosed_hole(inputs, tmp_path):
    terrain = read_terrain(path=Path(inputs.dem), vertical_datum='local')
    mask = np.ones(terrain.active.shape)
    mask[4:6, 6:8] = 0.
    mask[:, :2] = 0.
    path = tmp_path/'mask.tif'
    write_field(path=path, config=inputs, values=mask)
    prepared = prepare_run(config=replace(inputs, domain_mode='raster', domain=str(path), boundary='free'))
    np.testing.assert_array_equal(actual=prepared.terrain.active, desired=terrain.active & mask[::-1].astype(bool))
    # Four enclosed inactive cells have closed surrounding interfaces.
    assert np.all(prepared.qx[6:8, 6] == 1)
    assert np.all(prepared.qx[6:8, 8] == 1)
    assert np.all(prepared.qy[6, 6:8] == 1)
    assert np.all(prepared.qy[8, 6:8] == 1)
    assert np.all(prepared.qx[:, 2] == 2)
    output = tmp_path/'masked-result'
    execute_run(config=replace(inputs, domain_mode='raster', domain=str(path)), output=output)
    assert json.loads((output/'status.json').read_text())['state'] == 'completed'


@pytest.mark.parametrize('text', [
    'time_h,discharge_m3_s\n0,1\n1,0\n', 'time_h,rainfall_mm_h\n0,0\n',
    'time_h,rainfall_mm_h\n0,0\n0,0\n', 'time_h,rainfall_mm_h\n0,nan\n0.0001,0\n',
    'time_h,rainfall_mm_h\n0,-1\n0.0001,0\n', 'time_h,rainfall_mm_h\n0,0\n1,0\n',
    'time_h,rainfall_mm_h\n0,0\n0.0001,2\n', 'time_h,rainfall_mm_h\n0,\n0.0001,0\n'])
def test_invalid_csv(inputs, tmp_path, text):
    path = tmp_path/'bad.csv'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_hyetograph(path=path, duration_s=inputs.duration_s)


@pytest.mark.parametrize('kind', ['shift', 'crs', 'negative', 'nodata', 'empty', 'nonbinary'])
def test_invalid_rasters(inputs, tmp_path, kind):
    with rasterio.open(inputs.dem) as source:
        values = np.ones(source.shape)
    path = tmp_path/'bad.tif'
    if kind == 'negative':
        values[5, 5] = -1.
    if kind == 'nodata':
        values[5, 5] = np.nan
    if kind == 'empty':
        values.fill(0.)
    if kind == 'nonbinary':
        values.fill(2.)
    write_field(path=path, config=inputs, values=values, shift=1. if kind == 'shift' else 0.,
        crs='EPSG:3857' if kind == 'crs' else 'EPSG:2193')
    config = replace(inputs, domain_mode='raster', domain=str(path)) if kind in ('empty', 'nonbinary') else replace(inputs, distribution='map', multiplier=str(path))
    with pytest.raises(ValueError):
        prepare_run(config=config)


def test_polygon_declared_crs_and_reprojection(inputs, tmp_path):
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely import to_wkb
    from pyogrio.raw import write
    import pyogrio
    terrain = read_terrain(path=Path(inputs.dem), vertical_datum='local')
    transform = terrain.metadata['transform']
    west, north = transform[2], transform[5]
    polygon = box(minx=west, miny=north-60., maxx=west+40., maxy=north)
    path = tmp_path/'domain.gpkg'
    write(path=path, geometry=np.asarray([to_wkb(geometry=polygon)], dtype=object),
        field_data=[], fields=[], driver='GPKG', crs='EPSG:2193', geometry_type='Polygon')
    mask = polygon_mask(path=path, terrain=terrain)
    assert mask[:, :8].all() and not mask[:, 8:].any()
    converter = Transformer.from_crs(crs_from='EPSG:2193', crs_to='EPSG:4326', always_xy=True)
    from shapely import transform as transform_polygon
    geographic = transform_polygon(geometry=polygon, transformation=converter.transform, interleaved=False)
    geojson = tmp_path/'domain.geojson'
    write(path=geojson, geometry=np.asarray([to_wkb(geometry=geographic)], dtype=object),
        field_data=[], fields=[], driver='GeoJSON', crs='EPSG:4326', geometry_type='Polygon')
    contents = json.loads(geojson.read_text())
    contents['crs'] = dict(type='name', properties=dict(name='EPSG:4326'))
    geojson.write_text(json.dumps(contents))
    np.testing.assert_array_equal(actual=polygon_mask(path=geojson, terrain=terrain), desired=mask)
    del contents['crs']
    geojson.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='explicit crs'):
        polygon_mask(path=geojson, terrain=terrain)


def test_failure_preserves_last_checkpoint(inputs, tmp_path, monkeypatch):
    import floodwarp.ui_runner as runner
    original_advance = runner.advance

    def fail_after_first(*, model, end_time):
        if end_time > 1.:
            raise RuntimeError('deliberate second-sample failure')
        return original_advance(model=model, end_time=end_time)

    monkeypatch.setattr(target=runner, name='advance', value=fail_after_first)
    output = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='deliberate'):
        execute_run(config=replace(inputs, interval_s=1.), output=output)
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'failed' and status['time_s'] == 1.
    metrics = json.loads((output/'metrics.json').read_text())
    assert not metrics['complete'] and metrics['actual_time_s'] == 1.
    assert 'deliberate' in status['error']
    fresh = create_run_model(prepared=prepare_run(config=replace(inputs, interval_s=1.)))
    load_checkpoint(model=fresh, path=output/'checkpoint.npz')
    assert fresh.solver.time == 1.


def test_multiplier_checkpoint_identity(inputs, tmp_path):
    from floodwarp.model import save_checkpoint
    prepared = prepare_run(config=inputs)
    original = create_run_model(prepared=prepared)
    save_checkpoint(model=original, path=tmp_path/'state.npz')
    path = tmp_path/'twos.tif'
    write_field(path=path, config=inputs, values=np.full(prepared.terrain.active.shape, 2.))
    changed = create_run_model(prepared=prepare_run(config=replace(inputs, distribution='map', multiplier=str(path))))
    with pytest.raises(ValueError, match='provenance'):
        load_checkpoint(model=changed, path=tmp_path/'state.npz')


def test_input_change_after_validation_fails(inputs, tmp_path):
    prepared = prepare_run(config=inputs)
    Path(inputs.rainfall_csv).write_text('time_h,rainfall_mm_h\n0,1\n0.0001,0\n')
    with pytest.raises(ValueError, match='changed after validation'):
        execute_run(config=inputs, output=tmp_path/'changed', expected_inputs=prepared.summary['inputs'])
    assert json.loads((tmp_path/'changed/status.json').read_text())['state'] == 'failed'


def test_ui_cuda_default_graph_path(inputs, tmp_path):
    import os
    if os.environ.get('OVERLANDER_TEST_FULL_SWE_CUDA') != '1':
        pytest.skip('UI CUDA run is explicitly opt-in')
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip('CUDA is unavailable')
    cpu_output, gpu_output = tmp_path/'cpu', tmp_path/'gpu'
    execute_run(config=inputs, output=cpu_output)
    execute_run(config=replace(inputs, device='cuda:0'), output=gpu_output)
    for filename in ('maximum_depth.tif', 'final_depth.tif'):
        with rasterio.open(cpu_output/filename) as cpu, rasterio.open(gpu_output/filename) as gpu:
            np.testing.assert_allclose(actual=gpu.read(1), desired=cpu.read(1), rtol=1.e-12, atol=1.e-14)
    cpu_ledger = json.loads((cpu_output/'diagnostics.json').read_text())[-1]
    gpu_ledger = json.loads((gpu_output/'diagnostics.json').read_text())[-1]
    assert gpu_ledger['time_s'] == cpu_ledger['time_s'] == 2.
    assert abs(gpu_ledger['residual_m3']) < 1.e-11
