"""Spatial coefficients, explicit coast faces and display-only map registration."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('pyogrio')
from floodwarp.inputs import read_terrain
from floodwarp.model import advance, create_model, diagnostics, save_checkpoint, load_checkpoint
from floodwarp.ui_fields import boundary_template
from floodwarp.ui_maps import map_overlay, boundary_features
from floodwarp.ui_presets import convert_orton_inputs
from floodwarp.ui_runner import RunConfig, file_identity, prepare_run, create_run_model, read_hyetograph


@pytest.fixture
def spatial_inputs(tmp_path):
    from rasterio.transform import from_origin
    dem = tmp_path/'dem.tif'
    elevation = np.zeros((4, 5))
    elevation[0] = -9999.
    with rasterio.open(dem, 'w', driver='GTiff', width=5, height=4, count=1,
            dtype='float64', crs='EPSG:2193', transform=from_origin(west=1574940., north=5167275., xsize=5., ysize=5.), nodata=-9999.) as target:
        target.write(elevation, 1)
        target.update_tags(vertical_datum='NZVD2016', units='m')
    rain = tmp_path/'rain.csv'
    rain.write_text('time_s,rainfall_m_s\n0,0.001\n1,0.001\n')
    return RunConfig(dem=str(dem), rainfall_csv=str(rain), output_parent=str(tmp_path),
        vertical_datum='NZVD2016', rainfall_units='seconds_m_s', rainfall_end='finite_support',
        duration_s=2.3, interval_s=1., boundary='closed', device='cpu')


def coefficient(*, config, path, values, shift=False):
    from affine import Affine
    with rasterio.open(config.dem) as source:
        profile = source.profile.copy()
        profile.update(transform=source.transform @ Affine.translation(xoff=int(shift), yoff=0.))
        with rasterio.open(path, 'w', **profile) as target:
            target.write(np.broadcast_to(values, source.shape).astype(np.float64), 1)


def test_spatial_roughness_and_capacity_match_direct_api(spatial_inputs, tmp_path):
    roughness, infiltration = tmp_path/'n.tif', tmp_path/'infil.tif'
    coefficient(config=spatial_inputs, path=roughness, values=np.linspace(.02, .08, 5))
    coefficient(config=spatial_inputs, path=infiltration, values=np.linspace(0., 5., 5))
    config = replace(spatial_inputs, roughness_mode='map', roughness_map=str(roughness),
        infiltration_mode='map', infiltration_map=str(infiltration))
    prepared = prepare_run(config=config)
    actual = create_run_model(prepared=prepared)
    direct = create_model(terrain=prepared.terrain, solver_method='fv1', manning_n=prepared.roughness,
        infiltration=prepared.infiltration, rainfall_series=(prepared.times_s, prepared.rates_m_s),
        rainfall_end_behavior='finite_support', boundary_qx=prepared.qx, boundary_qy=prepared.qy)
    for model in (actual, direct):
        advance(model=model, end_time=config.duration_s)
    np.testing.assert_array_equal(actual=actual.solver.get_conservative_state(), desired=direct.solver.get_conservative_state())
    budget = diagnostics(model=actual)
    assert abs(budget['residual_m3']) < 1.e-11
    assert budget['rain'] == pytest.approx(.001*15*25, abs=1.e-13)
    assert prepared.summary['roughness_range'] == [.02, .08]
    assert prepared.summary['infiltration_range_mm_h'] == pytest.approx([0., 5.])
    assert prepared.summary['inputs']['roughness']['sha256'] == file_identity(path=roughness)['sha256']
    save_checkpoint(model=actual, path=tmp_path/'checkpoint.npz')
    coefficient(config=spatial_inputs, path=roughness, values=.09)
    changed = create_run_model(prepared=prepare_run(config=config))
    with pytest.raises(ValueError, match='provenance'):
        load_checkpoint(model=changed, path=tmp_path/'checkpoint.npz')


@pytest.mark.parametrize(('mode', 'value', 'shift'), [('roughness', 0., False), ('roughness', np.nan, False),
    ('roughness', .03, True), ('infiltration', -1., False), ('infiltration', np.nan, False)])
def test_invalid_spatial_fields(spatial_inputs, tmp_path, mode, value, shift):
    path = tmp_path/'field.tif'
    coefficient(config=spatial_inputs, path=path, values=value, shift=shift)
    config = replace(spatial_inputs, roughness_mode='map', roughness_map=str(path)) if mode == 'roughness' else replace(spatial_inputs, infiltration_mode='map', infiltration_map=str(path))
    with pytest.raises(ValueError):
        prepare_run(config=config)


def test_explicit_boundary_stage_and_map_coordinates(spatial_inputs, tmp_path):
    path = tmp_path/'boundaries.csv'
    path.write_text('axis,row,col,type,stage_m\nqy,3,2,stage,0.02\nqx,0,0,free,\n')
    prepared = prepare_run(config=replace(spatial_inputs, boundary='custom', boundary_csv=str(path), boundary_stage_datum='NZVD2016'))
    assert prepared.qy[3, 2] == 3
    assert prepared.qy[3, 1] == 1
    assert prepared.summary['stage_faces'] == 1
    model = create_run_model(prepared=prepared)
    advance(model=model, end_time=.1)
    assert diagnostics(model=model)['boundary_inflow'] > 0.
    assert abs(diagnostics(model=model)['residual_m3']) < 1.e-11
    features = boundary_features(terrain=prepared.terrain, qx=prepared.qx, qy=prepared.qy)
    stage = next(feature for feature in features['features'] if feature['properties']['closure'] == 'stage')
    from pyproj import Transformer
    converter = Transformer.from_crs(crs_from='EPSG:2193', crs_to='EPSG:4326', always_xy=True)
    expected = converter.transform(xx=1574940.+10., yy=5167275.-5.)
    np.testing.assert_allclose(actual=stage['geometry']['coordinates'][0], desired=expected, rtol=0., atol=1.e-10)
    overlay = map_overlay(terrain=prepared.terrain, values=prepared.terrain.elevation, label='Terrain')
    assert overlay['shape'][0] <= 768 and overlay['shape'][1] <= 768
    assert overlay['bounds'][0][0] < overlay['bounds'][1][0]
    assert overlay['display_crs'] == 'EPSG:3857'
    assert 'axis,row,col,type,stage_m' in boundary_template(terrain=prepared.terrain)


@pytest.mark.parametrize('rows', ['qx,-1,0,free,\n', 'qx,0,99,free,\n', 'qx,0,1,free,\n',
    'qy,3,2,stage,\n', 'qy,3,2,stage,nan\n', 'qy,3,2,free,1\n',
    'qy,3,2,free,\nqy,3,2,closed,\n'])
def test_boundary_rules_rejected(spatial_inputs, tmp_path, rows):
    path = tmp_path/'boundary.csv'
    path.write_text('axis,row,col,type,stage_m\n'+rows)
    with pytest.raises(ValueError):
        prepare_run(config=replace(spatial_inputs, boundary='custom', boundary_csv=str(path), boundary_stage_datum='NZVD2016'))


def test_stage_datum_is_explicit(spatial_inputs, tmp_path):
    path = tmp_path/'coast.csv'
    path.write_text('axis,row,col,type,stage_m\nqy,3,2,stage,0\n')
    with pytest.raises(ValueError, match='datum'):
        prepare_run(config=replace(spatial_inputs, boundary='custom', boundary_csv=str(path)))


def test_finite_support_is_explicit_and_restart_bound(spatial_inputs, tmp_path):
    terrain = read_terrain(path=Path(spatial_inputs.dem), vertical_datum='NZVD2016')
    forcing = ([0., 1.], [.001, .001])
    ordinary = create_model(terrain=terrain, rainfall_series=forcing)
    with pytest.raises(ValueError, match='cover simulation'):
        advance(model=ordinary, end_time=2.)
    finite = create_model(terrain=terrain, rainfall_series=forcing, rainfall_end_behavior='finite_support')
    advance(model=finite, end_time=2.)
    np.testing.assert_allclose(actual=finite.solver.get_water_depth()[terrain.active], desired=.001, rtol=1.e-13)
    save_checkpoint(model=finite, path=tmp_path/'finite.npz')
    with pytest.raises(ValueError, match='provenance'):
        load_checkpoint(model=ordinary, path=tmp_path/'finite.npz')


def test_preset_conversion_preserves_exact_original_arrays(spatial_inputs, tmp_path):
    source, output = tmp_path/'archive', tmp_path/'converted'
    (source/'inputs').mkdir(parents=True)
    (source/'outputs').mkdir()
    import shutil
    shutil.copyfile(src=spatial_inputs.dem, dst=source/'outputs/orton_bradley_dem_5m.tif')
    terrain = read_terrain(path=Path(spatial_inputs.dem), vertical_datum='NZVD2016')
    shape = terrain.active.shape
    fields = np.zeros((6, *shape))
    fields[0] = terrain.active
    fields[1] = .543210987654321
    fields[2] = .000012345678901234
    fields[3] = .123456789012345
    fields[4] = .000001234567890123
    fields[5, 3, 2] = 1.
    times = np.array([0., 1.])
    rates = np.array([.000012345678901234, .000023456789012345])
    with (source/'inputs/inputs_f64.bin').open('wb') as stream:
        np.array([*shape, 2], dtype='<i4').tofile(file=stream)
        times.tofile(file=stream); rates.tofile(file=stream)
        fields[:, ::-1].copy().tofile(file=stream)
    np.savetxt(fname=source/'inputs/roughness.asc', X=np.full(shape, .035),
        header='header\nheader\nheader\nheader\nheader\nheader', comments='')
    names = ['inputs/inputs_f64.bin', 'inputs/roughness.asc', 'outputs/orton_bradley_dem_5m.tif']
    digests = {name: file_identity(path=source/name)['sha256'] for name in names}
    manifest = convert_orton_inputs(directory=source, output=output, expected_digests=digests)
    assert manifest['coastal_stage_faces'] == 1
    from floodwarp.inputs import read_aligned_field
    for name, index in [('multiplier',1), ('infiltration_f0',2), ('infiltration_fd',3), ('infiltration_k',4)]:
        values = read_aligned_field(path=output/(name+'.tif'), terrain=terrain)
        np.testing.assert_array_equal(actual=values[terrain.active], desired=fields[index][terrain.active])
    actual_times, actual_rates = read_hyetograph(path=output/'rainfall.csv', duration_s=3., units='seconds_m_s', end_behavior='finite_support')
    np.testing.assert_array_equal(actual=actual_times, desired=times)
    np.testing.assert_array_equal(actual=actual_rates, desired=rates)
    assert convert_orton_inputs(directory=source, output=output, expected_digests=digests) == manifest
    (output/'rainfall.csv').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        convert_orton_inputs(directory=source, output=output, expected_digests=digests)


def test_exponential_spatial_soils_and_invalid_order(spatial_inputs, tmp_path):
    paths = {name: tmp_path/(name+'.tif') for name in ('f0','fd','k')}
    coefficient(config=spatial_inputs, path=paths['f0'], values=np.linspace(5., 10., 5))
    coefficient(config=spatial_inputs, path=paths['fd'], values=.2)
    coefficient(config=spatial_inputs, path=paths['k'], values=2.)
    config = replace(spatial_inputs, infiltration_mode='exponential',
        infiltration_f0_map=str(paths['f0']), infiltration_fd_map=str(paths['fd']), infiltration_k_map=str(paths['k']))
    prepared = prepare_run(config=config)
    actual = create_run_model(prepared=prepared)
    reference = create_model(terrain=prepared.terrain, rainfall_series=(prepared.times_s,prepared.rates_m_s),
        rainfall_end_behavior='finite_support', manning_n=prepared.roughness,
        boundary_qx=prepared.qx, boundary_qy=prepared.qy)
    reference.solver.set_exponential_infiltration(**prepared.exponential_soil)
    for model in (actual, reference):
        advance(model=model, end_time=2.3)
    np.testing.assert_array_equal(actual=actual.solver.get_water_depth(), desired=reference.solver.get_water_depth())
    np.testing.assert_array_equal(actual=actual.solver.cumulative_infiltration_m(), desired=reference.solver.cumulative_infiltration_m())
    assert diagnostics(model=actual)['infiltration'] > 0.
    assert abs(diagnostics(model=actual)['residual_m3']) < 1.e-11
    coefficient(config=spatial_inputs, path=paths['k'], values=20.)
    with pytest.raises(ValueError, match='f0 >='):
        prepare_run(config=config)


def test_independent_checkpoint_cadence_and_final_save(spatial_inputs, tmp_path, monkeypatch):
    import floodwarp.ui_runner as runner
    original_save = runner.save_checkpoint
    saved_times = []

    def record_checkpoint(*, model, path):
        saved_times.append(model.solver.time)
        return original_save(model=model,path=path)

    monkeypatch.setattr(target=runner,name='save_checkpoint',value=record_checkpoint)
    runner.execute_run(config=replace(spatial_inputs,checkpoint_interval_s=10.),output=tmp_path/'run')
    assert saved_times == [0.,2.3]
    assert json.loads((tmp_path/'run/run_config.json').read_text())['actual_final_time_s'] == 2.3
