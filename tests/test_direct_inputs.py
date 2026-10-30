from pathlib import Path

import numpy as np
import pytest

from floodwarp import ExperimentalSGSWarning
from floodwarp.inputs import terrain_from_array, boundary_faces, read_terrain, write_raster, read_aligned_field
from floodwarp.model import create_model, advance, diagnostics, save_checkpoint, load_checkpoint


def terrain():
    return terrain_from_array(elevation=np.zeros((4, 5)), dx=2., dy=2.,
                              row_order='south_to_north', vertical_datum='local')


@pytest.mark.parametrize('method', ['li', 'fv1'])
def test_array_rain_mass_and_disk_restart(method, tmp_path):
    grid = terrain()
    options = dict(terrain=grid, solver_method=method, rainfall_series=([0., .5, 1.], [0., .001, 0.]),
                   explicit_maximum_dt=.05)
    model = create_model(**options)
    advance(model=model, end_time=.5)
    checkpoint = tmp_path/'state.npz'
    save_checkpoint(model=model, path=checkpoint)
    fresh = create_model(**options)
    load_checkpoint(model=fresh, path=checkpoint)
    advance(model=model, end_time=1.)
    advance(model=fresh, end_time=1.)
    np.testing.assert_array_equal(model.solver.get_water_depth(), fresh.solver.get_water_depth())
    budget = diagnostics(model=model)
    assert abs(budget['residual_m3']) < 1.e-12
    assert budget['rain'] == pytest.approx(.04, abs=1.e-12)
    assert budget['numerical'].get('validation_profile', 'physical_candidate') == 'physical_candidate'


@pytest.mark.parametrize('mode', ['traditional', 'enhanced'])
def test_sgs_warning_mass_and_restart(mode, tmp_path):
    options = dict(terrain=terrain(), hydraulic_geometry_mode=mode,
        subgrid_bed_elevation=np.zeros((8, 10)), subgrid_resolution_m=1.,
        rainfall=.001, explicit_maximum_dt=.01)
    with pytest.warns(ExperimentalSGSWarning, match=mode):
        model = create_model(**options)
    advance(model=model, end_time=.1)
    save_checkpoint(model=model, path=tmp_path/'sgs.npz')
    with pytest.warns(ExperimentalSGSWarning):
        fresh = create_model(**options)
    load_checkpoint(model=fresh, path=tmp_path/'sgs.npz')
    advance(model=model, end_time=.2)
    advance(model=fresh, end_time=.2)
    np.testing.assert_array_equal(model.solver.get_water_depth(), fresh.solver.get_water_depth())
    assert abs(diagnostics(model=model)['residual_m3']) < 1.e-12
    assert diagnostics(model=model)['numerical']['hydraulic_geometry_mode'] == mode
    assert model.solver.validation_profile == 'default'


@pytest.mark.parametrize('method', ['li', 'fv1'])
def test_partial_stage_edge_and_invalid_geometry(method):
    grid = terrain()
    x, y = boundary_faces(terrain=grid)
    x[1:3, -1] = 3
    with pytest.raises(ValueError, match='datum'):
        create_model(terrain=grid, solver_method=method, boundary_qx=x, boundary_qy=y)
    model = create_model(terrain=grid, solver_method=method, boundary_qx=x, boundary_qy=y,
        stage_datum='local', stage_series_qx=([0., .1], [.05, .1]), explicit_maximum_dt=.01)
    advance(model=model, end_time=.1)
    assert diagnostics(model=model)['boundary_inflow'] > 0
    assert abs(diagnostics(model=model)['residual_m3']) < 1.e-11
    x[0, 1] = 2
    with pytest.raises(ValueError, match='interface'):
        create_model(terrain=grid, boundary_qx=x)


def test_orientation_nodata_and_ambiguous_inputs():
    grid = terrain_from_array(elevation=np.array([[1., 2.], [3., -9999.]]), dx=2., dy=3.,
                              row_order='north_to_south', vertical_datum='local', nodata=-9999.)
    np.testing.assert_array_equal(grid.elevation, [[3., -9999.], [1., 2.]])
    np.testing.assert_array_equal(grid.active, [[True, False], [True, True]])
    with pytest.raises(ValueError, match='ambiguous'):
        create_model(terrain=grid, manning_n=np.ones((1, 2))*.03)
    with pytest.raises(ValueError, match='never both'):
        create_model(terrain=grid, initial_depth=.1, initial_stage=.2)


def test_geotiff_roundtrip_and_shift_rejection(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    path = tmp_path/'terrain.tif'
    values = np.array([[1., 2.], [3., -9999.]])
    with rasterio.open(path, 'w', driver='GTiff', count=1, width=2, height=2,
            dtype='float64', crs='EPSG:2193', transform=from_origin(100., 200., 2., 3.), nodata=-9999.) as target:
        target.write(values, 1)
        target.update_tags(vertical_datum='NZVD2016')
    grid = read_terrain(path=path, vertical_datum='NZVD2016')
    write_raster(path=tmp_path/'roundtrip.tif', terrain=grid, values=grid.elevation, quantity='elevation')
    np.testing.assert_array_equal(read_aligned_field(path=tmp_path/'roundtrip.tif', terrain=grid), grid.elevation)
    with pytest.raises(ValueError, match='datum'):
        read_terrain(path=path, vertical_datum='local')
    with rasterio.open(tmp_path/'roundtrip.tif', 'r+') as target:
        target.transform = from_origin(101., 200., 2., 3.)
    with pytest.raises(ValueError, match='geometry'):
        read_aligned_field(path=tmp_path/'roundtrip.tif', terrain=grid)


@pytest.mark.parametrize('method', ['li', 'fv1'])
def test_inactive_perimeter_mask_and_independent_snapshots(method, tmp_path):
    from examples.common import run_example
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    grid = terrain_from_array(elevation=np.zeros((4, 5)), active=mask, dx=2., dy=2.,
                              row_order='south_to_north', vertical_datum='local')
    model = create_model(terrain=grid, solver_method=method, rainfall=.001,
                         explicit_maximum_dt=.01)
    run_example(model=model, output=tmp_path, sample_times=np.array([.1, .2]),
                make_plots=False, use_cuda_graph=False)
    with np.load(tmp_path/'depths.npz') as data:
        np.testing.assert_allclose(data['depth'][0][mask], .0001, atol=1.e-14)
        np.testing.assert_allclose(data['depth'][1][mask], .0002, atol=1.e-14)
    assert abs(diagnostics(model=model)['residual_m3']) < 1.e-12


@pytest.mark.parametrize('method', ['li', 'fv1'])
def test_restart_rejects_numerical_drift_before_mutation(method, tmp_path):
    model = create_model(terrain=terrain(), solver_method=method, rainfall=.001)
    advance(model=model, end_time=.1)
    save_checkpoint(model=model, path=tmp_path/'state.npz')
    fresh = create_model(terrain=terrain(), solver_method=method, rainfall=.001)
    fresh.solver.gravity = type(fresh.solver.gravity)(9.)
    before = fresh.solver.get_water_depth().copy()
    with pytest.raises(ValueError, match='provenance'):
        load_checkpoint(model=fresh, path=tmp_path/'state.npz')
    np.testing.assert_array_equal(before, fresh.solver.get_water_depth())
