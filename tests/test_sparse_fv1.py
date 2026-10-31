"""Execution equivalence, activation, complete donor budgets and restart."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from floodwarp import create_overland_solver
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model


def make_public_solver(*, entrypoint, device, execution_options):
    shape = (73, 91)
    bed = np.zeros(shape=shape)
    if entrypoint == 'model':
        terrain = terrain_from_array(elevation=bed, dx=1., dy=1.3,
            row_order='south_to_north', vertical_datum='local')
        solver = create_model(terrain=terrain, solver_method='fv1', device=device,
            explicit_minimum_dt=0., explicit_maximum_dt=.1, **execution_options).solver
    else:
        common = dict(ny=shape[0], nx=shape[1], dx=1., dy=1.3, bed_elevation=bed,
            manning_n=.03, active_mask=np.ones(shape=shape, dtype=bool), device=device)
        if entrypoint == 'native':
            solver = FullSWESolver(**common, maximum_dt=.1, **execution_options)
        else:
            solver = create_overland_solver(solver_method='fv1', **common,
                explicit_minimum_dt=0., explicit_maximum_dt=.1, **execution_options)
    depth = np.zeros(shape=shape)
    depth[29:35, 37:43] = .7
    depth[38, 47] = 1.e-10
    solver.set_water_depth(depth=depth)
    return solver


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
@pytest.mark.parametrize('entrypoint', ('native', 'factory', 'model'))
def test_default_sparse_matches_explicit_dense_and_restart(device, entrypoint, tmp_path):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    default = make_public_solver(entrypoint=entrypoint, device=device, execution_options={})
    dense = make_public_solver(entrypoint=entrypoint, device=device,
        execution_options=dict(execution_mode='dense'))
    assert dense.execution_diagnostics()['execution'] == 'dense_uniform'
    for solver in (default, dense):
        solver.run_device_loop(duration=.1)
    activity = default.execution_diagnostics()
    assert activity['execution'] == 'sparse_uniform'
    assert activity['tile_size'] == 8
    assert not activity['dense_fallback']
    assert activity['evaluated_cells'] < default.nx * default.ny
    assert_equivalent(dense=dense, sparse=default)
    checkpoint = tmp_path / 'default.npz'
    default.save_checkpoint(path=checkpoint)
    restarted = make_public_solver(entrypoint=entrypoint, device=device,
        execution_options=dict(execution_mode='dense'))
    restarted.load_checkpoint(path=checkpoint)
    dense_checkpoint = tmp_path / 'dense.npz'
    dense.save_checkpoint(path=dense_checkpoint)
    restarted_default = make_public_solver(entrypoint=entrypoint, device=device, execution_options={})
    restarted_default.load_checkpoint(path=dense_checkpoint)
    for solver in (default, dense, restarted, restarted_default):
        solver.run_device_loop(duration=.1)
        assert solver.time == .2
    assert_equivalent(dense=dense, sparse=default)
    assert_equivalent(dense=restarted, sparse=default)
    assert_equivalent(dense=dense, sparse=restarted_default)


def make_solver(*, device, case, mode, tile_size, dense_fraction):
    shape = (73, 91)
    row, column = np.indices(dimensions=shape)
    active = np.ones(shape=shape, dtype=bool)
    bed = .002 * row + .001 * column
    depth = np.zeros(shape=shape)
    if case in ('front', 'masked', 'heterogeneous'):
        depth[29:35, 37:43] = .7
        depth[38, 47] = 1.e-10
    elif case == 'lake':
        bed = np.where((row - 31)**2 + (column - 39)**2 < 100, .2 + .01 * (column % 4), 1.)
        depth = np.maximum(.6 - bed, 0.)
    elif case == 'shallow':
        depth[20:25, 32:38] = 1.e-8
    if case == 'masked':
        active[15:55, 45:49] = False
        active[:4] = False
    solver = FullSWESolver(ny=shape[0], nx=shape[1], dx=1., dy=1.3,
        bed_elevation=bed, manning_n=.03, active_mask=active, device=device, maximum_dt=.1,
        execution_mode=mode, tile_size=tile_size, dense_fraction=dense_fraction)
    solver.set_water_depth(depth=depth)
    if case in ('front', 'masked', 'heterogeneous'):
        hu = .3 * depth
        hv = -.2 * depth
        hu[depth <= solver.dry_depth] = 0.
        hv[depth <= solver.dry_depth] = 0.
        solver.set_momentum(hu=hu, hv=hv)
    if case == 'stage':
        bx = np.zeros(shape=(shape[0], shape[1] + 1), dtype=np.int32)
        by = np.zeros(shape=(shape[0] + 1, shape[1]), dtype=np.int32)
        bx[10:15, 0] = 3
        bx[45:50, -1] = 3
        by[0, 20:25] = 3
        by[-1, 60:65] = 3
        solver.set_boundary_type_qx(boundary_type=bx)
        solver.set_boundary_type_qy(boundary_type=by)
        solver.set_boundary_stage_qx(stage=np.full(shape=bx.shape, fill_value=.8))
        solver.set_boundary_stage_qy(stage=np.full(shape=by.shape, fill_value=.8))
    if case in ('infiltrating', 'ponding', 'heterogeneous'):
        solver.set_uniform_rainfall_rate(rainfall_rate=1.e-3)
        capacity = 2.e-3 if case == 'infiltrating' else 2.e-4
        if case == 'heterogeneous':
            capacity = np.where(column % 3 == 0, .002, .0002)
        solver.set_exponential_infiltration(f0=capacity, fd=.01, conductivity=capacity)
    return solver


def assert_equivalent(*, dense, sparse):
    assert dense.time == sparse.time
    assert dense.steps == sparse.steps
    for name, reference in dense.capture_state()['arrays'].items():
        np.testing.assert_allclose(actual=sparse._fields[name].numpy(), desired=reference,
                                   rtol=2.e-14, atol=2.e-14, err_msg=name)
    assert dense.numerical_configuration() == sparse.numerical_configuration()
    assert dense.cumulative_accounting() == sparse.cumulative_accounting()


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
@pytest.mark.parametrize('case', ('front', 'masked', 'lake', 'shallow', 'stage',
                                 'infiltrating', 'ponding', 'heterogeneous'))
@pytest.mark.parametrize('tile_size', (8, 16, 32))
def test_execution_equivalence(device, case, tile_size):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    dense = make_solver(device=device, case=case, mode='dense', tile_size=tile_size, dense_fraction=1.)
    sparse = make_solver(device=device, case=case, mode='sparse', tile_size=tile_size, dense_fraction=1.)
    for dt in (.001, .003, .002, .004):
        assert dense.step(fixed_dt=dt) == sparse.step(fixed_dt=dt)
        assert_equivalent(dense=dense, sparse=sparse)
    dense.run_device_loop(duration=.3, use_cuda_graph=False)
    sparse.run_device_loop(duration=.3, use_cuda_graph=False)
    assert_equivalent(dense=dense, sparse=sparse)
    if case == 'shallow':
        assert sparse.get_water_depth().sum() > 0.
    if case == 'infiltrating':
        assert sparse.execution_diagnostics()['work_tiles'] == 0
        assert sparse.cumulative_infiltration_m().sum() > 0.
    if case == 'lake':
        hu, hv = sparse.get_momentum()
        assert np.max(np.abs(hu)) < 1.e-15
        assert np.max(np.abs(hv)) < 1.e-15


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_deactivation_reactivation_graph_restart_and_switch(device, tmp_path: Path):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    dense = make_solver(device=device, case='front', mode='dense', tile_size=8, dense_fraction=.01)
    sparse = make_solver(device=device, case='front', mode='sparse', tile_size=8, dense_fraction=.01)
    for solver in (dense, sparse):
        solver.run_device_loop(duration=.2)
        solver.set_source_rates(infil=100.)
        solver.run_device_loop(duration=.2)
    assert_equivalent(dense=dense, sparse=sparse)
    assert sparse.execution_diagnostics()['work_tiles'] == 0
    maxima = sparse.get_max_water_depth()
    volumes = sparse.get_cumulative_face_volumes()
    assert np.max(maxima) > 0.
    assert np.max(np.abs(volumes['qx'])) > 0.
    checkpoint = tmp_path / 'restart.npz'
    sparse.save_checkpoint(path=checkpoint)
    restarted = make_solver(device=device, case='front', mode='sparse', tile_size=16, dense_fraction=.7)
    restarted.load_checkpoint(path=checkpoint)
    for solver in (dense, sparse, restarted):
        solver.set_source_rates(infil=0.)
        solver.set_uniform_rainfall_rate(rainfall_rate=.002)
        solver.run_device_loop(duration=.4)
    assert_equivalent(dense=dense, sparse=sparse)
    assert_equivalent(dense=dense, sparse=restarted)
    assert sparse.execution_diagnostics()['dense_fallback']
    sparse.configure_execution(mode='dense')
    dense.run_device_loop(duration=.1)
    sparse.run_device_loop(duration=.1)
    assert_equivalent(dense=dense, sparse=sparse)


def test_execution_options_reject_invalid_settings():
    solver = make_solver(device='cpu', case='front', mode='dense', tile_size=8, dense_fraction=.7)
    with pytest.raises(ValueError, match='execution mode'):
        solver.configure_execution(mode='amr')
    with pytest.raises(ValueError, match='tile_size'):
        solver.configure_execution(mode='sparse', tile_size=4)
    with pytest.raises(ValueError, match='dense_fraction'):
        solver.configure_execution(mode='sparse', dense_fraction=0.)


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_fully_infiltrating_rain_retains_source_aware_cfl(device):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    dense = make_solver(device=device, case='infiltrating', mode='dense', tile_size=8, dense_fraction=.65)
    sparse = make_solver(device=device, case='infiltrating', mode='sparse', tile_size=8, dense_fraction=.65)
    dense.maximum_dt = 100.
    sparse.maximum_dt = 100.
    dt = dense.step()
    assert dt == sparse.step()
    assert dt < 1.
    assert sparse.execution_diagnostics()['work_tiles'] == 0
    assert not sparse.get_water_depth().any()
    assert sparse.cumulative_infiltration_m().sum() > 0.
    assert_equivalent(dense=dense, sparse=sparse)
