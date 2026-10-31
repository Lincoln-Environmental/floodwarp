"""Static leaves, conservative subfaces, transfers, geometry vetoes and restart."""

import numpy as np
import pytest
import warp as wp

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.static_fv1 import StaticBlockFV1Solver
from floodwarp.static_fv1_mesh import restrict_state, prolong_state
from floodwarp import static_fv1_kernels as kernels


def make_uniform(*, case, device):
    shape = (64, 64)
    row, column = np.indices(dimensions=shape)
    bed = np.zeros(shape=shape)
    depth = np.full(shape=shape, fill_value=.4)
    protected = np.zeros(shape=shape, dtype=bool)
    if case == 'lake':
        bed = .002 * row + .003 * column + np.where(column >= 32, .1, 0.)
        depth = .8 - bed
        protected[:, 31:33] = True
    elif case == 'front':
        depth[:, 32:] = 0.
        protected[:, 24:40] = True
    elif case in ('ponding', 'infiltrating', 'heterogeneous'):
        depth.fill(0.)
    elif case == 'disconnected':
        bed[:, 30:34] = 2.
        depth = np.maximum(.4 - bed, 0.)
        protected[:, 30:34] = True
    solver = FullSWESolver(ny=shape[0], nx=shape[1], dx=1., dy=1., bed_elevation=bed,
        manning_n=.03, active_mask=np.ones(shape=shape, dtype=bool), device=device, maximum_dt=.05)
    solver.set_water_depth(depth=depth)
    if case == 'front':
        solver.set_momentum(hu=.2 * depth, hv=.1 * depth)
    if case in ('ponding', 'infiltrating', 'heterogeneous'):
        solver.set_uniform_rainfall_rate(rainfall_rate=.001)
        capacity = .002 if case == 'infiltrating' else .0002
        if case == 'heterogeneous':
            capacity = np.where(column % 3 == 0, .002, .0002)
        solver.set_exponential_infiltration(f0=capacity, fd=.01, conductivity=capacity)
    return solver, protected


def make_static(*, solver, protected, requested_roots=None):
    return StaticBlockFV1Solver.from_solver(solver=solver, block_size=4, protected=protected,
        surface_tolerance=1.e-12, momentum_tolerance=1.e-12, geometry_tolerance=1.e-12,
        source_tolerance=0., requested_roots=requested_roots)


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
@pytest.mark.parametrize('case', ('lake', 'front', 'ponding', 'infiltrating', 'heterogeneous', 'disconnected'))
def test_static_physical_controls(case, device):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    uniform, protected = make_uniform(case=case, device=device)
    static = make_static(solver=uniform, protected=protected)
    initial = static.get_conservative_state()
    diag = static.execution_diagnostics()
    if case == 'heterogeneous':
        assert diag['coarse_roots'] == 0
    else:
        assert diag['coarse_leaves'] > 0
        assert diag['hydraulic_leaves'] < uniform.nx * uniform.ny
    for dt in (.01, .015, .02, .005):
        uniform.step(fixed_dt=dt)
        static.step(fixed_dt=dt)
    static.run_device_loop(duration=.15)
    uniform.run_device_loop(duration=.15)
    assert np.isfinite(static.get_conservative_state()).all()
    assert static.get_water_depth().min() >= 0.
    balance = static.cumulative_accounting()
    assert abs(balance['residual']) < 1.e-8 * max(1., static.initial_volume + balance['rain'])
    if case in ('lake', 'disconnected'):
        np.testing.assert_allclose(actual=static.get_conservative_state(), desired=initial, rtol=0., atol=1.e-12)
    if case in ('lake', 'ponding', 'infiltrating', 'heterogeneous', 'disconnected'):
        np.testing.assert_allclose(actual=static.get_conservative_state(), desired=uniform.get_conservative_state(), rtol=0., atol=1.e-12)
        np.testing.assert_allclose(actual=static.cumulative_infiltration_m(), desired=uniform.cumulative_infiltration_m(), rtol=0., atol=1.e-12)


def test_transfer_conservation_equilibrium_and_vetoes():
    uniform, protected = make_uniform(case='lake', device='cpu')
    static = make_static(solver=uniform, protected=protected)
    mesh = static._static_execution.mesh
    fine = uniform.q.numpy().copy()
    leaves = restrict_state(q=fine, mesh=mesh)
    np.testing.assert_allclose(actual=np.sum(leaves * mesh.counts[:, None], axis=0), desired=fine.sum(axis=0), rtol=1.e-14, atol=1.e-12)
    projected = prolong_state(leaves=leaves, mesh=mesh, bed=uniform._fields['bed'].numpy(), dry_depth=uniform.dry_depth)
    np.testing.assert_allclose(actual=projected, desired=fine, rtol=0., atol=1.e-15)
    coarse = np.flatnonzero(mesh.counts > 1)
    leaves[coarse[0], 0] = .0001
    with pytest.raises(ValueError, match='mixed wet/dry'):
        prolong_state(leaves=leaves, mesh=mesh, bed=uniform._fields['bed'].numpy(), dry_depth=uniform.dry_depth)
    requested = np.ones_like(mesh.coarse_roots)
    with pytest.raises(ValueError, match='veto'):
        make_static(solver=uniform, protected=protected, requested_roots=requested)


def test_shared_coarse_donor_budget_all_orientations():
    uniform, protected = make_uniform(case='flat', device='cpu')
    static = make_static(solver=uniform, protected=protected)
    engine = static._static_execution
    mesh, s, f = engine.mesh, engine.fields, static._fields
    leaf = int(mesh.owner[24, 24])
    assert mesh.counts[leaf] == 4
    flux_x = np.zeros(shape=(static.ny, static.nx + 1, 3))
    flux_y = np.zeros(shape=(static.ny + 1, static.nx, 3))
    flux_x[24:26, 24, 0] = -1.
    flux_x[24:26, 26, 0] = 1.
    flux_y[24, 24:26, 0] = -1.
    flux_y[26, 24:26, 0] = 1.
    f['flux_x'].assign(flux_x.reshape(-1, 3))
    f['flux_y'].assign(flux_y.reshape(-1, 3))
    ctrl = np.zeros(shape=static.ctrl.size)
    ctrl[2] = 1.
    static.ctrl.assign(ctrl)
    s['delta'].zero_()
    s['outgoing'].zero_()
    for axis, index in (('x', 0), ('y', 1)):
        wp.launch(kernels.collect_outflows, dim=s['faces_' + axis].size,
            inputs=[s['faces_' + axis], f['flux_' + axis], s['owner'], s['outgoing'], 64, 64, index, 1.], device='cpu')
    wp.launch(kernels.donor_budgets, dim=s['leaves'].size,
        inputs=[s['leaves'], s['counts'], s['outgoing'], s['theta'], static.ctrl, 1.], device='cpu')
    assert s['outgoing'].numpy()[leaf] == 8.
    assert s['theta'].numpy()[leaf] == pytest.approx(.2)
    for axis, index in (('x', 0), ('y', 1)):
        wp.launch(kernels.accept_subfaces, dim=s['faces_' + axis].size,
            inputs=[s['faces_' + axis], f['flux_' + axis], f['pressure_' + axis], s['owner'], s['theta'], s['delta'],
                    f['volume_' + axis], f['inflow'], f['outflow'], static.ctrl, 64, 64, index, 1.], device='cpu')
    delta = s['delta'].numpy()
    assert delta[leaf, 0] == pytest.approx(-1.6)
    assert np.sum(delta[:, 0]) == pytest.approx(0., abs=1.e-15)
    accepted = s['leaves'].numpy()[:, 0] + delta[:, 0] / mesh.counts
    assert accepted.min() >= -1.e-15


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_static_disk_restart_and_identity(tmp_path, device):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    uniform, protected = make_uniform(case='front', device=device)
    first = make_static(solver=uniform, protected=protected)
    second = make_static(solver=uniform, protected=protected)
    first.run_device_loop(duration=.2, use_cuda_graph=False)
    path = tmp_path / 'static.npz'
    first.save_checkpoint(path=path)
    second.load_checkpoint(path=path)
    first.run_device_loop(duration=.3, use_cuda_graph=False)
    second.run_device_loop(duration=.3, use_cuda_graph=False)
    np.testing.assert_allclose(actual=first.get_conservative_state(), desired=second.get_conservative_state(), rtol=2.e-14, atol=2.e-13)
    for name in first._fields:
        np.testing.assert_allclose(actual=first._fields[name].numpy(), desired=second._fields[name].numpy(), rtol=2.e-14, atol=2.e-13)
    with pytest.raises(ValueError, match='not qualified'):
        first.configure_execution(mode='sparse')


def test_source_changes_cannot_bypass_coarsening_vetoes():
    uniform, protected = make_uniform(case='flat', device='cpu')
    solver = make_static(solver=uniform, protected=protected)
    variable = np.ones(shape=uniform._shape)
    variable[10, 10] = 2.
    before = solver._fields['multiplier'].numpy().copy()
    with pytest.raises(ValueError, match='heterogeneity'):
        solver.set_rainfall_spatial_multiplier(multiplier=variable)
    np.testing.assert_array_equal(actual=solver._fields['multiplier'].numpy(), desired=before)
    with pytest.raises(ValueError, match='heterogeneity'):
        solver.set_source_rates(rain=variable)
    with pytest.raises(ValueError, match='heterogeneity'):
        solver.set_exponential_infiltration(f0=variable, fd=1., conductivity=0.)
    with pytest.raises(ValueError, match='initial state'):
        solver.set_water_depth(depth=.1)
    with pytest.raises(ValueError, match='initial state'):
        solver.set_momentum(hu=0., hv=0.)
    solver.set_rainfall_spatial_multiplier(multiplier=2.)


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_static_ordinary_cfl_controller(device):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    uniform, protected = make_uniform(case='front', device=device)
    uniform.maximum_dt = 1.
    solver = make_static(solver=uniform, protected=protected)
    result = solver.run_device_loop(duration=1.)
    assert result['dt_min_s'] < uniform.maximum_dt
    assert solver.steps > 1
    assert solver.get_water_depth().min() >= 0.
    assert abs(solver.cumulative_accounting()['residual']) < 1.e-9


def test_repeated_representable_transfers_preserve_momentum_and_water():
    uniform, protected = make_uniform(case='lake', device='cpu')
    solver = make_static(solver=uniform, protected=protected)
    mesh = solver._static_execution.mesh
    leaves = restrict_state(q=uniform.q.numpy(), mesh=mesh)
    leaves[:, 1] = .02 * leaves[:, 0]
    leaves[:, 2] = -.01 * leaves[:, 0]
    expected = np.sum(leaves * mesh.counts[:, None], axis=0)
    for cycle in range(32):
        fine = prolong_state(leaves=leaves, mesh=mesh, bed=uniform._fields['bed'].numpy(), dry_depth=uniform.dry_depth)
        assert fine[:, 0].min() >= 0.
        leaves = restrict_state(q=fine, mesh=mesh)
    np.testing.assert_allclose(actual=np.sum(leaves * mesh.counts[:, None], axis=0), desired=expected, rtol=1.e-14, atol=1.e-12)


@pytest.mark.parametrize('rotation', (0, 1, 2, 3))
def test_moving_front_crosses_interfaces_in_every_orientation(rotation):
    source, protected = make_uniform(case='front', device='cpu')
    state = source.get_conservative_state()
    depth = np.rot90(state[..., 0], k=rotation).copy()
    hu = np.rot90(state[..., 1], k=rotation).copy()
    hv = np.rot90(state[..., 2], k=rotation).copy()
    for turn in range(rotation):
        hu, hv = hv.copy(), -hu.copy()
    source.set_water_depth(depth=depth)
    source.set_momentum(hu=hu, hv=hv)
    protected = np.rot90(protected, k=rotation).copy()
    # Narrow the fixed fine band so the front reaches a coarse interface.
    protected.fill(False)
    if rotation % 2 == 0:
        protected[:, 28:36] = True
    else:
        protected[28:36, :] = True
    solver = make_static(solver=source, protected=protected)
    source.run_device_loop(duration=6., use_cuda_graph=False)
    solver.run_device_loop(duration=6., use_cuda_graph=False)
    assert solver.get_water_depth().min() >= 0.
    assert abs(solver.cumulative_accounting()['residual']) < 1.e-8
    relative = np.linalg.norm(solver.get_water_depth() - source.get_water_depth()) / np.linalg.norm(source.get_water_depth())
    assert relative < .02
    assert solver._static_execution.mesh.coarse_roots.any()


def test_mask_interfaces_stay_fine_without_explicit_boundary_codes():
    active = np.ones(shape=(64, 64), dtype=bool)
    active[24:32, 24:32] = False
    source = FullSWESolver(ny=64, nx=64, dx=1., dy=1., bed_elevation=np.zeros_like(active, dtype=float),
        manning_n=.03, active_mask=active, device='cpu', maximum_dt=.05)
    source.set_water_depth(depth=.4)
    solver = make_static(solver=source, protected=np.zeros_like(active))
    mesh = solver._static_execution.mesh
    assert mesh.level[mesh.owner[23, 26]] == 0
    assert mesh.level[mesh.owner[26, 23]] == 0
    assert mesh.level[mesh.owner[32, 26]] == 0
    assert mesh.level[mesh.owner[26, 32]] == 0
    assert int(mesh.counts.sum()) == int(active.sum())
    solver.run_device_loop(duration=.1, use_cuda_graph=False)
    np.testing.assert_allclose(actual=solver.get_water_depth()[active], desired=.4, rtol=0., atol=1.e-15)


def test_coarse_flat_bed_preserves_nonzero_shallow_source_water_at_large_datum():
    source = FullSWESolver(ny=64, nx=64, dx=1., dy=1., bed_elevation=900., manning_n=.03,
        active_mask=np.ones(shape=(64, 64), dtype=bool), device='cpu', maximum_dt=.05)
    source.set_water_depth(depth=0.)
    source.set_uniform_rainfall_rate(rainfall_rate=1.e-12)
    solver = make_static(solver=source, protected=np.zeros(shape=(64, 64), dtype=bool))
    assert solver.execution_diagnostics()['coarse_roots'] > 0
    source.step()
    solver.step()
    assert solver.get_water_depth().min() > 0.
    np.testing.assert_allclose(actual=solver.get_water_depth(), desired=source.get_water_depth(), rtol=1.e-14, atol=0.)


def test_auto_coarsening_keeps_unrepresentable_positive_children_fine():
    row, column = np.indices(dimensions=(64, 64))
    bed = .002 * (row + column)
    depth = 1. - bed
    depth[16:24, 16:24] = bed[23, 23] - .0004 - bed[16:24, 16:24]
    depth[23, 23] = .00015
    assert depth.min() > .0001
    source = FullSWESolver(ny=64, nx=64, dx=1., dy=1., bed_elevation=bed, manning_n=.03,
        active_mask=np.ones(shape=(64, 64), dtype=bool), device='cpu')
    source.set_water_depth(depth=depth)
    solver = StaticBlockFV1Solver.from_solver(solver=source, block_size=4,
        protected=np.zeros(shape=(64, 64), dtype=bool), surface_tolerance=.001,
        momentum_tolerance=.001, geometry_tolerance=1.e-12, source_tolerance=0.)
    assert not solver._static_execution.mesh.coarse_roots[2, 2]
    assert solver.execution_diagnostics()['veto_counts']['unrepresentable_transfer'] > 0
    np.testing.assert_array_equal(actual=solver.get_water_depth()[16:24, 16:24], desired=depth[16:24, 16:24])
