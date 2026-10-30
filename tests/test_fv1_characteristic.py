"""Analytical characteristic/regime checks, independent of LISFLOOD agreement."""
import numpy as np
import pytest
import warp as wp

from floodwarp import full_swe_kernels as kernels
from floodwarp.full_swe_reference import face_fluxes, numpy_step
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.solver_factory import create_overland_solver


@wp.kernel
def characteristic_samples(q: wp.array(dtype=wp.vec3d), h: wp.array(dtype=wp.float64),
                           g: wp.float64, dry: wp.float64, sign: wp.float64,
                           result: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    result[k] = kernels.stage_boundary_state(q[k], h[k], dry, 2, g, sign)


def make_solver(*, device, shape, depth, velocity, stage, side, roughness, cfl, maximum_dt, mode):
    solver = FullSWESolver(ny=shape[0], nx=shape[1], dx=1., dy=1., bed_elevation=np.zeros(shape),
        active_mask=np.ones(shape, bool), manning_n=roughness, gravity=9.80665,
        device=device, cfl=cfl, maximum_dt=maximum_dt, stage_boundary_momentum=mode)
    solver.set_water_depth(depth=np.full(shape, depth))
    solver.set_momentum(hu=np.full(shape, depth*velocity[0]), hv=np.full(shape, depth*velocity[1]))
    bx = np.zeros((shape[0], shape[1]+1), int); bx[:, [0, -1]] = 1
    by = np.zeros((shape[0]+1, shape[1]), int); by[[0, -1]] = 1
    if side == 'east': bx[:, -1] = 3
    elif side == 'west': bx[:, 0] = 3
    elif side == 'north': by[-1] = 3
    else: by[0] = 3
    sx, sy = np.full(bx.shape, stage), np.full(by.shape, stage)
    solver.set_boundary_type_qx(boundary_type=bx); solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=sx); solver.set_boundary_stage_qy(stage=sy)
    return solver, (bx, by, sx, sy)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_characteristic_invariant_regimes_and_tangent(device, sign):
    if device != 'cpu' and not wp.is_cuda_available(): pytest.skip('CUDA unavailable')
    g = 9.80665; c = np.sqrt(g)
    # Outflow, inflow with incoming transverse characteristic, supercritical
    # outflow, supercritical inflow, dry drainage, dry/dry, unsupported dry inflow.
    data = np.array([[1., sign*.5, .7], [1., -sign*.5, .7], [1., sign*4., .7],
        [1., -sign*4., .7], [1., 0., .7], [0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    depths = np.array([.9, 1.1, 20., 1., 0., 0., 1., 9.])
    result = wp.zeros(len(data), dtype=wp.vec3d, device=device)
    wp.launch(kernel=characteristic_samples, dim=len(data),
        inputs=[wp.array(data, dtype=wp.vec3d, device=device), wp.array(depths, dtype=wp.float64, device=device), g, 1e-4, sign, result], device=device)
    q = result.numpy()
    for k in (0, 1, 4):
        assert sign*q[k, 1]/q[k, 0]+2*np.sqrt(g*q[k, 0]) == pytest.approx(sign*data[k, 1]+2*c, rel=2e-14)
    assert q[0, 2]/q[0, 0] == pytest.approx(.7)
    assert q[1, 2] == 0.
    np.testing.assert_array_equal(actual=q[2], desired=data[2])
    assert q[3, 0] < 0 and q[6, 0] < 0 and q[7, 0] < 0
    assert q[4, 0] == pytest.approx(4/9)
    assert sign*q[4, 1]/q[4, 0] == pytest.approx(np.sqrt(g*q[4, 0]))
    np.testing.assert_array_equal(actual=q[5], desired=np.zeros(3))


@pytest.mark.parametrize('side,velocity', [('east', (.5, 0.)), ('west', (-.5, 0.)), ('north', (0., .5)), ('south', (0., -.5))])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_all_orientations_match_numpy_and_conserve(side, velocity, device):
    if device != 'cpu' and not wp.is_cuda_available(): pytest.skip('CUDA unavailable')
    solver, (bx, by, sx, sy) = make_solver(device=device, shape=(5, 6), depth=1., velocity=velocity,
        stage=.9, side=side, roughness=.04, cfl=.4, maximum_dt=.1, mode='characteristic')
    initial = solver.get_conservative_state()
    dt = solver.step(fixed_dt=.002)
    expected, budget = numpy_step(state=initial, bed=np.zeros((5, 6)), active=np.ones((5, 6), bool),
        manning=np.full((5, 6), .04), dx=1., dy=1., gravity=9.80665, dry_depth=1e-4, dt=dt,
        boundary_type_x=bx, boundary_type_y=by, boundary_stage_x=sx, boundary_stage_y=sy,
        rain_rate=np.zeros((5, 6)), infiltration_rate=np.zeros((5, 6)), stage_boundary_momentum='characteristic')
    np.testing.assert_allclose(actual=solver.get_conservative_state(), desired=expected, rtol=3e-13, atol=3e-15)
    assert solver.cumulative_accounting()['boundary_outflow'] == pytest.approx(budget['boundary_outflow'], abs=3e-14)
    assert abs(solver.cumulative_accounting()['residual']) < 3e-14


@pytest.mark.parametrize('device,graph', [('cpu', False), ('cuda:0', False), ('cuda:0', True)])
@pytest.mark.parametrize('depth,velocity,stage', [(1., (-4., 0.), 1.), (1., (0., 0.), 9.), (0., (0., 0.), 1.)])
def test_unsupported_inflow_rejected_before_state_or_budget_changes(device, graph, depth, velocity, stage):
    if device != 'cpu' and not wp.is_cuda_available(): pytest.skip('CUDA unavailable')
    solver, _ = make_solver(device=device, shape=(3, 4), depth=depth, velocity=velocity,
        stage=stage, side='east', roughness=0., cfl=.4, maximum_dt=.1, mode='characteristic')
    initial = solver.get_conservative_state(); budget = solver.cumulative_accounting()
    arrays = solver.capture_state()['arrays']
    with pytest.raises(ArithmeticError, match='incoming momentum'):
        solver.run_device_loop(duration=.1, use_cuda_graph=graph)
    np.testing.assert_array_equal(actual=solver.get_conservative_state(), desired=initial)
    assert solver.cumulative_accounting() == budget
    assert solver.steps == 0 and solver.time == 0.
    for key, value in arrays.items():
        np.testing.assert_array_equal(actual=solver.capture_state()['arrays'][key], desired=value)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('rain,infil', [(0., 0.), (5., 0.), (0., 5.), (5., 5.)])
def test_pre_source_cfl_bounds_actual_hll_state(device, rain, infil):
    if device != 'cpu' and not wp.is_cuda_available(): pytest.skip('CUDA unavailable')
    solver, (bx, by, sx, sy) = make_solver(device=device, shape=(3, 4), depth=1., velocity=(.5, .1),
        stage=.8, side='east', roughness=0., cfl=.4, maximum_dt=.2, mode='characteristic')
    solver.set_source_rates(rain=rain, infil=infil)
    original = solver.get_conservative_state()
    dt = solver.step(fixed_dt=.2)
    available = original[..., 0]+rain*dt
    q = original.copy(); q[..., 0] = np.maximum(available-infil*dt, 0.)
    q[..., 1:] *= (q[..., 0]/available)[..., None]
    # Record actual inputs to the independent HLL and their endpoint speeds.
    from unittest.mock import patch
    from floodwarp import full_swe_reference as reference
    original_flux = reference.hll_flux
    speeds = []
    def sampled_flux(*, left, right, gravity, dry_depth,flux_scheme='legacy_hll'):
        maximum = 0.
        for state in (left, right):
            velocity = np.divide(state[..., 1], state[..., 0], out=np.zeros_like(state[..., 0]), where=state[..., 0] > dry_depth)
            maximum = max(maximum, float(np.max(np.abs(velocity)+np.sqrt(gravity*state[..., 0]))))
        speeds.append(maximum)
        return original_flux(left=left, right=right, gravity=gravity, dry_depth=dry_depth,flux_scheme=flux_scheme)
    with patch.object(reference, 'hll_flux', sampled_flux):
        for axis, boundary, stages in ((1, bx, sx), (0, by, sy)):
            face_fluxes(state=q, bed=np.zeros((3, 4)), active=np.ones((3, 4), bool), boundary_type=boundary,
                boundary_stage=stages, axis=axis, gravity=9.80665, dry_depth=1e-4, stage_boundary_momentum='characteristic')
    assert dt*sum(speeds) <= .4*(1+3e-14)


def test_graph_cpu_cuda_and_complete_disk_restart(tmp_path):
    if not wp.is_cuda_available(): pytest.skip('CUDA unavailable')
    models = []
    for device, graph in (('cpu', False), ('cuda:0', False), ('cuda:0', True)):
        solver, _ = make_solver(device=device, shape=(6, 8), depth=1., velocity=(.5, 0.), stage=.9,
            side='east', roughness=.04, cfl=.4, maximum_dt=.03, mode='characteristic')
        solver.run_device_loop(duration=.12, use_cuda_graph=graph)
        solver.save_checkpoint(path=tmp_path/(device.replace(':', '_')+str(graph)+'.npz'))
        continued, _ = make_solver(device=device, shape=(6, 8), depth=1., velocity=(.5, 0.), stage=.9,
            side='east', roughness=.04, cfl=.4, maximum_dt=.03, mode='characteristic')
        continued.load_checkpoint(path=tmp_path/(device.replace(':', '_')+str(graph)+'.npz'))
        solver.run_device_loop(duration=.12, use_cuda_graph=graph)
        continued.run_device_loop(duration=.12, use_cuda_graph=graph)
        for key, value in solver.capture_state()['arrays'].items():
            np.testing.assert_array_equal(actual=continued.capture_state()['arrays'][key], desired=value)
        models.append(solver)
    np.testing.assert_array_equal(actual=models[1].get_conservative_state(), desired=models[2].get_conservative_state())
    np.testing.assert_allclose(actual=models[0].get_conservative_state(), desired=models[2].get_conservative_state(), rtol=3e-13, atol=3e-15)
    assert models[0].steps == models[1].steps == models[2].steps


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('datum', [0., 1000.])
def test_irregular_lake_dry_shore_and_embedded_stages(device, datum):
    if device != 'cpu' and not wp.is_cuda_available(): pytest.skip('CUDA unavailable')
    y, x = np.indices((9, 11))
    bed = datum+.2*np.sin(x)+.3*np.cos(y)
    stage = datum+.15; h = np.maximum(stage-bed, 0.)
    active = np.ones(h.shape, bool); active[4, 5] = False
    solver = FullSWESolver(ny=9, nx=11, dx=1., dy=1., bed_elevation=bed,
        manning_n=0., active_mask=active, device=device, stage_boundary_momentum='characteristic', maximum_dt=.03)
    solver.set_water_depth(depth=h)
    ax = np.pad(active, ((0, 0), (1, 1))); ay = np.pad(active, ((1, 1), (0, 0)))
    bx = np.where(ax[:, :-1] != ax[:, 1:], 3, 0)
    by = np.where(ay[:-1] != ay[1:], 3, 0)
    solver.set_boundary_type_qx(boundary_type=bx); solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=np.full(bx.shape, stage)); solver.set_boundary_stage_qy(stage=np.full(by.shape, stage))
    solver.run_device_loop(duration=.3, use_cuda_graph=True)
    np.testing.assert_allclose(actual=solver.get_water_depth(), desired=np.where(active, h, 0.), atol=4e-13, rtol=0.)
    assert np.max(np.abs(solver.get_conservative_state()[..., 1:])) < 5e-13
    assert abs(solver.cumulative_accounting()['residual']) < 2e-12


@pytest.mark.parametrize('velocity', [.5, -.5])
def test_subcritical_rarefaction_boundary_volume_improves_against_exact_physics(velocity):
    errors = []
    g = 9.80665; hb = .9; duration = .5; width = 4
    ub = velocity+2*(np.sqrt(g)-np.sqrt(g*hb))
    exact_net = width*duration*hb*ub
    for mode in ('discharge_extrapolation', 'velocity_extrapolation', 'characteristic'):
        solver, _ = make_solver(device='cpu', shape=(width, 24), depth=1., velocity=(velocity, 0.),
            stage=hb, side='east', roughness=0., cfl=.05, maximum_dt=.1, mode=mode)
        solver.run_device_loop(duration=duration, use_cuda_graph=False)
        budget = solver.cumulative_accounting()
        errors.append(abs(budget['boundary_outflow']-budget['boundary_inflow']-exact_net))
    assert errors[-1] < min(errors[:-1])


def test_dry_drainage_converges_to_exact_critical_rarefaction_volume():
    g = 9.80665; duration = .5; width = 2; length = 12.
    exact = width*duration*8*np.sqrt(g)/27
    errors = []
    for nx in (24, 48, 96):
        solver = FullSWESolver(ny=width, nx=nx, dx=length/nx, dy=1., bed_elevation=np.zeros((width, nx)),
            manning_n=0., active_mask=np.ones((width, nx), bool), stage_boundary_momentum='characteristic',
            cfl=.05, maximum_dt=.1, gravity=g)
        solver.set_water_depth(depth=np.ones((width, nx)))
        bx = np.zeros((width, nx+1), int); bx[:, 0] = 1; bx[:, -1] = 3
        by = np.zeros((width+1, nx), int); by[[0, -1]] = 1
        solver.set_boundary_type_qx(boundary_type=bx); solver.set_boundary_type_qy(boundary_type=by)
        solver.run_device_loop(duration=duration, use_cuda_graph=False)
        budget = solver.cumulative_accounting()
        errors.append(abs(budget['boundary_outflow']-exact))
        assert solver.get_water_depth().min() >= 0 and abs(budget['residual']) < 1e-12
    assert errors[2] < errors[1] < errors[0]


def test_factory_and_legacy_checkpoint_remain_explicit():
    options = dict(ny=2, nx=3, dx=1., dy=1., bed_elevation=np.zeros((2, 3)), manning_n=0., active_mask=np.ones((2, 3), bool), device='cpu')
    legacy = create_overland_solver(solver_method='fv1', stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', **options)
    characteristic = create_overland_solver(solver_method='fv1', stage_boundary_momentum='characteristic', **options)
    assert legacy.numerical_configuration()['stage_boundary_momentum'] == 'discharge_extrapolation'
    assert characteristic.numerical_configuration()['stage_boundary_momentum'] == 'characteristic'
    historical = legacy.capture_state(); historical['configuration'].pop('stage_boundary_momentum')
    legacy.restore_state(state=historical)
    with pytest.raises(ValueError, match='configuration'):
        characteristic.restore_state(state=historical)
