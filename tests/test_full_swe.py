"""Historical full-SWE controls, explicitly pinned without changing tolerances."""

from pathlib import Path
import os

import numpy as np
import pytest

from floodwarp.full_swe_reference import (numpy_step, stable_timestep,
    exponential_infiltration_capacity, exact_ponded_infiltration)
from floodwarp.full_swe_solver import FullSWESolver


def build_solver(*, bed, depth, hu=None, hv=None, roughness=0., device="cpu", spacing=1., cap=.03):
    active = np.ones(bed.shape, dtype=bool)
    solver = FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=bed.shape[0], nx=bed.shape[1], dx=spacing, dy=spacing,
        bed_elevation=bed, manning_n=roughness, active_mask=active, device=device, maximum_dt=cap)
    solver.set_water_depth(depth=depth)
    if hu is not None:
        solver.set_momentum(hu=hu, hv=hv)
    return solver


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_numpy_warp_step_parity_and_budget(axis, boundary):
    rng = np.random.default_rng(seed=235)
    shape = (7, 9)
    bed = rng.uniform(-.2, .2, shape)
    depth = rng.uniform(.01, .6, shape)
    hu, hv = rng.uniform(-.05, .05, (2, *shape))
    roughness = rng.uniform(.01, .08, shape)
    solver = build_solver(bed=bed, depth=depth, hu=hu, hv=hv, roughness=roughness)
    bx, by = np.zeros((7, 10), int), np.zeros((8, 9), int)
    bx[:, [0, -1]], by[[0, -1], :] = boundary, boundary
    sx, sy = np.full(bx.shape, .1), np.full(by.shape, .1)
    if axis == 0:
        by[:] = 0
    else:
        bx[:] = 0
    solver.set_boundary_type_qx(boundary_type=bx)
    solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=sx)
    solver.set_boundary_stage_qy(stage=sy)
    solver.set_source_rates(rain=np.full(shape, 3.e-5), infil=np.full(shape, 2.e-5))
    q = solver.get_conservative_state()
    dt = solver.step(fixed_dt=.003)
    expected, budget = numpy_step(stage_boundary_momentum='discharge_extrapolation',flux_scheme='legacy_hll', state=q, bed=bed, active=np.ones(shape, bool), manning=roughness,
        dx=1., dy=1., gravity=9.81, dry_depth=1.e-4, dt=dt,
        boundary_type_x=bx, boundary_type_y=by, boundary_stage_x=sx, boundary_stage_y=sy,
        rain_rate=np.full(shape, 3.e-5), infiltration_rate=np.full(shape, 2.e-5))
    np.testing.assert_allclose(solver.get_conservative_state(), expected, atol=3.e-16, rtol=3.e-14)
    account = solver.cumulative_accounting()
    for key in budget:
        assert account[key] == pytest.approx(budget[key], abs=3.e-15)
    assert abs(account["residual"]) < 2.e-14


@pytest.mark.parametrize("datum", [0., 1000.])
@pytest.mark.parametrize("irregular", [False, True])
def test_lake_at_rest_including_dry_shore(datum, irregular):
    y, x = np.indices((9, 14))
    bed = datum + (.2 * np.sin(x) + .3 * np.cos(y) if irregular else np.zeros(x.shape))
    depth = np.maximum(0., datum + .15 - bed)
    solver = build_solver(bed=bed, depth=depth)
    solver.run_device_loop(duration=1.)
    np.testing.assert_allclose(solver.get_water_depth(), depth, atol=4.e-13, rtol=0.)
    hu, hv = solver.get_momentum()
    assert max(np.abs(hu).max(), np.abs(hv).max()) < 5.e-13
    assert abs(solver.cumulative_accounting()["residual"]) < 2.e-12


@pytest.mark.parametrize("bed_value,depth", [(.2, 0.), (-.2, .2), (-.2, 0.), (.2, .2)])
def test_zero_stage_hydrostatic_boundary(bed_value, depth):
    bed = np.full((5, 8), bed_value)
    solver = build_solver(bed=bed, depth=np.full(bed.shape, depth))
    types = np.zeros((6, 8), int)
    types[-1] = 3
    solver.set_boundary_type_qy(boundary_type=types)
    solver.run_device_loop(duration=.3)
    account = solver.cumulative_accounting()
    if bed_value > 0 and depth == 0:
        assert np.count_nonzero(solver.get_water_depth()) == 0
        assert account["boundary_inflow"] == account["boundary_outflow"] == 0.
    elif bed_value < 0 and depth > 0:
        np.testing.assert_allclose(solver.get_water_depth(), depth, atol=2.e-16)
    elif depth == 0:
        assert account["boundary_inflow"] > 0
    else:
        assert account["boundary_outflow"] > 0
    assert abs(account["residual"]) < 2.e-14


def test_mask_and_exact_source_integral():
    shape = (5, 9)
    active = np.ones(shape, bool)
    active[2, 3:6] = False
    solver = FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=5, nx=9, dx=2., dy=3., bed_elevation=np.zeros(shape),
        manning_n=0., active_mask=active, maximum_dt=.12)
    solver.set_rainfall_hyetograph(times_s=np.array([0., .37, .81, 1.]), rates_m_s=np.array([0., .001, .002, 0.]))
    solver.set_exponential_infiltration(f0=1.e-5, fd=.04, conductivity=1.e-6)
    solver.run_device_loop(duration=1.)
    account = solver.cumulative_accounting()
    exact = np.trapezoid(solver._rates, solver._times) * active.sum() * 6.
    assert account["rain"] == pytest.approx(exact, abs=3.e-16)
    assert account["infiltration"] > 0
    assert abs(account["residual"]) < 3.e-16
    assert np.count_nonzero(solver.get_water_depth()[~active]) == 0


@pytest.mark.parametrize('terminal_rate', [0., .002])
@pytest.mark.parametrize('segmented', [False, True])
def test_finite_storm_allows_zero_rain_recession(terminal_rate, segmented):
    shape = (2, 3)
    solver = build_solver(bed=np.zeros(shape), depth=np.zeros(shape), cap=.12)
    times = np.array([0., .37, .81, 1.])
    rates = np.array([0., .001, .002, terminal_rate])
    solver.set_rainfall_hyetograph(times_s=times, rates_m_s=rates)
    if segmented:
        solver.run_device_loop(duration=1.)
        rain_at_end = solver.cumulative_accounting()['rain']
        solver.run_device_loop(duration=1.37)
        assert solver.cumulative_accounting()['rain'] == pytest.approx(rain_at_end, abs=1e-14)
    else:
        solver.run_device_loop(duration=2.37)
    expected_depth = float(np.trapezoid(rates, times))
    np.testing.assert_allclose(solver.get_water_depth(), expected_depth, atol=1e-12, rtol=1e-10)
    ledger = solver.cumulative_accounting()
    assert ledger['rain'] == pytest.approx(expected_depth * np.prod(shape), abs=1e-12)
    assert abs(ledger['residual']) < 1e-12
    assert solver.time == pytest.approx(2.37, abs=1e-12)


def test_step_after_storm_uses_zero_rain():
    solver = build_solver(bed=np.zeros((2, 3)), depth=np.zeros((2, 3)), cap=.1)
    solver.set_rainfall_hyetograph(times_s=np.array([0., .2]), rates_m_s=np.array([.001, .002]))
    solver.run_device_loop(duration=.2)
    before = solver.get_water_depth().copy()
    solver.step(fixed_dt=.1)
    np.testing.assert_allclose(solver.get_water_depth(), before, atol=1e-12, rtol=1e-10)


def test_exactly_dry_faces_clear_flux_and_pressure_buffers():
    import warp as wp
    active=np.ones((3,4),dtype=bool);active[1,1]=False
    solver=FullSWESolver(ny=3,nx=4,dx=1.,dy=1.,bed_elevation=np.zeros((3,4)),
        manning_n=.035,active_mask=active,maximum_dt=.1)
    for axis in ('x','y'):
        solver._fields['flux_'+axis].fill_(wp.vec3d(2.,3.,4.))
        solver._fields['pressure_'+axis].fill_(wp.vec2d(5.,6.))
    solver.run_device_loop(duration=.1)
    np.testing.assert_allclose(solver.get_conservative_state(),0.,atol=1e-12,rtol=0.)
    for axis in ('x','y'):
        np.testing.assert_allclose(solver._fields['flux_'+axis].numpy(),0.,atol=1e-12,rtol=0.)
        np.testing.assert_allclose(solver._fields['pressure_'+axis].numpy(),0.,atol=1e-12,rtol=0.)
    boundary=np.zeros((3,5),dtype=np.int32);boundary[:,0]=3
    solver.set_boundary_type_qx(boundary_type=boundary)
    solver.set_boundary_stage_qx(stage=np.full((3,5),.2))
    solver.run_device_loop(duration=.1)
    assert solver.cumulative_accounting()['boundary_inflow']>0.
    assert solver.get_water_depth()[:,0].min()>0.


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('scheme',['legacy_hll','contact_hll','star_hll','star_contact_hll'])
def test_dry_reconstructed_faces_preserve_numpy_operator(device,scheme):
    import warp as wp
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(seed=931)
    shape=(4,7)
    active=np.ones(shape,dtype=bool);active[1:3,2:4]=False
    bed=rng.uniform(-.1,.4,shape)
    depth=rng.choice(np.array([0.,.02,.3]),size=shape)
    solver=FullSWESolver(ny=shape[0],nx=shape[1],dx=1.,dy=1.,bed_elevation=bed,
        manning_n=.035,active_mask=active,device=device,maximum_dt=.1,
        stage_boundary_momentum='discharge_extrapolation',flux_scheme=scheme)
    solver.set_water_depth(depth=depth)
    depth=solver.get_water_depth().copy()
    solver.set_momentum(hu=rng.uniform(-.1,.1,shape)*depth,hv=rng.uniform(-.1,.1,shape)*depth)
    initial=solver.get_conservative_state().copy()
    expected,_=numpy_step(state=initial,bed=bed,active=active,manning=np.full(shape,.035),
        dx=1.,dy=1.,gravity=9.81,dry_depth=.0001,dt=.001,
        stage_boundary_momentum='discharge_extrapolation',flux_scheme=scheme,
        boundary_type_x=np.zeros((shape[0],shape[1]+1),dtype=np.int32),
        boundary_type_y=np.zeros((shape[0]+1,shape[1]),dtype=np.int32),
        boundary_stage_x=np.zeros((shape[0],shape[1]+1)),
        boundary_stage_y=np.zeros((shape[0]+1,shape[1])),
        rain_rate=np.zeros(shape),infiltration_rate=np.zeros(shape))
    solver.step(fixed_dt=.001)
    np.testing.assert_allclose(solver.get_conservative_state(),expected,atol=1e-12,rtol=1e-10)
    assert abs(solver.cumulative_accounting()['residual'])<1e-12


def test_vector_manning_decay_in_oblique_flow():
    shape = (9, 11)
    solver = build_solver(bed=np.zeros(shape), depth=np.ones(shape), hu=np.full(shape, .3),
        hv=np.full(shape, .4), roughness=.05, cap=.001)
    # Matching exterior depths and extrapolated momenta preserve the uniform
    # state; reflecting walls would launch waves into this friction control.
    bx, by = np.zeros((9, 12), int), np.zeros((10, 11), int)
    bx[:, [0, -1]], by[[0, -1], :] = 3, 3
    solver.set_boundary_type_qx(boundary_type=bx)
    solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=np.ones(bx.shape))
    solver.set_boundary_stage_qy(stage=np.ones(by.shape))
    solver.run_device_loop(duration=.1)
    hu, hv = solver.get_momentum()
    factor = 1. + .1 * 9.81 * .05**2 * .5
    assert hu[4, 5] == pytest.approx(.3 / factor, abs=2.e-12)
    assert hv[4, 5] == pytest.approx(.4 / factor, abs=2.e-12)


def test_cfl_includes_both_velocity_components():
    q = np.zeros((3, 4, 3))
    q[..., 0], q[..., 1], q[..., 2] = 1., 30., 40.
    expected = .4 / (70. + 2. * np.sqrt(9.81))
    dt = stable_timestep(state=q, active=np.ones(q.shape[:2], bool), dx=1., dy=1., gravity=9.81,
        dry_depth=1.e-4, cfl=.4, maximum_dt=1., rain_rate=0.)
    assert dt == pytest.approx(expected)
    solver = build_solver(bed=np.zeros((3, 4)), depth=q[..., 0], hu=q[..., 1], hv=q[..., 2], cap=1.)
    assert solver.step(fixed_dt=1.) == pytest.approx(expected)


@pytest.mark.parametrize("right_depth", [0., .25])
def test_dam_break_rotation_positivity_and_transport(right_depth):
    y, x = np.indices((30, 30))
    depth = np.where(x < 15, 1., right_depth)
    solver = build_solver(bed=np.zeros(depth.shape), depth=depth)
    rotated = build_solver(bed=np.zeros(depth.shape), depth=depth.T)
    solver.run_device_loop(duration=.8)
    rotated.run_device_loop(duration=.8)
    q, r = solver.get_conservative_state(), rotated.get_conservative_state()
    np.testing.assert_allclose(q[..., 0], r[..., 0].T, atol=3.e-15)
    np.testing.assert_allclose(q[..., 1], r[..., 2].T, atol=3.e-15)
    assert q[..., 0].min() >= 0
    assert q[:, 15:, 1].max() > .1
    assert abs(solver.cumulative_accounting()["residual"]) < 3.e-12


def test_two_dimensional_dam_break_over_step_and_recession():
    y, x = np.indices((25, 25))
    bed = np.where(x > 15, .15, 0.)
    depth = np.where((x - 12)**2 + (y - 12)**2 < 25, .8, 0.)
    solver = build_solver(bed=bed, depth=depth, roughness=.03)
    types = np.zeros((26, 25), int)
    types[-1] = 2
    solver.set_boundary_type_qy(boundary_type=types)
    solver.run_device_loop(duration=4.)
    q = solver.get_conservative_state()
    assert np.isfinite(q).all() and q[..., 0].min() >= 0
    assert q[:, 17:, 0].max() > .01
    assert np.max(np.abs(q[..., 2])) > .01
    assert np.all(solver.get_max_water_depth() >= q[..., 0])
    assert abs(solver.cumulative_accounting()["residual"]) < 2.e-12


def test_complete_disk_restart_and_configuration_rejection(tmp_path: Path):
    y, x = np.indices((8, 13))
    bed = .01 * y
    depth = np.where(x < 6, .2, .01)
    first = build_solver(bed=bed, depth=depth, roughness=.04)
    first.set_source_rates(rain=1.e-4, infil=1.e-5)
    first.run_device_loop(duration=.3)
    path = tmp_path / "state.npz"
    first.save_checkpoint(path=path)
    restored = build_solver(bed=bed, depth=depth, roughness=.04)
    restored.load_checkpoint(path=path)
    first.run_device_loop(duration=.3)
    restored.run_device_loop(duration=.3)
    for name, array in first.capture_state()["arrays"].items():
        np.testing.assert_array_equal(restored.capture_state()["arrays"][name], array)
    bad = build_solver(bed=bed + 1., depth=depth, roughness=.04)
    with pytest.raises(ValueError, match="physical input"):
        bad.load_checkpoint(path=path)


def test_factory_fv1_rejects_unsupported_physics():
    from floodwarp.solver_factory import create_overland_solver
    options = dict(solver_method="fv1", ny=3, nx=4, dx=1., dy=1., bed_elevation=np.zeros((3, 4)),
        manning_n=np.zeros((3, 4)), active_mask=np.ones((3, 4), bool))
    assert isinstance(create_overland_solver(**options), FullSWESolver)
    with pytest.raises(ValueError, match="SGS"):
        create_overland_solver(**options, hydraulic_geometry_mode="traditional")


@pytest.mark.skipif(os.environ.get("OVERLANDER_TEST_FULL_SWE_CUDA") != "1", reason="GPU validation is explicitly opt-in")
def test_cuda_graph_matches_cpu_and_complete_restart(tmp_path: Path):
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    y, x = np.indices((17, 23))
    bed = .03 * np.sin(x / 4.) + .02 * np.cos(y / 3.)
    depth = np.maximum(.15 - bed + .1 * (x < 10), 0.)
    cpu = build_solver(bed=bed, depth=depth, roughness=.04, device="cpu")
    gpu = build_solver(bed=bed, depth=depth, roughness=.04, device="cuda:0")
    restored = build_solver(bed=bed, depth=depth, roughness=.04, device="cuda:0")
    for solver in (cpu, gpu):
        solver.set_rainfall_hyetograph(times_s=np.array([0., .17, .5, 1.]), rates_m_s=np.array([0., .001, .0001, 0.]))
        solver.set_exponential_infiltration(f0=1.e-5, fd=.04, conductivity=1.e-6)
        solver.run_device_loop(duration=.4)
    np.testing.assert_allclose(gpu.get_conservative_state(), cpu.get_conservative_state(), atol=2.e-15, rtol=2.e-13)
    path = tmp_path / "gpu_checkpoint.npz"
    gpu.save_checkpoint(path=path)
    restored.load_checkpoint(path=path)
    for solver in (cpu, gpu, restored):
        solver.run_device_loop(duration=.3)
    np.testing.assert_allclose(gpu.get_conservative_state(), cpu.get_conservative_state(), atol=3.e-15, rtol=3.e-13)
    np.testing.assert_array_equal(gpu.get_conservative_state(), restored.get_conservative_state())
    assert abs(gpu.cumulative_accounting()["residual"]) < 5.e-13


def test_endpoint_clipping_is_not_a_minimum_cfl_failure():
    solver = FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=2, nx=3, dx=1., dy=1., bed_elevation=np.zeros((2, 3)),
        manning_n=0., active_mask=np.ones((2, 3), bool), maximum_dt=.1, minimum_dt=.05)
    solver.set_water_depth(depth=np.ones((2, 3)))
    result = solver.run_device_loop(duration=.101, use_cuda_graph=False)
    assert result["dt_min_s"] < .05
    assert sum(solver.get_timestep_histogram()["counts"]) == result["steps"]


def test_finite_availability_removes_water_and_vector_momentum():
    bed = np.zeros((4, 6))
    solver = build_solver(bed=bed, depth=np.full(bed.shape, .001), hu=np.full(bed.shape, .0002), hv=np.full(bed.shape, -.0003))
    solver.set_source_rates(rain=.001, infil=1.)
    solver.run_device_loop(duration=.01, use_cuda_graph=False)
    assert np.count_nonzero(solver.get_conservative_state()) == 0
    account = solver.cumulative_accounting()
    assert account["infiltration"] == pytest.approx(.00101 * bed.size)
    assert abs(account["residual"]) < 1.e-17


def test_returned_fields_and_captures_are_independent_snapshots():
    solver = build_solver(bed=np.zeros((3, 4)), depth=np.ones((3, 4)))
    snapshot = solver.capture_state()
    depth = solver.get_water_depth()
    depth[:] = 0.
    solver.set_source_rates(infil=.001)
    solver.run_device_loop(duration=.03, use_cuda_graph=False)
    assert np.all(snapshot["arrays"]["q"][:, 0] == 1.)
    assert np.all(solver.get_water_depth() > .99)


def test_wet_stage_ghost_limits_initially_dry_inflow():
    solver = build_solver(bed=np.zeros((3, 4)), depth=np.zeros((3, 4)), cap=10.)
    types = np.zeros((4, 4), int)
    types[-1] = 3
    solver.set_boundary_type_qy(boundary_type=types)
    solver.set_boundary_stage_qy(stage=np.full(types.shape, 4.))
    expected = .4 / np.sqrt(9.81 * 4.)
    assert solver.step(fixed_dt=10.) == pytest.approx(expected)
    assert solver.cumulative_accounting()["boundary_inflow"] > 0


def test_shallow_wet_ghost_extrapolated_momentum_participates_in_cfl():
    depth = np.full((3, 4), .1)
    solver = build_solver(bed=np.zeros(depth.shape), depth=depth, hu=np.zeros(depth.shape), hv=np.full(depth.shape, -.1), cap=1.)
    types = np.zeros((4, 4), int)
    types[-1] = 3
    solver.set_boundary_type_qy(boundary_type=types)
    solver.set_boundary_stage_qy(stage=np.full(types.shape, .001))
    expected = .4 / (np.sqrt(9.81 * .1) + .1 / .001 + np.sqrt(9.81 * .001))
    assert solver.step(fixed_dt=1.) == pytest.approx(expected)
    assert abs(solver.cumulative_accounting()["residual"]) < 1.e-15


def test_minimum_dt_aborts_without_overriding_stability():
    solver = FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=2, nx=3, dx=1., dy=1., bed_elevation=np.zeros((2, 3)),
        manning_n=0., active_mask=np.ones((2, 3), bool), maximum_dt=1., minimum_dt=.01)
    solver.set_water_depth(depth=np.ones((2, 3)))
    solver.set_momentum(hu=np.full((2, 3), 10000.), hv=np.zeros((2, 3)))
    with pytest.raises(ArithmeticError, match="below configured"):
        solver.run_device_loop(duration=1., max_steps=3, use_cuda_graph=False)
    assert solver.steps == 1 and solver.time < .01


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("high_side", [False, True])
def test_embedded_stage_matches_same_physical_outer_face(axis, high_side):
    shape = (4, 5)
    padding = [(0, 0), (0, 0)]
    padding[axis] = (0, 2) if high_side else (2, 0)
    active = np.pad(np.ones(shape, bool), padding)
    bed = np.where(active, .1, 1000.)
    masked = FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=bed.shape[0], nx=bed.shape[1], dx=1., dy=1., bed_elevation=bed,
        manning_n=.04, active_mask=active, maximum_dt=.03)
    masked.set_water_depth(depth=np.where(active, .2, 0.))
    cropped = build_solver(bed=np.full(shape, .1), depth=np.full(shape, .2), roughness=.04)
    for solver, interface in ((masked, shape[axis] if high_side else 2), (cropped, shape[axis] if high_side else 0)):
        face_shape = (solver.ny + 1, solver.nx) if axis == 0 else (solver.ny, solver.nx + 1)
        types = np.zeros(face_shape, int)
        index = [slice(None), slice(None)]
        index[axis] = interface
        types[tuple(index)] = 3
        if axis == 0:
            solver.set_boundary_type_qy(boundary_type=types)
        else:
            solver.set_boundary_type_qx(boundary_type=types)
        solver.run_device_loop(duration=.3, use_cuda_graph=False)
    np.testing.assert_array_equal(masked.get_conservative_state()[active], cropped.get_conservative_state().reshape(-1, 3))
    assert masked.cumulative_accounting()["boundary_outflow"] == pytest.approx(cropped.cumulative_accounting()["boundary_outflow"], abs=3.e-16)


@pytest.mark.parametrize("normal_momentum", [-.2, .2])
def test_free_boundary_is_one_way_and_conservative(normal_momentum):
    shape = (4, 5)
    solver = build_solver(bed=np.zeros(shape), depth=np.ones(shape), hu=np.zeros(shape), hv=np.full(shape, normal_momentum))
    types = np.zeros((5, 5), int)
    types[-1] = 2
    solver.set_boundary_type_qy(boundary_type=types)
    dt = solver.step(fixed_dt=.001)
    budget = solver.cumulative_accounting()
    assert budget["boundary_inflow"] == 0.
    assert budget["boundary_outflow"] == pytest.approx(max(0., normal_momentum) * shape[1] * dt)
    assert abs(budget["residual"]) < 4.e-15




























@pytest.mark.parametrize("conductivity", [0., .0001])
def test_ponded_exponential_infiltration_converges_to_exact_solution(conductivity):
    shape = (2, 3)
    duration = 10.
    initial_capacity, decay_depth = .001, .005
    exact = float(exact_ponded_infiltration(time_s=duration, initial_capacity_m_s=initial_capacity,
        decay_depth_m=decay_depth, conductivity_m_s=conductivity))
    errors = []
    for cap in (.4, .2, .1, .05):
        solver = build_solver(bed=np.zeros(shape), depth=np.full(shape, .1), spacing=10., cap=cap)
        solver.set_exponential_infiltration(f0=initial_capacity, fd=decay_depth, conductivity=conductivity)
        solver.run_device_loop(duration=duration, use_cuda_graph=False)
        soil = solver.cumulative_infiltration_m()
        errors.append(abs(float(soil[0, 0]) - exact))
        np.testing.assert_allclose(solver.get_water_depth() + soil, .1, atol=2.e-16, rtol=0.)
        np.testing.assert_array_equal(soil, np.full(shape, soil[0, 0]))
        assert abs(solver.cumulative_accounting()["residual"]) < 5.e-14
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all((orders > .9) & (orders < 1.1))


def test_spatial_exponential_soil_and_vector_withdrawal_match_numpy():
    rng = np.random.default_rng(seed=246)
    shape = (5, 7)
    bed = rng.uniform(0., .0005, shape)
    depth = rng.uniform(.0006, .002, shape)
    hu, hv = rng.uniform(-.00002, .00002, (2, *shape))
    f0 = rng.uniform(.001, .003, shape)
    fd = rng.uniform(.0001, .0003, shape)
    conductivity = rng.uniform(.00001, .0001, shape)
    rain = rng.uniform(0., .0002, shape)
    roughness = rng.uniform(.02, .06, shape)
    soil = np.zeros(shape)
    solver = build_solver(bed=bed, depth=depth, hu=hu, hv=hv, roughness=roughness, cap=.005)
    solver.set_exponential_infiltration(f0=f0, fd=fd, conductivity=conductivity)
    solver.set_source_rates(rain=rain)
    state = solver.get_conservative_state()
    bx, by = np.zeros((5, 8), int), np.zeros((6, 7), int)
    for _ in range(100):
        capacity = exponential_infiltration_capacity(cumulative_infiltration_m=soil,
            initial_capacity_m_s=f0, decay_depth_m=fd, conductivity_m_s=conductivity)
        dt = solver.step(fixed_dt=.005)
        removed = np.minimum(state[..., 0] + rain * dt, capacity * dt)
        state, _ = numpy_step(stage_boundary_momentum='discharge_extrapolation',flux_scheme='legacy_hll', state=state, bed=bed, active=np.ones(shape, bool), manning=roughness,
            dx=1., dy=1., gravity=9.81, dry_depth=1.e-4, dt=dt,
            boundary_type_x=bx, boundary_type_y=by, boundary_stage_x=np.zeros(bx.shape), boundary_stage_y=np.zeros(by.shape),
            rain_rate=rain, infiltration_rate=capacity)
        soil += removed
    np.testing.assert_allclose(solver.get_conservative_state(), state, atol=1.e-16, rtol=3.e-13)
    np.testing.assert_allclose(solver.cumulative_infiltration_m(), soil, atol=1.e-16, rtol=3.e-13)
    assert np.max(capacity / f0) < .75
    budget = solver.cumulative_accounting()
    # Bound accumulated FP64 additions and the four domain reductions. A fixed
    # 2e-17 absolute bound was smaller than three ulps of this basin's volume.
    additions = 8 * solver.steps + 4 * np.prod(shape)
    epsilon = np.finfo(np.float64).eps
    gamma = additions * epsilon / (1. - additions * epsilon)
    assert abs(budget["residual"]) <= gamma * (solver.initial_volume + budget["rain"])


def test_spreading_wet_front_recedes_to_dry_without_losing_volume():
    shape = (3, 21)
    columns = np.indices(shape)[1]
    depth = np.where(columns < 7, .003, 0.)
    solver = build_solver(bed=np.zeros(shape), depth=depth, roughness=.04, cap=.05)
    solver.set_source_rates(infil=.001)
    solver.run_device_loop(duration=1., use_cuda_graph=False)
    assert np.any(solver.get_water_depth()[:, 7:] > 0.)
    solver.run_device_loop(duration=3., use_cuda_graph=False)
    assert np.count_nonzero(solver.get_conservative_state()) == 0
    budget = solver.cumulative_accounting()
    assert budget["infiltration"] == pytest.approx(float(depth.sum()), abs=3.e-17)
    assert abs(budget["residual"]) < 3.e-17
