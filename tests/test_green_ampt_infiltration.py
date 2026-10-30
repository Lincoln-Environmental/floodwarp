import numpy as np
import pytest

from floodwarp.diagnostics import compute_diagnostics
from floodwarp.infiltration import (
    green_ampt_infiltration_increment_m,
    green_ampt_rainfall_increment_m,
)
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.rainfall import Hyetograph


def scalar_reference(f0, ks, psi, delta_theta, dt):
    a = psi * delta_theta
    if ks == 0.0 or dt == 0.0:
        return 0.0
    if a == 0.0:
        return ks * dt
    lower = 0.0
    upper = ks * dt + 2.0 * np.sqrt(a * ks * dt) + 2.0 * a
    for _ in range(160):
        middle = 0.5 * (lower + upper)
        residual = middle - a * np.log1p(middle / (f0 + a))
        if residual < ks * dt:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def make_solver(*, method="green_ampt", ks=0.001, psi=0.1, deficit=0.2,
                ny=1, nx=1, device="cpu", **kwargs):
    return LocalInertialSolver(
        ny=ny, nx=nx, dx=1.0, dy=1.0,
        bed_elevation=np.zeros((ny, nx)),
        manning_n=np.full((ny, nx), 0.05),
        active_mask=np.ones((ny, nx), dtype=np.uint8),
        device=device, infiltration_method=method,
        infiltration_ks_m_per_s=ks,
        infiltration_psi_f_m=psi,
        infiltration_delta_theta=deficit,
        **kwargs,
    )


def test_numpy_finite_step_matches_high_accuracy_reference_and_limits():
    actual = green_ampt_infiltration_increment_m(0.012, 0.0004, 0.2, 0.31, 120.0)
    expected = scalar_reference(0.012, 0.0004, 0.2, 0.31, 120.0)
    assert actual == pytest.approx(expected, rel=1e-13)
    assert green_ampt_infiltration_increment_m(0.0, 0.002, 0.5, 0.0, 10.0) == pytest.approx(0.02)
    assert green_ampt_infiltration_increment_m(0.0, 0.0, 0.5, 0.2, 10.0) == 0.0


def test_numpy_spatial_monotonicity_and_capacity_tends_to_ks():
    ks = np.array([[1e-4, 2e-4], [3e-4, 4e-4]])
    increment = green_ampt_infiltration_increment_m(
        np.array([[0.0, 0.01], [0.1, 1.0]]), ks, 0.1, 0.25, 60.0
    )
    assert np.all(increment >= 0.0)
    assert np.all(increment >= ks * 60.0 - 1e-15)
    small = green_ampt_infiltration_increment_m(0.01, 0.001, 0.2, 0.3, 10.0)
    large = green_ampt_infiltration_increment_m(10.0, 0.001, 0.2, 0.3, 10.0)
    assert large / 10.0 == pytest.approx(0.001, rel=0.01)
    assert small > large


def test_dry_preponding_and_within_step_ponding_match_independent_threshold():
    ks = 0.001
    rain = 0.002
    psi = 0.1
    deficit = 0.2
    ponding_f = ks * psi * deficit / (rain - ks)
    ponding_time = ponding_f / rain
    assert ponding_f == pytest.approx(0.02)
    assert ponding_time == pytest.approx(10.0)
    assert green_ampt_rainfall_increment_m(0.0, ks, psi, deficit, rain, 5.0) == pytest.approx(0.01)
    expected = ponding_f + scalar_reference(ponding_f, ks, psi, deficit, 10.0)
    actual = green_ampt_rainfall_increment_m(0.0, ks, psi, deficit, rain, 20.0)
    assert actual == pytest.approx(expected, rel=1e-12)
    assert actual < rain * 20.0


def test_dry_rain_below_capacity_and_wetter_soil_response():
    assert green_ampt_rainfall_increment_m(0.0, 0.002, 0.1, 0.3, 0.001, 10.0) == pytest.approx(0.01)
    wetter = green_ampt_rainfall_increment_m(0.0, 0.001, 0.1, 0.1, 0.005, 20.0)
    drier = green_ampt_rainfall_increment_m(0.0, 0.001, 0.1, 0.4, 0.005, 20.0)
    assert wetter < drier
    assert green_ampt_rainfall_increment_m(0.0, 0.001, 0.1, 0.0, 0.005, 20.0) == pytest.approx(0.02)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_single_cell_solver_matches_analytical_ponding_transition(device):
    import warp as wp

    if device == "cuda:0" and not wp.is_device_available(device):
        pytest.skip("CUDA unavailable")
    solver = make_solver(ks=0.001, psi=0.1, deficit=0.2, device=device)
    solver.set_uniform_rainfall_rate(0.002)
    solver.step(20.0, fixed_dt=20.0)
    ponding_f = 0.02
    expected = ponding_f + scalar_reference(ponding_f, 0.001, 0.1, 0.2, 10.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(expected, rel=2e-4)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.04 - expected, abs=2e-6)
    assert abs(compute_diagnostics(solver)["mass_balance_residual"]) < 1e-9


def test_ponding_transition_is_stable_under_timestep_refinement():
    coarse = make_solver(ks=0.001, psi=0.1, deficit=0.2)
    fine = make_solver(ks=0.001, psi=0.1, deficit=0.2)
    for solver in (coarse, fine):
        solver.set_uniform_rainfall_rate(0.002)
    coarse.step(20.0, fixed_dt=20.0)
    for _ in range(4):
        fine.step(5.0, fixed_dt=5.0)
    np.testing.assert_allclose(
        fine.cumulative_infiltration_m, coarse.cumulative_infiltration_m,
        rtol=3e-4, atol=2e-6,
    )


def test_solver_caps_to_available_water_and_accounts_mass():
    solver = make_solver(ks=0.001, psi=0.2, deficit=0.3)
    solver.set_water_depth(np.array([[0.001]]))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.001)
    assert solver.get_water_depth()[0, 0] == 0.0
    diag = compute_diagnostics(solver)
    assert abs(diag["mass_balance_residual"]) < 1e-12


def test_solver_rain_ordering_spatial_arrays_and_reset():
    ks = np.array([[0.001, 0.002]])
    solver = make_solver(ks=ks, psi=0.1, deficit=np.array([[0.0, 0.3]]), nx=2)
    solver.set_uniform_rainfall_rate(0.01)
    solver.set_water_depth(np.zeros((1, 2)))
    solver.step(1.0, fixed_dt=1.0)
    actual = solver.cumulative_infiltration_m
    assert actual[0, 0] == pytest.approx(0.001)
    assert 0.002 < actual[0, 1] <= 0.01
    assert np.all(solver.get_water_depth() >= 0.0)
    solver.set_green_ampt_infiltration(ks, 0.1, 0.3)
    np.testing.assert_array_equal(solver.cumulative_infiltration_m, np.zeros((1, 2)))


def test_zero_conductivity_has_zero_reported_capacity_at_event_start():
    solver = make_solver(ks=np.array([[0.0, 0.001]]), psi=0.1, deficit=0.2, nx=2)
    capacity = solver.current_infiltration_capacity_m_per_s
    assert capacity[0, 0] == 0.0
    assert np.isinf(capacity[0, 1])


def test_reset_restores_zero_cumulative_infiltration_and_antecedent_deficit():
    solver = make_solver(ks=0.001, psi=0.1, deficit=0.25)
    solver.set_uniform_rainfall_rate(0.003)
    solver.step(1.0, fixed_dt=1.0)
    first = solver.cumulative_infiltration_m.copy()
    assert first[0, 0] > 0.0
    solver.reset()
    np.testing.assert_array_equal(solver.cumulative_infiltration_m, [[0.0]])
    np.testing.assert_array_equal(solver.green_ampt_delta_theta.numpy(), [[0.25]])
    solver.step(1.0, fixed_dt=1.0)
    np.testing.assert_allclose(solver.cumulative_infiltration_m, first, atol=1e-12)


def test_segmented_hyetograph_continuation_preserves_green_ampt_state():
    storm = Hyetograph(
        node_times_s=np.array([0.0, 1.0, 2.0, 3.0]),
        node_rates_m_per_s=np.array([0.001, 0.004, 0.002, 0.0]),
        storm_type="synthetic", metadata={},
    )

    def build():
        return make_solver(ks=0.001, psi=0.1, deficit=0.25, maximum_dt=0.5,
                           adaptive_dt_growth_factor=None)

    continuous = build()
    continuous.run_device_loop(3.0, sync_interval=4, rainfall_hyetograph=storm)
    segmented = build()
    segmented.run_device_loop(1.0, sync_interval=4, rainfall_hyetograph=storm)
    initial_segment_f = float(segmented.cumulative_infiltration_m[0, 0])
    segmented.run_device_loop(2.0, sync_interval=4, rainfall_hyetograph=storm)
    assert float(segmented.cumulative_infiltration_m[0, 0]) > initial_segment_f
    np.testing.assert_allclose(segmented.cumulative_infiltration_m, continuous.cumulative_infiltration_m, atol=2e-6)
    np.testing.assert_allclose(segmented.get_water_depth(), continuous.get_water_depth(), atol=2e-6)


def test_cpu_warp_source_parity_when_cuda_is_available():
    pytest.importorskip("warp")
    cpu = make_solver(ks=0.001, psi=0.2, deficit=0.3, device="cpu")
    cpu.set_uniform_rainfall_rate(0.003)
    cpu.set_water_depth(np.zeros((1, 1)))
    cpu.step(1.0, fixed_dt=1.0)
    try:
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA unavailable")
        gpu = make_solver(ks=0.001, psi=0.2, deficit=0.3, device="cuda:0")
        gpu.set_uniform_rainfall_rate(0.003)
        gpu.set_water_depth(np.zeros((1, 1)))
        gpu.step(1.0, fixed_dt=1.0)
        np.testing.assert_allclose(gpu.get_water_depth(), cpu.get_water_depth(), atol=2e-6)
        np.testing.assert_allclose(gpu.cumulative_infiltration_m, cpu.cumulative_infiltration_m, atol=2e-6)
    except RuntimeError as exc:
        pytest.skip(f"CUDA initialization unavailable: {exc}")


def test_constructor_validates_all_three_green_ampt_fields():
    with pytest.raises(ValueError, match="requires Ks"):
        LocalInertialSolver(
            ny=1, nx=1, dx=1.0, dy=1.0,
            bed_elevation=np.zeros((1, 1)), manning_n=np.ones((1, 1)) * 0.05,
            active_mask=np.ones((1, 1)), infiltration_method="green_ampt",
            infiltration_ks_m_per_s=0.001, infiltration_psi_f_m=0.1,
        )


@pytest.mark.parametrize("use_fused", [False, True])
def test_fused_and_unfused_source_paths_match(use_fused):
    solver = LocalInertialSolver(
        ny=1, nx=1, dx=1.0, dy=1.0,
        bed_elevation=np.zeros((1, 1)), manning_n=np.full((1, 1), 0.05),
        active_mask=np.ones((1, 1), dtype=np.uint8), device="cpu",
        use_fused_kernels=use_fused, infiltration_method="green_ampt",
        infiltration_ks_m_per_s=0.001, infiltration_psi_f_m=0.2,
        infiltration_delta_theta=0.3, maximum_dt=1.0,
        adaptive_dt_growth_factor=None,
    )
    solver.set_uniform_rainfall_rate(0.02)
    solver.set_water_depth(np.zeros((1, 1)))
    solver.step(1.0, fixed_dt=1.0)
    expected = green_ampt_rainfall_increment_m(0.0, 0.001, 0.2, 0.3, 0.02, 1.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(expected, abs=2e-6)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.02 - expected, abs=2e-6)


def test_device_loop_and_lts_match_global_green_ampt_source_update():
    def build():
        solver = make_solver(
            ks=0.001, psi=0.2, deficit=0.3, maximum_dt=0.25,
            adaptive_dt_growth_factor=None,
        )
        solver.set_uniform_rainfall_rate(0.003)
        solver.set_water_depth(np.zeros((1, 1)))
        return solver

    global_solver = build()
    for _ in range(4):
        global_solver.step(0.25, fixed_dt=0.25)
    device_loop = build()
    device_loop.run_device_loop(1.0, sync_interval=4)
    lts = build()
    lts.run_spatially_adaptive_loop(
        duration=1.0, max_levels=0, wet_max_level=0,
        accuracy_fraction=0.25, accuracy_depth_scale=1e-3,
    )
    for solver in (device_loop, lts):
        np.testing.assert_allclose(
            solver.get_water_depth(), global_solver.get_water_depth(), atol=2e-6
        )
        np.testing.assert_allclose(
            solver.cumulative_infiltration_m,
            global_solver.cumulative_infiltration_m,
            atol=2e-6,
        )


def test_subgrid_green_ampt_uses_available_water_and_conserves_mass():
    native = np.zeros((5, 5), dtype=np.float64)
    solver = LocalInertialSolver(
        ny=1, nx=1, dx=5.0, dy=5.0,
        bed_elevation=np.zeros((1, 1)), manning_n=np.full((1, 1), 0.05),
        active_mask=np.ones((1, 1), dtype=np.uint8), device="cpu",
        hydraulic_geometry_mode="traditional", subgrid_bed_elevation=native,
        subgrid_resolution_m=1.0, infiltration_method="green_ampt",
        infiltration_ks_m_per_s=1e-5, infiltration_psi_f_m=0.2,
        infiltration_delta_theta=0.3, maximum_dt=1.0,
        adaptive_dt_growth_factor=None,
    )
    solver.set_water_depth(np.array([[0.001]]))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.cumulative_infiltration_m[0, 0] <= 0.001 + 1e-12
    assert solver.get_water_depth()[0, 0] >= 0.0
    assert abs(compute_diagnostics(solver)["mass_balance_residual"]) < 1e-10


def test_ponded_timestep_refinement_converges_to_integrated_solution():
    def run(step_s: float) -> float:
        solver = make_solver(
            ks=2.0e-5, psi=0.25, deficit=0.35, maximum_dt=step_s,
            adaptive_dt_growth_factor=None,
        )
        solver.set_water_depth(np.array([[5.0]]))
        for _ in range(round(3600.0 / step_s)):
            solver.step(step_s, fixed_dt=step_s)
        return float(solver.cumulative_infiltration_m[0, 0])

    coarse = run(60.0)
    fine = run(10.0)
    expected = green_ampt_infiltration_increment_m(0.0, 2.0e-5, 0.25, 0.35, 3600.0)
    assert fine == pytest.approx(expected, rel=2e-4)
    assert coarse == pytest.approx(fine, rel=2e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_green_ampt_solver_matches_numpy_over_wide_parameter_range(device):
    import warp as wp

    if device == "cuda:0" and not wp.is_device_available(device):
        pytest.skip("CUDA unavailable")
    f0 = np.array([[0.0, 1e-12, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0]])
    ks = np.array([[1e-3, 0.0, 1e-10, 1e-4, 1e-3, 1e-5, 1e-3, 1e-4]])
    psi = np.array([[0.01, 0.1, 0.5, 1.0, 0.061, 0.168, 1.0, 10.0]])
    deficit = np.array([[0.0, 0.9, 0.3, 0.1, 0.4, 0.49, 1.0, 0.01]])
    dt = 10000.0
    solver = make_solver(ks=ks, psi=psi, deficit=deficit, nx=8, device=device)
    solver.set_water_depth(np.zeros((1, 8)))
    wp.copy(
        solver.cumulative_infiltration,
        wp.array(f0, dtype=solver.cumulative_infiltration.dtype, device=device),
    )
    solver.set_uniform_rainfall_rate(10.0)
    solver.step(dt, fixed_dt=dt)
    expected = green_ampt_rainfall_increment_m(f0, ks, psi, deficit, 10.0, dt)
    actual = solver.cumulative_infiltration_m - f0
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-7)
