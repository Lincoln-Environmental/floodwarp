import numpy as np
import pytest
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.infiltration import exponential_infiltration_capacity
from floodwarp.local_inertial_kernels import (
    DL_DT,
    DL_RAIN_RATE,
    DL_SIZE,
    device_loop_apply_sources_kernel,
    lts_apply_sources_indexed_kernel,
)
from floodwarp.local_inertial_solver import LocalInertialSolver


def make_solver(
    ny=1,
    nx=1,
    *,
    f0=0.01,
    fd=0.1,
    k=0.001,
    active_mask=None,
    **kwargs,
):
    if active_mask is None:
        active_mask = np.ones((ny, nx), dtype=np.uint8)
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=1.0,
        dy=1.0,
        bed_elevation=np.zeros((ny, nx)),
        manning_n=np.full((ny, nx), 0.05),
        active_mask=active_mask,
        device="cpu",
        infiltration_method="exponential",
        infiltration_f0_m_per_s=f0,
        infiltration_fd_m=fd,
        infiltration_k_m_per_s=k,
        **kwargs,
    )


def test_capacity_formula_initial_exact_decay_and_limiting_values():
    f0 = 0.01
    fd = 0.04
    k = 0.001
    assert exponential_infiltration_capacity(0.0, f0, fd, k) == pytest.approx(f0)
    assert exponential_infiltration_capacity(fd, f0, fd, k) == pytest.approx(
        k + (f0 - k) / np.e
    )
    assert exponential_infiltration_capacity(1.0e6, f0, fd, k) == pytest.approx(k)


def test_no_available_water_leaves_cumulative_state_unchanged():
    solver = make_solver()
    solver.set_water_depth(np.zeros((1, 1)))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.0)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.0)


def test_rainfall_below_capacity_is_fully_infiltrated():
    solver = make_solver(f0=0.01)
    solver.set_uniform_rainfall_rate(0.002)
    solver.set_water_depth(np.zeros((1, 1)))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.002)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.0)


def test_rainfall_above_capacity_leaves_excess_surface_water():
    solver = make_solver(f0=0.01)
    solver.set_uniform_rainfall_rate(0.02)
    solver.set_water_depth(np.zeros((1, 1)))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.01)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.01)


def test_ponded_water_infiltrates_without_rainfall():
    solver = make_solver(f0=0.005)
    solver.set_water_depth(np.array([[0.01]]))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.005)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.005)


def test_scalar_grid_and_mixed_parameter_inputs():
    solver = make_solver(
        ny=1,
        nx=2,
        f0=np.array([[0.005, 0.02]]),
        fd=0.1,
        k=np.array([[0.001, 0.01]]),
    )
    solver.set_uniform_rainfall_rate(0.02)
    solver.set_water_depth(np.zeros((1, 2)))
    solver.step(1.0, fixed_dt=1.0)
    np.testing.assert_allclose(solver.cumulative_infiltration_m, [[0.005, 0.02]])
    assert float(solver.get_water_depth().sum()) == pytest.approx(0.015)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fd": 0.0},
        {"f0": -0.01},
        {"k": -0.01},
        {"f0": 0.001, "k": 0.002},
        {"f0": np.zeros((2, 2)), "fd": np.ones((1, 1)), "k": 0.0},
        {"f0": np.array([[np.nan]]), "fd": 0.1, "k": 0.0},
    ],
)
def test_invalid_exponential_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_solver(**kwargs)


def test_mass_balance_and_diagnostics_for_exponential_mode():
    solver = make_solver(maximum_dt=0.25)
    solver.set_uniform_rainfall_rate(0.02)
    solver.set_water_depth(np.zeros((1, 1)))
    solver.run_spatially_adaptive_loop(
        duration=2.0,
        max_levels=0,
        wet_max_level=0,
        accuracy_fraction=0.25,
        accuracy_depth_scale=1.0e-3,
    )
    diagnostics = compute_diagnostics(solver)
    assert diagnostics["total_infiltration_volume_m3"] > 0.0
    assert diagnostics["maximum_cumulative_infiltration_m"] > 0.0
    assert abs(diagnostics["mass_balance_residual"]) < 1.0e-12
    assert diagnostics["cumulative_rainfall_volume"] == pytest.approx(
        diagnostics["total_surface_water_volume"]
        + diagnostics["total_infiltration_volume_m3"]
    )


def test_constant_mode_regression_path_remains_available():
    solver = LocalInertialSolver(
        ny=1,
        nx=1,
        dx=1.0,
        dy=1.0,
        bed_elevation=np.zeros((1, 1)),
        manning_n=np.full((1, 1), 0.05),
        active_mask=np.ones((1, 1), dtype=np.uint8),
        device="cpu",
    )
    solver.set_source_rates(rain=np.array([[0.002]]), infil=np.array([[0.001]]))
    solver.set_water_depth(np.zeros((1, 1)))
    solver.step(1.0, fixed_dt=1.0)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.001)
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.0)


def test_lts_matches_small_global_steps_for_a_flat_source_case():
    kwargs = dict(maximum_dt=0.25, adaptive_dt_growth_factor=None)
    global_solver = make_solver(**kwargs)
    lts_solver = make_solver(**kwargs)
    for solver in (global_solver, lts_solver):
        solver.set_uniform_rainfall_rate(0.02)
        solver.set_water_depth(np.zeros((1, 1)))
    for _ in range(4):
        global_solver.step(0.25, fixed_dt=0.25)
    lts_solver.run_spatially_adaptive_loop(
        duration=1.0,
        max_levels=0,
        wet_max_level=0,
        accuracy_fraction=0.25,
        accuracy_depth_scale=1.0e-3,
    )
    np.testing.assert_allclose(
        lts_solver.get_water_depth(), global_solver.get_water_depth(), rtol=0.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        lts_solver.cumulative_infiltration_m,
        global_solver.cumulative_infiltration_m,
        rtol=0.0,
        atol=2.0e-7,
    )


def make_partially_wet_subgrid_solver(
    *,
    ny=1,
    nx=1,
    device="cpu",
    f0=30.0 / 1000.0 / 3600.0,
    fd=1000.0,
    k=0.0,
    maximum_dt=3600.0,
):
    """Build cells with five low native subcells (a 20% wet fraction)."""
    factor = 5
    native = np.full((ny * factor, nx * factor), 10.0)
    for j in range(ny):
        for i in range(nx):
            native[j * factor, i * factor:(i + 1) * factor] = 0.0
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.full((ny, nx), 8.0),
        manning_n=np.full((ny, nx), 0.05),
        active_mask=np.ones((ny, nx), dtype=np.int32),
        device=device,
        maximum_dt=maximum_dt,
        adaptive_dt_growth_factor=None,
        hydraulic_geometry_mode="traditional",
        subgrid_bed_elevation=native,
        subgrid_resolution_m=1.0,
        infiltration_method="exponential",
        infiltration_f0_m_per_s=f0,
        infiltration_fd_m=fd,
        infiltration_k_m_per_s=k,
    )


def test_subgrid_partially_wet_ponded_infiltration_uses_wet_fraction():
    solver = make_partially_wet_subgrid_solver()
    solver.set_water_depth(np.array([[0.2]]))
    solver.step(3600.0, fixed_dt=3600.0)

    # C = 30 mm over the coarse plan area; only 5 / 25 subcells are wet.
    expected_depth = 0.2 * 0.03
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(
        expected_depth, rel=2.0e-5
    )
    assert solver.get_water_depth()[0, 0] == pytest.approx(
        0.2 - expected_depth, rel=2.0e-5
    )


def test_subgrid_dry_cell_rainfall_infiltrates_before_wet_fraction_limit():
    solver = make_partially_wet_subgrid_solver()
    solver.set_water_depth(np.zeros((1, 1)))
    solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
    solver.step(3600.0, fixed_dt=3600.0)

    # Dry terrain still admits rainfall over its full plan area.
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.01, rel=2.0e-5)
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.0, abs=1.0e-12)


def test_subgrid_rainfall_and_ponding_split_remaining_capacity():
    solver = make_partially_wet_subgrid_solver()
    solver.set_water_depth(np.array([[0.2]]))
    solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
    solver.step(3600.0, fixed_dt=3600.0)

    # 10 mm rain infiltrates across the whole cell, then 20% of the remaining
    # 20 mm capacity is drawn from ponded water: 10 + 0.2 * 20 = 14 mm.
    expected_depth = 0.014
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(
        expected_depth, rel=2.0e-5
    )
    assert solver.get_water_depth()[0, 0] == pytest.approx(
        0.2 + 0.01 - expected_depth, rel=2.0e-5
    )


def test_subgrid_fully_wet_matches_coarse_exponential_source_update():
    f0 = 30.0 / 1000.0 / 3600.0
    common = dict(
        f0=f0,
        fd=1000.0,
        k=0.0,
        maximum_dt=3600.0,
    )
    coarse = LocalInertialSolver(
        ny=1,
        nx=1,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((1, 1)),
        manning_n=np.full((1, 1), 0.05),
        active_mask=np.ones((1, 1), dtype=np.int32),
        device="cpu",
        adaptive_dt_growth_factor=None,
        infiltration_method="exponential",
        infiltration_f0_m_per_s=common["f0"],
        infiltration_fd_m=common["fd"],
        infiltration_k_m_per_s=common["k"],
        maximum_dt=common["maximum_dt"],
    )
    native = np.zeros((5, 5))
    subgrid = LocalInertialSolver(
        ny=1,
        nx=1,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((1, 1)),
        manning_n=np.full((1, 1), 0.05),
        active_mask=np.ones((1, 1), dtype=np.int32),
        device="cpu",
        adaptive_dt_growth_factor=None,
        hydraulic_geometry_mode="traditional",
        subgrid_bed_elevation=native,
        subgrid_resolution_m=1.0,
        infiltration_method="exponential",
        infiltration_f0_m_per_s=common["f0"],
        infiltration_fd_m=common["fd"],
        infiltration_k_m_per_s=common["k"],
        maximum_dt=common["maximum_dt"],
    )
    for solver in (coarse, subgrid):
        solver.set_water_depth(np.array([[0.05]]))
        solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
        solver.step(3600.0, fixed_dt=3600.0)

    np.testing.assert_allclose(
        subgrid.get_water_depth(), coarse.get_water_depth(), rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        subgrid.cumulative_infiltration_m,
        coarse.cumulative_infiltration_m,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_subgrid_exponential_mass_balance_with_rain_and_other_source():
    solver = make_partially_wet_subgrid_solver()
    solver.set_water_depth(np.array([[0.2]]))
    solver.set_source_rates(other=np.array([[5.0 / 1000.0 / 3600.0]]))
    solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
    solver.step(3600.0, fixed_dt=3600.0)

    diagnostics = compute_diagnostics(solver)
    assert abs(diagnostics["mass_balance_residual"]) < 1.0e-12
    assert diagnostics["cumulative_rainfall_volume"] + diagnostics["cumulative_externally_imposed_source_volume"] == pytest.approx(
        diagnostics["total_surface_water_volume"]
        + diagnostics["total_infiltration_volume_m3"]
        - 0.2 * 25.0,
        abs=1.0e-12,
    )


def test_subgrid_exponential_lts_multilevel_matches_small_global_steps():
    global_solver = make_partially_wet_subgrid_solver(
        ny=8, nx=8, maximum_dt=0.05
    )
    lts_solver = make_partially_wet_subgrid_solver(
        ny=8, nx=8, maximum_dt=60.0
    )
    initial = np.zeros((8, 8))
    initial[:2, :2] = 0.2
    for solver in (global_solver, lts_solver):
        solver.set_water_depth(initial.copy())

    for _ in range(200):
        global_solver.step(0.05, fixed_dt=0.05)
    stats = lts_solver.run_spatially_adaptive_loop(
        duration=10.0,
        max_levels=4,
        wet_max_level=0,
        drainage_max_level=-1,
        accuracy_fraction=0.25,
        accuracy_depth_scale=1.0e-3,
    )
    assert sum(stats["level_presence"][1:]) > 0
    # Genuine multilevel momentum introduces spatial field differences, but
    # its wet-fraction-aware source state remains close to the fine global
    # reference in domain-integrated infiltration and storage.
    assert lts_solver.cumulative_infiltration_m.sum() == pytest.approx(
        global_solver.cumulative_infiltration_m.sum(), rel=0.1
    )
    assert lts_solver.get_water_depth().sum() == pytest.approx(
        global_solver.get_water_depth().sum(), rel=2.0e-3
    )


def test_subgrid_exponential_device_loop_matches_host_stepping():
    host = make_partially_wet_subgrid_solver(maximum_dt=0.25)
    device_loop = make_partially_wet_subgrid_solver(maximum_dt=0.25)
    for solver in (host, device_loop):
        solver.set_water_depth(np.array([[0.2]]))

    for _ in range(4):
        host.step(0.25, fixed_dt=0.25)
    device_loop.run_device_loop(1.0, sync_interval=4)

    np.testing.assert_allclose(
        device_loop.get_water_depth(), host.get_water_depth(), rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        device_loop.cumulative_infiltration_m,
        host.cumulative_infiltration_m,
        rtol=0.0,
        atol=1.0e-12,
    )


def _committed_subgrid_source_result(device: str, execution_path: str) -> tuple[float, float]:
    """Run the host or device-loop source kernel with a prescribed outflow."""
    solver = make_partially_wet_subgrid_solver(
        device=device,
        f0=0.1,
        fd=1000.0,
        k=0.1,
        maximum_dt=1.0,
    )
    # At 2.2 m storage, the 20% low subcells and 80% high subcells are all
    # wet. A 2.0 m committed outflow leaves 0.2 m in the low 20% channel.
    solver.set_water_depth(np.array([[2.2]]))
    committed_qx = wp.array(
        np.array([[0.0, 10.0]]), dtype=WP_FLOAT, device=device
    )
    committed_qy = wp.zeros((2, 1), dtype=WP_FLOAT, device=device)

    if execution_path == "host_source":
        solver._apply_sources(
            dt_wp=WP_FLOAT(1.0),
            defer=False,
            flush=True,
            committed_qx=committed_qx,
            committed_qy=committed_qy,
        )
    elif execution_path == "device_loop_source":
        control_host = np.zeros(DL_SIZE, dtype=np.float64)
        control_host[DL_DT] = 1.0
        control_host[DL_RAIN_RATE] = 0.0
        control = wp.array(control_host, dtype=wp.float64, device=device)
        wp.launch(
            kernel=device_loop_apply_sources_kernel,
            dim=(1, 1),
            inputs=[
                solver.water_depth,
                solver.rainfall_rate,
                solver.rainfall_spatial_multiplier,
                solver.other_source_rate,
                solver.infiltration_capacity,
                solver.infiltration_f0,
                solver.infiltration_fd,
                solver.infiltration_k,
                solver.green_ampt_ks,
                solver.green_ampt_psi,
                solver.green_ampt_delta_theta,
                solver.cumulative_infiltration,
                solver.infiltration_method,
                solver._source_subgrid_infiltration_enabled,
                solver._source_subgrid_cell_z,
                solver._source_subgrid_cell_breakpoints,
                solver._source_subgrid_cell_count,
                solver.active_mask,
                control,
                0,
                solver.source_mode,
                solver.dry_depth,
                committed_qx,
                committed_qy,
                1,
                solver.applied_rain,
                solver.applied_other,
                solver.applied_infil,
                solver.dx,
                solver.dy,
                solver.cumulative_totals,
                1,
                0,
                solver.additional_source_rate,
                solver.infiltration_exclusion_mask,
                solver.infiltration_cap_active,
                solver.infiltration_cap_remaining_m,
            ],
            device=device,
        )
    else:
        raise ValueError(f"Unknown execution path: {execution_path}")

    wp.synchronize_device(device)
    return (
        float(solver.get_water_depth()[0, 0]),
        float(solver.cumulative_infiltration_m[0, 0]),
    )


@pytest.mark.parametrize("execution_path", ["host_source", "device_loop_source"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_subgrid_committed_outflow_uses_post_outflow_wet_fraction(
    device, execution_path
):
    if device == "cuda:0" and not wp.is_device_available(device):
        pytest.skip("CUDA is unavailable")

    source_depth, cumulative = _committed_subgrid_source_result(
        device=device,
        execution_path=execution_path,
    )
    committed_outflow_depth = 2.0
    expected_infiltration = 0.2 * 0.1

    # Source application itself retains the committed flux for continuity.
    assert source_depth == pytest.approx(2.2 - expected_infiltration, abs=1.0e-12)
    assert cumulative == pytest.approx(expected_infiltration, abs=1.0e-12)
    assert source_depth - committed_outflow_depth >= 0.0

    # Once continuity consumes the prescribed outflow, H_initial - Q_out =
    # H_final + I exactly. This distinguishes the correct 20% result from
    # the erroneous pre-outflow full-wet result of 0.1 m.
    post_continuity_depth = source_depth - committed_outflow_depth
    assert 2.2 - committed_outflow_depth == pytest.approx(
        post_continuity_depth + cumulative,
        abs=1.0e-12,
    )


def test_lts_subgrid_source_uses_effective_pending_storage_for_wet_fraction():
    solver = make_partially_wet_subgrid_solver(
        f0=0.1,
        fd=1000.0,
        k=0.1,
        maximum_dt=1.0,
    )
    # The committed 0.2 m depth wets only the five low subcells. Pending
    # hydraulic volume raises effective storage to 2.2 m, which is fully wet.
    solver.set_water_depth(np.array([[0.2]]))
    cell_area = 25.0
    pending = wp.array(np.array([[2.0 * cell_area]]), dtype=wp.float64, device="cpu")
    effective_depth = wp.zeros((1, 1), dtype=WP_FLOAT, device="cpu")
    indices = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    error_flag = wp.zeros(1, dtype=wp.int32, device="cpu")

    wp.launch(
        kernel=lts_apply_sources_indexed_kernel,
        dim=1,
        inputs=[
            indices,
            0,
            solver.water_depth,
            pending,
            effective_depth,
            solver.rainfall_rate,
            solver.rainfall_spatial_multiplier,
            solver.other_source_rate,
            solver.infiltration_capacity,
            solver.infiltration_f0,
            solver.infiltration_fd,
            solver.infiltration_k,
            solver.green_ampt_ks,
            solver.green_ampt_psi,
            solver.green_ampt_delta_theta,
            solver.cumulative_infiltration,
            solver.infiltration_method,
            solver._source_subgrid_infiltration_enabled,
            solver._source_subgrid_cell_z,
            solver._source_subgrid_cell_breakpoints,
            solver._source_subgrid_cell_count,
            solver.active_mask,
            WP_FLOAT(1.0),
            WP_FLOAT(0.0),
            0,
            solver.source_mode,
            solver.dry_depth,
            solver.applied_rain,
            solver.applied_other,
            solver.applied_infil,
            solver.dx,
            solver.dy,
            solver.negative_depth_tolerance,
            error_flag,
        ],
        device="cpu",
    )
    wp.synchronize_device("cpu")

    # Effective storage 2.2 m is fully wet, so all 0.1 m of capacity
    # infiltrates. Using committed water_depth (0.2 m) would incorrectly
    # remove only 0.02 m at the 20% wet fraction.
    expected_infiltration = 0.1
    assert float(effective_depth.numpy()[0, 0]) == pytest.approx(
        2.2 - expected_infiltration, abs=1.0e-12
    )
    assert float(solver.cumulative_infiltration_m[0, 0]) == pytest.approx(
        expected_infiltration, abs=1.0e-12
    )
    assert float(solver.applied_infil.numpy()[0, 0]) == pytest.approx(
        expected_infiltration, abs=1.0e-12
    )
    assert float(pending.numpy()[0, 0]) == pytest.approx(
        (2.0 - expected_infiltration) * cell_area, abs=1.0e-12
    )
    assert int(error_flag.numpy()[0]) == 0

    # The conservative pending register represents a 0.1 m plan-area loss:
    # h_committed + V_pending / A falls from 2.2 m to 2.1 m.
    final_effective_storage = (
        float(solver.get_water_depth()[0, 0]) + float(pending.numpy()[0, 0]) / cell_area
    )
    assert 2.2 - final_effective_storage == pytest.approx(
        expected_infiltration, abs=1.0e-12
    )
