"""Constant-method subgrid infiltration regressions.

The constant method must apply its capacity through the same shared
rain-first subgrid logic as the exponential method (rainfall over the full
plan area, remaining capacity scaled by the subgrid wet fraction of the
effective ponded storage), without maintaining cumulative infiltration
state. These tests mirror ``test_exponential_infiltration.py``.
"""

import numpy as np
import pytest
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_kernels import (
    DL_DT,
    DL_RAIN_RATE,
    DL_SIZE,
    device_loop_apply_sources_kernel,
    lts_apply_sources_indexed_kernel,
)
from floodwarp.local_inertial_solver import LocalInertialSolver


def make_constant_subgrid_solver(
    *,
    ny=1,
    nx=1,
    device="cpu",
    rate_m_per_s=30.0 / 1000.0 / 3600.0,
    maximum_dt=3600.0,
):
    """Build cells with five low native subcells (a 20% wet fraction)."""
    factor = 5
    native = np.full((ny * factor, nx * factor), 10.0)
    for j in range(ny):
        for i in range(nx):
            native[j * factor, i * factor:(i + 1) * factor] = 0.0
    solver = LocalInertialSolver(
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
        infiltration_method="constant",
    )
    solver.set_source_rates(infil=np.full((ny, nx), rate_m_per_s))
    return solver, rate_m_per_s


def _set_constant_rate(solver, rate_m_per_s, ny=1, nx=1):
    solver.set_source_rates(
        infil=np.full((ny, nx), rate_m_per_s)
    )


def test_constant_has_no_cumulative_state_and_uses_shared_application():
    solver, rate = make_constant_subgrid_solver()
    _set_constant_rate(solver, rate)
    solver.set_water_depth(np.array([[0.2]]))
    solver.step(3600.0, fixed_dt=3600.0)

    # Only the 20% wet fraction of the 30 mm capacity draws from ponded
    # storage: 0.2 * 0.03 = 0.006 m; the constant method keeps no cumulative
    # infiltration state.
    expected_depth = 0.006
    assert solver.get_water_depth()[0, 0] == pytest.approx(
        0.2 - expected_depth, rel=2.0e-5
    )
    assert solver.cumulative_infiltration_m[0, 0] == pytest.approx(0.0)


def test_constant_dry_cell_rainfall_infiltrates_before_wet_fraction_limit():
    solver, rate = make_constant_subgrid_solver()
    solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
    solver.set_water_depth(np.zeros((1, 1)))
    solver.step(3600.0, fixed_dt=3600.0)

    # Dry terrain still admits rainfall over its full plan area.
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert float(solver.applied_infil.numpy()[0, 0]) == pytest.approx(
        0.01, rel=2.0e-5
    )


def test_constant_rainfall_and_ponding_split_remaining_capacity():
    solver, rate = make_constant_subgrid_solver()
    _set_constant_rate(solver, rate)
    solver.set_water_depth(np.array([[0.2]]))
    solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
    solver.step(3600.0, fixed_dt=3600.0)

    # 10 mm rain infiltrates across the whole cell, then 20% of the remaining
    # 20 mm capacity is drawn from ponded water: 10 + 0.2 * 20 = 14 mm.
    expected_depth = 0.014
    assert float(solver.applied_infil.numpy()[0, 0]) == pytest.approx(
        expected_depth, rel=2.0e-5
    )
    assert solver.get_water_depth()[0, 0] == pytest.approx(
        0.2 + 0.01 - expected_depth, rel=2.0e-5
    )


def test_constant_subgrid_fully_wet_matches_coarse_constant_source_update():
    rate = 30.0 / 1000.0 / 3600.0
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
        infiltration_method="constant",
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
        infiltration_method="constant",
    )
    for solver in (coarse, subgrid):
        solver.set_source_rates(infil=np.array([[rate]]))
        solver.set_water_depth(np.array([[0.05]]))
        solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
        solver.step(3600.0, fixed_dt=3600.0)

    np.testing.assert_allclose(
        subgrid.get_water_depth(), coarse.get_water_depth(), rtol=0.0, atol=1.0e-12
    )


def test_constant_subgrid_mass_balance_with_rain_and_other_source():
    solver, rate = make_constant_subgrid_solver()
    _set_constant_rate(solver, rate)
    solver.set_water_depth(np.array([[0.2]]))
    solver.set_source_rates(other=np.array([[5.0 / 1000.0 / 3600.0]]))
    solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)
    solver.step(3600.0, fixed_dt=3600.0)

    diagnostics = compute_diagnostics(solver)
    assert abs(diagnostics["mass_balance_residual"]) < 1.0e-12
    assert diagnostics["cumulative_rainfall_volume"] + diagnostics[
        "cumulative_externally_imposed_source_volume"
    ] == pytest.approx(
        diagnostics["total_surface_water_volume"]
        + diagnostics["total_infiltration_volume_m3"]
        - 0.2 * 25.0,
        abs=1.0e-12,
    )


def test_constant_coarse_mode_regression_values_are_unchanged():
    """Constant + coarse keeps the historical source update exactly."""
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


def _committed_constant_source_result(device: str, execution_path: str):
    """Run the host or device-loop source kernel with a prescribed outflow."""
    solver, rate = make_constant_subgrid_solver(device=device, rate_m_per_s=0.1)
    _set_constant_rate(solver, rate)
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
        float(solver.applied_infil.numpy()[0, 0]),
    )


@pytest.mark.parametrize("execution_path", ["host_source", "device_loop_source"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_constant_committed_outflow_uses_post_outflow_wet_fraction(
    device, execution_path
):
    if device == "cuda:0" and not wp.is_device_available(device):
        pytest.skip("CUDA is unavailable")

    source_depth, applied_infil = _committed_constant_source_result(
        device=device,
        execution_path=execution_path,
    )
    committed_outflow_depth = 2.0
    capacity_depth = 0.1  # rate 0.1 m/s over the 1 s source interval
    expected_infiltration = 0.2 * capacity_depth

    # Source application itself retains the committed flux for continuity.
    assert source_depth == pytest.approx(2.2 - expected_infiltration, abs=1.0e-12)
    assert applied_infil == pytest.approx(expected_infiltration, abs=1.0e-12)
    assert source_depth - committed_outflow_depth >= 0.0

    # Once continuity consumes the prescribed outflow, H_initial - Q_out =
    # H_final + I exactly. This distinguishes the correct 20% result from
    # the erroneous pre-outflow full-wet result.
    post_continuity_depth = source_depth - committed_outflow_depth
    assert 2.2 - committed_outflow_depth == pytest.approx(
        post_continuity_depth + applied_infil,
        abs=1.0e-12,
    )


def test_constant_lts_source_uses_effective_pending_storage_for_wet_fraction():
    solver, rate = make_constant_subgrid_solver(rate_m_per_s=0.1)
    _set_constant_rate(solver, rate)
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

    # Effective storage 2.2 m is fully wet, so the full 0.1 m of capacity
    # (rate 0.1 m/s x dt 1 s would be 0.1 m) infiltrates. Using the committed
    # water_depth (0.2 m) would instead remove only 0.02 m at the 20% wet
    # fraction.
    expected_infiltration = min(0.1, 2.2)
    assert float(effective_depth.numpy()[0, 0]) == pytest.approx(
        2.2 - expected_infiltration, abs=1.0e-12
    )
    # The constant method keeps no cumulative infiltration state.
    assert float(solver.cumulative_infiltration_m[0, 0]) == pytest.approx(0.0)
    assert float(solver.applied_infil.numpy()[0, 0]) == pytest.approx(
        expected_infiltration, abs=1.0e-12
    )
    assert float(pending.numpy()[0, 0]) == pytest.approx(
        (2.0 - expected_infiltration) * cell_area, abs=1.0e-12
    )
    assert int(error_flag.numpy()[0]) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_constant_subgrid_device_matches_host(device):
    if device == "cuda:0" and not wp.is_device_available(device):
        pytest.skip("CUDA is unavailable")

    host, rate = make_constant_subgrid_solver(device="cpu")
    device_solver, _ = make_constant_subgrid_solver(device=device)
    for solver in (host, device_solver):
        _set_constant_rate(solver, rate)
        solver.set_water_depth(np.array([[0.2]]))
        solver.set_uniform_rainfall_rate(10.0 / 1000.0 / 3600.0)

    for _ in range(4):
        host.step(0.25, fixed_dt=0.25)
    device_solver.run_device_loop(1.0, sync_interval=4)

    np.testing.assert_allclose(
        device_solver.get_water_depth(), host.get_water_depth(), rtol=0.0, atol=1.0e-12
    )
