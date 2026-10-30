"""Regression tests: ACC-profile infiltration must respect committed outfluxes.

Profiles with ``apply_sources_after_momentum=True`` (``lisflood_fp_acc``,
``lisflood_fp_acc_parity``) run the donor-cell outflow limiter BEFORE sources
are applied.  The limiter budgets outfluxes against pre-source depth, so an
infiltration sink applied afterwards could push a fully draining cell
negative by up to ``infil_rate * dt`` and trip the negative-depth invariant
(observed in the nested terrain benchmark, straight_valley child at
t=4210 s).  The source kernel now additionally caps infiltration at the
depth remaining after the committed (post-limiter) outfluxes, mirroring
LISFLOOD-FP's FPInfiltration cap at the post-update depth.
"""
import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import BOUNDARY_CLOSED
from floodwarp.config import _float_env
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_solver import LocalInertialSolver


def get_tol():
    if _float_env == "float64":
        return {"mass_atol": 1e-12}
    return {"mass_atol": 1e-5}


def build_draining_case(device):
    """One thin-film cell forced to drain fully within a single step."""
    wp.init()
    ny, nx = 1, 2
    bed = np.array([[50.0, 0.0]])  # extreme slope drives a limiter-capped drain
    manning = np.full((ny, nx), 0.035)
    mask = np.ones((ny, nx), dtype=np.int32)
    solver = LocalInertialSolver(
        ny=ny, nx=nx, dx=10.0, dy=10.0,
        bed_elevation=bed, manning_n=manning, active_mask=mask,
        device=device, validation_profile="lisflood_fp_acc",
    )
    # Thin film above the ACC dry/infiltration gate (1e-3 m).
    solver.set_water_depth(np.array([[2.0e-3, 0.0]]))
    # Strong infiltration: uncapped this removes 5e-5 m in the step below,
    # far beyond the 1e-8 m negative-depth tolerance.
    solver.set_source_rates(infil=np.full((ny, nx), 1.0e-4))
    return solver


def run_draining_step(device):
    solver = build_draining_case(device)
    solver.step(remaining_time=0.5, fixed_dt=0.5)
    return solver


def assert_draining_step_survives(solver):
    tol = get_tol()
    h_final = solver.get_water_depth()
    assert np.all(h_final >= 0.0)
    diag = compute_diagnostics(solver)
    assert abs(diag["mass_balance_residual"]) < tol["mass_atol"]
    # The infiltration sink must have been capped below its nominal 5e-5 m
    # (only the limiter slack remaining after the committed drain applies).
    full_infiltration = 1.0e-4 * 0.5 * 100.0
    assert 0.0 <= diag["cumulative_infiltration_volume"] < full_infiltration


def test_acc_infiltration_capped_at_committed_outflow_cpu():
    assert_draining_step_survives(run_draining_step("cpu"))


def test_acc_infiltration_capped_at_committed_outflow_cuda():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    assert_draining_step_survives(run_draining_step("cuda:0"))


def run_host_stepped(solver, duration):
    end_time = solver.simulation_time + duration
    while solver.simulation_time < end_time:
        remaining = end_time - solver.simulation_time
        dt = solver.calculate_stable_timestep(remaining_time=remaining)
        solver.step(remaining_time=remaining, fixed_dt=dt)


def compare_host_vs_device_loop(device):
    duration = 2.0
    host = build_draining_case(device)
    device_solver = build_draining_case(device)
    run_host_stepped(host, duration)
    device_solver.run_device_loop(duration, sync_interval=4)
    assert host.simulation_time == device_solver.simulation_time == duration
    assert np.array_equal(host.get_water_depth(), device_solver.get_water_depth())
    assert np.array_equal(host.get_qx(), device_solver.get_qx())
    assert np.array_equal(host.get_qy(), device_solver.get_qy())
    host_totals = host.cumulative_accounting()
    device_totals = device_solver.cumulative_accounting()
    for key in host_totals:
        assert device_totals[key] == pytest.approx(
            host_totals[key], rel=1.0e-9, abs=1.0e-12
        )


def test_acc_infiltration_draining_device_loop_matches_host_cpu():
    compare_host_vs_device_loop("cpu")


def test_acc_infiltration_draining_device_loop_matches_host_cuda():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    compare_host_vs_device_loop("cuda:0")


def test_acc_exponential_infiltration_is_capped_at_committed_outflow_cpu():
    """The rain/ponding split must retain the post-limiter availability cap."""
    solver = build_draining_case("cpu")
    solver.set_exponential_infiltration(
        infiltration_f0_m_per_s=1.0e-4,
        infiltration_fd_m=1000.0,
        infiltration_k_m_per_s=1.0e-4,
    )
    solver.step(remaining_time=0.5, fixed_dt=0.5)
    assert_draining_step_survives(solver)
