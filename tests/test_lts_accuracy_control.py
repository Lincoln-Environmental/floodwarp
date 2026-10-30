"""Regression tests for the accuracy-controlled spatial LTS classifier.

These tests exercise the device-side classifier state machine (history,
hysteresis, wetting-front protection, reset behaviour) and the conservative
invariants of the multirate scheme on small CPU cases.
"""

import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import BOUNDARY_FREE_OUTFLOW
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.rainfall import Hyetograph, build_uniform_hyetograph


def create_valley_solver(ny=8, nx=8, device="cpu"):
    wp.init()
    downstream_slope = 0.002
    bed = np.zeros((ny, nx), dtype=np.float64)
    for j in range(ny):
        bed[j, :] = downstream_slope * (ny - 1 - j) * 10.0
    manning = np.full((ny, nx), 0.035, dtype=np.float64)
    mask = np.ones((ny, nx), dtype=np.int32)
    solver = LocalInertialSolver(
        ny, nx, 10.0, 10.0, bed, manning, mask, device=device,
        adaptive_dt_growth_factor=None,
    )
    boundary = solver.boundary_type_qy.numpy()
    boundary[ny, :] = BOUNDARY_FREE_OUTFLOW
    solver.set_boundary_type_qy(boundary)
    return solver


def run_lts(solver, duration, rainfall_hyetograph=None, **kwargs):
    defaults = {"max_levels": 6, "wet_max_level": 2}
    defaults.update(kwargs)
    return solver.run_spatially_adaptive_loop(
        duration=duration,
        rainfall_hyetograph=rainfall_hyetograph,
        **defaults,
    )


def test_accuracy_control_conserves_mass_with_rainfall():
    solver = create_valley_solver()
    hyetograph = build_uniform_hyetograph(duration_s=3600.0, total_depth_m=0.02)
    result = run_lts(solver, duration=1800.0, rainfall_hyetograph=hyetograph)
    diagnostics = compute_diagnostics(solver)
    assert abs(diagnostics["relative_mass_balance_error"]) < 1.0e-9
    assert result["macro_steps"] > 0
    assert result["base_steps"] >= result["macro_steps"]


def test_accuracy_control_reports_classification_statistics():
    solver = create_valley_solver()
    # A deep pond in one corner pins the CFL/base timestep while the rest of
    # the domain starts dry on coarse levels.  Rain from t=0 then forces the
    # coarse dry cells to demote as their effective storage grows.
    initial_depth = np.zeros((8, 8), dtype=np.float64)
    initial_depth[:2, :2] = 0.5
    solver.set_water_depth(initial_depth)
    hyetograph = build_uniform_hyetograph(duration_s=3600.0, total_depth_m=0.1)
    result = run_lts(solver, duration=600.0, rainfall_hyetograph=hyetograph)
    control = result["accuracy_control"]
    active_cells = 8 * 8
    macro_steps = result["macro_steps"]
    # One classification per active cell per macrostep, accumulated across the
    # run since the last depth reset.
    assert control["classifications"] == active_cells * macro_steps
    limited = control["cfl_limited"] + control["accuracy_limited"]
    assert limited == control["classifications"]
    # Quiescent coarse cells must demote when rain wets the domain.
    assert control["demotions"] > 0
    assert control["accuracy_limited"] > 0
    indicator = control["indicator_relative_change"]
    assert indicator["samples"] == active_cells * (macro_steps - 1)
    assert indicator["p50"] is not None
    assert indicator["max"] is not None
    assert indicator["p50"] <= indicator["p90"] <= indicator["p99"]


def test_history_initialised_and_reset_with_depth():
    solver = create_valley_solver()
    hyetograph = build_uniform_hyetograph(duration_s=3600.0, total_depth_m=0.02)
    run_lts(solver, duration=600.0, rainfall_hyetograph=hyetograph)
    previous_levels = solver._lts_prev_cell_levels.numpy()
    assert np.all(previous_levels >= 0)
    assert solver._lts_last_classify_time is not None
    stats_after_run = solver._lts_classify_stats.numpy().copy()
    assert stats_after_run.sum() > 0

    solver.set_water_depth(np.zeros((8, 8), dtype=np.float64))
    assert np.all(solver._lts_prev_cell_levels.numpy() == -1)
    assert np.all(solver._lts_prev_effective_depth.numpy() == 0.0)
    assert np.all(solver._lts_classify_stats.numpy() == 0)
    assert solver._lts_last_classify_time is None


def test_dry_cells_never_exceed_front_guard_level_near_wetting():
    solver = create_valley_solver()
    hyetograph = build_uniform_hyetograph(duration_s=3600.0, total_depth_m=0.04)
    wet_max_level = 2
    run_lts(
        solver,
        duration=600.0,
        rainfall_hyetograph=hyetograph,
        wet_max_level=wet_max_level,
    )
    cell_levels = solver._lts_cell_levels.numpy()
    depths = solver.get_water_depth()
    wet_max_observed = int(
        cell_levels[depths > float(solver.dry_depth)].max(initial=-1)
    )
    # Smoothing can only tighten levels, so wet cells must remain at or below
    # the guard level at the final schedule.
    assert wet_max_observed <= wet_max_level


def test_stricter_accuracy_fraction_is_not_faster_than_relaxed():
    hyetograph = build_uniform_hyetograph(duration_s=3600.0, total_depth_m=0.02)
    strict = create_valley_solver()
    relaxed = create_valley_solver()
    strict_result = run_lts(
        strict, duration=900.0, rainfall_hyetograph=hyetograph,
        accuracy_fraction=0.05,
    )
    relaxed_result = run_lts(
        relaxed, duration=900.0, rainfall_hyetograph=hyetograph,
        accuracy_fraction=1.0,
    )
    assert (
        strict_result["cell_updates"] >= relaxed_result["cell_updates"]
    )
    # Both remain mass-conservative regardless of the accuracy setting.
    for solver in (strict, relaxed):
        diagnostics = compute_diagnostics(solver)
        assert abs(diagnostics["relative_mass_balance_error"]) < 1.0e-9


def test_invalid_accuracy_parameters_rejected():
    solver = create_valley_solver()
    with pytest.raises(ValueError, match="accuracy_fraction"):
        run_lts(solver, duration=10.0, accuracy_fraction=0.0)
    with pytest.raises(ValueError, match="accuracy_depth_scale"):
        run_lts(solver, duration=10.0, accuracy_depth_scale=-1.0)
    with pytest.raises(ValueError, match="promotion_margin"):
        run_lts(solver, duration=10.0, promotion_margin=0.5)
