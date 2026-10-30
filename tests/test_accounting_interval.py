"""Tests for the deferred source-mass accounting interval (Phase A)."""
import numpy as np
import pytest
import warp as wp

from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_solver import LocalInertialSolver


def build(interval, monkeypatch=None, reduce_mass="1"):
    if monkeypatch is not None:
        monkeypatch.setenv("OVERLANDER_REDUCE_MASS", reduce_mass)
    rng = np.random.default_rng(3)
    ny, nx = 30, 40
    bed = rng.random((ny, nx)) * 0.3
    manning = np.full((ny, nx), 0.04)
    mask = np.ones((ny, nx), dtype=np.int32)
    solver = LocalInertialSolver(
        ny, nx, 10.0, 10.0, bed, manning, mask,
        device="cpu", accounting_interval=interval,
    )
    solver.set_water_depth(rng.random((ny, nx)) * 0.1)
    solver.set_source_rates(rain=np.full((ny, nx), 2.0e-5), infil=np.full((ny, nx), 1.0e-7))
    return solver


def test_deferred_accounting_preserves_trajectory_and_totals(monkeypatch):
    wp.init()
    reference = build(1, monkeypatch)
    deferred = build(7, monkeypatch)
    assert deferred._deferred_mass_accounting

    for _ in range(20):
        reference.step(fixed_dt=0.5)
        deferred.step(fixed_dt=0.5)

    assert np.array_equal(reference.get_water_depth(), deferred.get_water_depth())
    assert np.array_equal(reference.get_qx(), deferred.get_qx())
    assert np.array_equal(reference.get_qy(), deferred.get_qy())
    reference_totals = reference.cumulative_accounting()
    deferred_totals = deferred.cumulative_accounting()
    for key in reference_totals:
        assert deferred_totals[key] == pytest.approx(reference_totals[key], rel=1.0e-9, abs=1.0e-12)


def test_deferred_accounting_flushes_on_final_step(monkeypatch):
    wp.init()
    solver = build(1000, monkeypatch)
    while solver.simulation_time < 9.0:
        solver.step(remaining_time=9.0 - solver.simulation_time)
    assert solver._steps_since_mass_accounting == 0
    totals = solver.cumulative_accounting()
    assert totals["rain"] > 0.0


def test_deferred_accounting_flushes_on_demand(monkeypatch):
    wp.init()
    reference = build(1, monkeypatch)
    deferred = build(1000, monkeypatch)
    for _ in range(5):
        reference.step(fixed_dt=0.5)
        deferred.step(fixed_dt=0.5)
    # Mid-window on-demand read must account for every step exactly.
    reference_totals = reference.cumulative_accounting()
    deferred_totals = deferred.cumulative_accounting()
    for key in reference_totals:
        assert deferred_totals[key] == pytest.approx(reference_totals[key], rel=1.0e-9, abs=1.0e-12)
    assert deferred._steps_since_mass_accounting == 0
    diagnostics = compute_diagnostics(deferred)
    assert abs(diagnostics["relative_mass_balance_error"]) < 1.0e-9


def test_accounting_interval_validation():
    wp.init()
    with pytest.raises(ValueError, match="accounting_interval"):
        build(0)
