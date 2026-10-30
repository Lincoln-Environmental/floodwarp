import pytest
import numpy as np
import warp as wp
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.local_inertial_reference import step_local_inertial_numpy
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.boundary_conditions import (
    BOUNDARY_INTERNAL, BOUNDARY_CLOSED,
    BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE
)
from floodwarp.config import _float_env

def get_tol():
    if _float_env == "float64":
        return {"atol": 1e-12, "rtol": 1e-10, "mass_atol": 1e-12, "mass_rtol": 1e-10}
    return {"atol": 1e-5, "rtol": 1e-5, "mass_atol": 1e-5, "mass_rtol": 1e-5}

def create_base_solver(ny, nx, dx, dy, device="cpu"):
    wp.init()
    bed = np.zeros((ny, nx))
    manning = np.full((ny, nx), 0.035)
    mask = np.ones((ny, nx), dtype=np.int32)
    return LocalInertialSolver(ny, nx, dx, dy, bed, manning, mask, device=device)

def test_rainfall_conservation():
    tol = get_tol()
    ny, nx = 5, 5
    dx, dy = 10.0, 10.0
    solver = create_base_solver(ny, nx, dx, dy)

    R = 1e-4
    source = np.full((ny, nx), R)
    solver.set_source_rates(rain=source)

    dt = 10.0
    steps = 5
    for _ in range(steps):
        solver.step(remaining_time=dt)

    diag = compute_diagnostics(solver)
    expected_h = R * dt * steps
    assert np.allclose(solver.get_water_depth(), expected_h, atol=tol["atol"])
    assert abs(diag["mass_balance_residual"]) < tol["mass_atol"]
