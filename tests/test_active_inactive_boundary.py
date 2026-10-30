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

def test_active_inactive_boundary():
    tol = get_tol()
    solver = create_base_solver(1, 2, 10.0, 10.0)
    mask = np.array([[1, 0]], dtype=np.int32)

    solver = LocalInertialSolver(1, 2, 10.0, 10.0, np.zeros((1,2)), np.full((1,2), 0.035), mask)

    h_init = np.array([[2.0, 0.0]])
    solver.set_water_depth(h_init)

    solver.step()

    h_final = solver.get_water_depth()
    qx = solver.get_qx()

    assert qx[0, 1] == 0.0
    assert h_final[0, 0] == 2.0
    assert h_final[0, 1] == 0.0

    diag = compute_diagnostics(solver)
    assert abs(diag["mass_balance_residual"]) < tol["mass_atol"]
