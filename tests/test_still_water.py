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

def test_still_water():
    tol = get_tol()
    ny, nx = 5, 5
    dx, dy = 10.0, 10.0
    solver = create_base_solver(ny, nx, dx, dy)

    y, x = np.mgrid[0:ny, 0:nx]
    bed = 0.1 * (x + y)
    solver.bed_elevation = wp.array(bed, dtype=solver.bed_elevation.dtype, device=solver.device)

    eta0 = 5.0
    h_init = np.maximum(0.0, eta0 - bed)
    solver.set_water_depth(h_init)

    for _ in range(10):
        solver.step()

    diag = compute_diagnostics(solver)
    assert np.allclose(solver.get_qx(), 0.0, atol=tol["atol"])
    assert np.allclose(solver.get_qy(), 0.0, atol=tol["atol"])
    assert np.allclose(solver.get_water_depth(), h_init, atol=tol["atol"])
    assert abs(diag["mass_balance_residual"]) < tol["mass_atol"]
