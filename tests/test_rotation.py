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

def test_rotation():
    tol = get_tol()
    solver_x = create_base_solver(1, 3, 10.0, 10.0)
    solver_y = create_base_solver(3, 1, 10.0, 10.0)

    h_x = np.array([[2.0, 1.0, 0.5]])
    h_y = np.array([[2.0], [1.0], [0.5]])

    solver_x.set_water_depth(h_x)
    solver_y.set_water_depth(h_y)

    solver_x.step()
    solver_y.step()

    h_final_x = solver_x.get_water_depth()
    h_final_y = solver_y.get_water_depth()

    qx_final = solver_x.get_qx()
    qy_final = solver_y.get_qy()

    assert np.allclose(h_final_x, h_final_y.T, atol=tol["atol"])
    assert np.allclose(qx_final, qy_final.T, atol=tol["atol"])
