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

def test_cfl_safety():
    solver = create_base_solver(1, 1, 10.0, 10.0)

    b_qx = solver.boundary_type_qx.numpy()
    b_qx[0, 0] = BOUNDARY_FREE_OUTFLOW
    solver.set_boundary_type_qx(b_qx)

    solver.set_water_depth(np.array([[1000.0]])) # Very deep water, fast celerity

    # dt_cfl = 0.4 * 10 / sqrt(9.8 * 1000) = 4 / 99 = 0.04
    # Set minimum_dt > 0.04 to trigger the error
    solver.minimum_dt = 0.1

    with pytest.raises(RuntimeError, match="Calculated stable time step .* is below minimum_dt"):
        solver.step()
