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
    return LocalInertialSolver(ny, nx, dx, dy, bed, manning, mask, device=device, validation_profile='default')

def test_cpu_warp_equivalence():
    tol = get_tol()
    ny, nx = 3, 3
    dx, dy = 10.0, 10.0

    device = "cuda:0" if wp.is_device_available("cuda:0") else "cpu"

    np.random.seed(42)
    bed = np.random.rand(ny, nx)
    manning = np.full((ny, nx), 0.035)
    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

    solver = LocalInertialSolver(ny, nx, dx, dy, bed, manning, mask, device=device, validation_profile='default')

    b_qx = solver.boundary_type_qx.numpy()
    b_qx[0, 0] = BOUNDARY_FREE_OUTFLOW
    b_qx[0, nx] = BOUNDARY_SPECIFIED_STAGE
    solver.set_boundary_type_qx(b_qx)

    b_stage = solver.boundary_stage_qx.numpy()
    b_stage[0, nx] = 2.0
    solver.set_boundary_stage_qx(b_stage)

    h_init = np.random.rand(ny, nx)
    h_init[mask == 0] = 0.0
    solver.set_water_depth(h_init)

    rain = np.random.rand(ny, nx) * 1e-4
    infil = np.random.rand(ny, nx) * 1e-5
    solver.set_source_rates(rain=rain, infil=infil)

    # Re-run numpy with actual accepted dt to ensure exact match step-by-step
    solver2 = LocalInertialSolver(ny, nx, dx, dy, bed, manning, mask, device="cpu", validation_profile='default')
    solver2.set_boundary_type_qx(b_qx)
    solver2.set_boundary_stage_qx(b_stage)
    solver2.set_water_depth(h_init)
    solver2.set_source_rates(rain=rain, infil=infil)

    h_np = h_init.copy()
    qx_np = solver2.get_qx().copy()
    qy_np = solver2.get_qy().copy()

    for _ in range(3):
        dt = solver2.step()

        res = step_local_inertial_numpy(
            water_depth=h_np,
            bed_elevation=bed,
            manning_n=manning,
            active_mask=mask,
            qx=qx_np,
            qy=qy_np,
            rainfall_rate=rain,
            other_source_rate=np.zeros((ny, nx)),
            infiltration_capacity=infil,
            boundary_type_qx=b_qx,
            boundary_stage_qx=b_stage,
            boundary_type_qy=solver2.boundary_type_qy.numpy(),
            boundary_stage_qy=solver2.boundary_stage_qy.numpy(),
            dx=dx,
            dy=dy,
            dt=dt, # Force numpy to use the exact same dt accepted by warp
            gravity=9.80665,
            dry_depth=1.0e-4,
            depth_epsilon=1.0e-6,
            cfl_number=0.4,
            limiter_epsilon=1e-12,
            negative_depth_tolerance=1e-8
        )
        h_np = res["water_depth"]
        qx_np = res["qx"]
        qy_np = res["qy"]

        assert np.allclose(solver2.get_water_depth(), h_np, atol=tol["atol"])
        assert np.allclose(solver2.get_qx(), qx_np, atol=tol["atol"])
        assert np.allclose(solver2.get_qy(), qy_np, atol=tol["atol"])
