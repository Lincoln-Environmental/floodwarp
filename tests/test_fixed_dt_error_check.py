"""Regression test for fixed-dt negative-depth error detection.

Phase 3 moved the default-profile negative-depth readback into
``calculate_cfl_dt()``. When ``fixed_dt`` is supplied, that function is skipped,
so ``step()`` must explicitly read the fused continuity error flag to preserve
the original per-step safety guarantee.
"""

import numpy as np
import pytest
import warp as wp

from floodwarp import local_inertial_solver as solver_module
from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_solver import LocalInertialSolver


@wp.kernel
def inject_negative_depth_error_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    neg_depth_tol: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    reduce_max_depth: int,
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    cumulative_totals: wp.array(dtype=wp.float64),
    rain_sum: wp.array(dtype=wp.float64),
    infil_sum: wp.array(dtype=wp.float64),
    other_sum: wp.array(dtype=wp.float64),
    cell_area: wp.float64,
    fold_source_mass: int,
):
    """Stand-in for continuity_update_and_check_kernel that always signals an error."""
    control[1] = wp.float64(1.0)


def test_fixed_dt_default_profile_checks_fused_error_flag():
    wp.init()
    device = "cuda:0" if wp.is_device_available("cuda:0") else "cpu"

    ny, nx = 4, 5
    bed = np.zeros((ny, nx), dtype=np.float64)
    manning = np.full((ny, nx), 0.03, dtype=np.float64)
    mask = np.ones((ny, nx), dtype=np.int32)

    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=10.0,
        dy=10.0,
        bed_elevation=bed,
        manning_n=manning,
        active_mask=mask,
        device=device,
        validation_profile=None,  # default profile
        use_fused_kernels=True,
    )
    solver.set_water_depth(np.full((ny, nx), 0.1))

    original_kernel = solver_module.continuity_update_and_check_kernel
    solver_module.continuity_update_and_check_kernel = inject_negative_depth_error_kernel
    try:
        with pytest.raises(RuntimeError, match="Significant negative depth encountered"):
            solver.step(fixed_dt=0.1)
    finally:
        solver_module.continuity_update_and_check_kernel = original_kernel
