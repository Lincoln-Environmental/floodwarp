"""Mixed-precision conditioning regressions for high-datum terrain."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from floodwarp.config import PRECISION_MODE
from floodwarp.local_inertial_solver import LocalInertialSolver


pytestmark = pytest.mark.skipif(
    PRECISION_MODE != "mixed",
    reason="conditioned face fields are specific to mixed precision",
)


def build_solver(bed: np.ndarray) -> LocalInertialSolver:
    """Build a closed mixed-precision solver with a supplied FP64 DEM."""
    ny, nx = bed.shape
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=10.0,
        dy=10.0,
        bed_elevation=bed,
        manning_n=np.full((ny, nx), 0.035),
        active_mask=np.ones((ny, nx), dtype=np.int32),
        device="cpu",
    )


def test_conditioned_face_differences_preserve_high_datum_slopes() -> None:
    """FP64 differencing precedes FP32 storage for a high absolute datum."""
    wp.init()
    bed = 1.0e6 + np.array(
        [[0.0000, 0.0011, 0.0024], [0.0007, 0.0019, 0.0034]],
        dtype=np.float64,
    )
    solver = build_solver(bed=bed)

    expected_qx = np.diff(bed, axis=1).astype(np.float32)
    expected_qy = np.diff(bed, axis=0).astype(np.float32)
    assert np.array_equal(solver.bed_slope_qx.numpy()[:, 1:-1], expected_qx)
    assert np.array_equal(solver.bed_slope_qy.numpy()[1:-1, :], expected_qy)


def test_high_datum_still_water_remains_still() -> None:
    """The relative formulation is invariant to an otherwise irrelevant datum."""
    wp.init()
    ny, nx = 4, 5
    y, x = np.mgrid[0:ny, 0:nx]
    bed = 1.0e6 + 0.001 * x + 0.0007 * y
    solver = build_solver(bed=bed)
    eta = 1.0e6 + 2.0
    initial_depth = eta - bed
    solver.set_water_depth(depth=initial_depth)

    for _ in range(5):
        solver.step(fixed_dt=0.1)

    assert np.allclose(solver.get_water_depth(), initial_depth, rtol=0.0, atol=2.0e-6)
    assert np.allclose(solver.get_qx(), 0.0, rtol=0.0, atol=2.0e-6)
    assert np.allclose(solver.get_qy(), 0.0, rtol=0.0, atol=2.0e-6)


def test_specified_stage_is_stored_as_datum_relative_depth() -> None:
    """External absolute stages are converted once before mixed stepping."""
    wp.init()
    bed = 1.0e6 + np.arange(12, dtype=np.float64).reshape(3, 4) * 0.002
    solver = build_solver(bed=bed)
    stage = np.zeros((3, 5), dtype=np.float64)
    stage[:, 0] = bed[:, 0] + 0.25
    stage[:, -1] = bed[:, -1] + 0.75
    solver.set_boundary_stage_qx(stage=stage)

    stored_depth = solver.boundary_stage_depth_qx.numpy()
    assert np.allclose(stored_depth[:, 0], 0.25, rtol=0.0, atol=2.0e-7)
    assert np.allclose(stored_depth[:, -1], 0.75, rtol=0.0, atol=2.0e-7)
