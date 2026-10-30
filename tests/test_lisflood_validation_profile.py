import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import BOUNDARY_FREE_OUTFLOW
from floodwarp.local_inertial_solver import LocalInertialSolver


def make_solver(ny: int, nx: int, bed: np.ndarray, manning: np.ndarray) -> LocalInertialSolver:
    wp.init()
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=10.0,
        dy=10.0,
        bed_elevation=bed,
        manning_n=manning,
        active_mask=np.ones((ny, nx), dtype=np.int32),
        validation_profile="lisflood_fp_acc",
    )


def test_lisflood_profile_only_infiltrates_preexisting_wet_cells_before_rain():
    solver = make_solver(
        ny=1,
        nx=1,
        bed=np.zeros((1, 1)),
        manning=np.full((1, 1), 0.03),
    )
    solver.set_water_depth(np.array([[0.001]]))
    solver.set_source_rates(rain=np.array([[0.01]]), infil=np.array([[1.0]]))

    solver.step(remaining_time=1.0)

    assert solver.validation_profile == "lisflood_fp_acc"
    assert solver.get_water_depth()[0, 0] == pytest.approx(0.011)
    assert solver.cumulative_accounting()["infiltration"] == pytest.approx(0.0)


def test_lisflood_profile_free_outflow_uses_interior_surface_slope():
    solver = make_solver(
        ny=1,
        nx=2,
        bed=np.zeros((1, 2)),
        manning=np.full((1, 2), 0.03),
    )
    boundary_type = solver.boundary_type_qx.numpy()
    boundary_type[0, 2] = BOUNDARY_FREE_OUTFLOW
    solver.set_boundary_type_qx(boundary_type)
    solver.set_water_depth(np.ones((1, 2)))

    solver.step(remaining_time=1.0)

    assert solver.get_qx()[0, 2] == pytest.approx(0.0, abs=1e-7)


def test_lisflood_profile_uses_face_average_and_two_dimensional_friction():
    solver = make_solver(
        ny=1,
        nx=2,
        bed=np.zeros((1, 2)),
        manning=np.array([[0.03, 0.05]]),
    )
    solver.set_water_depth(np.ones((1, 2)))
    solver.qx = wp.array(np.array([[0.0, 1.0, 0.0]]), dtype=solver.qx.dtype, device=solver.device)
    solver.qy = wp.array(np.ones((2, 2)), dtype=solver.qy.dtype, device=solver.device)

    solver.step(remaining_time=1.0)

    expected = 1.0 / (1.0 + 9.80665 * 0.04**2 * np.sqrt(2.0))
    assert solver.get_qx()[0, 1] == pytest.approx(expected, rel=1e-5)
