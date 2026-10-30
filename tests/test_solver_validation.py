import numpy as np
import pytest

from floodwarp.boundary_conditions import BOUNDARY_INTERNAL
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_solver import LocalInertialSolver


def create_solver(ny: int = 1, nx: int = 2) -> LocalInertialSolver:
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=10.0,
        dy=10.0,
        bed_elevation=np.zeros((ny, nx)),
        manning_n=np.full((ny, nx), 0.035),
        active_mask=np.ones((ny, nx), dtype=np.int32),
    )


def test_sources_constrain_cfl_before_the_first_momentum_update():
    solver = create_solver()
    solver.set_source_rates(rain=np.array([[0.01, 0.0]]))

    solver.step()

    assert solver.current_dt < solver.maximum_dt
    assert solver.current_dt == pytest.approx(0.4 * 10.0 / np.sqrt(9.80665 * 0.6), rel=1e-5)


def test_lisflood_acc_profile_uses_global_depth_timestep_without_velocity_cfl():
    solver = LocalInertialSolver(
        ny=1,
        nx=2,
        dx=10.0,
        dy=10.0,
        bed_elevation=np.zeros((1, 2)),
        manning_n=np.full((1, 2), 0.035),
        active_mask=np.ones((1, 2), dtype=np.int32),
        validation_profile="lisflood_fp_acc",
    )
    solver.set_water_depth(np.array([[0.1, 0.4]]))

    timestep_s = solver.calculate_stable_timestep()

    assert timestep_s == pytest.approx(0.4 * 10.0 / np.sqrt(9.80665 * 0.4), rel=1.0e-6)
    assert solver.last_timestep_max_depth_m == pytest.approx(0.4)


def test_lisflood_acc_reuses_post_step_maximum_depth_for_next_timestep():
    solver = LocalInertialSolver(
        ny=1,
        nx=1,
        dx=10.0,
        dy=10.0,
        bed_elevation=np.zeros((1, 1)),
        manning_n=np.full((1, 1), 0.035),
        active_mask=np.ones((1, 1), dtype=np.int32),
        validation_profile="lisflood_fp_acc",
    )
    solver.set_source_rates(rain=np.full((1, 1), 0.01))

    solver.step()
    timestep_s = solver.calculate_stable_timestep()

    assert solver._lisflood_cached_maximum_depth_m == pytest.approx(0.6)
    assert solver.last_timestep_max_depth_m == pytest.approx(0.6)
    assert timestep_s == pytest.approx(0.4 * 10.0 / np.sqrt(9.80665 * 0.6), rel=1.0e-6)


def test_boundary_configuration_rejects_internal_exterior_face():
    solver = create_solver()
    boundary_type = solver.boundary_type_qx.numpy()
    boundary_type[0, 0] = BOUNDARY_INTERNAL

    with pytest.raises(ValueError, match="only valid on interior qx faces"):
        solver.set_boundary_type_qx(boundary_type)


def test_setting_depth_reinitializes_mass_accounting_and_fluxes():
    solver = create_solver()
    solver.set_source_rates(rain=np.full((1, 2), 1e-4))
    solver.step(remaining_time=10.0)

    solver.set_water_depth(np.full((1, 2), 1.0))

    diagnostics = compute_diagnostics(solver)
    assert solver.simulation_time == 0.0
    assert np.allclose(solver.get_qx(), 0.0)
    assert np.allclose(solver.get_qy(), 0.0)
    assert diagnostics["cumulative_rainfall_volume"] == 0.0
    assert diagnostics["mass_balance_residual"] == pytest.approx(0.0)
