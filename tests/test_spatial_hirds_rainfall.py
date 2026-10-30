"""Analytical checks for separable reference-hyetograph spatial rainfall."""
import numpy as np
import pytest

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.local_inertial_kernels import (
    DL_MAX_DT,
    DL_RAIN_RATE,
    DL_SIZE,
    device_loop_predict_depth_kernel,
    lts_predict_depth_kernel,
)
from floodwarp.rainfall import Hyetograph


def _storm() -> Hyetograph:
    return Hyetograph(
        node_times_s=np.array([0.0, 10.0]),
        node_rates_m_per_s=np.array([0.001, 0.001]),
        storm_type="uniform",
        metadata={},
    )


def _solver(mask=None, device="cpu") -> LocalInertialSolver:
    if mask is None:
        mask = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.int32)
    ny, nx = mask.shape
    return LocalInertialSolver(
        ny=ny, nx=nx, dx=10.0, dy=10.0,
        bed_elevation=np.zeros((ny, nx)),
        manning_n=np.full((ny, nx), 0.04),
        active_mask=mask,
        device=device,
        infiltration_method="none",
        maximum_dt=2.0,
        adaptive_dt_growth_factor=None,
    )


def _multiplier() -> np.ndarray:
    return np.array([[0.5, 0.0, 1.5], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64)


def _assert_expected_rain(solver: LocalInertialSolver, multiplier: np.ndarray) -> None:
    expected_depth = 0.01 * multiplier
    np.testing.assert_allclose(solver.get_water_depth(), expected_depth, rtol=2e-5, atol=2e-7)
    assert solver.cumulative_accounting()["rain"] == pytest.approx(float(np.sum(expected_depth) * 100.0), rel=2e-5)
    assert solver.cumulative_accounting()["infiltration"] == 0.0


def test_normal_step_spatial_rates_and_mass():
    solver = _solver()
    multiplier = _multiplier()
    solver.set_rainfall_spatial_multiplier(multiplier)
    solver.set_uniform_rainfall_rate(0.001)
    solver.step(10.0, fixed_dt=10.0)
    _assert_expected_rain(solver, multiplier)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_loop_spatial_rain_and_storm_end(device):
    import warp as wp

    if device == "cuda:0" and not wp.is_device_available(device):
        pytest.skip("CUDA unavailable")
    solver = _solver(device=device)
    multiplier = _multiplier()
    solver.set_rainfall_spatial_multiplier(multiplier)
    solver.run_device_loop(10.0, rainfall_hyetograph=_storm(), sync_interval=4)
    _assert_expected_rain(solver, multiplier)
    rain_before = solver.cumulative_accounting()["rain"]
    solver.run_device_loop(2.0, rainfall_hyetograph=_storm(), sync_interval=4)
    assert solver.cumulative_accounting()["rain"] == pytest.approx(rain_before)


def test_lts_spatial_rain_and_segmented_continuation():
    multiplier = _multiplier()
    continuous = _solver()
    segmented = _solver()
    for solver in (continuous, segmented):
        solver.set_rainfall_spatial_multiplier(multiplier)
    continuous.run_spatially_adaptive_loop(10.0, rainfall_hyetograph=_storm())
    segmented.run_spatially_adaptive_loop(5.0, rainfall_hyetograph=_storm())
    segmented.run_spatially_adaptive_loop(5.0, rainfall_hyetograph=_storm())
    _assert_expected_rain(continuous, multiplier)
    np.testing.assert_allclose(segmented.get_water_depth(), continuous.get_water_depth(), rtol=2e-5, atol=2e-7)
    assert segmented.cumulative_accounting()["rain"] == pytest.approx(continuous.cumulative_accounting()["rain"], rel=2e-5)


def test_cuda_lts_graph_applies_spatial_multiplier_when_available():
    import warp as wp

    if not wp.is_device_available("cuda:0"):
        pytest.skip("CUDA unavailable")
    solver = _solver(device="cuda:0")
    multiplier = _multiplier()
    solver.set_rainfall_spatial_multiplier(multiplier)
    solver.run_spatially_adaptive_loop(
        10.0, rainfall_hyetograph=_storm(), use_device_graph=True,
    )
    _assert_expected_rain(solver, multiplier)


def test_uniform_multiplier_matches_old_uniform_path():
    baseline = _solver()
    spatial = _solver()
    spatial.set_rainfall_spatial_multiplier(_solver_default_mask().astype(np.float64))
    baseline.run_device_loop(10.0, rainfall_hyetograph=_storm(), sync_interval=4)
    spatial.run_device_loop(10.0, rainfall_hyetograph=_storm(), sync_interval=4)
    np.testing.assert_array_equal(baseline.get_water_depth(), spatial.get_water_depth())
    assert baseline.cumulative_accounting() == spatial.cumulative_accounting()


def _solver_default_mask() -> np.ndarray:
    return np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.int32)


def test_offshore_cell_multiplier_zero_and_invalid_multiplier_rejected():
    mask = np.array([[1, 1], [1, 0]], dtype=np.int32)
    solver = _solver(mask=mask)
    with pytest.raises(ValueError, match="zero outside active"):
        solver.set_rainfall_spatial_multiplier(np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        solver.set_rainfall_spatial_multiplier(np.array([[1.0, -1.0], [1.0, 0.0]]))
    multiplier = np.array([[0.5, 1.5], [1.0, 0.0]])
    solver.set_rainfall_spatial_multiplier(multiplier)
    solver.run_device_loop(10.0, rainfall_hyetograph=_storm(), sync_interval=4)
    assert solver.get_water_depth()[1, 1] == 0.0
    assert solver.cumulative_accounting()["rain"] == pytest.approx(3.0, rel=2e-5)


def test_device_and_lts_cfl_prediction_use_local_rain_rates():
    import warp as wp

    solver = _solver()
    multiplier = _multiplier()
    solver.set_rainfall_spatial_multiplier(multiplier)
    control_values = np.zeros(DL_SIZE, dtype=np.float64)
    control_values[DL_MAX_DT] = 10.0
    control_values[DL_RAIN_RATE] = 0.001
    control = wp.array(control_values, dtype=wp.float64, device="cpu")
    wp.launch(
        kernel=device_loop_predict_depth_kernel,
        dim=(solver.ny, solver.nx),
        inputs=[solver.water_depth, solver.rainfall_rate,
                solver.rainfall_spatial_multiplier, solver.other_source_rate,
                solver.additional_source_rate,
                solver.active_mask, control, 1, solver.cfl_depth],
        device="cpu",
    )
    np.testing.assert_allclose(solver.cfl_depth.numpy(), 0.01 * multiplier)
    wp.launch(
        kernel=lts_predict_depth_kernel,
        dim=(solver.ny, solver.nx),
        inputs=[solver.water_depth, solver.rainfall_rate,
                solver.rainfall_spatial_multiplier, solver.other_source_rate,
                solver.additional_source_rate,
                solver.active_mask, 10.0, 0.001, 1, solver.cfl_depth],
        device="cpu",
    )
    np.testing.assert_allclose(solver.cfl_depth.numpy(), 0.01 * multiplier)
