"""Standalone opt-in per-cell maximum-water-depth accumulator (event labels).

Covers the ``track_max_water_depth`` flag added for the maximum-flood-extent
surrogate label pipeline: opt-in allocation, brute-force parity against a
dense host-readback series, off-mode trajectory purity, ``set_water_depth``
reset, host-stepped vs device-loop bit-exactness, and spatial-LTS rejection.
"""

import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver


def _devices():
    devices = ["cpu"]
    if wp.is_device_available("cuda:0"):
        devices.append("cuda:0")
    return devices


def _make_solver(*, track: bool, device: str = "cpu") -> LocalInertialSolver:
    """2x4 closed domain on an eastward-downslope bed with a partial mask."""
    ny, nx = 2, 4
    mask = np.ones((ny, nx), dtype=np.int32)
    mask[0, 0] = 0
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=2.0,
        dy=2.0,
        bed_elevation=np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (ny, 1)),
        manning_n=np.full((ny, nx), 0.05),
        active_mask=mask,
        device=device,
        maximum_dt=0.05,
        adaptive_dt_growth_factor=None,
        track_max_water_depth=track,
    )


@pytest.mark.parametrize("device", _devices())
def test_max_depth_tracker_is_opt_in(device: str):
    disabled = _make_solver(track=False, device=device)
    assert disabled.max_water_depth_m is None
    with pytest.raises(RuntimeError, match="track_max_water_depth"):
        disabled.get_max_water_depth()

    enabled = _make_solver(track=True, device=device)
    field = enabled.get_max_water_depth()
    assert field.shape == (2, 4)
    assert np.all(field == 0.0)


@pytest.mark.parametrize("device", _devices())
def test_max_depth_matches_dense_brute_force(device: str):
    """The accumulator must equal the max of the post-step host readback."""
    solver = _make_solver(track=True, device=device)
    solver.set_water_depth(np.where(solver._active_mask_host == 1, 0.3, 0.0))
    brute = np.zeros((2, 4), dtype=np.float64)
    for _ in range(25):
        solver.step(remaining_time=0.02, fixed_dt=0.02)
        depth = solver.get_water_depth()
        brute = np.maximum(brute, depth)

    tracked = solver.get_max_water_depth()
    np.testing.assert_array_equal(tracked, brute)
    # Inactive cells never enter the accumulator.
    assert np.all(tracked[solver._active_mask_host == 0] == 0.0)
    # The maximum dominates the final state everywhere.
    final = solver.get_water_depth()
    assert np.all(tracked >= final - 0.0)


@pytest.mark.parametrize("device", _devices())
def test_off_mode_trajectory_is_bit_identical(device: str):
    """Flag off: nothing allocated, nothing launched, identical trajectory."""
    enabled = _make_solver(track=True, device=device)
    disabled = _make_solver(track=False, device=device)
    for solver in (enabled, disabled):
        solver.set_water_depth(np.where(solver._active_mask_host == 1, 0.25, 0.0))
        solver.set_uniform_rainfall_rate(1.0e-3)

    for _ in range(15):
        enabled.step(remaining_time=0.02, fixed_dt=0.02)
        disabled.step(remaining_time=0.02, fixed_dt=0.02)

    np.testing.assert_array_equal(enabled.get_water_depth(), disabled.get_water_depth())
    np.testing.assert_array_equal(enabled.get_qx(), disabled.get_qx())
    np.testing.assert_array_equal(enabled.get_qy(), disabled.get_qy())
    assert enabled.simulation_time == disabled.simulation_time


@pytest.mark.parametrize("device", _devices())
def test_set_water_depth_resets_max_depth(device: str):
    solver = _make_solver(track=True, device=device)
    solver.set_water_depth(np.where(solver._active_mask_host == 1, 0.3, 0.0))
    for _ in range(10):
        solver.step(remaining_time=0.02, fixed_dt=0.02)
    assert np.max(solver.get_max_water_depth()) > 0.0

    solver.set_water_depth(np.zeros((2, 4), dtype=np.float64))
    np.testing.assert_array_equal(solver.get_max_water_depth(), np.zeros((2, 4)))

    # Tracking resumes after the reset.
    solver.set_water_depth(np.where(solver._active_mask_host == 1, 0.1, 0.0))
    solver.step(remaining_time=0.02, fixed_dt=0.02)
    assert np.max(solver.get_max_water_depth()) > 0.0


@pytest.mark.parametrize("device", _devices())
def test_device_loop_max_depth_matches_host_stepped(device: str):
    """FP64 fused path: the device loop is bit-exact with host step()."""
    duration = 0.5

    def _run(use_device_loop: bool) -> tuple[np.ndarray, np.ndarray]:
        solver = _make_solver(track=True, device=device)
        solver.set_water_depth(np.where(solver._active_mask_host == 1, 0.2, 0.0))
        solver.set_uniform_rainfall_rate(5.0e-4)
        if use_device_loop:
            solver.run_device_loop(duration=duration, sync_interval=8)
        else:
            while solver.simulation_time < duration - 1.0e-9:
                solver.step(remaining_time=duration - solver.simulation_time)
        return solver.get_water_depth(), solver.get_max_water_depth()

    host_depth, host_max = _run(use_device_loop=False)
    loop_depth, loop_max = _run(use_device_loop=True)
    np.testing.assert_array_equal(loop_depth, host_depth)
    np.testing.assert_array_equal(loop_max, host_max)
    assert np.max(host_max) > 0.0


@pytest.mark.parametrize("device", _devices())
def test_spatial_lts_rejects_max_depth_tracking(device: str):
    solver = _make_solver(track=True, device=device)
    with pytest.raises(ValueError, match="track_max_water_depth"):
        solver.run_spatially_adaptive_loop(duration=0.2)
