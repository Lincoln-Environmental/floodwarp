"""Opt-in cumulative face-volume / peak-discharge / max-depth diagnostics."""

import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import (
    BOUNDARY_FREE_OUTFLOW,
    BOUNDARY_MASKED_WEST_FREE,
    BOUNDARY_SPECIFIED_STAGE,
)
from floodwarp.local_inertial_solver import LocalInertialSolver


def _devices():
    devices = ["cpu"]
    if wp.is_device_available("cuda:0"):
        devices.append("cuda:0")
    return devices


def _make_solver(*, track: bool, device: str = "cpu") -> LocalInertialSolver:
    return LocalInertialSolver(
        ny=2,
        nx=3,
        dx=2.0,
        dy=3.0,
        bed_elevation=np.array(((2.0, 1.0, 0.0), (2.0, 1.0, 0.0))),
        manning_n=np.full((2, 3), 0.05),
        active_mask=np.ones((2, 3), dtype=np.int32),
        device=device,
        maximum_dt=0.01,
        adaptive_dt_growth_factor=None,
        track_cumulative_face_volumes=track,
    )


def _make_sloped_row_solver(*, device: str, track: bool) -> LocalInertialSolver:
    """2x4 closed domain on an eastward-downslope bed."""
    ny, nx = 2, 4
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=2.0,
        dy=2.0,
        bed_elevation=np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (ny, 1)),
        manning_n=np.full((ny, nx), 0.05),
        active_mask=np.ones((ny, nx), dtype=np.int32),
        device=device,
        maximum_dt=0.05,
        adaptive_dt_growth_factor=None,
        track_cumulative_face_volumes=track,
    )


@pytest.mark.parametrize("device", _devices())
def test_face_volume_accumulator_is_opt_in_and_tracks_accepted_global_fluxes(
    device: str,
):
    disabled = _make_solver(track=False, device=device)
    with pytest.raises(RuntimeError, match="track_cumulative_face_volumes"):
        disabled.get_cumulative_face_volumes()

    solver = _make_solver(track=True, device=device)
    solver.set_water_depth(np.full((2, 3), 0.1))
    solver.step(0.01, fixed_dt=0.01)
    result = solver.get_cumulative_face_volumes()

    qx = solver.get_qx()
    qy = solver.get_qy()
    np.testing.assert_allclose(
        result["cumulative_net_qx_volume_m3"], qx * 3.0 * 0.01,
    )
    np.testing.assert_allclose(
        result["cumulative_net_qy_volume_m3"], qy * 2.0 * 0.01,
    )
    np.testing.assert_allclose(
        result["cumulative_abs_qx_volume_m3"],
        np.abs(result["cumulative_net_qx_volume_m3"]),
    )
    np.testing.assert_allclose(
        result["cumulative_abs_qy_volume_m3"],
        np.abs(result["cumulative_net_qy_volume_m3"]),
    )
    assert result["peak_discharge_m2_per_s"] >= np.max(np.abs(qx))
    assert np.max(result["diagnostic_max_depth_m"]) > 0.0

    solver.reset_cumulative_face_volumes()
    reset = solver.get_cumulative_face_volumes()
    assert not np.any(reset["cumulative_net_qx_volume_m3"])
    assert not np.any(reset["cumulative_net_qy_volume_m3"])


@pytest.mark.parametrize("device", _devices())
def test_signed_face_volume_matches_recorded_accepted_flux(device: str):
    # Reference method: step() launches the accumulate_face_volume_*_kernel
    # pair on qx_next/qy_next AFTER the donor-cell outflow limiter (and the
    # optional seam hook), then _swap_q_buffers publishes exactly those
    # arrays as the current q state.  get_qx()/get_qy() after each step
    # therefore return the identical post-limiter flux the accumulators
    # integrated, so with a fixed dt the exact reference is
    # sum_k get_q*_k * face_width * dt recorded from the tracked solver
    # itself — no second-solver trajectory assumption is needed.
    solver = _make_sloped_row_solver(device=device, track=True)
    solver.set_water_depth(np.full((2, 4), 0.3))
    dt = 0.02
    qx_steps = []
    qy_steps = []
    for _ in range(20):
        solver.step(remaining_time=dt, fixed_dt=dt)
        qx_steps.append(solver.get_qx().copy())
        qy_steps.append(solver.get_qy().copy())

    volumes = solver.get_cumulative_face_volumes()
    dy = float(solver.dy)
    dx = float(solver.dx)
    qx_steps = np.asarray(qx_steps)
    qy_steps = np.asarray(qy_steps)
    np.testing.assert_allclose(
        volumes["cumulative_net_qx_volume_m3"],
        dy * dt * np.sum(qx_steps, axis=0),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        volumes["cumulative_net_qy_volume_m3"],
        dx * dt * np.sum(qy_steps, axis=0),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        volumes["cumulative_abs_qx_volume_m3"],
        dy * dt * np.sum(np.abs(qx_steps), axis=0),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        volumes["cumulative_abs_qy_volume_m3"],
        dx * dt * np.sum(np.abs(qy_steps), axis=0),
        rtol=1e-9,
        atol=1e-12,
    )
    # One-direction drainage: net and absolute coincide on interior faces.
    np.testing.assert_allclose(
        volumes["cumulative_abs_qx_volume_m3"],
        np.abs(volumes["cumulative_net_qx_volume_m3"]),
        rtol=1e-9,
        atol=1e-12,
    )
    expected_peak = max(
        float(np.max(np.abs(qx_steps))), float(np.max(np.abs(qy_steps)))
    )
    assert volumes["peak_discharge_m2_per_s"] == pytest.approx(
        expected_peak, rel=1e-12
    )


@pytest.mark.parametrize("device", _devices())
def test_absolute_exceeds_net_under_flow_reversal(device: str):
    # Ponded water against an eastern SPECIFIED_STAGE boundary: the stage is
    # first held above the pond surface (inflow, qx < 0 on the east face),
    # then lowered to bed level (drainage, qx > 0), so the face reverses
    # sign mid-run.
    ny, nx = 2, 3
    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=2.0,
        dy=2.0,
        bed_elevation=np.zeros((ny, nx)),
        manning_n=np.full((ny, nx), 0.05),
        active_mask=np.ones((ny, nx), dtype=np.int32),
        device=device,
        maximum_dt=0.05,
        adaptive_dt_growth_factor=None,
        track_cumulative_face_volumes=True,
    )
    boundary = solver.boundary_type_qx.numpy()
    boundary[:, nx] = BOUNDARY_SPECIFIED_STAGE
    solver.set_boundary_type_qx(boundary_type=boundary)
    stage = np.zeros((ny, nx + 1))
    stage[:, nx] = 0.55
    solver.set_boundary_stage_qx(stage=stage)
    solver.set_water_depth(np.full((ny, nx), 0.4))

    dt = 0.05
    east_fluxes = []
    for _ in range(5):
        solver.step(remaining_time=dt, fixed_dt=dt)
        east_fluxes.append(solver.get_qx()[:, nx].copy())
    stage[:, nx] = 0.0
    solver.set_boundary_stage_qx(stage=stage)
    # The inflow phase builds inward momentum, so the drainage phase needs
    # more steps before the face flux changes sign.
    for _ in range(30):
        solver.step(remaining_time=dt, fixed_dt=dt)
        east_fluxes.append(solver.get_qx()[:, nx].copy())

    east_fluxes = np.asarray(east_fluxes)
    assert np.any(east_fluxes < 0.0)
    assert np.any(east_fluxes > 0.0)

    volumes = solver.get_cumulative_face_volumes()
    dy = float(solver.dy)
    net = volumes["cumulative_net_qx_volume_m3"][:, nx]
    absolute = volumes["cumulative_abs_qx_volume_m3"][:, nx]
    np.testing.assert_allclose(
        absolute,
        dy * dt * np.sum(np.abs(east_fluxes), axis=0),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        net,
        dy * dt * np.sum(east_fluxes, axis=0),
        rtol=1e-9,
        atol=1e-12,
    )
    # Strictly greater where the flow reversed sign during the run.
    assert np.all(absolute > np.abs(net))


@pytest.mark.parametrize("device", _devices())
def test_accumulator_sees_post_limiter_flux_on_hard_limited_drain(device: str):
    # One shallow donor cell draining through its only outflow face with a
    # deliberately oversized fixed dt.  The raw local-inertial momentum
    # candidate (q_old = 0, so the friction denominator is 1) would remove
    # more than 5x the donor's stored volume in the first step, so the
    # donor-cell limiter binds hard.  Because the donor loses volume only
    # through the interior face, the accepted face volume equals the donor's
    # per-step storage loss exactly; a pre-limiter accumulator would record
    # the raw candidate and exceed the donor storage by an order of
    # magnitude.
    solver = LocalInertialSolver(
        ny=1,
        nx=2,
        dx=1.0,
        dy=1.0,
        bed_elevation=np.array([[5.0, 0.0]]),
        manning_n=np.full((1, 2), 0.05),
        active_mask=np.ones((1, 2), dtype=np.int32),
        device=device,
        adaptive_dt_growth_factor=None,
        track_cumulative_face_volumes=True,
    )
    donor_depth0 = 5.0e-3
    solver.set_water_depth(np.array([[donor_depth0, 0.0]]))

    dt = 0.5
    eta_drop = 5.0 + donor_depth0  # eta_donor - eta_receiver
    raw_q = float(solver.gravity) * donor_depth0 * dt * eta_drop / 1.0
    raw_volume = raw_q * 1.0 * dt
    donor_volume0 = donor_depth0 * 1.0 * 1.0
    assert raw_volume > 5.0 * donor_volume0

    donor_volumes = [donor_volume0]
    for _ in range(3):
        solver.step(remaining_time=dt, fixed_dt=dt)
        donor_volumes.append(float(solver.get_water_depth()[0, 0]))

    volumes = solver.get_cumulative_face_volumes()
    recorded = float(volumes["cumulative_abs_qx_volume_m3"][0, 1])
    donor_removed = donor_volumes[0] - donor_volumes[-1]
    assert donor_removed > 0.0
    assert recorded == pytest.approx(donor_removed, rel=1e-9)
    assert recorded < raw_volume
    assert recorded == pytest.approx(
        float(volumes["cumulative_net_qx_volume_m3"][0, 1]), rel=1e-12
    )


@pytest.mark.parametrize("device", _devices())
def test_boundary_face_volumes_close_with_mass_accounting(device: str):
    # 4x4 grid, western nodata strip with masked FREE interior faces and a
    # FREE_OUTFLOW eastern outer edge, ponded water draining both ways.
    ny = nx = 4
    active_mask = np.ones((ny, nx), dtype=np.int32)
    active_mask[:, 0] = 0
    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.tile(0.2 * np.array([3.0, 2.0, 1.0, 0.0]), (ny, 1)),
        manning_n=np.full((ny, nx), 0.04),
        active_mask=active_mask,
        device=device,
        maximum_dt=0.05,
        adaptive_dt_growth_factor=None,
        track_cumulative_face_volumes=True,
    )
    boundary = solver.boundary_type_qx.numpy()
    boundary[:, 1] = BOUNDARY_MASKED_WEST_FREE
    boundary[:, nx] = BOUNDARY_FREE_OUTFLOW
    solver.set_boundary_type_qx(boundary_type=boundary)
    depth0 = np.where(active_mask == 1, 0.4, 0.0)
    solver.set_water_depth(depth0)
    for _ in range(20):
        solver.step(remaining_time=0.05, fixed_dt=0.05)

    volumes = solver.get_cumulative_face_volumes()
    accounting = solver.cumulative_accounting()
    abs_qx = volumes["cumulative_abs_qx_volume_m3"]
    net_qx = volumes["cumulative_net_qx_volume_m3"]
    masked_volume = float(abs_qx[:, 1].sum())
    free_volume = float(abs_qx[:, nx].sum())
    assert masked_volume > 0.0
    assert free_volume > 0.0
    # Both boundary families are outward-only, so per-face net magnitude
    # equals the absolute volume.
    np.testing.assert_allclose(
        abs_qx[:, 1], np.abs(net_qx[:, 1]), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        abs_qx[:, nx], np.abs(net_qx[:, nx]), rtol=1e-9, atol=1e-12
    )
    # Every outward boundary volume is folded into the accounting ledger.
    assert masked_volume + free_volume == pytest.approx(
        accounting["boundary_outflow"], rel=1e-9
    )
    assert accounting["boundary_inflow"] == 0.0
    cell_area = float(solver.dx) * float(solver.dy)
    initial_volume = float(depth0.sum()) * cell_area
    final_volume = float(solver.get_water_depth().sum()) * cell_area
    # No sources: boundary outflow closes the storage difference.
    assert accounting["boundary_outflow"] == pytest.approx(
        initial_volume - final_volume, rel=1e-9
    )


@pytest.mark.parametrize("device", _devices())
def test_diagnostic_max_depth_is_running_post_step_maximum(device: str):
    # Rain-then-drain on an eastward-downslope bed with a FREE_OUTFLOW east
    # edge.  The reference is the elementwise running maximum of the
    # post-step (post-continuity) depth fields of an identically configured
    # non-tracking solver; the diagnostic starts from zeros and sees only
    # post-step states, so the initial depth is deliberately excluded.
    ny, nx = 2, 4

    def _configure(track: bool) -> LocalInertialSolver:
        solver = _make_sloped_row_solver(device=device, track=track)
        boundary = solver.boundary_type_qx.numpy()
        boundary[:, nx] = BOUNDARY_FREE_OUTFLOW
        solver.set_boundary_type_qx(boundary_type=boundary)
        solver.set_water_depth(np.full((ny, nx), 0.1))
        return solver

    tracked = _configure(track=True)
    reference = _configure(track=False)
    dt = 0.02
    running_max = np.zeros((ny, nx))
    for k in range(20):
        rate = 2.0e-3 if k < 6 else 0.0
        tracked.set_uniform_rainfall_rate(rate)
        reference.set_uniform_rainfall_rate(rate)
        tracked.step(remaining_time=dt, fixed_dt=dt)
        reference.step(remaining_time=dt, fixed_dt=dt)
        running_max = np.maximum(running_max, reference.get_water_depth())

    volumes = tracked.get_cumulative_face_volumes()
    np.testing.assert_allclose(
        volumes["diagnostic_max_depth_m"], running_max, rtol=1e-12, atol=1e-15
    )
    # The pulse makes the running maximum nontrivial: it exceeds the ponded
    # 0.1 m start depth somewhere and bounds every post-step depth.
    assert np.max(running_max) > 0.1
    assert np.all(running_max >= tracked.get_water_depth())


@pytest.mark.parametrize("device", _devices())
def test_disabled_mode_allocates_nothing_and_matches_bit_exactly(device: str):
    off = _make_sloped_row_solver(device=device, track=False)
    assert off.cumulative_net_qx_volume_m3 is None
    assert off.cumulative_net_qy_volume_m3 is None
    assert off.cumulative_abs_qx_volume_m3 is None
    assert off.cumulative_abs_qy_volume_m3 is None
    assert off._diagnostic_peak_discharge_m2_per_s is None
    assert off._diagnostic_max_depth_m is None
    with pytest.raises(RuntimeError, match="track_cumulative_face_volumes"):
        off.get_cumulative_face_volumes()
    with pytest.raises(RuntimeError, match="track_cumulative_face_volumes"):
        off.reset_cumulative_face_volumes()

    on = _make_sloped_row_solver(device=device, track=True)
    depth0 = np.full((2, 4), 0.3)
    off.set_water_depth(depth0)
    on.set_water_depth(depth0)
    for _ in range(15):
        off.step(remaining_time=0.02, fixed_dt=0.02)
        on.step(remaining_time=0.02, fixed_dt=0.02)
    # Tracking is pure observation: the trajectories are bit-identical.
    np.testing.assert_array_equal(off.get_water_depth(), on.get_water_depth())
    np.testing.assert_array_equal(off.get_qx(), on.get_qx())
    np.testing.assert_array_equal(off.get_qy(), on.get_qy())


@pytest.mark.parametrize("device", _devices())
def test_device_loop_matches_host_step_with_tracking(device: str):
    """Tracked device stepping matches the host trajectory and ledgers."""
    duration = 0.1
    depth0 = np.full((2, 4), 0.3)
    host = _make_sloped_row_solver(device=device, track=True)
    device_solver = _make_sloped_row_solver(device=device, track=True)
    host.set_water_depth(depth0)
    device_solver.set_water_depth(depth0)
    host_dt = []
    while host.simulation_time < duration:
        host_dt.append(host.step(remaining_time=duration - host.simulation_time))
    device_result = device_solver.run_device_loop(duration=duration, sync_interval=2, dt_history_capacity=128)
    for host_field, device_field in ((host.get_water_depth(), device_solver.get_water_depth()), (host.get_qx(), device_solver.get_qx()), (host.get_qy(), device_solver.get_qy())):
        np.testing.assert_allclose(host_field, device_field, rtol=1e-12, atol=1e-14)
    host_volumes = host.get_cumulative_face_volumes()
    device_volumes = device_solver.get_cumulative_face_volumes()
    for key in ("cumulative_net_qx_volume_m3", "cumulative_net_qy_volume_m3", "cumulative_abs_qx_volume_m3", "cumulative_abs_qy_volume_m3", "diagnostic_max_depth_m"):
        np.testing.assert_allclose(host_volumes[key], device_volumes[key], rtol=1e-12, atol=1e-14)
    assert host_volumes["peak_discharge_m2_per_s"] == pytest.approx(device_volumes["peak_discharge_m2_per_s"], rel=1e-12, abs=1e-14)
    host_mass = host.cumulative_accounting()
    device_mass = device_solver.cumulative_accounting()
    for key in host_mass:
        assert host_mass[key] == pytest.approx(device_mass[key], rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(host_dt, device_result["dt_history"][: device_result["steps"]], rtol=1e-12, atol=1e-14)
    assert device_result["steps"] == len(host_dt)


@pytest.mark.parametrize("device", _devices())
def test_set_water_depth_resets_diagnostic_transport_state(device: str):
    solver = _make_sloped_row_solver(device=device, track=True)
    depth0 = np.full((2, 4), 0.3)
    solver.set_water_depth(depth0)
    for _ in range(10):
        solver.step(remaining_time=0.02, fixed_dt=0.02)
    first = solver.get_cumulative_face_volumes()
    assert np.any(first["cumulative_abs_qx_volume_m3"] > 0.0)
    assert first["peak_discharge_m2_per_s"] > 0.0
    assert np.any(first["diagnostic_max_depth_m"] > 0.0)

    solver.set_water_depth(depth0)
    reset = solver.get_cumulative_face_volumes()
    for key in (
        "cumulative_net_qx_volume_m3",
        "cumulative_net_qy_volume_m3",
        "cumulative_abs_qx_volume_m3",
        "cumulative_abs_qy_volume_m3",
        "diagnostic_max_depth_m",
    ):
        assert not np.any(reset[key])
    assert reset["peak_discharge_m2_per_s"] == 0.0

    for _ in range(10):
        solver.step(remaining_time=0.02, fixed_dt=0.02)
    second = solver.get_cumulative_face_volumes()

    fresh = _make_sloped_row_solver(device=device, track=True)
    fresh.set_water_depth(depth0)
    for _ in range(10):
        fresh.step(remaining_time=0.02, fixed_dt=0.02)
    reference = fresh.get_cumulative_face_volumes()

    # No carry-over: the reset run reproduces a fresh solver's first run
    # bit-for-bit.
    for key in second:
        np.testing.assert_array_equal(second[key], reference[key])
    np.testing.assert_array_equal(solver.get_water_depth(), fresh.get_water_depth())
