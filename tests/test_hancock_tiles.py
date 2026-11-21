"""Exact transactional parity across active, cleared and newly seeded tiles."""

import numpy as np
import pytest
import warp as wp

from floodwarp.fv2_solver import HancockSolver


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_moving_rain_stage_and_thin_channel(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    y, x = np.indices((19, 193))
    channel = (y == 9) | (x < 32) | (x > 155)
    settings = dict(
        nx=193,
        ny=19,
        dx=1.0,
        dy=1.0,
        bed_elevation=-0.002 * x,
        active_mask=channel,
        maximum_dt=1.0,
        manning_n=0.03,
        device=device,
    )
    dense, tiled = (
        HancockSolver(execution_mode=m, **settings) for m in ("dense", "tiles")
    )
    for s in (dense, tiled):
        bx = np.zeros((19, 194), dtype=np.int32)
        bx[9, 0] = 3
        s.set_boundary_type_qx(boundary_type=bx)
        s.set_stage_series(
            axis="x", series=([0.0, 1.0, 3.0, 30.0], [0.0, 0.15, 0.0, 0.0])
        )
        s.set_exponential_infiltration(f0=5.0e-6, fd=0.01, conductivity=2.0e-7)
    for step, centre in enumerate((175, 175, 0, 45, 75, 0, 129, 171)):
        rain = (
            np.where(channel & (abs(x - centre) <= 1), 2.0e-5, 0.0)
            if centre
            else np.zeros(x.shape)
        )
        for s in (dense, tiled):
            s.set_source_rates(rain=rain)
        for duration in (0.125, 0.875):
            stats = [s.run_device_loop(duration=duration) for s in (dense, tiled)]
            for key in ("steps", "dt_min_s", "dt_max_s", "rejected_cfl_trials"):
                assert stats[0][key] == stats[1][key]
            a, b = dense.capture_state(), tiled.capture_state()
            for name in a["arrays"]:
                np.testing.assert_array_equal(
                    a["arrays"][name], b["arrays"][name], err_msg=name
                )
            np.testing.assert_array_equal(
                dense.predictor_fallback_cells(), tiled.predictor_fallback_cells()
            )
            if step == 0:
                film = tiled.get_water_depth()[9, 175]
                assert 0.0 < film < tiled.dry_depth
        if step == 3:
            tiled.restore_state(state=dense.capture_state())
    assert tiled.cumulative_accounting()["boundary_inflow"] > 0.0
    assert abs(tiled.cumulative_accounting()["residual"]) < 1.0e-10


def test_directional_characteristic_sum_and_checkpoint_identity():
    settings = dict(
        nx=65,
        ny=13,
        dx=2.0,
        dy=4.0,
        bed_elevation=0.0,
        active_mask=np.ones((13, 65)),
        manning_n=0.0,
        maximum_dt=2.0,
        device="cpu",
        cfl_policy="directional",
    )
    solvers = [
        HancockSolver(execution_mode=mode, **settings) for mode in ("dense", "tiles")
    ]
    for s in solvers:
        s.set_water_depth(depth=0.2)
        s.set_momentum(hu=0.4, hv=0.0)
        dt = s.step()
        c = np.sqrt(s.gravity * 0.2)
        np.testing.assert_allclose(
            dt, 0.95 * s.cfl / ((2.0 + c) / s.dx + c / s.dy), rtol=1e-14
        )
    np.testing.assert_array_equal(
        solvers[0].get_conservative_state(), solvers[1].get_conservative_state()
    )
    old = HancockSolver(**dict(settings, cfl_policy="legacy"))
    with pytest.raises(ValueError, match="configuration"):
        old.restore_state(state=solvers[0].capture_state())


def test_positive_film_crosses_tile_edge_on_first_accepted_step():
    mask = np.zeros((17, 161), dtype=bool)
    mask[8, :] = True
    h = np.zeros_like(mask, dtype=float)
    h[8, 96] = 3.0e-6
    settings = dict(
        nx=161,
        ny=17,
        dx=1.0,
        dy=1.0,
        active_mask=mask,
        bed_elevation=0.0,
        manning_n=0.035,
        device="cpu",
        maximum_dt=0.5,
    )
    dense, tiled = (
        HancockSolver(execution_mode=mode, **settings) for mode in ("dense", "tiles")
    )
    for s in (dense, tiled):
        # Release the work list, then introduce positive water between tile columns.
        s.run_device_loop(duration=1.0)
        s.set_source_rates(rain=h)
    for _ in range(8):
        assert dense.step() == tiled.step()
        np.testing.assert_array_equal(
            tiled.get_conservative_state(), dense.get_conservative_state()
        )
        assert 0.0 < tiled.get_water_depth()[8, 95] < tiled.dry_depth
        assert tiled.get_cumulative_face_volumes()["qx"][8, 96] < 0.0
    assert abs(tiled.cumulative_accounting()["residual"]) < 1.0e-20
