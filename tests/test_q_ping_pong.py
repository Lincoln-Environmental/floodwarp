"""Regression tests for persistent q-face ping-pong ownership."""

import numpy as np
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver


def build_solver():
    ny, nx = 4, 5
    bed = np.zeros((ny, nx), dtype=np.float64)
    manning = np.full((ny, nx), 0.035, dtype=np.float64)
    active = np.ones((ny, nx), dtype=np.int32)
    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=10.0,
        dy=10.0,
        bed_elevation=bed,
        manning_n=manning,
        active_mask=active,
        device="cpu",
        validation_profile="lisflood_fp_acc",
    )
    solver.set_water_depth(np.full((ny, nx), 0.1, dtype=np.float64))
    solver.set_source_rates(rain=np.full((ny, nx), 1.0e-6, dtype=np.float64))
    return solver


def test_q_ping_pong_swaps_handles_without_step_copy(monkeypatch):
    wp.init()
    solver = build_solver()
    qx_initial = solver.qx
    qy_initial = solver.qy
    copied_during_step = []

    original_copy = wp.copy

    def record_copy(*args, **kwargs):
        copied_during_step.append(True)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(wp, "copy", record_copy)
    solver.step(fixed_dt=0.1)

    assert not copied_during_step
    assert solver.qx is solver.qx_next or solver.qx_next is qx_initial
    assert solver.qy is solver.qy_next or solver.qy_next is qy_initial
    assert solver.qx_next is qx_initial
    assert solver.qy_next is qy_initial


def test_q_ping_pong_public_flux_state_is_current_after_two_steps():
    wp.init()
    solver = build_solver()
    solver.step(fixed_dt=0.1)
    first_qx = solver.get_qx().copy()
    first_qy = solver.get_qy().copy()
    solver.step(fixed_dt=0.1)

    assert np.array_equal(first_qx, solver.qx_next.numpy())
    assert np.array_equal(first_qy, solver.qy_next.numpy())
    assert np.array_equal(solver.get_qx(), solver.qx.numpy())
    assert np.array_equal(solver.get_qy(), solver.qy.numpy())


def test_set_face_fluxes_restores_both_ping_pong_buffers():
    wp.init()
    solver = build_solver()
    qx = np.linspace(-0.2, 0.2, solver.ny * (solver.nx + 1)).reshape(
        solver.ny,
        solver.nx + 1,
    )
    qy = np.linspace(-0.1, 0.1, (solver.ny + 1) * solver.nx).reshape(
        solver.ny + 1,
        solver.nx,
    )

    solver.set_face_fluxes(qx=qx, qy=qy)

    assert np.array_equal(solver.get_qx(), qx)
    assert np.array_equal(solver.get_qy(), qy)
    assert np.array_equal(solver.qx_next.numpy(), qx)
    assert np.array_equal(solver.qy_next.numpy(), qy)
