"""Bit-exact equivalence between the fused and unfused kernel paths.

The fused kernels (update_momentum, apply_outflow_limiter, continuity_update_and_check)
must reproduce the unfused path exactly. This guards Phase 1 and later phases
against regressions and supports the bisect-at-the-kernel-level strategy via the
OVERLANDER_FUSED_{MOMENTUM,LIMITER,CONTINUITY} environment overrides.
"""
import os

import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE
from floodwarp.local_inertial_solver import LocalInertialSolver


def _build(device, profile, seed, fused):
    rng = np.random.default_rng(seed)
    ny, nx = 6, 7
    bed = rng.random((ny, nx)) * 0.5
    manning = rng.uniform(0.02, 0.06, (ny, nx))
    mask = np.ones((ny, nx), dtype=np.int32)
    mask[2, 3] = 0
    solver = LocalInertialSolver(
        ny=ny, nx=nx, dx=10.0, dy=10.0,
        bed_elevation=bed, manning_n=manning, active_mask=mask,
        device=device, validation_profile=profile, use_fused_kernels=fused,
    )
    b_qx = solver.boundary_type_qx.numpy()
    b_qx[0, nx] = BOUNDARY_FREE_OUTFLOW
    b_qx[1, 0] = BOUNDARY_SPECIFIED_STAGE
    solver.set_boundary_type_qx(boundary_type=b_qx)
    b_stage = solver.boundary_stage_qx.numpy()
    b_stage[1, 0] = 1.2
    solver.set_boundary_stage_qx(stage=b_stage)
    h = rng.random((ny, nx)) * 0.3
    h[mask == 0] = 0.0
    solver.set_water_depth(h)
    solver.set_source_rates(rain=rng.random((ny, nx)) * 2e-4, infil=rng.random((ny, nx)) * 1e-6)
    return solver


_FUSION_ENVS = [
    pytest.param(None, id="full"),
    pytest.param({"OVERLANDER_FUSED_MOMENTUM": "1", "OVERLANDER_FUSED_LIMITER": "0", "OVERLANDER_FUSED_CONTINUITY": "0"}, id="momentum-only"),
    pytest.param({"OVERLANDER_FUSED_MOMENTUM": "0", "OVERLANDER_FUSED_LIMITER": "1", "OVERLANDER_FUSED_CONTINUITY": "0"}, id="limiter-only"),
    pytest.param({"OVERLANDER_FUSED_MOMENTUM": "0", "OVERLANDER_FUSED_LIMITER": "0", "OVERLANDER_FUSED_CONTINUITY": "1"}, id="continuity-only"),
]


@pytest.mark.parametrize("profile", [None, "lisflood_fp_acc"])
@pytest.mark.parametrize("sut_env", _FUSION_ENVS)
def test_fused_path_matches_unfused_bit_exact(profile, sut_env):
    wp.init()
    device = "cuda:0" if wp.is_device_available("cuda:0") else "cpu"
    for k in ("OVERLANDER_FUSED_MOMENTUM", "OVERLANDER_FUSED_LIMITER", "OVERLANDER_FUSED_CONTINUITY"):
        os.environ.pop(k, None)
    ref = _build(device, profile, seed=7, fused=False)
    fixed_dt = ref.calculate_stable_timestep()

    if sut_env:
        for k, v in sut_env.items():
            os.environ[k] = v
    sut = _build(device, profile, seed=7, fused=True)
    for k in ("OVERLANDER_FUSED_MOMENTUM", "OVERLANDER_FUSED_LIMITER", "OVERLANDER_FUSED_CONTINUITY"):
        os.environ.pop(k, None)

    for _ in range(40):
        ref.step(fixed_dt=fixed_dt)
        sut.step(fixed_dt=fixed_dt)

    assert np.array_equal(ref.water_depth.numpy(), sut.water_depth.numpy())
    assert np.array_equal(ref.qx.numpy(), sut.qx.numpy())
    assert np.array_equal(ref.qy.numpy(), sut.qy.numpy())


def test_reduced_mass_accounting_matches_atomic_accounting(monkeypatch):
    """The Phase 2 reduction must not alter the hydraulic update."""
    wp.init()
    device = "cuda:0" if wp.is_device_available("cuda:0") else "cpu"

    monkeypatch.setenv("OVERLANDER_REDUCE_MASS", "0")
    atomic = _build(device, "lisflood_fp_acc", seed=11, fused=True)
    monkeypatch.setenv("OVERLANDER_REDUCE_MASS", "1")
    reduced = _build(device, "lisflood_fp_acc", seed=11, fused=True)

    for _ in range(20):
        atomic.step(fixed_dt=0.25)
        reduced.step(fixed_dt=0.25)

    assert np.array_equal(atomic.water_depth.numpy(), reduced.water_depth.numpy())
    assert np.array_equal(atomic.qx.numpy(), reduced.qx.numpy())
    assert np.array_equal(atomic.qy.numpy(), reduced.qy.numpy())
    atomic_totals = atomic.cumulative_accounting()
    reduced_totals = reduced.cumulative_accounting()
    for key in atomic_totals:
        assert reduced_totals[key] == pytest.approx(atomic_totals[key], rel=1.0e-9, abs=1.0e-12)
