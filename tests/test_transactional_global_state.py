"""Exact global-step rollback, including deferred accounting and constraints."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_global_snapshot_restores_all_arrays_and_device_exclusion(device):
    if device.startswith("cuda") and not wp.is_device_available(device):
        pytest.skip("CUDA unavailable")
    shape = (3, 4)
    solver = LocalInertialSolver(ny=shape[0], nx=shape[1], dx=10.0, dy=10.0,
        bed_elevation=np.zeros(shape), manning_n=np.full(shape, 0.035),
        active_mask=np.ones(shape, dtype=np.int32), device=device,
        infiltration_method="green_ampt", infiltration_ks_m_per_s=1e-5,
        infiltration_psi_f_m=0.05, infiltration_delta_theta=0.2,
        track_max_water_depth=True, track_cumulative_face_volumes=True,
        accounting_interval=4)
    solver.set_uniform_rainfall_rate(rainfall_rate=3e-5)
    mask = np.eye(*shape, dtype=bool)
    solver.set_infiltration_exclusion_mask(mask)
    solver.set_infiltration_storage_limit_m(np.full(shape, 0.002))
    solver.step(fixed_dt=1.0)
    state = solver.capture_state()
    solver.set_infiltration_exclusion_mask(~mask)
    solver.set_additional_source_rate(np.full(shape, 1e-5))
    solver.step(fixed_dt=2.0)
    solver.restore_state(state)
    restored = solver.capture_state()
    for name, value in state["host"].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, restored["host"][name])
        else:
            assert value is restored["host"][name] or value == restored["host"][name]
    for name, value in state["device"].items():
        if value is not None:
            np.testing.assert_array_equal(value.numpy(), restored["device"][name].numpy(), err_msg=name)
    np.testing.assert_array_equal(solver.infiltration_exclusion_mask.numpy().astype(bool), mask)


def test_lts_snapshot_is_explicitly_unsupported():
    shape = (2, 2)
    solver = LocalInertialSolver(ny=2, nx=2, dx=10.0, dy=10.0,
        bed_elevation=np.zeros(shape), manning_n=np.full(shape, 0.035),
        active_mask=np.ones(shape, dtype=np.int32), device="cpu")
    solver._lts_cell_levels = wp.zeros(shape, dtype=wp.int32, device="cpu")
    with pytest.raises(RuntimeError, match="global stepping"):
        solver.capture_state()
