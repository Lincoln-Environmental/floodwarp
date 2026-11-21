"""Constant and decaying infiltration capacities share a moving wet/dry footprint."""

import numpy as np
import pytest
import warp as wp

from floodwarp.fv2_solver import HancockSolver
from validation.hancock_long_storm_review.capacity import ReferenceHancock


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_constant_capacity_uses_same_accepted_state_as_midpoint(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    y, x = np.indices((8, 9))
    states = []
    for constructor in (ReferenceHancock, HancockSolver):
        solver = constructor(
            nx=9,
            ny=8,
            dx=5.0,
            dy=5.0,
            bed_elevation=0.0,
            manning_n=0.03,
            active_mask=np.ones(x.shape),
            device=device,
            execution_mode="tiles",
            maximum_dt=1.0,
        )
        solver.set_water_depth(depth=np.where(x % 3 == 0, 0.0, 0.15))
        solver.set_exponential_infiltration(
            f0=np.where(x % 2, 0.0001, 0.0002), fd=0.001, conductivity=0.0001
        )
        solver.set_source_rates(rain=np.where(y % 2, 0.0005, 0.0))
        solver.run_device_loop(duration=150.0)
        states.append(solver.capture_state()["arrays"])
    for key, reference in states[0].items():
        np.testing.assert_array_equal(states[1][key], reference, err_msg=key)
