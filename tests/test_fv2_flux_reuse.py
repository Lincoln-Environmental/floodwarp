"""Check that FV2's precomputed-speed flux preserves existing contact-HLL."""

import numpy as np
import pytest
import warp as wp

from floodwarp.full_swe_kernels import hll, hll_speeds
from floodwarp.fv2_kernels import hll_fv2_precomputed


@wp.kernel
def paired_flux(
    left: wp.array(dtype=wp.vec3d),
    right: wp.array(dtype=wp.vec3d),
    reference: wp.array(dtype=wp.vec3d),
    candidate: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    g = wp.float64(9.81)
    dry = wp.float64(1.0e-4)
    bounds = hll_speeds(left[i], right[i], g, dry, 1)
    reference[i] = hll(left[i], right[i], g, dry, 1)
    candidate[i] = hll_fv2_precomputed(left[i], right[i], g, dry, 1, bounds)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_fv2_precomputed_hll_flux_matches_original(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(27591)
    count = 512
    depths = 10.0 ** rng.uniform(-7.0, 1.5, size=(2, count))
    velocities = rng.uniform(-9.0, 9.0, size=(2, count, 2))
    states = np.zeros((2, count, 3), dtype=np.float64)
    states[..., 0] = depths
    states[..., 1:] = depths[..., None] * velocities
    states[..., 1:] *= (depths > 1.0e-4)[..., None]
    states[:, :4] = [
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, -14.0, 7.0], [1.0, 0.5, -2.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 14.0, -7.0], [1.0, 0.5, 2.0]],
    ]
    inputs_left = wp.array(states[0], dtype=wp.vec3d, device=device)
    inputs_right = wp.array(states[1], dtype=wp.vec3d, device=device)
    reference = wp.empty(count, dtype=wp.vec3d, device=device)
    candidate = wp.empty(count, dtype=wp.vec3d, device=device)
    wp.launch(paired_flux, dim=count, inputs=[inputs_left, inputs_right, reference, candidate], device=device)
    np.testing.assert_allclose(candidate.numpy(), reference.numpy(), rtol=1.0e-14, atol=1.0e-14)
