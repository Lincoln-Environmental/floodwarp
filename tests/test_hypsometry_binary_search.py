"""Binary queries must preserve the original linear scan bit for bit."""

import numpy as np
import warp as wp

from floodwarp.config import BED_FLOAT, WP_FLOAT, NP_FLOAT
from floodwarp.subgrid_hypsometry import subgrid_stage_from_storage, subgrid_wet_fraction_from_storage


@wp.kernel
def compare_queries(
    volumes: wp.array2d(dtype=WP_FLOAT), elevations: wp.array3d(dtype=BED_FLOAT),
    knots: wp.array3d(dtype=BED_FLOAT), count: int,
    expected: wp.array2d(dtype=WP_FLOAT), actual: wp.array2d(dtype=WP_FLOAT),
    expected_wet: wp.array2d(dtype=WP_FLOAT), actual_wet: wp.array2d(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    storage = volumes[j, i]
    eta = elevations[j, i, 0]
    if storage > WP_FLOAT(0):
        eta = elevations[j, i, count - 1] + BED_FLOAT(storage - knots[j, i, count - 1])
        found = int(0)
        for m in range(count - 1):
            if found == 0:
                if BED_FLOAT(storage) <= knots[j, i, m + 1]:
                    eta = elevations[j, i, m] + BED_FLOAT((storage - knots[j, i, m]) * BED_FLOAT(count) / BED_FLOAT(m + 1))
                    found = 1
    wet = int(0)
    for m in range(count):
        if eta > elevations[j, i, m]:
            wet += 1
    expected[j, i] = WP_FLOAT(eta)
    expected_wet[j, i] = WP_FLOAT(wet) / WP_FLOAT(count)
    actual[j, i] = WP_FLOAT(subgrid_stage_from_storage(storage, elevations, knots, j, i, count))
    actual_wet[j, i] = subgrid_wet_fraction_from_storage(storage, elevations, knots, j, i, count)


def test_binary_queries_preserve_linear_arithmetic_at_every_knot_and_plateau():
    generator = np.random.default_rng(seed=985)
    for count in (1, 25, 100, 625):
        heights = np.sort(generator.integers(0, 40, size=count).astype(float) / 7 + 1e3)
        breakpoints = (heights * np.arange(1, count + 1) - np.cumsum(heights)) / count
        values = np.concatenate(([0, -1, 10], breakpoints,
                                 np.nextafter(breakpoints, np.inf), np.nextafter(breakpoints, -np.inf)))
        volumes = values.astype(NP_FLOAT).reshape(1, -1)
        elevation = np.broadcast_to(heights, (*volumes.shape, count)).copy()
        knots = np.broadcast_to(breakpoints, elevation.shape).copy()
        outputs = [wp.empty(volumes.shape, dtype=WP_FLOAT, device="cpu") for unused in range(4)]
        wp.launch(compare_queries, dim=volumes.shape,
                  inputs=[wp.array(volumes, dtype=WP_FLOAT, device="cpu"),
                          wp.array(elevation, dtype=BED_FLOAT, device="cpu"),
                          wp.array(knots, dtype=BED_FLOAT, device="cpu"), count, *outputs], device="cpu")
        assert outputs[0].numpy().tobytes() == outputs[1].numpy().tobytes()
        assert outputs[2].numpy().tobytes() == outputs[3].numpy().tobytes()
