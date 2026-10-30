"""Interval hints must reproduce an independent uncached upper-bound lookup."""
from typing import Any

import numpy as np
import pytest
import warp as wp

from floodwarp.enhanced_kernels import cached_scalar_arm


@wp.kernel
def compare_intervals(stages: wp.array(dtype=wp.float64),
                      begins: wp.array(dtype=int), ends: wp.array(dtype=int),
                      rows: wp.array2d(dtype=wp.float64), indices: wp.array(dtype=Any),
                      values: wp.array2d(dtype=Any), hints: wp.array2d(dtype=int),
                      actual: wp.array2d(dtype=wp.float64),
                      expected: wp.array2d(dtype=wp.float64)):
    c = wp.tid()
    for i in range(stages.shape[0]):
        stage = stages[i]
        begin = begins[i]
        end = ends[i]
        # Keep the pre-optimisation search and interpolation as the reference.
        value = wp.float64(values[begin, 0])
        if end-begin > 1:
            lo = begin
            hi = end
            while lo < hi:
                mid = (lo+hi)//2
                if rows[int(indices[mid]), 0] <= stage:
                    lo = mid+1
                else:
                    hi = mid
            k = wp.max(begin, wp.min(end-2, lo-1))
            left = rows[int(indices[k]), 0]
            right = rows[int(indices[k+1]), 0]
            fraction = wp.clamp((stage-left)/(right-left), wp.float64(0.0), wp.float64(1.0))
            a = wp.float64(values[k, 0])
            b = wp.float64(values[k+1, 0])
            value = a+fraction*(b-a)
        expected[c, i] = value
        actual[c, i] = cached_scalar_arm(stage, begin, end, rows, indices, values, hints, c, 0)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('index_type', [np.int32, np.uint16])
@pytest.mark.parametrize('value_type', [np.float32, np.float64])
def test_exact_intervals_after_reversals_knots_and_invalid_hints(device, index_type, value_type):
    if device != 'cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    knots = np.array([0., .01, .13, .5, 2., 4.])
    # Exercise exact knots, either adjacent FP64 value, extrapolation clamps,
    # changing table identities and constant tables without hint storage.
    sequence = np.concatenate(([-1.], knots, np.nextafter(knots, np.inf),
                               np.nextafter(knots, -np.inf), [10.], knots[::-1]))
    stages = np.tile(sequence, 3)
    begins = np.repeat([0, 6, 12], len(sequence)).astype(np.int32)
    ends = np.repeat([6, 12, 13], len(sequence)).astype(np.int32)
    rows = wp.array(np.tile(knots, 3)[:, None], dtype=wp.float64, device=device)
    indices = wp.array(np.arange(13, dtype=index_type), device=device)
    values = wp.array(np.random.default_rng(42).uniform(.01, 5., (13, 1)).astype(value_type), device=device)
    hints = wp.array(np.array([[-1], [1000000], [3], [0]], np.int32), device=device)
    actual = wp.zeros((4, len(stages)), dtype=wp.float64, device=device)
    expected = wp.zeros_like(actual)
    wp.launch(kernel=compare_intervals, dim=4, inputs=[
        wp.array(stages, dtype=wp.float64, device=device),
        wp.array(begins, dtype=int, device=device), wp.array(ends, dtype=int, device=device),
        rows, indices, values, hints, actual, expected], device=device)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
