"""Absolute output times shared by public examples and local runs."""
import numpy as np


def sample_targets(*, duration, interval):
    """Include the exact requested final time once, without overshoot."""
    if not np.isfinite([duration, interval]).all() or min(duration, interval) <= 0.:
        raise ValueError('Positive finite duration and sample interval required')
    interior = np.arange(1, int(np.floor(duration / interval)) + 1, dtype=np.float64) * interval
    return np.r_[interior[interior < duration], float(duration)]
