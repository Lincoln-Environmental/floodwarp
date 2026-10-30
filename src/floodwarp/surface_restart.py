"""Numeric JSON serialization for standalone checkpoint metadata.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp


def json_value(value):
    """Encode numeric metadata without pickling Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported checkpoint metadata {type(value)}")
