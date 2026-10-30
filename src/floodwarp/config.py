from __future__ import annotations

import os
import numpy as np

_legacy_float_env = os.environ.get("FLOODWARP_FLOAT", "float64")
# Backwards-compatible name used by the precision-parameterized test suite.
_float_env = _legacy_float_env
_precision_env = os.environ.get("FLOODWARP_PRECISION")

if _precision_env is None:
    if _legacy_float_env == "float64":
        PRECISION_MODE = "fp64"
    elif _legacy_float_env == "float32":
        PRECISION_MODE = "fp32"
    else:
        raise ValueError("FLOODWARP_FLOAT must be 'float32' or 'float64'")
else:
    _precision_aliases = {
        "fp64": "fp64",
        "float64": "fp64",
        "fp32": "fp32",
        "float32": "fp32",
        "mixed": "mixed",
    }
    try:
        PRECISION_MODE = _precision_aliases[_precision_env.lower()]
    except KeyError as error:
        raise ValueError(
            "FLOODWARP_PRECISION must be 'fp64', 'fp32', or 'mixed'"
        ) from error

# Tests and reporting utilities use this compatibility value to choose
# hydraulic tolerances. Mixed retains FP32 hydraulic state, so it uses the
# FP32 tolerance family even though bed/stage arrays are FP64.
_float_env = "float64" if PRECISION_MODE == "fp64" else "float32"

# Mixed precision uses a separately conditioned FP32 face-kernel path.  The
# legacy direct-gradient branch remains available only for compatibility with
# cached kernels; normal mixed stepping must not enter it.
USE_DIRECT_GRADIENT = 0
USE_CONDITIONED_MIXED_KERNELS = int(PRECISION_MODE == "mixed")

__all__ = [
    "NP_FLOAT", "WP_FLOAT", "BED_FLOAT", "STAGE_FLOAT", "PRECISION_MODE",
    "USE_DIRECT_GRADIENT", "USE_CONDITIONED_MIXED_KERNELS",
]

def __getattr__(name: str):
    if name == "WP_FLOAT":
        import warp as wp
        if PRECISION_MODE == "fp64":
            return wp.float64
        if PRECISION_MODE in ("fp32", "mixed"):
            return wp.float32
        raise AssertionError(f"unhandled precision mode {PRECISION_MODE!r}")

    if name == "NP_FLOAT":
        if PRECISION_MODE == "fp64":
            return np.float64
        if PRECISION_MODE in ("fp32", "mixed"):
            return np.float32
        raise AssertionError(f"unhandled precision mode {PRECISION_MODE!r}")

    if name in ("BED_FLOAT", "STAGE_FLOAT"):
        import warp as wp
        # Mixed precision keeps the DEM and elevation-dependent stage forcing
        # in FP64 while water depth/q/source fields remain FP32. This avoids
        # quantising a high-datum, shallow-gradient DEM at input time.
        return wp.float64 if PRECISION_MODE in ("fp64", "mixed") else wp.float32

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
