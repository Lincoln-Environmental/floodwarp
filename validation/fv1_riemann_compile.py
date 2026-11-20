"""Measure first CUDA compilation and load in an empty Warp kernel cache."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json

import warp as wp


def measure(*, device, cache):
    wp.config.kernel_cache_dir = str(cache)
    wp.init()
    from floodwarp import full_swe_kernels

    wp.synchronize_device(device=device)
    began = perf_counter()
    wp.load_module(module=full_swe_kernels, device=device)
    wp.synchronize_device(device=device)
    return dict(device=device, warp=wp.__version__, module=full_swe_kernels.__name__,
                empty_warp_cache_compile_and_load_seconds=perf_counter()-began)


if __name__ == "__main__":
    device = 'cuda:0'
    output = Path('validation/fv1_riemann_review/corrected/compile.json')
    with TemporaryDirectory() as cache:
        measurement = measure(device=device, cache=Path(cache))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(measurement, indent=2) + '\n')
