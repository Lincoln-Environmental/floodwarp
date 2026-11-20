"""Time first module loads in a fresh Warp cache separately from steady-state execution."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import warp as wp

from validation.fv1_riemann import report_json


def measure(*, cache, device, size, output):
    wp.config.kernel_cache_dir=str(cache)
    wp.init()
    from floodwarp import full_swe_kernels,fv2_kernels
    from floodwarp.fv2_solver import FV2Solver
    modules={}
    for module in (full_swe_kernels,fv2_kernels):
        start=perf_counter()
        wp.load_module(module=module,device=device)
        wp.synchronize_device(device=device)
        modules[module.__name__]=perf_counter()-start
    s=FV2Solver(ny=size,nx=size,dx=5.,dy=5.,bed_elevation=0.,manning_n=.03,
                active_mask=np.ones((size,size),bool),device=device)
    s.set_water_depth(depth=1.)
    first=s.run_device_loop(duration=.1)
    warm=s.run_device_loop(duration=.1)
    report_json(path=output,content=dict(warp=wp.__version__,device=device,
        empty_warp_cache_compile_and_load_seconds=modules,first=first,warm=warm))


if __name__ == "__main__":
    device="cuda:0"
    size=32
    output=Path("validation/fv2_review/compile.json")
    with TemporaryDirectory(prefix="floodwarp-fv2-compile-") as cache:
        measure(cache=Path(cache),device=device,size=size,output=output)

