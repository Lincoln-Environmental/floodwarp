"""Compare accepted FP64 arrays with an unmodified dev checkout in separate processes."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.fv2_solver import FV2Solver


def fingerprint(*, array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def trial(*, factory, mode, shape, device, seed):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    active = rng.uniform(size=shape) > 0.06
    h = rng.uniform(0.0, 0.3, shape)
    h[(rng.uniform(size=shape) < 0.3) | ~active] = 0.0
    s = factory(
        ny=ny,
        nx=nx,
        dx=1.0,
        dy=2.0,
        bed_elevation=rng.uniform(-0.05, 0.05, shape),
        manning_n=rng.uniform(0.02, 0.07, shape),
        active_mask=active,
        device=device,
        execution_mode=mode,
        maximum_dt=0.03,
    )
    s.set_water_depth(depth=h)
    s.set_momentum(hu=h * rng.uniform(-1.0, 1.0, shape), hv=h * rng.uniform(-0.8, 0.8, shape))
    s.set_source_rates(rain=rng.uniform(0.0, 0.002, shape), infil=0.0001)
    stats = s.run_device_loop(duration=1.5)
    arrays = dict(
        state=s.get_conservative_state(),
        maximum=s.get_max_water_depth(),
        soil=s.cumulative_infiltration_m(),
        **s.get_cumulative_face_volumes(),
    )
    return dict(
        method=factory.__name__,
        mode=mode,
        steps=stats["steps"],
        hashes={k: fingerprint(array=v) for k, v in arrays.items()},
    )


if __name__ == "__main__":
    output = Path(os.environ["FV_PARITY_OUTPUT"])
    device = "cuda:0"
    seed = 77402
    shape = (32, 48)
    modes = ("dense", "sparse")
    factories = (FullSWESolver, FV2Solver)
    rows = [trial(factory=f, mode=m, shape=shape, device=device, seed=seed) for f in factories for m in modes]
    output.write_text(json.dumps(rows, indent=2) + "\n")
