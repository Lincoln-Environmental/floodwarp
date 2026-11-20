"""One uncaptured hydraulic step for Nsight Compute kernel counters."""

import os
from validation.fv2_local_inertia import build, domain


def probe(*, dx, device, length, width, duration, method="mc", fused=True):
    fields = domain(
        case="stoker", dx=dx, duration=duration, gravity=9.80665, near_zero_n=1e-7, length=length, width=width
    )
    solver = build(d=fields, method=method, dx=dx, cap=duration, device=device, gravity=9.80665)
    if method.startswith("hancock_"):
        solver._fuse_prediction = fused
    solver.run_device_loop(duration=duration, use_cuda_graph=False)


if __name__ == "__main__":
    method=os.environ.get("FV_PROFILE_METHOD","mc")
    fused=os.environ.get("FV_PROFILE_FUSE","1")=="1"
    probe(method=method,fused=fused,dx=5.0, device="cuda:0", length=1920.0, width=1280.0, duration=0.05)
