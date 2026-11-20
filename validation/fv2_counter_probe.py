"""One uncaptured hydraulic step for Nsight Compute kernel counters."""

from validation.fv2_local_inertia import build, domain


def probe(*, dx, device, length, width, duration):
    fields = domain(
        case="stoker", dx=dx, duration=duration, gravity=9.80665, near_zero_n=1e-7, length=length, width=width
    )
    solver = build(d=fields, method="mc", dx=dx, cap=duration, device=device, gravity=9.80665)
    solver.run_device_loop(duration=duration, use_cuda_graph=False)


if __name__ == "__main__":
    probe(dx=5.0, device="cuda:0", length=1920.0, width=1280.0, duration=0.05)
