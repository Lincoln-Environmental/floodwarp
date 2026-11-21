"""Warmed fixed-step GPU controls and occupancy sweep on identical bed and fronts."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from validation.fv2_cost import reset
from validation.fv2_local_inertia import build, domain
from validation.hancock_default_profile import measure


def sweep(
    *, directory, device, fractions, methods, length, width, dx, cap, duration, repeats
):
    rows = []
    for fraction in fractions:
        data = domain(
            case="stoker",
            dx=dx,
            duration=duration,
            gravity=9.80665,
            near_zero_n=1.0e-7,
            length=length,
            width=width,
        )
        data["initial"].fill(0.0)
        data["initial"][..., 0] = np.where(
            abs(data["x"]) < fraction * length / 2.0, 0.45, 0.0
        )
        for method, mode in methods:
            s = build(
                d=data, method=method, dx=dx, cap=cap, device=device, gravity=9.80665
            )
            s.configure_execution(mode=mode)
            initial = s.capture_state()
            warm = s.run_device_loop(duration=duration)
            times = []
            for _ in range(repeats):
                reset(solver=s, state=initial)
                wp.synchronize_device(device=device)
                start = perf_counter()
                stats = s.run_device_loop(duration=duration)
                wp.synchronize_device(device=device)
                times.append(perf_counter() - start)
            expected = s.get_conservative_state()
            diag = s.execution_diagnostics()
            reset(solver=s, state=initial)
            direct = s.run_device_loop(duration=duration, use_cuda_graph=False)
            np.testing.assert_array_equal(s.get_conservative_state(), expected)
            rows.append(
                dict(
                    method=method,
                    mode=mode,
                    seed_fraction=fraction,
                    initial_fraction=float((data["initial"][..., 0] > 0.0).mean()),
                    final_fraction=float((expected[..., 0] > 0.0).mean()),
                    active=diag,
                    steps=stats["steps"],
                    graph_s=float(np.median(times)),
                    repeats_s=times,
                    uncaptured_loop_s=direct["loop_seconds"],
                    graph_loop_s=stats["loop_seconds"],
                    capture_s=warm["graph_capture_seconds"],
                )
            )
        print(fraction, flush=True)
    (directory / "occupancy.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).parent
    cases = ("stoker", "dry_patch")
    methods = (
        ("contact_hll", "sparse"),
        ("hancock_adaptive", "dense"),
        ("hancock_adaptive", "sparse"),
        ("hancock_adaptive", "dynamic"),
        ("hancock_adaptive", "tiles"),
    )
    device = "cuda:0"
    length, width, dx, cap, duration, repeats = 1024.0, 640.0, 1.0, 1 / 32, 2.0, 7
    fractions = (0.0, 0.005, 0.03, 0.12, 0.35, 0.55, 0.7, 0.85, 1.0)
    measure(
        output=directory / "final_profile.json",
        cases=cases,
        methods=methods,
        length=length,
        width=width,
        spacing=dx,
        cap=cap,
        duration=duration,
        repeats=repeats,
        device=device,
    )
    sweep(
        directory=directory,
        device=device,
        fractions=fractions,
        methods=(methods[0], methods[1], methods[-1]),
        length=length,
        width=width,
        dx=dx,
        cap=cap,
        duration=duration,
        repeats=repeats,
    )
