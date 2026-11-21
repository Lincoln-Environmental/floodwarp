"""Measure work and positive-depth support during the original 64-step benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np

from validation.fv2_local_inertia import build, domain


def sample(*, review, shape, cases, methods, dx, duration, cap, cadence, device):
    rows = []
    baseline = json.loads((review / "final_profile.json").read_text())
    for case in cases:
        length, width = shape
        inputs = domain(
            case="stoker",
            dx=dx,
            duration=duration,
            gravity=9.80665,
            near_zero_n=1e-7,
            length=length,
            width=width,
        )
        if case == "dry_patch":
            inputs["initial"][..., 0] = np.where(
                abs(inputs["x"]) < length / 32, 0.45, 0.0
            )
        for method, mode in methods:
            s = build(
                d=inputs, method=method, dx=dx, cap=cap, device=device, gravity=9.80665
            )
            s.configure_execution(mode=mode)
            initial = s.capture_state()
            series = []
            for t in np.arange(cap * cadence, duration + cap / 2, cap * cadence):
                stats = s.run_device_loop(duration=t - s.time)
                work = s.execution_diagnostics()
                dynamic = []
                if mode == "dynamic":
                    dynamic = [int(a.numpy()[0]) for a in s._face_tiles]
                series.append(
                    dict(
                        t_s=t,
                        steps=stats["steps"],
                        wet_fraction=float((s.get_water_depth() > 0.0).mean()),
                        dynamic_face_tiles_xy=dynamic,
                        hardware_work=work,
                    )
                )
            sha = hashlib.sha256(s.get_conservative_state().tobytes()).hexdigest()
            ref = next(
                r
                for r in baseline
                if r["case"] == case and r["method"] == method and r["mode"] == mode
            )
            if sha != ref["field_sha256"]:
                raise ArithmeticError("Sampling altered the 64-step trajectory")
            graph_control = {}
            for use_graph in (True, False):
                s.restore_state(state=initial)
                replay = s.run_device_loop(duration=duration, use_cuda_graph=use_graph)
                if (
                    hashlib.sha256(s.get_conservative_state().tobytes()).hexdigest()
                    != sha
                ):
                    raise ArithmeticError("Graph replay changed the final solution")
                graph_control["captured" if use_graph else "uncaptured"] = dict(
                    hydraulic_loop_s=replay["loop_seconds"],
                    steps=replay["steps"],
                    rejected=replay.get("rejected_cfl_trials", 0),
                    capture_s=replay["graph_capture_seconds"],
                )
            rows.append(
                dict(
                    case=case,
                    method=method,
                    mode=mode,
                    initial_wet_fraction=float(
                        (inputs["initial"][..., 0] > 0.0).mean()
                    ),
                    graph_control=graph_control,
                    samples=series,
                )
            )
    (review / "strip_activity.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    shape = (1024.0, 640.0)
    cases = ("stoker", "dry_patch")
    methods = (
        ("contact_hll", "sparse"),
        ("hancock_adaptive", "dense"),
        ("hancock_adaptive", "sparse"),
        ("hancock_adaptive", "dynamic"),
        ("hancock_adaptive", "tiles"),
    )
    sample(
        review=review,
        shape=shape,
        cases=cases,
        methods=methods,
        dx=1.0,
        duration=2.0,
        cap=1 / 32,
        cadence=8,
        device="cuda:0",
    )
