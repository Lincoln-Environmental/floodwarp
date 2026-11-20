"""Controlled warm FP64 timing and state fingerprints for each optimisation revision."""

import hashlib
import os
from pathlib import Path
import numpy as np
import warp as wp
from validation.fv1_riemann import report_json
from validation.fv2_local_inertia import build, domain
from validation.fv2_cost import reset, timed
from validation.fv2_storm_cost import profile


def record(*, method, case, mode, device, dx, length, width, seconds, cap, repeats):
    d = domain(
        case="stoker", dx=dx, duration=seconds, gravity=9.80665, near_zero_n=1e-7, length=length, width=width
    )
    if case == "dry_patch":
        x = d["x"]
        d["initial"][..., 0] = np.where(abs(x) < length / 32, 0.45, 0.0)
    s = build(d=d, method=method, dx=dx, cap=cap, device=device, gravity=9.80665)
    s.configure_execution(mode=mode)
    initial = s.capture_state()
    warm = timed(solver=s, duration=seconds, samples=1)
    expected = s.get_conservative_state()
    times = []
    for _ in range(repeats):
        reset(solver=s, state=initial)
        times.append(timed(solver=s, duration=seconds, samples=1))
        np.testing.assert_array_equal(s.get_conservative_state(), expected)
    kernels = [profile(solver=s, state=initial, duration=min(seconds, cap * 8)) for _ in range(3)]
    arrays = [*s._fields.values(), s.ctrl, s.condition, s._rain_rates, s._rain_times]
    if hasattr(s, "_slopes"):
        arrays.extend(
            [s._start, s._stage, s._end, s._amounts, *s._slopes, *s._flux[0], *s._flux[1], *s._samples]
        )
        for series in s._stage_series.values():
            arrays.extend(series)
        if hasattr(s, "_shock_flags"):
            arrays.append(s._shock_flags)
        if hasattr(s, "_inverse"):
            arrays.append(s._inverse)
    owned = sum({a.ptr: a.capacity for a in arrays}.values()) + s.execution_diagnostics()["scheduling_bytes"]
    return dict(
        owned_device_bytes=owned,
        method=method,
        case=case,
        mode=mode,
        dx=dx,
        shape=list(expected.shape[:2]),
        duration=seconds,
        warm=warm,
        times=times,
        median_s=float(np.median([r["full_hydraulic_s"] for r in times])),
        kernels=kernels,
        pool_bytes=wp.get_mempool_used_mem_current(device=device),
        process_pool_peak_bytes=wp.get_mempool_used_mem_high(device=device),
        field_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
    )


def measure(*, output, methods, cases, device, repeats):
    rows = []
    for case, length, width, dx, seconds, cap, mode in cases:
        for method in methods:
            r = record(
                method=method,
                case=case,
                mode=mode,
                device=device,
                dx=dx,
                length=length,
                width=width,
                seconds=seconds,
                cap=cap,
                repeats=repeats,
            )
            rows.append(r)
            print(case, mode, method, r["median_s"], flush=True)
    report_json(path=output, content=rows)


if __name__ == "__main__":
    tag = os.environ.get("FV2_STUDY_REVISION", "baseline")
    methods = os.environ.get("FV2_STUDY_METHODS", "contact_hll,mc").split(",")
    cases = (
        ("stoker", 960, 640, 5, 24, 5 / 64, "dense"),
        ("wet_large", 1920, 1280, 5, 8, 5 / 64, "dense"),
        ("dry_patch", 1920, 1280, 5, 8, 5 / 64, "sparse"),
    )
    measure(
        output=Path("validation/fv2_review/regimes") / ("profile_" + tag + ".json"),
        methods=methods,
        cases=cases,
        device="cuda:0",
        repeats=7,
    )
