"""Isolated profiling of a single-flux update, fusion and scheduling."""

from pathlib import Path
from validation.fv2_regime_profile import record
from validation.fv1_riemann import report_json


def profile(*, output, setups, methods, repeats, device, dx, length, width, duration, cap):
    rows = []
    for case, mode in setups:
        for method in methods:
            fingerprints = []
            for fuse in (False, True) if method == "hancock_mc" else (None,):
                r = record(
                    method=method,
                    case=case,
                    mode=mode,
                    device=device,
                    dx=dx,
                    length=length,
                    width=width,
                    seconds=duration,
                    cap=cap,
                    repeats=repeats,
                    fuse_prediction=fuse,
                )
                fingerprints.append(r["field_sha256"])
                rows.append(dict(fused=fuse, **r))
                print(case, mode, method, fuse, r["median_s"], flush=True)
            if len(set(fingerprints)) != 1:
                raise ArithmeticError("Fusion changed the accepted state")
        report_json(path=output, content=rows)


if __name__ == "__main__":
    output = Path("validation/fv1_diffusion_review/profile.json")
    setups = (("stoker", "dense"), ("dry_patch", "dense"), ("dry_patch", "sparse"))
    methods = (
        "contact_hll",
        "low_froude_hll",
        "euler_mc",
        "hancock_mc",
        "hancock_low_froude",
        "hancock_adaptive",
        "mc",
    )
    profile(
        output=output,
        setups=setups,
        methods=methods,
        repeats=7,
        device="cuda:0",
        dx=1.0,
        length=400.0,
        width=240.0,
        duration=4.0,
        cap=1 / 32,
    )
