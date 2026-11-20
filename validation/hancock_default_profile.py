"""Matched production-scheduler timings on larger synthetic wet and mostly dry grids."""
from pathlib import Path

from validation.fv1_riemann import report_json
from validation.fv2_regime_profile import record


def measure(*, output, cases, methods, length, width, spacing, cap, duration, repeats, device):
    rows = []
    for case in cases:
        for method, execution in methods:
            rows.append(record(method=method, case=case, mode=execution, device=device,
                               dx=spacing, length=length, width=width, seconds=duration,
                               cap=cap, repeats=repeats))
    report_json(path=output, content=rows)


if __name__ == "__main__":
    output = Path("validation/hancock_default_review/large_grid_timing.json")
    cases = ("stoker", "dry_patch")
    methods = (("contact_hll", "sparse"), ("hancock_adaptive", "dense"))
    measure(output=output, cases=cases, methods=methods, length=1024., width=640.,
            spacing=1., cap=1 / 32, duration=2., repeats=7, device="cuda:0")
