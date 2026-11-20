"""Resolve synthetic reference uncertainty with a separate Roe/HLL flux and nested subcells."""

from pathlib import Path

from validation.fv1_riemann import report_json
from validation.fv2_terrain import compare, refined, run_domain, terrain


def reference_check(*, cases, spacings, base_dx, fine_grids, length, width, device, samples, wet, output):
    rows = []
    for case in cases:
        for fixed in (False, True):
            base = terrain(name=case, dx=base_dx, length=length, width=width)
            refs = {}

            def calculate(*, dx, method, ratio):
                d = (
                    refined(domain=base, factor=round(base_dx / dx))
                    if fixed
                    else terrain(name=case, dx=dx, length=length, width=width)
                )
                return run_domain(
                    domain=d, method=method, cap=dx / ratio, device=device, samples=samples, wet=wet
                )

            for dx in fine_grids:
                for m in ("roe", "mc"):
                    refs[dx, m] = calculate(dx=dx, method=m, ratio=32.0)
            finest = fine_grids[-1]
            ref = refs[finest, "mc"]

            def differences(*, run, truth, spacing):
                return compare(result=run, reference=truth, wet=wet, dx=spacing, duration=base.duration)

            gap = differences(run=refs[fine_grids[0], "mc"], truth=ref, spacing=fine_grids[0])
            roe_gap = differences(
                run=refs[fine_grids[0], "roe"], truth=refs[finest, "roe"], spacing=fine_grids[0]
            )
            spread = differences(run=refs[finest, "roe"], truth=ref, spacing=finest)
            small = calculate(dx=finest, method="mc", ratio=64.0)
            time_error = differences(run=ref, truth=small, spacing=finest)
            tests = []
            for dx in (base_dx,) if fixed else spacings:
                for m in ("fv1", "mc"):
                    result = calculate(dx=dx, method=m, ratio=32.0)
                    tests.append(
                        dict(
                            dx=dx,
                            method=m,
                            against_mc=differences(run=result, truth=ref, spacing=dx),
                            against_roe_hll=differences(run=result, truth=refs[finest, "roe"], spacing=dx),
                        )
                    )
            rows.append(
                dict(
                    case=case,
                    terrain_fixed_5m=fixed,
                    reference_grids_m=fine_grids,
                    mc_refinement_gap=gap,
                    roe_hll_refinement_gap=roe_gap,
                    cross_formulation_spread=spread,
                    finest_mc_dt_halving=time_error,
                    candidates=tests,
                )
            )
            report_json(path=output, content=rows)
            print(f"{case}, fixed={fixed}, reference comparison complete", flush=True)


if __name__ == "__main__":
    output = Path("validation/fv2_review/synthetic_reference.json")
    cases = ("plane", "irregular", "steep", "expansion", "rainfall", "barrier", "oblique_channel")
    spacings = (20.0, 10.0, 5.0, 2.5)
    reference_check(
        cases=cases,
        spacings=spacings,
        base_dx=5.0,
        fine_grids=(0.625, 0.3125),
        length=320.0,
        width=160.0,
        device="cuda:0",
        samples=4,
        wet=0.01,
        output=output,
    )
