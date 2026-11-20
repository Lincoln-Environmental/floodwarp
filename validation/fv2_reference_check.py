"""Resolve synthetic reference uncertainty with a separate Roe/HLL flux and nested subcells."""

from pathlib import Path

import json

from validation.fv2_figures import rows_table

from validation.fv1_riemann import report_json
from validation.fv2_terrain import compare, refined, run_domain, terrain


def reference_check(
    *, cases, spacings, base_dx, fine_grids, length, width, device, samples, wet, output, candidates=("fv1", "mc")
):
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
                    domain=d,
                    method=method,
                    cap=dx / ratio,
                    device=device,
                    samples=samples,
                    wet=wet,
                )

            for dx in fine_grids:
                for m in ("roe", "mc"):
                    refs[dx, m] = calculate(dx=dx, method=m, ratio=32.0)
            finest = fine_grids[-1]
            ref = refs[finest, "mc"]

            def differences(*, run, truth, spacing):
                return compare(
                    result=run,
                    reference=truth,
                    wet=wet,
                    dx=spacing,
                    duration=base.duration,
                )

            gap = differences(
                run=refs[fine_grids[0], "mc"], truth=ref, spacing=fine_grids[0]
            )
            roe_gap = differences(
                run=refs[fine_grids[0], "roe"],
                truth=refs[finest, "roe"],
                spacing=fine_grids[0],
            )
            spread = differences(run=refs[finest, "roe"], truth=ref, spacing=finest)
            small = calculate(dx=finest, method="mc", ratio=64.0)
            time_error = differences(run=ref, truth=small, spacing=finest)
            tests = []
            for dx in (base_dx,) if fixed else spacings:
                for m in candidates:
                    result = calculate(dx=dx, method=m, ratio=32.0)
                    tests.append(
                        dict(
                            dx=dx,
                            method=m,
                            against_mc=differences(run=result, truth=ref, spacing=dx),
                            against_roe_hll=differences(
                                run=result, truth=refs[finest, "roe"], spacing=dx
                            ),
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


def summarize_references(*, source, output, dx):
    groups = json.loads(source.read_text())
    rows = []
    for group in groups:
        candidates = {r["method"]: r for r in group["candidates"] if r["dx"] == dx}
        values = [
            group[k]["depth_rmse_m"] * 1000
            for k in (
                "mc_refinement_gap",
                "roe_hll_refinement_gap",
                "cross_formulation_spread",
                "finest_mc_dt_halving",
            )
        ]
        for reference in ("against_mc", "against_roe_hll"):
            values.extend(
                candidates[m][reference]["depth_rmse_m"] * 1000 for m in ("fv1", "mc")
            )
        rows.append(
            [
                group["case"],
                "Fixed 5 m" if group["terrain_fixed_5m"] else "Resampled",
                *values,
            ]
        )
    table = rows_table(
        title="Fine numerical references for the 5 m synthetic models",
        columns=[
            "Case",
            "Bed",
            "MC grid gap",
            "Roe/HLL grid gap",
            "Fine inter-method gap",
            "MC half-dt change",
            "FV1 vs MC",
            "FV2 vs MC",
            "FV1 vs Roe/HLL",
            "FV2 vs Roe/HLL",
        ],
        rows=rows,
        note="All entries are final depth RMSE in millimetres. Grid gaps compare 0.625 and 0.3125 m results on the 0.625 m grid. The inter-method and timestep comparisons use 0.3125 m support. The last four columns conservatively average the fine reference to the 5 m candidate grid. Resampled terrain evaluates the same mathematical surface at cell centres. Fixed terrain preserves every 5 m cell's input, including rain and the initial condition, under subdivision. The gaps measure sensitivity and have no formal a posteriori error-bound interpretation.",
    )
    output.write_text(table)


if __name__ == "__main__":
    output = Path("validation/fv2_review/synthetic_reference.json")
    cases = (
        "plane",
        "irregular",
        "steep",
        "expansion",
        "rainfall",
        "barrier",
        "oblique_channel",
    )
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
    summarize_references(
        source=output, output=output.with_name("SYNTHETIC_REFERENCES.md"), dx=5.0
    )
