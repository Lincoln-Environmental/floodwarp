"""Observable-specific numerical-reference checks and independent equation controls."""

import json
from collections import defaultdict
from statistics import median

from research.floodwarp_methods_paper.metrics import OBJECTIVES, csv_read, csv_write


def independent(*, study):
    original = study.parents[1] / "validation/fv1_diffusion_review/local_inertia.json"
    records = []
    for i, r in enumerate(json.loads(original.read_text())):
        for truth, metrics in r["errors"].items():
            if not metrics:
                continue
            records.append(
                dict(
                    archival_row=i,
                    source=str(original.relative_to(study.parents[1])),
                    case=r["case"],
                    method=r["method"],
                    dx=r["dx"],
                    reference_equations=truth,
                    depth_rmse_m=metrics["depth_rmse_m"],
                    momentum_x_rmse_m2s=metrics["momentum_rmse_m2_s"][0],
                    momentum_y_rmse_m2s=metrics["momentum_rmse_m2_s"][1],
                    steps=r["steps"],
                )
            )
    csv_write(
        file=study / "processed/metrics/archival_reduced_equations.csv", rows=records
    )


def assess(*, study):
    processed = study / "processed"
    rows = csv_read(processed / "metrics/storms.csv")
    refs = {
        (r["case"], r["representation"]): r
        for r in csv_read(processed / "metrics/reference_uncertainty.csv")
    }
    candidates = {}
    cross = []
    for r in rows:
        if r["stage"] not in ("long-storms", "resolution"):
            continue
        key = (r["case"], r["representation"])
        ref = refs[key]
        if r["method"] == "hancock_adaptive" and float(r["dx"]) == float(
            ref["fine_dx"]
        ):
            obj = dict(
                case=key[0],
                representation=key[1],
                dx=r["dx"],
                method=r["method"],
                run_id=r["experiment_id"],
                fv2_id=r["reference_id"],
            )
            obj.update({k: float(r[v]) for k, v in OBJECTIVES.items() if k != "memory"})
            cross.append(obj)
        if r["reference_setup"] == "False":
            candidates.setdefault((*key, float(r["dx"])), {})[r["method"]] = r
    pairs = []
    for (case, representation, dx), methods in sorted(candidates.items()):
        ref = refs[case, representation]
        for m in ("fv1", "adaptive", "li"):
            if m not in methods or "hancock_adaptive" not in methods:
                continue
            a, b = methods["hancock_adaptive"], methods[m]
            for objective, metric in OBJECTIVES.items():
                if objective == "memory":
                    continue
                qualified = (
                    a["qualified_" + objective] == "True"
                    and a["reference_self"] == "False"
                    and b["reference_self"] == "False"
                )
                diff = float(a[metric]) - float(b[metric])
                delta = float(ref[objective])
                # Norm difference may be smaller than uncertainty in the reference.
                label = (
                    "unresolved reference"
                    if not qualified
                    else "within refinement difference"
                )
                if qualified and abs(diff) > 2 * delta:
                    label = "Hancock" if diff < 0 else m
                pairs.append(
                    dict(
                        case=case,
                        representation=representation,
                        dx=dx,
                        formulation=m,
                        objective=objective,
                        hancock_error=float(a[metric]),
                        alternative_error=float(b[metric]),
                        difference=diff,
                        reference_change=delta,
                        decision=label,
                        cost_ratio=float(a["runtime_s"]) / float(b["runtime_s"]),
                        hancock_id=a["experiment_id"],
                        alternative_id=b["experiment_id"],
                        reference_id=a["reference_id"],
                    )
                )
    csv_write(file=processed / "metrics/independent_fine_formulation.csv", rows=cross)
    csv_write(file=processed / "comparisons/reference_aware_pairs.csv", rows=pairs)


def integrity_by_family(*, study):
    rows = [
        r
        for r in csv_read(study / "processed/metrics/storms.csv")
        if r["stage"] == "long-storms"
        and r["reference_setup"] == "False"
        and r["representation"] == "sampled"
    ]
    g = defaultdict(dict)
    for r in rows:
        g[r["landform"], r["case"]][r["method"]] = r
    result = []
    for land in sorted({k[0] for k in g}):
        groups = [d for (l, c), d in g.items() if l == land]

        def ratios(metric):
            return [
                float(d["hancock_adaptive"][metric]) / float(d["fv1"][metric])
                for d in groups
            ]

        q = [d for d in groups if d["adaptive"]["qualified_discharge"] == "True"]
        result.append(
            dict(
                landform=land,
                events=len(groups),
                median_hancock_fv1_time_ratio=median(ratios("runtime_s")),
                median_hancock_fv1_steps_ratio=median(ratios("steps")),
                discharge_qualified=len(q),
                hancock_discharge_rmse_m3s=median(
                    float(d["hancock_adaptive"]["outlet_rmse_m3s"]) for d in q
                )
                if q
                else None,
                fv2_discharge_rmse_m3s=median(
                    float(d["adaptive"]["outlet_rmse_m3s"]) for d in q
                )
                if q
                else None,
                maximum_absolute_residual_m3=max(
                    abs(float(r["residual_m3"])) for d in groups for r in d.values()
                ),
            )
        )
    csv_write(file=study / "processed/comparisons/landform_results.csv", rows=result)
