"""Equal-budget and threshold comparisons among measured configurations, plus analytical audits."""

from collections import defaultdict
import json


from research.floodwarp_methods_paper.metrics import (
    OBJECTIVES,
    csv_read,
    csv_write,
    load,
)


def time_key(row):
    return float(row["runtime_s"])


def pairwise(*, study, config):
    rows = csv_read(study / "processed/metrics/storms.csv")
    group = defaultdict(list)
    for r in rows:
        if (
            r["stage"] in ("long-storms", "resolution")
            and r["reference_setup"] == "False"
        ):
            group[r["case"], r["representation"]].append(r)
    operational_path = study / "processed/million/observations.csv"
    if operational_path.exists():
        for r in csv_read(operational_path):
            if r["stage"] in ("million-storms", "million-auto", "recession-controls"):
                group[r["case"], r["representation"]].append(r)
    pairs = []
    feasible = []
    for (case, rep), records in group.items():
        cohort = (
            "operational"
            if records[0]["stage"].startswith("million-")
            else "120 × 80 m controls"
        )
        for objective, metric in OBJECTIVES.items():
            if objective == "memory":
                continue
            eligible = [
                r
                for r in records
                if r["qualified_" + objective] == "True"
                and r["reference_self"] == "False"
            ]
            met = [
                r
                for r in eligible
                if float(r[metric]) <= config["advisor"]["tolerances"][objective]
            ]
            won = min(met, key=time_key) if met else None
            feasible.append(
                dict(
                    cohort=cohort,
                    case=case,
                    representation=rep,
                    objective=objective,
                    tolerance=config["advisor"]["tolerances"][objective],
                    qualified_options=len(eligible),
                    feasible=len(met),
                    method=won["method"] if won else None,
                    dx=won["dx"] if won else None,
                    runtime_s=won["runtime_s"] if won else None,
                    run_id=won["experiment_id"] if won else None,
                )
            )
            combos = [("fv1", d, "hancock_adaptive", d) for d in (1, 5, 10)] + [
                ("fv1", 5, "hancock_adaptive", 10),
                ("fv1", 1, "hancock_adaptive", 5),
                ("li", 10, "hancock_adaptive", 10),
            ]
            combos += [
                (a, ad, mode, bd)
                for a, ad, _, bd in combos
                for mode in (
                    "hancock_dense",
                    "hancock_auto",
                    "fastest qualified Hancock",
                )
            ]
            for am, ad, bm, bd in combos:
                a = next(
                    (r for r in records if r["method"] == am and float(r["dx"]) == ad),
                    None,
                )
                b = next(
                    (r for r in records if r["method"] == bm and float(r["dx"]) == bd),
                    None,
                )
                if bm == "fastest qualified Hancock":
                    options = [
                        r
                        for r in eligible
                        if float(r["dx"]) == bd
                        and r["method"]
                        in ("hancock_dense", "hancock_adaptive", "hancock_auto")
                    ]
                    b = min(options, key=time_key) if options else None
                if not (a and b):
                    continue
                pairs.append(
                    dict(
                        cohort=cohort,
                        case=case,
                        representation=rep,
                        objective=objective,
                        method_a=am,
                        dx_a=ad,
                        method_b=bm,
                        execution_b=b.get("mode", b.get("execution_mode")),
                        dx_b=bd,
                        error_a=a.get(metric),
                        error_b=b.get(metric),
                        time_a_s=a["runtime_s"],
                        time_b_s=b["runtime_s"],
                        cost_ratio=float(b["runtime_s"]) / float(a["runtime_s"]),
                        qualified=a in eligible and b in eligible,
                        run_a=a["experiment_id"],
                        run_b=b["experiment_id"],
                    )
                )
            for a in eligible:
                if a["method"] != "fv1":
                    continue
                affordable = [
                    r
                    for r in eligible
                    if r["method"] == "adaptive"
                    and float(r["runtime_s"]) <= float(a["runtime_s"])
                ]
                b = min(affordable, key=metric_value(metric)) if affordable else None
                pairs.append(
                    dict(
                        cohort=cohort,
                        case=case,
                        representation=rep,
                        objective=objective,
                        method_a="fv1",
                        dx_a=a["dx"],
                        method_b="FV2 within FV1 budget",
                        dx_b=b["dx"] if b else None,
                        error_a=a.get(metric),
                        error_b=b[metric] if b else None,
                        time_a_s=a["runtime_s"],
                        time_b_s=b["runtime_s"] if b else None,
                        cost_ratio=float(b["runtime_s"]) / float(a["runtime_s"])
                        if b
                        else None,
                        qualified=b is not None,
                        run_a=a["experiment_id"],
                        run_b=b["experiment_id"] if b else None,
                    )
                )
    csv_write(file=study / "processed/comparisons/equal_cost.csv", rows=pairs)
    csv_write(file=study / "processed/pareto/least_cost.csv", rows=feasible)


def metric_value(key):
    def get(row):
        return float(row[key])

    return get


def exact_check(*, study):
    complete = load(study=study, stages=("analytical", "convergence"))
    primary = [
        r
        for meta, r, _ in complete
        if meta["stage"] == "analytical" and "depth_rmse" in r
    ]
    if len(primary) != 342:
        raise ValueError("Expected 342 core analytical runs")
    minimum = min(r["min_depth_m"] for r in primary)
    maxres = max(
        abs(r["residual_m3"]) / max(1, r["initial_m3"] + r["ledger"]["rain"])
        for r in primary
    )
    if minimum < 0 or maxres > 1e-8:
        raise ArithmeticError("Analytical positivity or mass balance failed")
    control = next(
        p for r, _, p in complete if r["spec"].get("kind") == "archived-controls"
    )
    lake = [
        r
        for r in json.loads((control / "analytical.json").read_text())
        if r["case"] == "lake"
    ]
    if max(r["depth_max_error_m"] for r in lake) > 1e-11:
        raise ArithmeticError("Lake baseline failed")
    return dict(
        primary_runs=len(primary),
        min_depth_m=minimum,
        maximum_relative_residual=maxres,
        lake_maximum_depth_error_m=max(r["depth_max_error_m"] for r in lake),
    )


def historical(*, root, study):
    old = root / "validation/fv1_diffusion_review/calibration"
    data = json.loads((old / "transfers.json").read_text())
    records = []
    for r in data:
        target = r["target"]
        records.append(
            dict(
                training=r["source"],
                heldout=r["test"],
                training_dx=r["source_dx"],
                test_dx=r["dx"],
                objective=target,
                bound=r["bound"],
                multiplier=r["multiplier"],
                improvement=r["baseline"][target] - r["scores"][target],
                source="validation/fv1_diffusion_review/calibration/transfers.json",
            )
        )
    csv_write(
        file=study / "processed/comparisons/archival_analytical_transfer.csv",
        rows=records,
    )
