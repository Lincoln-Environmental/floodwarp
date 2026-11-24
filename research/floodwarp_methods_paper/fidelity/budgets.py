"""Observed accuracy at actual FV1 budgets and least measured cost at application tolerances."""

from collections import defaultdict
from research.floodwarp_methods_paper import metrics


def conservative_error(*, value, sensitivity):
    return abs(float(value)) + float(sensitivity)


def feasible_cost(*, rows, target, tolerances, spacings, name="fidelity"):
    groups = defaultdict(list)
    fields = (
        "cohort",
        "case",
        "representation",
        "reference_id",
        "station",
        "cap",
        "interval_s",
        "end_h",
        "objective",
    )
    for row in rows:
        item = dict(row)
        if item.get("interval_s") is not None:
            item["interval_s"] = float(item["interval_s"])
        groups[tuple(item.get(k) for k in fields)].append(item)
    decisions = []
    budgets = []
    for key, points in groups.items():
        meta = dict(zip(fields, key, strict=True))
        valid = []
        for r in points:
            if r.get("eligible") is not True:
                continue
            sensitivity = r.get("reference_sensitivity")
            if sensitivity is None and r["reference_id"] in ("analytical", "exact"):
                sensitivity = 0.
            if sensitivity is not None:
                valid.append(dict(r, certified_error=conservative_error(value=r["absolute_error"], sensitivity=sensitivity)))
        limits = tolerances[meta["objective"]] if isinstance(tolerances, dict) else tolerances
        for tol in limits:
            feasible = [r for r in valid if r["certified_error"] <= tol]
            t = min((r["runtime_s"] for r in feasible), default=None)
            winners = [r for r in feasible if r["runtime_s"] == t]
            for r in winners or [None]:
                decisions.append(
                    dict(
                        meta,
                        tolerance_pct=tol if name == "fidelity" else None,
                        observations=len(points),
                        assessable=len(valid),
                        feasible=len(feasible),
                        nominal_feasible=sum(r["absolute_error"] <= tol for r in valid),
                        tolerance=tol,
                        conservative_error=r["certified_error"] if r else None,
                        reference_sensitivity=r.get("reference_sensitivity") if r else None,
                        method=r["method"] if r else None,
                        dx=r["dx"] if r else None,
                        execution=r["execution_mode"] if r else None,
                        id=r["experiment_id"] if r else None,
                        runtime_s=t,
                        error_pct=r["absolute_error"] if r and name == "fidelity" else None,
                        measured_error=r["absolute_error"] if r else None,
                        steps=r.get("steps") if r else None,
                        hydraulic_s_per_step=t/r["steps"] if r and r.get("steps") else None,
                    )
                )
        for fv in points:
            if fv["method"] != "fv1" or fv["dx"] not in spacings:
                continue
            allowed = [r for r in valid if r["runtime_s"] <= fv["runtime_s"]]
            best = min((r["certified_error"] for r in allowed), default=None)
            for r in [q for q in allowed if q["certified_error"] == best] or [None]:
                budgets.append(
                    dict(
                        meta,
                        budget_method="fv1",
                        budget_dx=fv["dx"],
                        budget_id=fv["experiment_id"],
                        budget_s=fv["runtime_s"],
                        fv1_error_pct=fv["absolute_error"] if fv["eligible"] and name == "fidelity" else None,
                        assessable_in_budget=len(allowed),
                        selected_method=r["method"] if r else None,
                        selected_dx=r["dx"] if r else None,
                        execution=r["execution_mode"] if r else None,
                        selected_id=r["experiment_id"] if r else None,
                        runtime_s=r["runtime_s"] if r else None,
                        error_pct=r["absolute_error"] if r and name == "fidelity" else None,
                        measured_error=r["absolute_error"] if r else None,
                        conservative_error=best,
                        reference_sensitivity=r.get("reference_sensitivity") if r else None,
                    )
                )
    metrics.csv_write(file=target / f"least_cost_{name}.csv", rows=decisions)
    metrics.csv_write(file=target / f"equal_cost_{name}.csv", rows=budgets)
