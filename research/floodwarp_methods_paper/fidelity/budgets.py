"""Observed accuracy at actual FV1 budgets and least measured cost at application tolerances."""

from collections import defaultdict
from research.floodwarp_methods_paper import metrics


def feasible_cost(*, rows, target, tolerances, spacings):
    groups = defaultdict(list)
    fields = (
        "cohort",
        "case",
        "representation",
        "reference_id",
        "station",
        "cap",
        "interval_s",
        "objective",
    )
    for row in rows:
        groups[tuple(row.get(k) for k in fields)].append(row)
    decisions = []
    budgets = []
    for key, points in groups.items():
        meta = dict(zip(fields, key, strict=True))
        valid = [r for r in points if r.get("eligible") is True]
        for tol in tolerances:
            feasible = [r for r in valid if r["absolute_error"] <= tol]
            t = min((r["runtime_s"] for r in feasible), default=None)
            winners = [r for r in feasible if r["runtime_s"] == t]
            for r in winners or [None]:
                decisions.append(
                    dict(
                        meta,
                        tolerance_pct=tol,
                        observations=len(points),
                        assessable=len(valid),
                        feasible=len(feasible),
                        method=r["method"] if r else None,
                        dx=r["dx"] if r else None,
                        execution=r["execution_mode"] if r else None,
                        id=r["experiment_id"] if r else None,
                        runtime_s=t,
                        error_pct=r["absolute_error"] if r else None,
                    )
                )
        for fv in points:
            if fv["method"] != "fv1" or fv["dx"] not in spacings:
                continue
            allowed = [r for r in valid if r["runtime_s"] <= fv["runtime_s"]]
            best = min((r["absolute_error"] for r in allowed), default=None)
            for r in [q for q in allowed if q["absolute_error"] == best] or [None]:
                budgets.append(
                    dict(
                        meta,
                        budget_method="fv1",
                        budget_dx=fv["dx"],
                        budget_id=fv["experiment_id"],
                        budget_s=fv["runtime_s"],
                        fv1_error_pct=fv["absolute_error"] if fv["eligible"] else None,
                        assessable_in_budget=len(allowed),
                        selected_method=r["method"] if r else None,
                        selected_dx=r["dx"] if r else None,
                        execution=r["execution_mode"] if r else None,
                        selected_id=r["experiment_id"] if r else None,
                        runtime_s=r["runtime_s"] if r else None,
                        error_pct=best,
                    )
                )
    metrics.csv_write(file=target / "least_cost_fidelity.csv", rows=decisions)
    metrics.csv_write(file=target / "equal_cost_fidelity.csv", rows=budgets)
