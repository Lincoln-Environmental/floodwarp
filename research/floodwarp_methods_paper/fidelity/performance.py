"""Measure cost to the physically checked end and distinguish it from the original forecast window."""

from collections import defaultdict
from statistics import median
import numpy as np
from research.floodwarp_methods_paper import metrics


def cost(row):
    return row["runtime_s"]


def costs(*, directory, rows):
    groups = defaultdict(dict)
    for r in rows:
        if (
            r["station"] == "outlet"
            and r["representation"] == "sampled"
            and r["cohort"] in ("complete-expanded", "complete-legacy")
        ):
            groups[r["cohort"], r["case"], r["dx"]][r["method"]] = r
    data = []
    for (cohort, case, dx), g in sorted(groups.items()):
        if "fv1" not in g:
            continue
        fv = g["fv1"]
        modes = [g[name] for name in ("hancock_dense", "hancock_adaptive") if name in g]
        if not modes:
            continue
        winner = min(modes, key=cost)
        primary = int(fv["cells"]) == 1048576
        label = (
            "small-48"
            if cohort == "complete-legacy"
            else (
                "primary-million"
                if primary
                and fv["seed"] in range(701, 709)
                and not case.endswith(("24h", "48h"))
                else "other-expanded"
            )
        )
        row = dict(
            cohort=cohort,
            suite=label,
            case=case,
            dx=dx,
            seed=fv["seed"],
            landform=fv["landform"],
            storm=fv["storm"],
            cells=fv["cells"],
            hours=fv["hours"],
            fv1_s=cost(fv),
            fv1_id=fv["experiment_id"],
            fastest_explicit=winner["execution_mode"],
            fastest_id=winner["experiment_id"],
            fastest_s=cost(winner),
            fastest_over_fv1=cost(winner) / cost(fv),
        )
        for name in (
            "hancock_dense",
            "hancock_adaptive",
            "hancock_auto",
            "adaptive",
            "li",
        ):
            if name in g:
                row[name + "_id"] = g[name]["experiment_id"]
                row[name + "_s"] = cost(g[name])
                row[name + "_over_fv1"] = cost(g[name]) / cost(fv)
        data.append(row)
    metrics.csv_write(file=directory / "results/completed_costs.csv", rows=data)
    summaries = []
    for suite in ("small-48", "primary-million"):
        use = [r for r in data if r["suite"] == suite]
        for k in (
            "fastest_over_fv1",
            "hancock_dense_over_fv1",
            "hancock_adaptive_over_fv1",
            "hancock_auto_over_fv1",
        ):
            a = [r[k] for r in use if k in r]
            if a:
                summaries.append(
                    dict(
                        suite=suite,
                        policy=k,
                        events=len(a),
                        median=median(a),
                        lower_quartile=np.percentile(a, 25),
                        upper_quartile=np.percentile(a, 75),
                        minimum=min(a),
                        maximum=max(a),
                    )
                )
    metrics.csv_write(
        file=directory / "results/completed_cost_summary.csv", rows=summaries
    )
    return summaries
