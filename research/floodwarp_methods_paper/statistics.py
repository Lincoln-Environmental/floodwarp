"""Secondary analyses preserving the grouping and physical support of observations."""

from collections import defaultdict
import json
import math
import sqlite3

import numpy as np

from research.floodwarp_methods_paper.metrics import csv_read, csv_write, load, score


def temporal_orders(*, study):
    (rec, row, directory) = next(
        item
        for item in load(study=study, stages=("convergence",))
        if item[0]["spec"]["kind"] == "temporal"
    )
    groups = defaultdict(list)
    for method, records in row["manufactured"].items():
        for r in records:
            design = r["study"]
            x = 8 / r["n"] if design == "space" else r["dt"]
            for name, value in zip(
                ("depth", "momentum_x", "momentum_y"), r["rmse"], strict=True
            ):
                groups["manufactured", design, method, name].append((x, value))
    fixed = json.loads((directory / "fixed_grid.json").read_text())
    for r in fixed["rows"]:
        for name in ("momentum", "transport"):
            groups["accelerating", "time", r["method"], name].append(
                (r["dt"], r[name + "_error"])
            )
    out = []
    for (case, design, method, metric), records in sorted(groups.items()):
        prev = None
        for x, e in sorted(records, reverse=True):
            order = (
                math.log(prev[1] / e) / math.log(prev[0] / x)
                if prev and prev[0] > x and min(prev[1], e) > 0
                else None
            )
            out.append(
                dict(
                    case=case,
                    design=design,
                    method=method,
                    metric=metric,
                    spacing=x,
                    error=e,
                    order=order,
                    experiment_id=rec["experiment_id"],
                )
            )
            prev = (x, e)
    csv_write(file=study / "processed/convergence/temporal.csv", rows=out)


def interval(*, values, settings):
    if not values:
        return None, None, None
    v = np.asarray([np.mean(g) for g in values], dtype=float)
    samples = np.random.default_rng(settings["bootstrap_seed"]).choice(
        v, size=(settings["bootstrap_repeats"], len(v)), replace=True
    )
    lo, hi = np.quantile(np.mean(samples, axis=1), [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def transfer_groups(*, study, settings):
    rows = csv_read(study / "processed/comparisons/transfer.csv")
    storm = csv_read(study / "processed/metrics/storms.csv")
    baselines = {
        (r["case"], float(r["dx"]), r["method"]): r
        for r in storm
        if r["stage"] in ("resolution", "long-storms")
        and r["representation"] == "sampled"
        and r["reference_self"] == "False"
    }
    grouped = defaultdict(dict)
    for r in rows:
        for method in ("hancock_adaptive", "adaptive"):
            base = baselines.get((r["case"], float(r["dx"]), method))
            if base:
                vals = {
                    k: float(base[k])
                    for k in ("depth_rmse_m", "outlet_rmse_m3s", "inundation_error")
                }
                r[method + "_true_n_score"] = score(row=vals, objective=r["objective"])
                r[method + "_true_n_id"] = base["experiment_id"]
        r["qualified_for_objective"] = (
            r["depth_qualified"] == "True"
            if r["objective"] == "space"
            else r["discharge_qualified"] == "True"
        )
        ref = next(
            s
            for s in storm
            if s["experiment_id"] == r["forecast_id"]
            and s["representation"] == "sampled"
        )
        if r["objective"] != "discharge":
            r["qualified_for_objective"] = (
                r["depth_qualified"] == "True"
                and ref["qualified_inundation"] == "True"
                and (r["objective"] == "space" or r["discharge_qualified"] == "True")
            )
        if r["qualified_for_objective"]:
            g = grouped[r["objective"], r["bound"]]
            g.setdefault((r["landform"], r["seed"]), []).append(float(r["improvement"]))
    csv_write(file=study / "processed/comparisons/transfer.csv", rows=rows)
    summaries = []
    for (obj, bound), g in grouped.items():
        mean, lo, hi = interval(values=list(g.values()), settings=settings)
        summaries.append(
            dict(
                objective=obj,
                bound=bound,
                terrain_groups=len(g),
                mean_improvement=mean,
                low=lo,
                high=hi,
            )
        )
    csv_write(
        file=study / "processed/comparisons/calibration_intervals.csv", rows=summaries
    )


def advisor_errors(*, study, settings):
    rows = csv_read(study / "processed/advisor/heldout.csv")
    groups = defaultdict(dict)
    for r in rows:
        if r["observed_error"] == "" or r["predicted_error"] == "":
            continue
        g = groups[r["objective"], r["rule"]]
        g.setdefault(r["terrain_id"], []).append(
            abs(float(r["observed_error"]) - float(r["predicted_error"]))
        )
    out = []
    for (objective, rule), g in sorted(groups.items()):
        mean, lo, hi = interval(values=list(g.values()), settings=settings)
        out.append(
            dict(
                objective=objective,
                rule=rule,
                terrain_groups=len(g),
                mean_absolute_prediction_error=mean,
                low=lo,
                high=hi,
            )
        )
    csv_write(file=study / "processed/advisor/prediction_error.csv", rows=out)


def operating(*, study):
    rows = load(study=study, stages=("performance",))
    groups = defaultdict(list)
    common = []
    for rec, r, folder in rows:
        if "samples" in r:
            groups[rec["spec"]["method"]].append((rec, r))
        else:
            for item in json.loads((folder / "common_cap.json").read_text()):
                ts = [w["full_hydraulic_s"] for w in item["times"]]
                common.append(
                    dict(
                        method=item["method"],
                        mode=item["mode"],
                        case=item["case"],
                        cells=int(np.prod(item["shape"])),
                        median_s=np.median(ts),
                        min_s=min(ts),
                        max_s=max(ts),
                        owned_mib=item["owned_device_bytes"] / 2**20,
                        pool_peak_mib=item["process_pool_peak_bytes"] / 2**20,
                        experiment_id=rec["experiment_id"],
                    )
                )
    measured = []
    for m, g in groups.items():
        t = [r["hydraulic_s"] for _, r in g]
        measured.append(
            dict(
                method=m,
                physical_case=g[0][0]["spec"]["case"],
                repeats=len(g),
                median_hydraulic_s=np.median(t),
                minimum_s=min(t),
                maximum_s=max(t),
                steps=g[0][1]["steps"],
                experiment_ids=" ".join(rec["experiment_id"] for rec, _ in g),
            )
        )
    csv_write(file=study / "processed/metrics/common_cap.csv", rows=common)
    csv_write(file=study / "processed/metrics/operational_repeats.csv", rows=measured)


def database(*, study):
    """Expose processed CSVs under deterministic table names alongside provenance."""
    base = study / "processed"
    with sqlite3.connect(base / "metrics/study.sqlite") as db:
        for path in sorted(base.rglob("*.csv")):
            rows = csv_read(path)
            name = "_".join(path.relative_to(base).with_suffix("").parts)
            if not rows:
                continue
            columns = list(rows[0])
            types = []
            for k in columns:
                populated = [r[k] for r in rows if r[k]]
                try:
                    for v in populated:
                        float(v)
                    kind = "REAL" if populated else "TEXT"
                except ValueError:
                    kind = "TEXT"
                types.append('"' + k + '" ' + kind)
            db.execute('DROP TABLE IF EXISTS "' + name + '"')
            db.execute('CREATE TABLE "' + name + '" (' + ",".join(types) + ")")
            values = [tuple(v if v != "" else None for v in r.values()) for r in rows]
            db.executemany(
                'INSERT INTO "'
                + name
                + '" VALUES ('
                + ",".join("?" for _ in columns)
                + ")",
                values,
            )
    return base / "metrics/study.sqlite"
