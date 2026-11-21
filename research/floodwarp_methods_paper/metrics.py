"""Objective-specific errors, resolution uncertainty, measured Pareto sets and bounded transfer."""

import csv
from collections import defaultdict
import json
import sqlite3

import numpy as np

from research.floodwarp_methods_paper.runner import archived
from research.floodwarp_methods_paper import evidence
from validation.hancock_long_storm_review.accuracy import compare, read_fields


OBJECTIVES = {
    "depth": "depth_rmse_m",
    "maximum_depth": "max_depth_rmse_m",
    "inundation": "inundation_error",
    "discharge": "outlet_rmse_m3s",
    "peak_discharge": "absolute_peak_error_m3s",
    "peak_timing": "absolute_peak_time_error_s",
    "memory": "memory_mib",
}


def csv_write(*, file, rows):
    file.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_read(file):
    with file.open() as handle:
        return list(csv.DictReader(handle))


def load(*, study, stages):
    runs = {}
    for rec in evidence.records(study=study, stages=stages):
        key = (rec["stage"], rec["experiment_id"])
        path = study / rec["directory"]
        if key not in runs or path.stat().st_mtime > runs[key][2].stat().st_mtime:
            runs[key] = (rec, json.loads((path / "result.json").read_text()), path)
    return list(runs.values())


def pair_metrics(*, item, ref, fields, thresholds):
    meta, row, _ = item
    base, truth, _ = ref
    m = compare(
        row=row,
        ref=truth,
        data=fields[meta["attempt"]],
        reference=fields[base["attempt"]],
        support=10.0,
        thresholds=thresholds,
    )
    m.update(
        inundation_error=1 - m["space_time_iou_1cm"],
        absolute_peak_error_m3s=abs(m["peak_error_m3s"]),
        absolute_peak_time_error_s=abs(m["peak_time_error_s"]),
    )
    return m


def storm_metrics(*, study, config):
    raw = load(
        study=study, stages=("long-storms", "resolution", "calibration", "performance")
    )
    groups = defaultdict(list)
    for item in raw:
        rec, row, _ = item
        if "samples" in row:
            groups[rec["spec"]["case"]].append(item)
    results = []
    uncertainties = []
    thresholds = config["storms"]["wet_thresholds_m"]
    for case, items in sorted(groups.items()):
        fields = {
            rec["attempt"]: read_fields(row=row, directory=folder, support=10.0)
            for rec, row, folder in items
        }
        for representation in ("sampled", "fixed"):
            records = [
                x
                for x in items
                if (
                    x[0]["spec"]["dx"] == x[0]["spec"]["dem"]
                    if representation == "sampled"
                    else x[0]["spec"]["dem"] == 10
                )
            ]
            refs = [
                x
                for x in records
                if x[0]["stage"] in ("long-storms", "resolution")
                and x[0]["spec"]["method"] == "adaptive"
            ]
            refs.sort(key=spacing)
            if len(refs) < 2:
                continue
            fine = refs[0]
            coarse = next((r for r in refs if spacing(r) > spacing(fine)), None)
            if coarse is None:
                continue
            delta = pair_metrics(
                item=coarse, ref=fine, fields=fields, thresholds=thresholds
            )
            qualified = dict(
                depth=delta["depth_rmse_m"] <= 0.001
                or (
                    delta["depth_relative_l2"] is not None
                    and delta["depth_relative_l2"] <= 0.02
                ),
                maximum_depth=delta["max_depth_rmse_m"] <= 0.01,
                inundation=delta["inundation_error"] <= 0.05,
                discharge=delta["outlet_rmse_m3s"]
                <= 0.05 * max(abs(s["outlet_m3s"]) for s in fine[1]["samples"]),
                peak_discharge=delta["absolute_peak_error_m3s"]
                <= 0.05
                * abs(
                    fine[1]["samples"][
                        np.argmax([s["outlet_m3s"] for s in fine[1]["samples"]])
                    ]["outlet_m3s"]
                ),
                peak_timing=delta["absolute_peak_time_error_s"] <= 900,
                memory=True,
            )
            uncertainty = dict(
                case=case,
                representation=representation,
                coarse_id=coarse[0]["experiment_id"],
                fine_id=fine[0]["experiment_id"],
                fine_dx=spacing(fine),
                coarse_dx=spacing(coarse),
            )
            uncertainty.update(
                {k: delta[v] for k, v in OBJECTIVES.items() if k != "memory"}
            )
            uncertainty.update({"qualified_" + k: v for k, v in qualified.items()})
            uncertainties.append(uncertainty)
            for item in records:
                rec, row, path = item
                s = rec["spec"]
                metrics = pair_metrics(
                    item=item, ref=fine, fields=fields, thresholds=thresholds
                )
                metrics.update(
                    case=case,
                    experiment_id=rec["experiment_id"],
                    attempt=rec["attempt"],
                    reference_id=fine[0]["experiment_id"],
                    source_group="prospective",
                    stage=rec["stage"],
                    method=s["method"],
                    mode=row["mode"],
                    seed=s["seed"],
                    initial=s["initial"],
                    hours=s["hours"],
                    split=s["split"],
                    representation=representation,
                    reference_self=rec["attempt"] == fine[0]["attempt"],
                    reference_setup=s.get("reference", False),
                    multiplier=s.get("multiplier", 1.0),
                    role=s.get("role", "forecast"),
                    steps=row["steps"],
                    rejected=sum(x["rejected"] for x in row["samples"]),
                    memory_mib=row["process_pool_mib"],
                    wet_fraction=np.mean([x["wet_fraction"] for x in row["samples"]]),
                    active_fraction=np.mean(
                        [x["active_fraction"] for x in row["samples"]]
                    ),
                    residual_m3=row["residual_m3"],
                    rain_m3=row["rain_m3"],
                    infiltration_m3=row["infiltration_m3"],
                    final_m3=row["final_m3"],
                    outflow_m3=row["outflow_m3"],
                    mean_dt_s=s["hours"] * 3600 / row["steps"],
                    minimum_dt_s=min(x["dt_min_s"] for x in row["samples"]),
                    graph_capture_s=sum(x["capture_s"] for x in row["samples"]),
                    output_s=row["extraction_s"] + row["field_write_s"],
                    end_to_end_s=row["end_to_end_s"],
                    control_sha=rec["context"]["source_sha256"],
                )
                metrics.update({"qualified_" + k: v for k, v in qualified.items()})
                results.append(metrics)
    return results, uncertainties


def spacing(item):
    return float(item[0]["spec"]["dx"])


def exact_metrics(*, study):
    rows = []
    for rec, r, _ in load(study=study, stages=("analytical", "convergence")):
        if "depth_rmse" not in r:
            continue
        s = rec["spec"]
        rows.append(
            dict(
                experiment_id=rec["experiment_id"],
                case=s["case"],
                stage=rec["stage"],
                method=s["method"],
                dx=s["dx"],
                policy=s["policy"],
                depth_rmse_m=r["depth_rmse"],
                maximum_depth_rmse_m=r["maximum_depth_rmse"],
                hydrograph_rmse_m2s=r["hydro_rmse"],
                momentum_rmse_m2s=(
                    r["momentum_x_rmse"] ** 2 + r["momentum_y_rmse"] ** 2
                )
                ** 0.5,
                inundation_error=1 - r["extent"]["0.01"]["iou"],
                peak_error_m2s=r["peak_error_rmse"],
                peak_timing_error_s=r["timing_error_rmse"],
                runtime_s=r["median_wall_s"],
                timing_min_s=min(r["warm_wall_s"]),
                timing_max_s=max(r["warm_wall_s"]),
                repeats=len(r["warm_wall_s"]),
                steps=r["steps"],
                rejected=r["rejects"],
                memory_mib=r["pool_mib"],
                residual_m3=r["residual_m3"],
                min_depth_m=r["min_depth_m"],
                true_n=r["model_n"],
            )
        )
    return rows


def orders(*, rows):
    groups = defaultdict(list)
    for r in rows:
        if r["stage"] == "convergence":
            groups[r["case"], r["method"]].append(r)
    result = []
    for (case, method), group in groups.items():
        by_grid = sorted(group, key=grid_key, reverse=True)
        for a, b in zip(by_grid, by_grid[1:]):
            for metric in ("depth_rmse_m", "momentum_rmse_m2s", "hydrograph_rmse_m2s"):
                left, right = a[metric], b[metric]
                order = (
                    np.log(left / right) / np.log(a["dx"] / b["dx"])
                    if min(left, right) > 0
                    else None
                )
                result.append(
                    dict(
                        case=case,
                        method=method,
                        metric=metric,
                        coarse=a["dx"],
                        fine=b["dx"],
                        order=order,
                        coarse_id=a["experiment_id"],
                        fine_id=b["experiment_id"],
                    )
                )
    return result


def grid_key(row):
    return row["dx"]


def config_fields(*, method, dx):
    policies = {
        "fv1": ("FV1", "contact HLL", "sparse"),
        "low_froude_hll": ("FV1", "low-Froude HLL", "sparse"),
        "euler_mc": ("MUSCL–Euler", "MC", "dense"),
        "hancock_adaptive": ("Hancock", "adaptive", "tiles"),
        "hancock_dense": ("Hancock", "adaptive", "dense"),
        "adaptive": ("FV2 RK2", "adaptive", "dense"),
        "li": ("LI", "native", "native"),
    }
    formulation, limiter, mode = policies[method]
    return dict(formulation=formulation, limiter_flux=limiter, execution_mode=mode,
                configuration=f"{formulation} / {limiter} / {mode} / {float(dx):g} m")


def is_true(value):
    return value is True or value == "True" or isinstance(value, np.bool_) and bool(value)


def pareto(*, rows, keys, objectives, cost, include_unresolved=False):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for group in groups.values():
        for objective, metric in objectives.items():
            measured = []
            for r in group:
                value = r.get(metric)
                valid = value not in ("", None) and np.isfinite(float(value))
                valid = valid and r.get(cost) not in (None, "") and np.isfinite(float(r[cost])) and float(r[cost]) > 0
                ref_ok = objective == "memory" or (
                    not is_true(r.get("reference_self")) and is_true(r.get("qualified_" + objective, True))
                )
                if valid and ref_ok:
                    measured.append((float(value), float(r[cost]), r["experiment_id"]))
            by_id = {key: (err, t) for err,t,key in measured}
            for r in group:
                pair = by_id.get(r["experiment_id"])
                if pair is None and not include_unresolved:
                    continue
                dominated = None if pair is None else sum(
                    error <= pair[0] and time <= pair[1] and (error < pair[0] or time < pair[1])
                    for error,time,_ in measured
                )
                result.append(dict(**{k:r[k] for k in keys}, objective=objective,method=r["method"],dx=r["dx"],
                                   **config_fields(method=r["method"],dx=r["dx"]),experiment_id=r["experiment_id"],
                                   error=r.get(metric),runtime_s=r.get(cost),eligible=pair is not None,
                                   nondominated=None if pair is None else dominated==0,dominator_count=dominated))
    return result


def fits(*, rows, config):
    candidates = [
        r
        for r in rows
        if r["stage"] == "calibration" and r["representation"] == "sampled"
    ]
    groups = defaultdict(list)
    for r in candidates:
        groups[r["case"], r["dx"]].append(r)
    calibration = []
    transfer = []
    settings = config["calibration"]
    fitted = [r for r in candidates if r["role"] == "fit"]
    if not fitted:
        return calibration, transfer
    for objective in settings["objectives"]:
        for bound in settings["bounds"]:
            subset = [r for r in fitted if abs(r["multiplier"] - 1) <= bound + 1e-12]
            scored = sorted(
                (
                    score(row=r, objective=objective),
                    abs(r["multiplier"] - 1),
                    r["multiplier"],
                    r["experiment_id"],
                )
                for r in subset
            )
            best, _, m, identifier = scored[0]
            base = next(r for r in fitted if r["multiplier"] == 1)
            calibration.append(
                dict(
                    objective=objective,
                    bound=bound,
                    multiplier=m,
                    n=m * config["storms"]["manning_n"],
                    fit_score=best,
                    baseline_score=score(row=base, objective=objective),
                    bounded_search_hydraulic_s=sum(r["runtime_s"] for r in subset),
                    unique_search_hydraulic_s=sum(r["runtime_s"] for r in fitted),
                    candidate_count=len(subset),
                    fit_id=identifier,
                )
            )
            for (case, dx), test in groups.items():
                calibrated = next((r for r in test if r["multiplier"] == m), None)
                native = next((r for r in test if r["multiplier"] == 1), None)
                if (
                    calibrated is None
                    or native is None
                    or calibrated["role"] != "transfer"
                ):
                    continue
                a, b = (
                    score(row=native, objective=objective),
                    score(row=calibrated, objective=objective),
                )
                transfer.append(
                    dict(
                        case=case,
                        landform=native["landform"],
                        seed=native["seed"],
                        dx=dx,
                        objective=objective,
                        bound=bound,
                        multiplier=m,
                        baseline=a,
                        fitted=b,
                        improvement=a - b,
                        fit_id=identifier,
                        forecast_id=calibrated["experiment_id"],
                        baseline_id=native["experiment_id"],
                        forecast_runtime_s=calibrated["runtime_s"],
                        depth_qualified=native["qualified_depth"],
                        discharge_qualified=native["qualified_discharge"],
                    )
                )
    return calibration, transfer


def score(*, row, objective):
    flow = row["outlet_rmse_m3s"] / 0.01
    space = np.sqrt(
        ((row["depth_rmse_m"] / 0.01) ** 2 + (row["inundation_error"] / 0.10) ** 2) / 2
    )
    return float(
        {"space": space, "discharge": flow, "joint": max(space, flow)}[objective]
    )


def bootstrap(*, rows, config):
    out = []
    for method in config["storms"]["methods"]:
        for objective, metric in OBJECTIVES.items():
            groups = defaultdict(list)
            for r in rows:
                if (
                    r["stage"] == "long-storms"
                    and r["dx"] == 10
                    and r["method"] == method
                    and r["representation"] == "sampled"
                    and r["qualified_" + objective]
                ):
                    groups[r["landform"], r["seed"]].append(r[metric])
            if not groups:
                continue
            values = np.array([np.mean(g) for g in groups.values()])
            rng = np.random.default_rng(config["advisor"]["bootstrap_seed"])
            draws = rng.choice(
                values,
                size=(config["advisor"]["bootstrap_repeats"], len(values)),
                replace=True,
            ).mean(axis=1)
            out.append(
                dict(
                    method=method,
                    objective=objective,
                    groups=len(values),
                    estate_mean=values.mean(),
                    low=np.quantile(draws, 0.025),
                    high=np.quantile(draws, 0.975),
                )
            )
    return out


def build(*, study, config):
    storm, uncertainty = storm_metrics(study=study, config=config)
    exact = exact_metrics(study=study)
    operating = [
        r
        for r in storm
        if r["stage"] in ("long-storms", "resolution") and not r["reference_setup"]
    ]
    pc = pareto(
        rows=operating,
        keys=("case", "landform", "representation"),
        objectives=OBJECTIVES,
        cost="runtime_s",
    )
    analytical_front = pareto(
        rows=[r for r in exact if r["stage"] == "analytical"],
        keys=("case",),
        objectives=dict(
            depth="depth_rmse_m",
            maximum_depth="maximum_depth_rmse_m",
            discharge="hydrograph_rmse_m2s",
            inundation="inundation_error",
            peak_discharge="peak_error_m2s",
            peak_timing="peak_timing_error_s",
            memory="memory_mib",
        ),
        cost="runtime_s",
    )
    calibration, transfer = fits(rows=storm, config=config)
    outputs = {
        "metrics/storms": storm,
        "metrics/analytical": exact,
        "metrics/reference_uncertainty": uncertainty,
        "convergence/orders": orders(rows=exact),
        "pareto/storms": pc,
        "pareto/storm_eligibility": pareto(rows=operating,keys=("case","landform","representation"),objectives=OBJECTIVES,cost="runtime_s",include_unresolved=True),
        "pareto/analytical": analytical_front,
        "comparisons/calibration": calibration,
        "comparisons/transfer": transfer,
        "comparisons/grouped_uncertainty": bootstrap(rows=storm, config=config),
    }
    for name, records in outputs.items():
        csv_write(file=study / "processed" / (name + ".csv"), rows=records)
    dbpath = study / "processed/metrics/study.sqlite"
    temp = dbpath.with_suffix(".build")
    with sqlite3.connect(temp) as db:
        for path, records in outputs.items():
            name = path.replace("/", "_")
            db.execute("DROP TABLE IF EXISTS " + name)
            if not records:
                continue
            cols = list(dict.fromkeys(k for r in records for k in r))
            col_list = ", ".join('"' + k + '"' for k in cols)
            db.execute(f"CREATE TABLE {name} ({col_list})")
            db.executemany(
                f"INSERT INTO {name} VALUES ({','.join('?' for k in cols)})",
                [tuple(r.get(k) for k in cols) for r in records],
            )
        db.execute("DROP TABLE IF EXISTS experiments")
        db.execute(
            "CREATE TABLE experiments (experiment_id TEXT, attempt TEXT PRIMARY KEY, stage TEXT, spec_json TEXT, provenance_json TEXT, directory TEXT)"
        )
        db.executemany(
            "INSERT INTO experiments VALUES (?,?,?,?,?,?)",
            [
                (
                    r["experiment_id"],
                    r["attempt"],
                    r["stage"],
                    json.dumps(r["spec"]),
                    json.dumps(r["context"]),
                    r["directory"],
                )
                for r in archived(
                    study=study,
                    stages=(
                        "analytical",
                        "convergence",
                        "long-storms",
                        "resolution",
                        "performance",
                        "calibration",
                        "synthetic",
                        "terrain-initial-controls",
                    ),
                )
            ],
        )
    temp.replace(dbpath)
    return {k: len(v) for k, v in outputs.items()}
