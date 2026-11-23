"""Apply hydraulic observables to matched intervals; retain reference and eligibility identities."""

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from research.floodwarp_methods_paper import evidence, metrics
from research.floodwarp_methods_paper.fidelity.hydrograph import evaluate
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import observations
from validation.fv1_diffusion_domains import design
from floodwarp.solver_advisor.model import configuration


def limits(*, row, key):
    candidate_missing = (
        row["reference_status"] == "model_insufficient_outward_or_reversing"
    )
    scalar_with_reference = (
        key in ("peak_bias_pct", "volume_bias_pct") and candidate_missing
    )
    if key not in row or (
        row["reference_status"] != "usable" and not scalar_with_reference
    ):
        return False
    # Full-event shape and volume require both limbs. Peak magnitude can be
    # assessed within a recorded window if a resolved interior crest is visible.
    if not row["width_resolved"]:
        return False
    if key in ("peak_bias_pct", "peak_delay_pct"):
        return not (row["reference_truncated"] or row["model_truncated"]) or (
            "peak_delay_s" in row
            and not row["model_active_at_start"]
            and not row["reference_active_at_start"]
        )
    return not any(
        row[k]
        for k in (
            "model_truncated",
            "reference_truncated",
            "model_active_at_start",
            "reference_active_at_start",
        )
    )


def numerical_identity(*, method, exact=False):
    if method in ("roe", "hllc"):
        return dict(formulation="FV1", limiter_flux=method, execution_mode="sparse")
    data = configuration(method)
    if method == "hancock_adaptive" and exact:
        data["execution_mode"] = "dense"
    return data


def item(*, spec, data, ident, path, cohort, factor=1.0):
    return dict(s=spec, d=data, id=ident, path=path, cohort=cohort, factor=factor)


def archive(*, directory, config, partial=False):
    study = directory.parent
    groups = defaultdict(list)
    for rec in evidence.records(study=study, stages=("long-storms", "resolution")):
        s = rec["spec"]
        src = study / rec["directory"]
        d = json.loads((src / "result.json").read_text())
        it = item(spec=s, data=d, ident=rec["experiment_id"], path=src, cohort="legacy")
        if s["dx"] == s["dem"]:
            if rec["stage"] == "long-storms":
                groups["legacy", s["case"], "sampled"].append(it)
        if rec["stage"] == "resolution" and s["dem"] == 10:
            groups["fixed", s["case"], "10m bed"].append(it)
    for (case, rep), items in observations.load_events(
        study=study, config=config
    ).items():
        for i in items:
            groups["expanded", case, rep].append(
                item(
                    spec=i["spec"],
                    data=i["data"],
                    ident=i["id"],
                    path=i["file"].parent,
                    cohort="expanded",
                )
            )
    folder = directory / "storm_runs"
    ids = (
        [p.parent.name for p in folder.glob("*/result.json")]
        if partial
        else json.loads((folder / "index.json").read_text())
        + json.loads((folder / "rolling_index.json").read_text())
    )
    for ident in ids:
        src = folder / ident
        record = json.loads((src / "result.json").read_text())
        s = record["spec"]
        cohort = record["cohort"]
        rep = "10m bed" if cohort == "double" else "sampled"
        groups[cohort, s["case"], rep].append(
            item(
                spec=s,
                data=record["data"],
                ident=ident,
                path=src,
                cohort=cohort,
                factor=record["factor"],
            )
        )
    return groups


def curve(*, it, station):
    d = it["d"]
    frames = d.get("frames", d.get("samples"))
    return np.r_[0.0, [f["t_s"] for f in frames]], np.array(
        [f[station + "_m3s"] for f in frames]
    )


def compare(*, it, ref, station, cfg):
    t, a = curve(it=it, station=station)
    tr, b = curve(it=ref, station=station)
    np.testing.assert_array_equal(t, tr)
    result = evaluate(edges=t, model=a, reference=b, config=cfg)
    for prefix, run in (("model", it), ("reference", ref)):
        if run["d"]["initial_m3"] == 0 and run["s"]["landform"] != "coastal":
            result[prefix + "_active_at_start"] = False
            result[prefix + "_known_dry_start"] = True
    return result


def spacing(it):
    return it["s"]["dx"]


def matched_reference_inputs(*, base, control):
    expected = base["d"]["forcing"]
    observed = control["d"]["forcing"]
    keys = (
        "terrain_sha256",
        "initial_depth_sha256",
        "initial_hu_sha256",
        "initial_hv_sha256",
        "rain_mask_sha256",
        "infiltration_sha256",
        "boundary_x_sha256",
        "boundary_y_sha256",
        "roughness",
        "stage_m",
        "coastal_stage_hours",
        "coastal_stage_m",
        "stage_hold_s",
    )
    for field in keys:
        if expected.get(field) != observed.get(field):
            raise ValueError("Fine-grid reference changed physical input " + field)
    end = 3600 * base["s"]["hours"]
    if end != 3600 * control["s"]["hours"]:
        raise ValueError("Reference comparisons require a common physical end")
    rain = []
    for item, f in ((base, expected), (control, observed)):
        forcing = item["d"].get("recession_capture", {}).get("actual_forcing", {})
        t, v = forcing.get("rainfall", (f["rain_times_s"], f["rain_rates_ms"]))
        t, v = np.asarray(t), np.asarray(v)
        if t[0] > 0 or t[-1] < end or np.any(np.diff(t) <= 0) or len(v) != len(t):
            raise ValueError("Incomplete reference rainfall coverage")
        rain.append((t, v))
    knots = np.unique(np.r_[0, end, rain[0][0], rain[1][0]])
    knots = knots[(knots >= 0) & (knots <= end)]
    np.testing.assert_array_equal(
        np.interp(knots, *rain[0]),
        np.interp(knots, *rain[1]),
        err_msg="Different physical hyetographs on the comparison window",
    )


def references(*, items):
    rk = sorted(
        (i for i in items if i["s"]["method"] == "adaptive" and i["factor"] == 1),
        key=spacing,
    )
    if not rk:
        return None, {}
    fine = rk[0]
    cross = next(
        (
            i
            for i in items
            if i["s"]["method"] in ("hancock_dense", "hancock_adaptive")
            and spacing(i) == spacing(fine)
        ),
        None,
    )
    time = next(
        (
            i
            for i in items
            if i["factor"] < 1
            and spacing(i) == spacing(fine)
            and i["s"]["method"] == "adaptive"
        ),
        None,
    )
    if cross:
        matched_reference_inputs(base=fine, control=cross)
    if time:
        matched_reference_inputs(base=fine, control=time)
        if time["d"]["steps"] <= fine["d"]["steps"]:
            time = None
    return fine, dict(
        space=next((i for i in rk if spacing(i) > spacing(fine)), None),
        cross=cross,
        time=time,
    )


def storm_rows(*, groups, cfg):
    rows, checks = [], []
    for (cohort, case, rep), items in sorted(groups.items()):
        fine, controls = references(items=items)
        for station in ("outlet", "section"):
            deltas = {
                k: compare(it=i, ref=fine, station=station, cfg=cfg)
                for k, i in controls.items()
                if i
            }
            gate = {}
            for metric, limit in cfg["reference_limits"].items():
                spreads = {
                    k: abs(v[metric])
                    for k, v in deltas.items()
                    if limits(row=v, key=metric)
                }
                cadence_floor = 0.0
                if (
                    metric == "peak_delay_pct"
                    and fine
                    and "time" in deltas
                    and deltas["time"].get("reference_width80_s")
                ):
                    cadence_floor = (
                        100
                        * deltas["time"]["interval_max_s"]
                        / deltas["time"]["reference_width80_s"]
                    )
                uncertainty = (
                    max(*spreads.values(), cadence_floor) if len(spreads) == 3 else None
                )
                gate[metric] = uncertainty
                checks.append(
                    dict(
                        cohort=cohort,
                        case=case,
                        representation=rep,
                        station=station,
                        metric=metric,
                        reference_id=fine["id"] if fine else None,
                        criterion=limit,
                        cadence_floor=cadence_floor,
                        uncertainty=uncertainty,
                        qualified=uncertainty is not None and uncertainty <= limit,
                        **{k + "_id": i["id"] for k, i in controls.items() if i},
                        **spreads,
                    )
                )
            for it in items:
                s, d = it["s"], it["d"]
                if it["factor"] != 1 or (
                    cohort in ("legacy", "complete-legacy") and s["dx"] != 10
                ):
                    continue
                t, q = curve(it=it, station=station)
                frames = d.get("samples", d.get("spatial"))
                dims = s.get("dimensions", d.get("dimensions_m"))
                row = dict(
                    cohort=cohort,
                    case=case,
                    representation=rep,
                    station=station,
                    reference_kind="numerical",
                    method=s["method"],
                    **numerical_identity(method=s["method"]),
                    dx=s["dx"],
                    dem=s["dem"],
                    experiment_id=it["id"],
                    parent_id=it.get("parent_id", it["id"]),
                    landform=s["landform"],
                    storm=s["storm"],
                    split=s["split"],
                    seed=s["seed"],
                    cells=round(np.prod(dims) / s["dx"] ** 2),
                    area_m2=np.prod(dims),
                    hours=s["hours"],
                    runtime_s=d["hydraulic_s"],
                    gradient=d["forcing"]["parameters"]["grade"],
                    wet_fraction=np.mean([f["wet_fraction"] for f in frames])
                    if frames
                    else None,
                    reference_id=fine["id"] if fine else "",
                    ref_dx=spacing(fine) if fine else None,
                    station_x_m=dims[0] * (1 if station == "outlet" else 0.5),
                    input_initial=s["initial"],
                    interval_max_s=max(np.diff(t)),
                    signed_window_volume_m3=q @ np.diff(t),
                    infiltration_m3=d["infiltration_m3"],
                    storage_m3=d.get("final_storage_m3", d.get("final_m3")),
                )
                froude = [
                    f["wet_froude_max"]
                    for f in frames
                    if f.get("wet_froude_max") is not None
                ]
                supercrit = [
                    f["supercritical_wet_fraction"]
                    for f in frames
                    if f.get("supercritical_wet_fraction") is not None
                ]
                row["flow_regime"] = (
                    "unmeasured"
                    if not froude or not supercrit
                    else "subcritical"
                    if max(supercrit) == 0
                    else "supercritical"
                    if min(supercrit) == 1
                    else "mixed"
                )
                if fine:
                    row.update(compare(it=it, ref=fine, station=station, cfg=cfg))
                    row["infiltration_delta_m3"] = (
                        d["infiltration_m3"] - fine["d"]["infiltration_m3"]
                    )
                    for k, limit in cfg["reference_limits"].items():
                        u = gate[k]
                        row["uncertainty_" + k] = u
                        row["qualified_" + k] = bool(
                            u is not None
                            and u <= limit
                            and spacing(it) > spacing(fine)
                            and limits(row=row, key=k)
                        )
                else:
                    row["reference_status"] = "missing_reference"
                rows.append(row)
    return rows, checks


def pulse_rows(*, directory, cfg, archive_name="long_pulses"):
    folder = directory / archive_name
    source = folder / (folder / "current.txt").read_text().strip()
    rows = []
    for file in sorted(source.glob("*.npz")):
        profile, dx, cap, method = file.stem.split("_", 3)
        with np.load(file) as d:
            for j, x in enumerate(d["stations"]):
                row = evaluate(
                    edges=d["edges"],
                    model=d["flow"][:, j],
                    reference=d["truth"][:, j],
                    config=cfg,
                )
                row.update(
                    case=profile,
                    cohort="pulse",
                    representation="flat",
                    reference_kind="analytical",
                    station=int(x),
                    distance_m=int(x) - cfg["pulse"]["centre"],
                    method=method,
                    **numerical_identity(method=method, exact=True),
                    dx=int(dx),
                    cap_courant=float(cap),
                    runtime_s=float(d["runtime_s"]),
                    steps=int(d["steps"]),
                    experiment_id=file.stem,
                    residual_m3=float(d["residual_m3"]),
                    cell_depth_m=cfg["pulse"]["depth"],
                )
                for k in cfg["reference_limits"]:
                    row["qualified_" + k] = limits(row=row, key=k)
                rows.append(row)
    return rows


def historical(*, study, cfg):
    cases = {c.name: c for c in design()}
    rows = []
    for rec in evidence.records(study=study, stages=("analytical",)):
        s = rec["spec"]
        if s.get("kind") != "exact":
            continue
        d = json.loads((study / rec["directory"] / "result.json").read_text())
        c = cases[s["case"]]
        for j, station in enumerate(c.gauges):
            # These archives use section discharge per metre of section width.
            width = 20 if c.family in ("pulse", "damped_pulse") else c.width
            row = evaluate(
                edges=np.r_[0, d["interval_end_s"]],
                model=np.asarray(d["model_m2_s"])[:, j] * width,
                reference=np.asarray(d["exact_m2_s"])[:, j] * width,
                config=cfg,
            )
            rows.append(
                dict(
                    row,
                    cohort="archived_exact",
                    case=c.name,
                    method=s["method"],
                    station=station,
                    policy=s["policy"],
                    dx=s["dx"],
                    experiment_id=rec["experiment_id"],
                    runtime_s=d["median_wall_s"],
                    depth_rmse_m=d["depth_rmse"],
                    momentum_x_rmse_m2s=d["momentum_x_rmse"],
                    momentum_y_rmse_m2s=d["momentum_y_rmse"],
                )
            )
    return rows


def process(*, directory):
    cfg = json.loads((directory / "design.json").read_text())
    config = read_config(path=directory.parent / "configs/study.json")
    groups = archive(directory=directory, config=config)
    from research.floodwarp_methods_paper.fidelity.extended_analysis import attach

    from research.floodwarp_methods_paper.fidelity import policy_recession

    completed = attach(directory=directory, groups=groups)
    policy_recession.attach(directory=directory, groups=groups, completed=completed)
    groups.update(completed)
    rows, checks = storm_rows(groups=groups, cfg=cfg)
    target = directory / "results"
    for name, data in (
        ("storms", rows),
        ("reference_checks", checks),
        ("pulses", pulse_rows(directory=directory, cfg=cfg)),
        (
            "original_pulses",
            pulse_rows(directory=directory, cfg=cfg, archive_name="pulse_runs"),
        ),
        ("archived_exact", historical(study=directory.parent, cfg=cfg)),
    ):
        metrics.csv_write(file=target / (name + ".csv"), rows=data)
    return groups, rows


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    process(directory=directory)
