"""Observable-specific reference gates and conservative, measured performance comparisons."""

from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.qualification import stages


def current(*, study, stage, config):
    source = stages.source(root=study.parents[1], study=study, stage=stage)
    target = runner.digest(config)
    gpu = os.environ.get("FLOODWARP_PAPER_RESULT_GPU", "").casefold()
    selected = [
        r
        for r in runner.archived(study=study, stages=(stage,))
        if r["config_sha256"] == target
        and r["context"]["source_sha256"] == source
        and gpu in r["context"]["gpu"].casefold()
    ]
    if (
        len(
            {
                (r["context"]["gpu"], r["context"]["device"], r["context"]["warp"])
                for r in selected
            }
        )
        > 1
    ):
        raise ValueError("Select one device cohort")
    return selected


def integral(*, record, seconds):
    frames = record["frames"]
    t = np.asarray([r["t_s"] for r in frames])
    q = np.asarray([r["outlet_m3s"] for r in frames])
    intervals = np.diff(np.r_[0, t])
    if t[-1] % seconds or np.any(seconds % intervals):
        raise ValueError("Unaligned interval products")
    bins = (t - 1) // seconds
    return np.bincount(bins, weights=q * intervals) / seconds


def rmse(a, b):
    return float(np.mean(np.square(a - b)) ** 0.5)


@lru_cache(maxsize=10)
def arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def differences(*, a, b):
    aa = arrays(a["file"])
    bb = arrays(b["file"])
    h, ref = aa["depth"], bb["depth"]
    wet = (h > 0.01) | (ref > 0.01)
    q = np.array([f["outlet_m3s"] for f in a["data"]["frames"]])
    qr = np.array([f["outlet_m3s"] for f in b["data"]["frames"]])
    t = np.array([f["t_s"] for f in a["data"]["frames"]])
    np.testing.assert_array_equal(t, [f["t_s"] for f in b["data"]["frames"]])
    out = dict(
        depth_rmse_m=rmse(h, ref),
        wet_depth_rmse_m=rmse(h[wet], ref[wet]) if wet.any() else None,
        max_depth_rmse_m=rmse(aa["maximum"], bb["maximum"]),
        outlet_rmse_m3s=rmse(q, qr),
        internal_rmse_m3s=rmse(
            np.array([f["section_m3s"] for f in a["data"]["frames"]]),
            np.array([f["section_m3s"] for f in b["data"]["frames"]]),
        ),
        absolute_peak_error_m3s=float(abs(q.max() - qr.max())),
        signed_volume_error_m3=float(np.sum((q - qr) * np.diff(np.r_[0, t]))),
        absolute_peak_time_error_s=float(abs(t[q.argmax()] - t[qr.argmax()]))
        if q.max() > 0 and qr.max() > 0
        else None,
        recession_rmse_m3s=rmse(q[t > 21600], qr[t > 21600]),
        component_count_error_1cm=a["data"]["components"]["1"]
        - b["data"]["components"]["1"],
    )
    for cm in (1, 5, 10):
        p = h > cm / 100
        z = ref > cm / 100
        union = np.count_nonzero(p | z)
        out[f"final_iou_{cm}cm"] = np.count_nonzero(p & z) / union if union else 1.0
        out[f"native_wet_area_difference_{cm}cm_m2"] = float(
            (aa[f"wet_fraction_{cm}cm"] - bb[f"wet_fraction_{cm}cm"]).sum()
            * a["spec"]["support"] ** 2
        )
    out["inundation_error"] = 1 - out["final_iou_1cm"]
    return out


def eligible(*, deltas, reference, objectives):
    """Each available refinement and cross-formulation comparison must pass."""
    h = arrays(reference["file"])["depth"]
    q = max(abs(f["outlet_m3s"]) for f in reference["data"]["frames"])
    peak = max(f["outlet_m3s"] for f in reference["data"]["frames"])
    limits = dict(
        depth=max(0.001, 0.02 * float(np.mean(h * h) ** 0.5)),
        maximum_depth=0.01,
        inundation=0.05,
        discharge=0.05 * q,
        peak_discharge=0.05 * abs(peak),
        peak_timing=reference["spec"]["interval"],
    )
    gates = {}
    spread = {}
    for obj, k in objectives.items():
        if obj == "memory":
            continue
        values = [d[k] for d in deltas]
        spread[obj] = (
            max(values) if values and all(v is not None for v in values) else None
        )
        gates[obj] = spread[obj] is not None and spread[obj] <= limits[obj]
        if obj in ("peak_discharge", "peak_timing", "discharge") and q == 0:
            gates[obj] = False
        if obj in ("peak_discharge", "peak_timing") and peak <= 0:
            gates[obj] = False
        if obj == "inundation" and not (h > 0.01).any():
            gates[obj] = False
    return gates, spread


def spacing(item):
    return item["spec"]["dx"]


def load_events(
    *,
    study,
    config,
    include=(
        "million-storms",
        "million-auto",
        "large-references",
        "large-refinements",
        "heldout-references",
        "recession-controls",
    ),
):
    group = defaultdict(list)
    for stage in include:
        for rec in current(study=study, stage=stage, config=config):
            folder = study / rec["directory"]
            d = json.loads((folder / "result.json").read_text())
            s = rec["spec"]
            item = dict(
                spec=s,
                data=d,
                file=folder / "spatial.npz",
                id=rec["experiment_id"],
                stage=stage,
                receipt=rec,
            )
            rep = (
                "sampled"
                if s["dx"] == s["dem"]
                else f"fixed-{s['dem']:g}m-{s['initial_grid']}"
            )
            group[s["case"], rep].append(item)
    return group


def performance(*, item, case, rep):
    s = item["spec"]
    v = item["data"]
    row = dict(
        case=case,
        representation=rep,
        landform=s["landform"],
        seed=s["seed"],
        storm=s["storm"],
        split=s["split"],
        method=s["method"],
        dx=s["dx"],
        dem=s["dem"],
        stage=item["stage"],
        experiment_id=item["id"],
        dimensions_m="x".join(str(n) for n in s["dimensions"]),
        interval_s=s["interval"],
        support_m=s["support"],
        hours=v["hours_completed"],
        runtime_s=v["hydraulic_s"],
        memory_mib=v["live_pool_mib"],
        **metrics.config_fields(method=s["method"], dx=s["dx"]),
    )
    for key in (
        "cells",
        "steps",
        "rejected",
        "dt_mean_s",
        "dt_min_s",
        "setup_s",
        "graph_capture_s",
        "preload_s",
        "extraction_s",
        "end_to_end_s",
        "wet_fraction",
        "scheduled_fraction",
        "rain_m3",
        "infiltration_m3",
        "initial_m3",
        "final_storage_m3",
        "inflow_m3",
        "outflow_m3",
        "residual_m3",
    ):
        row[key] = v[key]
    row["end_to_end_before_receipt_s"] = row["end_to_end_s"]
    row["end_to_end_s"] = item["receipt"]["wall_s"]
    coarse = integral(record=v, seconds=900)
    fine = [f["outlet_m3s"] for f in v["frames"]]
    row.update(
        hydraulic_identity=runner.digest(
            dict(
                fields=v["hashes"],
                spacing=s["dx"],
                q=fine,
                section=[f["section_m3s"] for f in v["frames"]],
            )
        ),
        outlet_peak_m3s=max(fine),
        outlet_net_m3=float(np.sum(coarse) * 900),
        dt_max_s=max(f["dt_max_s"] for f in v["frames"]),
        peak_wet_fraction=max(f["wet_fraction"] for f in v["spatial"]),
        peak_scheduled_fraction=max(f["active_fraction"] for f in v["spatial"]),
        final_wet_fraction=v["spatial"][-1]["wet_fraction"],
        final_area_1cm_m2=v["spatial"][-1]["cm_fraction"] * np.prod(s["dimensions"]),
        outlet_peak_900s_m3s=float(coarse.max()),
        peak_end_s=v["frames"][int(np.argmax(fine))]["t_s"] if max(fine) > 0 else None,
        peak_900s_end_s=900 * (1 + int(coarse.argmax())) if coarse.max() > 0 else None,
        control_sha=item["receipt"]["context"]["source_sha256"],
        input_sha=runner.digest(v["forcing"]),
        graph=all(f["graph"] for f in v["frames"]),
    )
    return row


def build(*, study, config):
    records = load_events(study=study, config=config)
    results = []
    refs = []
    phases = []
    hourly = []
    inventory = []
    for (case, rep), items in records.items():
        references = sorted(
            (i for i in items if i["spec"]["method"] == "adaptive"), key=spacing
        )
        fine = references[0] if references else None
        coarse = (
            next((i for i in references if spacing(i) > spacing(fine)), None)
            if fine
            else None
        )
        cross = (
            next(
                (
                    i
                    for i in items
                    if i["spec"]["method"] == "hancock_dense"
                    and spacing(i) == spacing(fine)
                ),
                None,
            )
            if fine
            else None
        )
        dimensions = items[0]["spec"]["dimensions"]
        area = int(np.prod(dimensions))
        cohort = (
            "120 × 80 m controls"
            if area == 9600
            else "1 km² resolution"
            if area == 1000000
            else "regional catchments"
        )
        valid = fine and coarse and cross
        gates = {}
        spread = {}
        if valid:
            ds = [differences(a=i, b=fine) for i in (coarse, cross)]
            gates, spread = eligible(
                deltas=ds, reference=fine, objectives=metrics.OBJECTIVES
            )
            for obj, ok in gates.items():
                refs.append(
                    dict(
                        case=case,
                        representation=rep,
                        objective=obj,
                        cohort=cohort,
                        area_m2=area,
                        coarse_dx_m=spacing(coarse),
                        fine_dx_m=spacing(fine),
                        qualified=ok,
                        coarse_id=coarse["id"],
                        fine_id=fine["id"],
                        independent_id=cross["id"],
                        refinement_change=ds[0][metrics.OBJECTIVES[obj]],
                        formulation_change=ds[1][metrics.OBJECTIVES[obj]],
                        uncertainty=spread[obj],
                    )
                )
        else:
            for obj in metrics.OBJECTIVES:
                if obj != "memory":
                    refs.append(
                        dict(
                            case=case,
                            representation=rep,
                            objective=obj,
                            cohort=cohort,
                            area_m2=area,
                            qualified=False,
                            reason="missing independent refinement or fine formulation",
                        )
                    )
        for i in items:
            row = performance(item=i, case=case, rep=rep)
            if fine:
                row.update(differences(a=i, b=fine))
            preflight = i["data"].get("preflight", {})
            row["selected_mode"] = preflight.get("selected", row["execution_mode"])
            row["preflight_estimated_work_fraction"] = preflight.get(
                "scheduled_estimate"
            )
            row["preflight_rain_fraction"] = preflight.get("rain_fraction")
            row["reference_id"] = fine["id"] if fine else None
            row["reference_self"] = bool(fine and i["id"] == fine["id"])
            row["reference_setup"] = i["stage"] not in (
                "million-storms",
                "million-auto",
                "recession-controls",
            ) and not (rep == "sampled" and spacing(i) == 10)
            row.update(
                {
                    "qualified_" + obj: bool(
                        gates.get(obj, False)
                        and fine
                        and spacing(i) > spacing(fine)
                        and row.get(metrics.OBJECTIVES[obj]) is not None
                    )
                    for obj in metrics.OBJECTIVES
                }
            )
            row.update({"reference_change_" + obj: v for obj, v in spread.items()})
            row["qualified_memory"] = True
            results.append(row)
            for frame in i["data"]["spatial"]:
                phases.append(dict(experiment_id=i["id"], **frame))
            for n, q in enumerate(integral(record=i["data"], seconds=900)):
                hourly.append(
                    dict(experiment_id=i["id"], end_s=(n + 1) * 900, outlet_m3s=q)
                )
            inventory.append(
                dict(
                    experiment_id=i["id"],
                    raw_directory=i["receipt"]["directory"],
                    attempt=i["receipt"]["attempt"],
                    git_sha=i["receipt"]["context"]["git_sha"],
                    hardware=i["receipt"]["context"]["gpu"],
                )
            )
    from research.floodwarp_methods_paper.qualification import intervals

    intervals.compare(events=records, study=study)
    path = study / "processed/million"
    for name, rows in (
        ("observations", results),
        ("reference_checks", refs),
        ("spatial_history", phases),
        ("hydrographs_900s", hourly),
        ("archive", inventory),
    ):
        metrics.csv_write(file=path / (name + ".csv"), rows=rows)
    fronts = metrics.pareto(
        rows=[
            r
            for r in results
            if r["stage"] in ("million-storms", "million-auto", "recession-controls")
        ],
        keys=("case", "representation", "landform"),
        objectives=metrics.OBJECTIVES,
        cost="runtime_s",
        include_unresolved=True,
    )
    metrics.csv_write(file=path / "pareto.csv", rows=fronts)
    short_controls = [
        row for row in results if short_interval_forecast(row=row, config=config)
    ]
    short_fronts = metrics.pareto(
        rows=short_controls,
        keys=("case", "representation", "landform", "interval_s"),
        objectives=metrics.OBJECTIVES,
        cost="runtime_s",
        include_unresolved=True,
    )
    metrics.csv_write(file=path / "short_interval_pareto.csv", rows=short_fronts)
    return results


def short_interval_forecast(*, row, config):
    dimensions = "x".join(str(n) for n in config["synthetic"]["dimensions_m"])
    return (
        row["stage"] == "large-references"
        and row["dimensions_m"] == dimensions
        and row["representation"] == "sampled"
        and not metrics.is_true(row["reference_setup"])
    )
