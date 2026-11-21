"""Matched within-event wetness, source, discharge and computational work."""

from collections import defaultdict
from statistics import median

import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.qualification import observations


def history(*, item, seconds):
    frames = item["data"]["frames"]
    spatial = {r["t_s"]: r for r in item["data"]["spatial"]}
    flux = observations.integral(record=item["data"], seconds=seconds)
    area = np.prod(item["spec"]["dimensions"])
    budget = 0.0
    rows = []
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame["t_s"] - 1) // seconds].append(frame)
    for i, q in enumerate(flux):
        t = (i + 1) * seconds
        state = spatial[t]
        cumulative = state["rain_m3"]
        rows.append(
            dict(
                case=item["spec"]["case"],
                method=item["spec"]["method"],
                dx=item["spec"]["dx"],
                experiment_id=item["id"],
                cells=item["data"]["cells"],
                end_s=t,
                rain_mm_h=(cumulative - budget) * 3.6e6 / (area * seconds),
                outlet_m3s=float(q),
                wet_fraction=state["wet_fraction"],
                work_fraction=state["active_fraction"],
                hydraulic_s=sum(r["hydraulic_s"] for r in groups[i]),
                steps=sum(r["steps"] for r in groups[i]),
                rejected=sum(r["rejected"] for r in groups[i]),
            )
        )
        budget = cumulative
    return rows


def fraction_bin(value):
    boundaries = (0.01, 0.05, 0.2, 0.5)
    labels = ("0–1%", "1–5%", "5–20%", "20–50%", "50–100%")
    return labels[int(np.searchsorted(boundaries, value, side="right"))]


def compare(*, events, study):
    records = []
    for items in events.values():
        for item in items:
            if item["stage"] in (
                "million-storms",
                "million-auto",
                "recession-controls",
            ):
                records.extend(history(item=item, seconds=900))
    peers = {
        (r["case"], r["dx"], r["end_s"]): r for r in records if r["method"] == "fv1"
    }
    groups = defaultdict(list)
    for row in records:
        fv = peers.get((row["case"], row["dx"], row["end_s"]))
        if fv is None or not row["method"].startswith("hancock"):
            continue
        row["fv1_hydraulic_s"] = fv["hydraulic_s"]
        row["fv1_id"] = fv["experiment_id"]
        for variable in ("wet_fraction", "work_fraction"):
            key = (
                row["case"],
                row["dx"],
                row["method"],
                variable,
                fraction_bin(row[variable]),
            )
            groups[key].append(row)
    summary = []
    for (case, dx, method, axis, interval), matched in sorted(groups.items()):
        h = sum(r["hydraulic_s"] for r in matched)
        fv = sum(r["fv1_hydraulic_s"] for r in matched)
        summary.append(
            dict(
                case=case,
                dx=dx,
                method=method,
                stratifier=axis,
                bin=interval,
                intervals=len(matched),
                hydraulic_s=h,
                fv1_s=fv,
                runtime_ratio=h / fv,
                median_interval_ratio=median(
                    r["hydraulic_s"] / r["fv1_hydraulic_s"] for r in matched
                ),
                hancock_id=matched[0]["experiment_id"],
                fv1_id=matched[0]["fv1_id"],
            )
        )
    folder = study / "processed/million"
    metrics.csv_write(file=folder / "event_intervals.csv", rows=records)
    metrics.csv_write(file=folder / "wetness_cost.csv", rows=summary)
