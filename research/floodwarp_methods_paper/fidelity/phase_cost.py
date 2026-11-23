"""Matched hydraulic cost during rain and recession at the available wet-footprint cadence."""

from collections import defaultdict
import numpy as np
from research.floodwarp_methods_paper import metrics


def intervals(*, item, cadence):
    data = item["d"]
    frames = data.get("frames", data.get("samples"))
    spatial = data.get("spatial", data.get("samples", []))
    footprint = {int(r["t_s"]): r for r in spatial if "wet_fraction" in r}
    bins = defaultdict(list)
    for f in frames:
        if "wet_fraction" in f:
            footprint[f["t_s"]] = f
        bins[((int(f["t_s"]) - 1) // cadence + 1) * cadence].append(f)
    out = {}
    for end, values in bins.items():
        if int(values[-1]["t_s"]) != end:
            raise ValueError("Unresolved cost reporting window")
        f = footprint.get(end, {})
        out[end] = dict(
            seconds=sum(
                v.get("hydraulic_s", v.get("hydraulic_wall_s")) for v in values
            ),
            steps=sum(v["steps"] for v in values),
            wet=f.get("wet_fraction"),
            active=f.get("active_fraction"),
            work_tiles=f.get("work_tiles", f.get("active_tiles")),
        )
    return out


def label(*, fraction, edges):
    if fraction is None:
        return "unmeasured"
    index = min(np.searchsorted(edges[1:], fraction, side="right"), len(edges) - 2)
    return f"{100 * edges[index]:g}–{100 * edges[index + 1]:g}%"


def collect(*, groups, directory, cadence, edges, phase_edges, target=None):
    if target is None:
        target = directory / "results"
    chunks = []
    totals = defaultdict(list)
    for (cohort, case, rep), items in groups.items():
        if not cohort.startswith("complete-") or rep != "sampled":
            continue
        grid = defaultdict(dict)
        for i in items:
            if i["factor"] == 1:
                grid[i["s"]["dx"]][i["s"]["method"]] = i
        for dx, methods in grid.items():
            if "fv1" not in methods:
                continue
            fv = methods["fv1"]
            baseline = intervals(item=fv, cadence=cadence)
            for method, it in methods.items():
                for end, v in intervals(item=it, cadence=cadence).items():
                    b = baseline[end]
                    phase_index = np.searchsorted(phase_edges, end / 3600, side="left")
                    phase = (
                        "rain",
                        "early recession",
                        "intermediate recession",
                        "late recession",
                    )[phase_index]
                    r = dict(
                        cohort=cohort,
                        case=case,
                        dx=dx,
                        method=method,
                        id=it["id"],
                        fv1_id=fv["id"],
                        end_s=end,
                        seconds=v["seconds"],
                        fv1_s=b["seconds"],
                        steps=v["steps"],
                        fv1_steps=b["steps"],
                        wet_fraction=v["wet"],
                        fv1_wet_fraction=b["wet"],
                        active_fraction=v["active"],
                        work_tiles=v["work_tiles"],
                        ratio=v["seconds"] / b["seconds"],
                        phase=phase,
                    )
                    chunks.append(r)
                    for axis, level in (
                        ("phase", phase),
                        ("fv1_wet", label(fraction=b["wet"], edges=edges)),
                        ("scheduled_work", label(fraction=v["active"], edges=edges)),
                    ):
                        totals[
                            cohort, case, dx, method, it["id"], fv["id"], axis, level
                        ].append(r)
    summary = []
    for (cohort, case, dx, method, ident, base, axis, level), g in totals.items():
        wall = sum(r["seconds"] for r in g)
        baseline = sum(r["fv1_s"] for r in g)
        steps = sum(r["steps"] for r in g)
        fv_steps = sum(r["fv1_steps"] for r in g)
        summary.append(
            dict(
                cohort=cohort,
                case=case,
                dx=dx,
                method=method,
                id=ident,
                fv1_id=base,
                axis=axis,
                level=level,
                windows=len(g),
                cadence_s=cadence,
                method_s=wall,
                fv1_s=baseline,
                ratio=wall / baseline,
                steps=steps,
                fv1_steps=fv_steps,
                per_step_ratio=(wall / steps) / (baseline / fv_steps),
            )
        )
    metrics.csv_write(file=target / "cost_intervals.csv", rows=chunks)
    metrics.csv_write(file=target / "cost_by_phase.csv", rows=summary)
