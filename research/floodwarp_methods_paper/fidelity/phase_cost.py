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
            old = footprint.get(f["t_s"], {})
            footprint[f["t_s"]] = {
                **old,
                **{k: v for k, v in f.items() if v is not None},
            }
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
            active_source=f.get("active_source", item["id"])
            if f.get("active_fraction") is not None
            else None,
            work_tiles=f.get("work_tiles", f.get("active_tiles")),
        )
    return out


def label(*, fraction, edges):
    if fraction is None:
        return "unmeasured"
    index = min(np.searchsorted(edges[1:], fraction, side="right"), len(edges) - 2)
    return f"{100 * edges[index]:g}–{100 * edges[index + 1]:g}%"


def event_diagnostics(*, item):
    data = item["d"]
    frames = data.get("frames", data.get("samples", []))
    sections = data.get("spatial", data.get("samples", []))
    steps = sum(r["steps"] for r in frames)
    end = frames[-1]["t_s"]
    result = dict(
        id=item["id"],
        parent_id=item.get("parent_id", item["id"]),
        end_s=end,
        accepted_steps=steps,
        mean_dt_s=end / steps,
        hydraulic_s=data["hydraulic_s"],
    )
    for field, old, reducer in (
        ("dt_min_s", "dt_min_s", min),
        ("dt_max_s", "dt_max_s", max),
        ("rejected", "rejected", sum),
        ("graph_capture_s", "capture_s", sum),
        ("compilation_s", "preload_s", sum),
        ("extraction_s", "extraction_s", sum),
    ):
        values = [r.get(field, r.get(old)) for r in frames]
        values = [v for v in values if v is not None]
        result[field + "_intervals"] = len(values)
        result[field] = reducer(values) if len(values) == len(frames) else None
    result["diagnostic_intervals"] = len(frames)
    mem = [r["live_pool_mib"] for r in sections if r.get("live_pool_mib") is not None]
    result["sampled_pool_peak_mib"] = max(mem) if mem else None
    result["pool_snapshots"] = len(mem)
    result["source_final_pool_mib"] = data.get(
        "live_pool_mib", data.get("process_pool_mib")
    )
    for k in (
        "inflow_m3",
        "outflow_m3",
        "rain_m3",
        "infiltration_m3",
        "initial_m3",
        "residual_m3",
    ):
        result[k] = data.get(k)
    result["storage_m3"] = data.get("final_storage_m3", data.get("final_m3"))
    return result


def collect(*, groups, directory, cadence, edges, phase_edges, target=None, representation="sampled"):
    if target is None:
        target = directory / "results"
    chunks = []
    diagnostics = []
    totals = defaultdict(list)
    for (cohort, case, rep), items in groups.items():
        if not cohort.startswith("complete-") or rep != representation:
            continue
        grid = defaultdict(dict)
        for item in items:
            diagnostics.append(
                dict(
                    case=case,
                    cohort=cohort,
                    dx=item["s"]["dx"],
                    method=item["s"]["method"],
                    cfl_factor=item["factor"],
                    **event_diagnostics(item=item),
                )
            )
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
                        scheduled_observation_id=v["active_source"],
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
    metrics.csv_write(file=target / "whole_storm_diagnostics.csv", rows=diagnostics)
    metrics.csv_write(file=target / "cost_intervals.csv", rows=chunks)
    metrics.csv_write(file=target / "cost_by_phase.csv", rows=summary)
