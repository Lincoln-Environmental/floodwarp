"""Event, landform, wet-fraction and uncertainty tables from measured results."""

import csv
import json
from pathlib import Path

import numpy as np


def write_rows(*, path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarise(*, root, edges):
    runs = [json.loads(p.read_text()) for p in sorted((root / "events").glob("*.json"))]
    base = {
        (
            r["landform"],
            r["storm"],
            r["dx"],
            r["terrain_dx"],
            tuple(r["dimensions_m"]),
            r["hours"],
        ): r
        for r in runs
        if r["method"] == "fv1"
    }
    rows = []
    bins = {}
    for r in runs:
        key = (
            r["landform"],
            r["storm"],
            r["dx"],
            r["terrain_dx"],
            tuple(r["dimensions_m"]),
            r["hours"],
        )
        fv = base.get(key)
        s = r["samples"]
        duration = r["hours"] * 3600
        rows.append(
            dict(
                group=r["group"],
                landform=r["landform"],
                storm=r["storm"],
                dx_m=r["dx"],
                terrain_dx_m=r["terrain_dx"],
                cells=int(np.prod(r["dimensions_m"]) / r["dx"] ** 2),
                method=r["method"],
                mode=r["mode"],
                duration_h=r["hours"],
                hydraulic_s=r["hydraulic_s"],
                loop_s=r["loop_s"],
                ratio_fv1=r["hydraulic_s"] / fv["hydraulic_s"] if fv else None,
                steps=r["steps"],
                dt_mean_s=duration / r["steps"],
                dt_min_s=min(i["dt_min_s"] for i in s),
                dt_max_s=max(i["dt_max_s"] for i in s),
                retries=sum(i["rejected"] for i in s),
                sampled_wet_mean=np.mean([i["wet_fraction"] for i in s]),
                sampled_wet_peak=max(i["wet_fraction"] for i in s),
                sampled_active_mean=np.mean([i["active_fraction"] for i in s]),
                preload_s=sum(i["preload_s"] or 0.0 for i in s)
                if s[0]["preload_s"] is not None
                else None,
                graph_capture_s=sum(i["capture_s"] for i in s),
                pool_mib=r["process_pool_mib"],
                extraction_s=r["extraction_s"],
                fields_write_s=r["field_write_s"],
                end_to_end_s=r["end_to_end_s"],
                peak_q_m3s=max(i["outlet_m3s"] for i in s),
                peak_interval_end_s=s[np.argmax([i["outlet_m3s"] for i in s])]["t_s"],
                initial_m3=r["initial_m3"],
                rain_mm=r["rain_m3"] * 1000 / np.prod(r["dimensions_m"]),
                rain_m3=r["rain_m3"],
                infiltration_m3=r["infiltration_m3"],
                out_m3=r["outflow_m3"],
                in_m3=r["inflow_m3"],
                final_m3=r["final_m3"],
                residual_m3=r["residual_m3"],
                max_speed_ms=max(i["max_speed_ms"] for i in s),
                balance_pass=r["balance_pass"],
            )
        )
        if r["method"] == "hancock" and r["mode"] == "tiles" and fv:
            for cand, control in zip(s, fv["samples"], strict=True):
                for measure, column in (
                    ("positive depth", "wet_fraction"),
                    ("scheduled cells", "active_fraction"),
                ):
                    index = min(
                        len(edges) - 2,
                        int(np.searchsorted(edges, cand[column], side="right") - 1),
                    )
                    group = (r["group"], measure, edges[index], edges[index + 1])
                    bins.setdefault(group, []).append(
                        (
                            cand["hydraulic_wall_s"],
                            control["hydraulic_wall_s"],
                            cand[column],
                            control[column],
                            cand["steps"],
                            control["steps"],
                        )
                    )
    write_rows(path=root / "event_summary.csv", rows=rows)
    grouped = []
    for (suite, measure, low, high), values in sorted(bins.items()):
        data = np.array(values)
        grouped.append(
            dict(
                group=suite,
                measure=measure,
                lower=low,
                upper=high,
                intervals=len(values),
                hancock_s=data[:, 0].sum(),
                fv1_s=data[:, 1].sum(),
                ratio=data[:, 0].sum() / data[:, 1].sum(),
                hancock_mean=data[:, 2].mean(),
                fv1_mean=data[:, 3].mean(),
                hancock_steps=int(data[:, 4].sum()),
                fv1_steps=int(data[:, 5].sum()),
            )
        )
    write_rows(path=root / "occupancy_events.csv", rows=grouped)


if __name__ == "__main__":
    root = Path(__file__).parent
    edges = (0.0, 0.01, 0.05, 0.2, 0.5, 1.0)
    summarise(root=root, edges=edges)
