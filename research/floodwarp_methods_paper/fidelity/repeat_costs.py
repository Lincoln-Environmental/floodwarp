"""Summarise independent timed executions on identical event horizons and numerical inputs."""

from collections import defaultdict
import numpy as np

from research.floodwarp_methods_paper import metrics


def collect(*, rows, target):
    groups = defaultdict(dict)
    for r in rows:
        label = r["horizon"]
        if (
            label.startswith("complete-native-repeat-")
            or label == "complete-native-metre"
        ):
            horizon = "completed recession"
        elif label == "original_window":
            horizon = "initial window"
        else:
            continue
        key = (
            r["case"],
            r["method"],
            r["execution_mode"],
            r["dx_m"],
            r["factor"],
            float(r["interval_s"]),
            float(r["end_h"]),
            r["native_sha"],
            r["solver_source"],
            r["hardware"],
            horizon,
        )
        identity = r["timing_repeat"] or 0
        if (
            identity in groups[key]
            and groups[key][identity]["source_id"] != r["source_id"]
        ):
            raise ValueError("Conflicting timings under a single execution identity")
        groups[key][identity] = r
    summaries = []
    for key, repeats in sorted(groups.items()):
        measurements = list(repeats.values())
        for key_field in (
            "steps",
            "net_outlet_m3",
            "infiltration_m3",
            "storage_m3",
            "outlet_peak_m3s",
        ):
            if len({r[key_field] for r in measurements}) != 1:
                raise ValueError(
                    f"{key_field} changed across nominally identical timing repeats"
                )
        times = np.array([r["hydraulic_s"] for r in measurements], dtype=float)
        total = [
            r["end_to_end_s"] for r in measurements if r["end_to_end_s"] is not None
        ]
        vals = dict(
            zip(
                (
                    "case",
                    "method",
                    "execution",
                    "dx_m",
                    "cfl_factor",
                    "interval_s",
                    "end_h",
                    "native_sha",
                    "solver_source",
                    "hardware",
                    "horizon",
                ),
                key,
                strict=True,
            )
        )
        summaries.append(
            dict(
                vals,
                independent_timings=len(repeats),
                campaigns=" ".join(map(str, repeats)),
                run_ids=" ".join(r["source_id"] for r in measurements),
                accepted_steps=measurements[0]["steps"],
                hydraulic_median_s=float(np.median(times)),
                hydraulic_min_s=float(times.min()),
                hydraulic_max_s=float(times.max()),
                hydraulic_q25_s=float(np.quantile(times, 0.25)),
                hydraulic_q75_s=float(np.quantile(times, 0.75)),
                end_to_end_timings=len(total),
                end_to_end_median_s=float(np.median(total)) if total else None,
                has_three_timings=len(repeats) >= 3,
            )
        )
    metrics.csv_write(file=target / "timing_distribution.csv", rows=summaries)
    return {
        ident: summary
        for summary in summaries
        if summary["horizon"] == "completed recession"
        for ident in summary["run_ids"].split()
    }
