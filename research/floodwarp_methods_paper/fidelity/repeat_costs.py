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
        usable = [r for r in measurements if r.get("raw_timing_status") != "shared GPU observed"]
        times = np.array([r["hydraulic_s"] for r in usable], dtype=float)
        total = [
            r["end_to_end_s"] for r in usable if r["end_to_end_s"] is not None
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
                independent_timings=len(usable),
                shared_gpu_executions=len(measurements) - len(usable),
                included_run_ids=" ".join(r["source_id"] for r in usable),
                campaigns=" ".join(map(str, repeats)),
                run_ids=" ".join(r["source_id"] for r in measurements),
                accepted_steps=measurements[0]["steps"],
                hydraulic_median_s=float(np.median(times)) if times.size else None,
                hydraulic_min_s=float(times.min()) if times.size else None,
                hydraulic_max_s=float(times.max()) if times.size else None,
                hydraulic_q25_s=float(np.quantile(times, 0.25)) if times.size else None,
                hydraulic_q75_s=float(np.quantile(times, 0.75)) if times.size else None,
                end_to_end_timings=len(total),
                end_to_end_median_s=float(np.median(total)) if total else None,
                has_three_timings=len(usable) >= 3,
            )
        )
    metrics.csv_write(file=target / "timing_distribution.csv", rows=summaries)
    return {
        ident: summary
        for summary in summaries
        if summary["horizon"] == "completed recession"
        for ident in summary["run_ids"].split()
    }
