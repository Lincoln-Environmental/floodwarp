"""Summarise comparisons relative to successive FV2 refinements and measured cost."""

import csv
import json
from pathlib import Path

import numpy as np


def rows(*, path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(*, path, records):
    if records:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)


def compare_pairs(*, review):
    accuracy = rows(path=review / "accuracy.csv")
    uncertain = rows(path=review / "reference_uncertainty.csv")
    uncertainty = {
        (r["landform"], r["storm"]): r
        for r in uncertain
        if r["representation"] == "sampled"
    }
    primary = {
        (r["landform"], r["storm"], r["method"]): r
        for r in accuracy
        if r["group"] == "primary"
        and r["representation"] == "sampled"
        and (r["method"] != "hancock" or r["mode"] == "tiles")
    }
    comparisons = []
    for (land, storm), ref in uncertainty.items():
        fv1 = primary[land, storm, "fv1"]
        for method in ("hancock", "fv2", "li"):
            target = primary.get((land, storm, method))
            if target is None:
                continue
            record = dict(
                landform=land,
                storm=storm,
                method=method,
                reference=ref["finer"],
                ref_depth_resolved=ref["depth_resolved"],
                ref_discharge_resolved=ref["discharge_resolved"],
                runtime_ratio_to_fv1=float(target["runtime_s"])
                / float(fv1["runtime_s"]),
            )
            for short, column, reference in (
                ("depth_m", "depth_rmse_m", "depth_difference_m"),
                ("max_depth_m", "max_depth_rmse_m", "max_depth_difference_m"),
                ("hydrograph_m3s", "outlet_rmse_m3s", "outlet_difference_m3s"),
            ):
                gain = float(fv1[column]) - float(target[column])
                # Difference of error norms can shift by twice the refinement difference.
                band = 2 * float(ref[reference])
                record[short + "_fv1_minus_method"] = gain
                record[short + "_twice_refinement_difference"] = band
                record[short + "_advantage"] = (
                    "method" if gain > band else "fv1" if gain < -band else "unresolved"
                )
            comparisons.append(record)
    save(path=review / "solver_tradeoffs.csv", records=comparisons)
    summary = []
    for land in sorted({k[0] for k in primary}):
        group = [
            r for r in comparisons if r["landform"] == land and r["method"] == "hancock"
        ]
        if not group:
            continue
        fav = [
            r
            for r in primary.values()
            if r["landform"] == land and r["method"] == "hancock"
        ]
        summary.append(
            dict(
                landform=land,
                pairs=len(group),
                runtime_ratio=float(
                    np.median([r["runtime_ratio_to_fv1"] for r in group])
                ),
                hancock_depth_rmse_mm=float(
                    np.median([1000 * float(r["depth_rmse_m"]) for r in fav])
                ),
                fv1_depth_rmse_mm=float(
                    np.median(
                        [
                            1000
                            * float(primary[land, r["storm"], "fv1"]["depth_rmse_m"])
                            for r in fav
                        ]
                    )
                ),
                hancock_outlet_rmse_m3s=float(
                    np.median([float(r["outlet_rmse_m3s"]) for r in fav])
                ),
                fv1_outlet_rmse_m3s=float(
                    np.median(
                        [
                            float(primary[land, r["storm"], "fv1"]["outlet_rmse_m3s"])
                            for r in fav
                        ]
                    )
                ),
                sampled_iou_1cm=float(
                    np.median([float(r["space_time_iou_1cm"]) for r in fav])
                ),
                hancock_depth_wins=sum(
                    r["depth_m_advantage"] == "method"
                    and r["ref_depth_resolved"] == "True"
                    for r in group
                ),
                fv1_depth_wins=sum(
                    r["depth_m_advantage"] == "fv1"
                    and r["ref_depth_resolved"] == "True"
                    for r in group
                ),
                hancock_discharge_wins=sum(
                    r["hydrograph_m3s_advantage"] == "method"
                    and r["ref_discharge_resolved"] == "True"
                    for r in group
                ),
                fv1_discharge_wins=sum(
                    r["hydrograph_m3s_advantage"] == "fv1"
                    and r["ref_discharge_resolved"] == "True"
                    for r in group
                ),
            )
        )
    save(path=review / "landform_tradeoffs.csv", records=summary)


def physical_summary(*, review):
    records = [json.loads(f.read_text()) for f in (review / "events").glob("*.json")]
    summary = {}
    for group in sorted({r["group"] for r in records}):
        data = [r for r in records if r["group"] == group]
        summary[group] = dict(
            runs=len(data),
            all_complete=all(
                r["hours"] >= 8 and r["samples"][-1]["t_s"] == r["hours"] * 3600
                for r in data
            ),
            all_balanced=all(r["balance_pass"] for r in data),
            worst_abs_final_residual_m3=max(abs(r["residual_m3"]) for r in data),
            worst_abs_rain_error_m3=max(
                abs(r["rain_m3"] - r["expected_rain_m3"]) for r in data
            ),
            maximum_sampled_speed_ms=max(
                s["max_speed_ms"] for r in data for s in r["samples"]
            ),
        )
    (review / "integrity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    compare_pairs(review=review)
    physical_summary(review=review)
