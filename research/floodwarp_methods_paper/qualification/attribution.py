"""Pair physical-domain geometry and initial-state controls, independent of timing."""

from collections import defaultdict
import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.qualification import observations


def build(*, study, config):
    cases = defaultdict(dict)
    for (case, rep), events in observations.load_events(
        study=study, config=config
    ).items():
        for e in events:
            s = e["spec"]
            cases[case][
                s["method"], float(s["dx"]), float(s["dem"]), s["initial_grid"]
            ] = e
    items = []
    for case, group in cases.items():
        for method in ("fv1", "hancock_dense", "adaptive"):
            pairs = [
                (
                    "grid on fixed 20 m bed and initial map",
                    (20, 20, "sampled"),
                    (10, 20, "parent"),
                ),
                (
                    "initial water mapping on fixed bed",
                    (10, 20, "parent"),
                    (10, 20, "sampled"),
                ),
                (
                    "resampled bed with matched water",
                    (10, 20, "sampled"),
                    (10, 10, "sampled"),
                ),
                ("grid on fixed 10 m bed", (5, 10, "sampled"), (1, 10, "sampled")),
                ("resampled bed at 1 m", (1, 10, "sampled"), (1, 1, "sampled")),
            ]
            for mechanism, a, b in pairs:
                left = group.get((method, *a))
                right = group.get((method, *b))
                if not left or not right:
                    continue
                delta = observations.differences(a=left, b=right)
                items.append(
                    dict(
                        case=case,
                        mechanism=mechanism,
                        method=method,
                        from_id=left["id"],
                        to_id=right["id"],
                        initial_storage_change_m3=right["data"]["initial_m3"]
                        - left["data"]["initial_m3"],
                        **delta,
                    )
                )
        for method in ("fv1", "adaptive"):
            seq = [
                group.get((m, 10.0, dem, "sampled"))
                for m in (method, "hancock_dense")
                for dem in (10.0, 20.0)
            ]
            if all(seq):
                a, b, c, d = seq
                h = [observations.arrays(e["file"])["depth"] for e in seq]
                q = [
                    np.array([r["outlet_m3s"] for r in e["data"]["frames"]])
                    for e in seq
                ]
                items.append(
                    dict(
                        case=case,
                        mechanism="bed-by-formulation difference of differences",
                        method=method + " vs Hancock",
                        from_id=a["id"] + " " + b["id"],
                        to_id=c["id"] + " " + d["id"],
                        depth_rmse_m=observations.rmse(h[0] - h[1], h[2] - h[3]),
                        outlet_rmse_m3s=observations.rmse(q[0] - q[1], q[2] - q[3]),
                    )
                )
    metrics.csv_write(
        file=study / "processed/million/resolution_components.csv", rows=items
    )
    old = {
        (r["case"], r["representation"]): r
        for r in metrics.csv_read(study / "processed/metrics/reference_uncertainty.csv")
    }
    rows = []
    for r in metrics.csv_read(study / "processed/million/reference_checks.csv"):
        base = old.get((r["case"], r["representation"]))
        if base:
            k = r["objective"]
            rows.append(
                dict(
                    **r,
                    original_qualified=base[
                        "qualified_final_inundation"
                        if k == "inundation"
                        else "qualified_" + k
                    ],
                    original_refinement_change=base[
                        "final_inundation_change" if k == "inundation" else k
                    ],
                    original_fine_dx=base["fine_dx"],
                )
            )
    metrics.csv_write(
        file=study / "processed/million/reference_reassessment.csv", rows=rows
    )
