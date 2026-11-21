"""Measure numerical, terrain and initial-state contrasts on identical spatial support."""

from collections import defaultdict
import numpy as np

from research.floodwarp_methods_paper.metrics import load, csv_write
from validation.hancock_long_storm_review.accuracy import read_fields


def fields(*, item):
    rec, row, folder = item
    data = read_fields(row=row, directory=folder, support=10.0)
    return dict(
        depth=data["depth"],
        maximum_depth=data["maximum"],
        discharge=np.array([r["outlet_m3s"] for r in row["samples"]]),
        inundation=data["snapshots"] > 0.01,
    )


def norm(array):
    return float(np.sqrt(np.mean(array**2)))


def iou(a, b):
    union = np.count_nonzero(a | b)
    return 1 - float(np.count_nonzero(a & b) / union) if union else 0.0


def compare(*, study):
    allrows = load(
        study=study, stages=("resolution", "long-storms", "terrain-initial-controls")
    )
    cases = {
        rec["spec"]["case"]
        for rec, row, folder in allrows
        if rec["stage"] == "resolution"
    }
    items = defaultdict(dict)
    for item in allrows:
        rec, row, folder = item
        s = rec["spec"]
        if s["case"] not in cases or s.get("reference", False):
            continue
        role = (
            "controlled" if rec["stage"] == "terrain-initial-controls" else "original"
        )
        items[s["case"], s["method"]][s["dx"], s["dem"], role] = item
    results = []
    for (case, method), runs in sorted(items.items()):
        values = {k: fields(item=v) for k, v in runs.items()}
        changes = {}
        rows = []
        for dx in (1, 5, 10):
            fixed = (dx, 10, "original")
            sampled = (dx, dx, "original")
            control = (
                (dx, 10, "controlled") if (dx, 10, "controlled") in runs else fixed
            )
            rec, raw, _ = runs[control]
            second, other, _ = runs[sampled]
            for key in (
                "roughness",
                "initial_depth_sha256",
                "initial_hu_sha256",
                "rain_mask_sha256",
                "rain_times_s",
                "rain_rates_ms",
                "infiltration_sha256",
                "boundary_x_sha256",
                "boundary_y_sha256",
                "stage_m",
            ):
                if raw["forcing"][key] != other["forcing"][key]:
                    raise ArithmeticError(
                        f"Geometry attribution changes {key} for {case}, {method}, {dx} m"
                    )
            if (
                raw["forcing"]["terrain_sha256"]
                != runs[fixed][1]["forcing"]["terrain_sha256"]
            ):
                raise ArithmeticError("Controlled bed mismatch")
            a = values[fixed]
            b = values[sampled]
            c = values[control]
            fine = values[1, 10, "original"]
            for metric in ("depth", "maximum_depth", "discharge"):
                difference = c[metric] - b[metric]
                if dx == 1:
                    changes[metric] = difference
                rows.append(
                    dict(
                        case=case,
                        family=rec["spec"]["landform"],
                        method=method,
                        dx=dx,
                        observable=metric,
                        unit="m3/s" if metric == "discharge" else "m",
                        fixed_geometry_numerical_rmse=norm(a[metric] - fine[metric]),
                        matched_initial_bed_effect_rmse=norm(difference),
                        initial_map_effect_rmse=norm(a[metric] - c[metric]),
                        signed_mean_bed_effect=float(np.mean(difference)),
                        bed_grid_interaction_rmse=norm(difference - changes[metric]),
                        initial_volume_difference_m3=runs[fixed][1]["initial_m3"]
                        - raw["initial_m3"],
                        bed_id=rec["experiment_id"],
                        sampled_id=second["experiment_id"],
                        original_id=runs[fixed][0]["experiment_id"],
                        fine_fixed_id=runs[1, 10, "original"][0]["experiment_id"],
                    )
                )
            rows.append(
                dict(
                    case=case,
                    family=rec["spec"]["landform"],
                    method=method,
                    dx=dx,
                    observable="1cm_space_time_inundation",
                    unit="1-IoU",
                    fixed_geometry_numerical_rmse=iou(
                        a["inundation"], fine["inundation"]
                    ),
                    matched_initial_bed_effect_rmse=iou(
                        c["inundation"], b["inundation"]
                    ),
                    initial_map_effect_rmse=iou(a["inundation"], c["inundation"]),
                    signed_mean_bed_effect=None,
                    bed_grid_interaction_rmse=None,
                    initial_volume_difference_m3=runs[fixed][1]["initial_m3"]
                    - raw["initial_m3"],
                    bed_id=rec["experiment_id"],
                    sampled_id=second["experiment_id"],
                    original_id=runs[fixed][0]["experiment_id"],
                    fine_fixed_id=runs[1, 10, "original"][0]["experiment_id"],
                )
            )
        results.extend(rows)
    csv_write(
        file=study / "processed/comparisons/terrain_attribution.csv", rows=results
    )
    return len(results)
