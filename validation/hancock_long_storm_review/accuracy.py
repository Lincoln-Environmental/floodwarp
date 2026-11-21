"""Hydraulic differences on equal areas and identical sampling windows."""

import csv
import json
from pathlib import Path

import numpy as np

from validation.hancock_long_storm_review.dataset import records
from scipy.ndimage import label


def block_mean(*, values, factor):
    ny, nx = values.shape[-2:]
    return values.reshape(
        *values.shape[:-2], ny // factor, factor, nx // factor, factor
    ).mean(axis=(-1, -3))


def read_fields(*, row, directory, support):
    ratio = support / row["dx"]
    if not ratio.is_integer():
        raise ValueError("Evaluation grid must nest all native cells")
    with np.load(directory / (row["name"] + ".npz")) as data:
        return {
            key: block_mean(values=data[key], factor=round(ratio))
            for key in ("depth", "maximum", "snapshots")
        }


def rmse(*, difference):
    return float(np.sqrt(np.mean(difference**2)))


def outlet_connection(*, masks):
    connected = np.zeros_like(masks)
    for j, wet in enumerate(masks):
        components, _ = label(wet)
        outlet_labels = np.unique(components[:, -1])
        connected[j] = np.isin(components, outlet_labels[outlet_labels != 0])
    return connected


def compare(*, row, ref, data, reference, support, thresholds):
    h, hr = data["depth"], reference["depth"]
    wet = (hr > 0.01) | (h > 0.01)
    delta = h - hr
    series = np.array([a["outlet_m3s"] for a in row["samples"]])
    truth = np.array([a["outlet_m3s"] for a in ref["samples"]])
    times = np.array([a["t_s"] for a in row["samples"]])
    result = dict(
        run=row["name"],
        reference=ref["name"],
        support_m=support,
        method=row["method"],
        mode=row["mode"],
        group=row["group"],
        dx=row["dx"],
        dem_m=row["terrain_dx"],
        storm=row["storm"],
        landform=row["landform"],
        runtime_s=row["hydraulic_s"],
        depth_rmse_m=rmse(difference=delta),
        wet_depth_rmse_m=rmse(difference=delta[wet]) if wet.any() else None,
        max_depth_rmse_m=rmse(difference=data["maximum"] - reference["maximum"]),
        depth_relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(hr))
        if hr.any()
        else None,
        outlet_rmse_m3s=rmse(difference=series - truth),
        peak_m3s=series.max(),
        peak_error_m3s=series.max() - truth.max(),
        peak_end_time_s=times[series.argmax()],
        peak_time_error_s=times[series.argmax()] - times[truth.argmax()],
        outlet_volume_error_m3=float((series - truth).sum() * row["sample_seconds"]),
        internal_rmse_m3s=rmse(
            difference=np.array([a["section_m3s"] for a in row["samples"]])
            - np.array([a["section_m3s"] for a in ref["samples"]])
        ),
        recession_rmse_m3s=rmse(difference=(series - truth)[times > 21600]),
        outlet_relative_rmse=float(rmse(difference=series - truth) / max(abs(truth)))
        if truth.any()
        else None,
    )
    for threshold in thresholds:
        a, b = data["snapshots"] > threshold, reference["snapshots"] > threshold
        union = a | b
        tag = str(round(threshold * 100)) + "cm"
        result["space_time_iou_" + tag] = (
            float((a & b).sum() / union.sum()) if union.any() else 1.0
        )
        result["final_iou_" + tag] = (
            float((a[-1] & b[-1]).sum() / union[-1].sum()) if union[-1].any() else 1.0
        )
        result["final_area_error_m2_" + tag] = (
            int(a[-1].sum()) - int(b[-1].sum())
        ) * support**2
        shared = a.any(axis=0) & b.any(axis=0)
        for name, aa, bb in (("arrival", a, b), ("last_wet", a[::-1], b[::-1])):
            result[name + "_rmse_s_" + tag] = (
                rmse(difference=(aa.argmax(axis=0) - bb.argmax(axis=0))[shared])
                * row["sample_seconds"]
                if shared.any()
                else None
            )
        result["component_error_" + tag] = label(a[-1])[1] - label(b[-1])[1]
        connected = outlet_connection(masks=a)
        connected_ref = outlet_connection(masks=b)
        combined = connected | connected_ref
        result["outlet_connected_iou_" + tag] = (
            float((connected & connected_ref).sum() / combined.sum())
            if combined.any()
            else 1.0
        )
        area_delta = (
            connected.sum(axis=(1, 2)) - connected_ref.sum(axis=(1, 2))
        ) * support**2
        result["outlet_connected_area_rmse_m2_" + tag] = rmse(difference=area_delta)
        result["outlet_connected_final_area_error_m2_" + tag] = int(area_delta[-1])

    return result


def large_parity(*, runs):
    comparisons = []
    state_keys = (
        "wet_fraction",
        "wet_cm_fraction",
        "section_m3s",
        "outlet_m3s",
        "rain_m3",
        "loss_m3",
        "storage_m3",
        "steps",
        "dt_min_s",
        "dt_mean_s",
        "dt_max_s",
        "rejected",
        "max_speed_ms",
        "fallback_cells_last_trial",
    )
    for dense in runs:
        if dense["group"] not in ("large", "terrain_large") or dense["mode"] != "dense":
            continue
        for other in runs:
            if other["method"] != "hancock" or other["mode"] not in ("sparse", "tiles"):
                continue
            identifiers = ("group", "landform", "storm", "dimensions_m", "dx", "hours")
            if any(other[key] != dense[key] for key in identifiers):
                continue
            differing = set()
            for a, b in zip(other["samples"], dense["samples"], strict=True):
                differing.update(key for key in state_keys if a[key] != b[key])
            comparisons.append(
                dict(
                    run=other["name"],
                    dense=dense["name"],
                    samples=len(dense["samples"]),
                    matched=not differing,
                    differing=sorted(differing),
                )
            )
    return comparisons


def assess(
    *, review, fields, support, thresholds, depth_tol, depth_relative_tol, outlet_tol
):
    rows = records(review=review)
    (review / "large_parity.json").write_text(
        json.dumps(large_parity(runs=rows), indent=2) + "\n"
    )
    out = []
    parity = []
    uncertainty = []
    for land, storm in sorted({(r["landform"], r["storm"]) for r in rows}):
        same = [
            r
            for r in rows
            if r["landform"] == land
            and r["storm"] == storm
            and r["group"] not in ("large", "terrain_large")
        ]
        for representation in ("sampled", "fixed"):
            subset = [
                r
                for r in same
                if (
                    r["terrain_dx"] == r["dx"]
                    if representation == "sampled"
                    else r["terrain_dx"] == 10
                )
            ]
            fine = [r for r in subset if r["method"] == "fv2" and r["dx"] < 10]
            if not fine:
                continue
            fine.sort(key=grid_spacing)
            reference = read_fields(row=fine[0], directory=fields, support=support)
            secondary = next((r for r in fine if r["dx"] > fine[0]["dx"]), None)
            for row in subset:
                record = compare(
                    row=row,
                    ref=fine[0],
                    data=read_fields(row=row, directory=fields, support=support),
                    reference=reference,
                    support=support,
                    thresholds=thresholds,
                )
                record["representation"] = representation
                out.append(record)
                if row is secondary:
                    rel = record["depth_relative_l2"]
                    hydro = record["outlet_relative_rmse"]
                    depth_resolved = record["depth_rmse_m"] <= depth_tol or (
                        rel is not None and rel <= depth_relative_tol
                    )
                    discharge_resolved = (
                        hydro <= outlet_tol
                        if hydro is not None
                        else record["outlet_rmse_m3s"] == 0.0
                    )
                    uncertainty.append(
                        dict(
                            landform=land,
                            storm=storm,
                            representation=representation,
                            coarser=secondary["name"],
                            finer=fine[0]["name"],
                            coarse_dx=secondary["dx"],
                            fine_dx=fine[0]["dx"],
                            depth_difference_m=record["depth_rmse_m"],
                            max_depth_difference_m=record["max_depth_rmse_m"],
                            outlet_difference_m3s=record["outlet_rmse_m3s"],
                            outlet_relative_rmse=hydro,
                            iou_1cm=record["space_time_iou_1cm"],
                            depth_resolved=depth_resolved,
                            discharge_resolved=discharge_resolved,
                            both_resolved=depth_resolved and discharge_resolved,
                        )
                    )

        dense = next(
            (
                r
                for r in same
                if r["group"] == "primary"
                and r["method"] == "hancock"
                and r["mode"] == "dense"
            ),
            None,
        )
        if dense:
            for r in same:
                if r["group"] != "primary" or r["method"] != "hancock" or r is dense:
                    continue
                with (
                    np.load(fields / (r["name"] + ".npz")) as a,
                    np.load(fields / (dense["name"] + ".npz")) as b,
                ):
                    exact = all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
                for a, b in zip(r["samples"], dense["samples"], strict=True):
                    for key in (
                        "section_m3s",
                        "outlet_m3s",
                        "storage_m3",
                        "rain_m3",
                        "loss_m3",
                        "steps",
                        "dt_min_s",
                        "dt_max_s",
                        "rejected",
                        "fallback_cells_last_trial",
                    ):
                        exact &= a[key] == b[key]
                parity.append(dict(run=r["name"], matched=bool(exact)))
    if out:
        with (review / "accuracy.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out[0].keys())
            writer.writeheader()
            writer.writerows(out)
    if uncertainty:
        with (review / "reference_uncertainty.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=uncertainty[0])
            writer.writeheader()
            writer.writerows(uncertainty)
    (review / "event_parity.json").write_text(json.dumps(parity, indent=2) + "\n")


def grid_spacing(row):
    return row["dx"]


if __name__ == "__main__":
    review = Path(__file__).parent
    fields = Path("outputs/hancock_long_storm_review/fields")
    support = 10.0
    thresholds = (0.01, 0.05, 0.10)
    depth_tol, depth_relative_tol, outlet_tol = 0.001, 0.02, 0.05
    assess(
        review=review,
        fields=fields,
        support=support,
        thresholds=thresholds,
        depth_tol=depth_tol,
        depth_relative_tol=depth_relative_tol,
        outlet_tol=outlet_tol,
    )
