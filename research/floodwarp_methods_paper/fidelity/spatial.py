"""Common-support maps and area histories using retained 15-minute depth snapshots."""

import numpy as np
from research.floodwarp_methods_paper import metrics
from validation.hancock_long_storm_review.accuracy import read_fields


def arrival(*, h, t, threshold):
    wet = h >= threshold
    out = t[np.argmax(wet, axis=0)].astype(float)
    out[(~wet.any(axis=0)) | wet[0]] = np.nan
    return out


def peak_time(*, h, t, threshold):
    peak = h.max(axis=0)
    top = h >= 0.99 * peak
    i = h.argmax(axis=0)
    result = t[i].astype(float)
    result[
        (peak < threshold) | (i == 0) | (i == len(t) - 1) | (top.sum(axis=0) > 2)
    ] = np.nan
    return result


def median_finite(*, field):
    finite = field[np.isfinite(field)]
    return np.median(finite) if finite.size else None


def run(*, groups, target):
    rows, areas = [], []
    maps = {}
    for (cohort, case, rep), items in groups.items():
        if cohort != "legacy":
            continue
        native = [i for i in items if "samples" in i["d"]]
        refs = [i for i in native if i["s"]["method"] == "adaptive"]
        ref = min(refs, key=spacing)
        fine = read_fields(row=ref["d"], directory=ref["path"], support=10.0)
        for i in native:
            if i["s"]["dx"] != 10:
                continue
            vals = read_fields(row=i["d"], directory=i["path"], support=10.0)
            t = np.array([f["t_s"] for f in i["d"]["samples"]])
            for threshold in (0.01, 0.05, 0.1):
                a = vals["snapshots"]
                b = fine["snapshots"]
                delta = arrival(h=a, t=t, threshold=threshold) - arrival(
                    h=b, t=t, threshold=threshold
                )
                peak = peak_time(h=a, t=t, threshold=threshold) - peak_time(
                    h=b, t=t, threshold=threshold
                )
                meta = dict(
                    case=case,
                    method=i["s"]["method"],
                    experiment_id=i["id"],
                    reference_id=ref["id"],
                    threshold_m=threshold,
                    dx=10,
                    reference_dx=spacing(ref),
                )
                rows.append(
                    dict(
                        meta,
                        arrival_delay_s=median_finite(field=delta),
                        arrival_mae_s=median_finite(field=abs(delta)),
                        time_to_max_delay_s=median_finite(field=peak),
                        arrival_assessable_cells=np.isfinite(delta).sum(),
                        peak_assessable_cells=np.isfinite(peak).sum(),
                        only_reference_wetted=(
                            (b.max(axis=0) >= threshold) & (a.max(axis=0) < threshold)
                        ).sum(),
                        only_model_wetted=(
                            (a.max(axis=0) >= threshold) & (b.max(axis=0) < threshold)
                        ).sum(),
                        maximum_bias_m=float(
                            np.mean(vals["maximum"] - fine["maximum"])
                        ),
                        final_depth_rmse_m=float(
                            np.mean((vals["depth"] - fine["depth"]) ** 2) ** 0.5
                        ),
                    )
                )
                for ts, aa, bb in zip(t, a >= threshold, b >= threshold, strict=True):
                    union = (aa | bb).sum()
                    areas.append(
                        dict(
                            meta,
                            t_s=ts,
                            model_wet_m2=aa.sum() * 100,
                            reference_wet_m2=bb.sum() * 100,
                            iou=(aa & bb).sum() / union if union else None,
                        )
                    )
                if (
                    case
                    in (
                        "incised-18217-moderate",
                        "floodplain-18217-intense",
                        "rolling-18217-back",
                    )
                    and threshold == 0.01
                ):
                    maps[case + "_" + i["s"]["method"]] = delta
    metrics.csv_write(file=target / "spatial.csv", rows=rows)
    metrics.csv_write(file=target / "inundation_history.csv", rows=areas)
    np.savez_compressed(target / "arrival_maps.npz", **maps)


def spacing(i):
    return i["s"]["dx"]
