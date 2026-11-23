"""Time-dependent depth, conserved momentum, inundation and front arrival on common spatial support."""

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import spatial


def times(*, folder):
    return sorted(
        int(file.stem) for file in folder.glob("*.npz") if file.stem.isdecimal()
    )


def summary(*, pair, reference, edges, threshold):
    a_name, a = pair
    b_name, b = reference
    if a["interval"] != b["interval"] or a["support"] != b["support"]:
        raise ValueError("Transient fields must use identical sampling supports")
    area = b["support"] ** 2
    rows = []
    first = {}
    peak = {}
    first_wet = {}
    meta = dict(
        case=a["case"],
        dx=a["dx"],
        ref_dx=b["dx"],
        dem=a["dem"],
        method=a["method"],
        reference_method=b["method"],
        cfl=a["cfl"],
        ref_cfl=b["cfl"],
        experiment_id=a_name.name,
        reference_id=b_name.name,
        interval_s=b["interval"],
        support_m=b["support"],
        threshold_m=threshold,
        common_end_s=edges[-1],
    )
    for t in edges:
        with np.load(a_name / f"{t}.npz") as model, np.load(b_name / f"{t}.npz") as ref:
            ha, hb = model["h"], ref["h"]
            for name, h in (("model", ha), ("ref", hb)):
                wet = h >= threshold
                if t == 0:
                    first[name] = np.full(h.shape, np.nan)
                    first_wet[name] = wet.copy()
                    peak[name] = h.copy()
                first[name][wet & ~np.isfinite(first[name]) & ~first_wet[name]] = t
                np.maximum(peak[name], h, out=peak[name])
            hu_error = model["hu"] - ref["hu"]
            hv_error = model["hv"] - ref["hv"]
            ww, wr = ha >= threshold, hb >= threshold
            union = ww | wr
            rows.append(
                dict(
                    meta,
                    t_s=t,
                    model_wet_m2=area * ww.sum(),
                    reference_wet_m2=area * wr.sum(),
                    iou=(ww & wr).sum() / union.sum() if union.any() else None,
                    depth_rmse_m=np.sqrt(np.mean((ha - hb) ** 2)),
                    momentum_vector_rmse_m2s=np.sqrt(
                        np.mean(hu_error**2 + hv_error**2)
                    ),
                    wet_union_momentum_vector_rmse_m2s=np.sqrt(
                        np.mean((hu_error**2 + hv_error**2)[union])
                    )
                    if union.any()
                    else None,
                )
            )
    delay = first["model"] - first["ref"]
    delta = peak["model"] - peak["ref"]
    row = dict(
        meta,
        co_arrival_cells=np.isfinite(delay).sum(),
        median_arrival_delay_s=spatial.median_finite(field=delay),
        median_absolute_arrival_difference_s=spatial.median_finite(field=abs(delay)),
        missed_arrivals=(
            (peak["model"] < threshold) & (peak["ref"] >= threshold)
        ).sum(),
        extra_arrivals=((peak["model"] >= threshold) & (peak["ref"] < threshold)).sum(),
        sampled_maximum_bias_m=np.mean(delta),
        sampled_maximum_rmse_m=np.sqrt(np.mean(delta**2)),
    )
    return rows, row, delay


def priority_ref(pair):
    spec = pair[1]
    return spec["dx"], -spec["cfl"]


def run(*, directory):
    config = json.loads((directory / "spatial_settings.json").read_text())
    groups = defaultdict(list)
    for file in sorted((directory / "transient").glob("*/spec.json")):
        spec = json.loads(file.read_text())
        groups[spec["case"]].append((file.parent, spec))
    series = []
    summaries = []
    maps = {}
    for case, items in groups.items():
        refs = [pair for pair in items if pair[1]["method"] == "adaptive"]
        if not refs:
            continue
        ref = min(refs, key=priority_ref)
        for pair in items:
            if ref == pair:
                continue
            edges = times(folder=pair[0])
            if (
                not edges
                or edges != times(folder=ref[0])
                or edges != list(range(0, edges[-1] + 1, config["interval_s"]))
            ):
                raise ValueError(
                    f"Incomplete transient archive for {case} {pair[0].name}"
                )
            rows, stats, delay = summary(
                pair=pair,
                reference=ref,
                edges=edges,
                threshold=config["arrival_depth_m"],
            )
            series.extend(rows)
            summaries.append(stats)
            if pair[1]["method"] in ("fv1", "hancock_adaptive", "hancock_dense"):
                maps[pair[0].name] = delay
    target = directory / "results"
    metrics.csv_write(file=target / "transient_sections.csv", rows=series)
    metrics.csv_write(file=target / "transient_arrival.csv", rows=summaries)
    np.savez_compressed(target / "transient_arrival.npz", **maps)
    runner.write_json(
        path=target / "transient_manifest.json",
        data={
            str(file.relative_to(directory)): runner.checksum(file)
            for file in sorted((directory / "transient").glob("*/*"))
            if file.is_file()
        },
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
