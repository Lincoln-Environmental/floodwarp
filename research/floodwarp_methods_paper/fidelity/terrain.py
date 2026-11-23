"""Conditioned flow-path diagnostics and fixed-grid contrasts on archived bed representations."""

from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from research.floodwarp_methods_paper import evidence, metrics
from research.floodwarp_methods_paper.fidelity.assessment import curve
from research.floodwarp_methods_paper.fidelity.hydrograph import evaluate
from floodwarp.terrain_advisor.diagnostics import drainage
import json


def paths(*, z, dx, threshold):
    valid = np.isfinite(z)
    filled, areas, _, dest = drainage(
        np.ascontiguousarray(z), valid, np.full_like(z, dx * dx), float(dx)
    )
    rec = dest.ravel()
    idx = np.flatnonzero((rec >= 0) & valid.ravel())
    next_cell = rec[idx]
    dr = next_cell // z.shape[1] - idx // z.shape[1]
    dc = next_cell % z.shape[1] - idx % z.shape[1]
    segment = dx * np.hypot(dr, dc)
    distance = np.zeros(z.size)
    counts = np.zeros(z.size)
    # In an acyclic D8 graph a downstream receiver has strictly larger contributing area.
    for i in np.argsort(areas.ravel()[idx])[::-1]:
        distance[idx[i]] = segment[i] + distance[next_cell[i]]
        counts[idx[i]] = 1 + counts[next_cell[i]]
    channel = areas.ravel()[idx] >= threshold
    grades = (filled.ravel()[idx] - filled.ravel()[next_cell]) / segment
    return dict(
        path_length_p90_m=np.percentile(distance[valid.ravel()], 90),
        path_cells_p90=np.percentile(counts[valid.ravel()], 90),
        channel_grade=np.median(grades[channel]) if channel.any() else None,
        diagonal_fraction=np.mean(((dr != 0) & (dc != 0))[channel])
        if channel.any()
        else None,
    )


def investigate(*, directory, cfg):
    study = directory.parent
    stats = json.loads((study / "processed/metrics/terrain.json").read_text())
    by_key = {}
    for r in evidence.records(study=study, stages=("synthetic",)):
        s = r["spec"]
        for dx in (1, 5, 10):
            key = s["landform"], s["seed"], dx
            m = stats[f"{key[0]}-{key[1]}-{dx}m"]["model"]
            with np.load(study / r["directory"] / f"terrain_{dx}m.npz") as a:
                extra = paths(
                    z=a["elevation_m"][::-1].copy(),
                    dx=dx,
                    threshold=m["channel_threshold_m2"],
                )
            by_key[key] = dict(
                landform=key[0],
                seed=key[1],
                dx=dx,
                relief_m=m["relief_m"],
                width_to_grid=m["width_to_grid"],
                storage_m3=m["depression_storage_m3"],
                valley_fraction=m["valley_fraction"],
                gradient=m["mean_slope"],
                **extra,
            )
    target = directory / "results"
    metrics.csv_write(
        file=target / "terrain_descriptors.csv", rows=list(by_key.values())
    )
    pairs = metrics.csv_read(target / "pairs.csv")
    out = []
    for k in cfg["reference_limits"]:
        joined = defaultdict(list)
        for r in pairs:
            if (r["cohort"], r["station"], r["first"], r["second"], r["metric"]) != (
                "complete-legacy",
                "outlet",
                "fv1",
                "hancock_dense",
                k,
            ):
                continue
            family, seed = r["terrain_id"].split("-")
            joined[family, int(seed), int(float(r["dx"]))].append(
                float(r["absolute_error_reduction"])
            )
        for feature in (
            "path_cells_p90",
            "path_length_p90_m",
            "channel_grade",
            "diagonal_fraction",
            "width_to_grid",
            "storage_m3",
            "valley_fraction",
            "relief_m",
            "gradient",
        ):
            x, y = [], []
            for key, v in joined.items():
                data = by_key[key][feature]
                if data is not None:
                    x.append(data)
                    y.append(np.median(v))
            rho = (
                float(spearmanr(x, y).statistic)
                if len(x) >= 4 and np.ptp(x) > 0 and np.ptp(y) > 0
                else None
            )
            out.append(
                dict(
                    metric=k,
                    descriptor=feature,
                    independent_terrains=len(x),
                    spearman_rho=rho,
                )
            )
    metrics.csv_write(file=target / "terrain_associations.csv", rows=out)
    runs = defaultdict(dict)
    for rec in evidence.records(study=study, stages=("resolution",)):
        s = rec["spec"]
        d = json.loads((study / rec["directory"] / "result.json").read_text())
        runs[s["case"], s["dx"]][s["method"], s["dem"]] = dict(
            d=d, id=rec["experiment_id"]
        )
    contrasts = []
    for (case, dx), group in runs.items():
        if dx >= 10:
            continue
        for method in ("hancock_adaptive", "fv1", "adaptive"):
            sampled = group.get((method, dx))
            fixed = group.get((method, 10))
            first = group.get(("fv1", dx))
            if not (sampled and fixed and first):
                continue
            for station in ("outlet", "section"):
                t, sq = curve(it=sampled, station=station)
                _, fq = curve(it=fixed, station=station)
                _, aq = curve(it=first, station=station)
                terrain = evaluate(edges=t, model=fq, reference=sq, config=cfg)
                solver = evaluate(edges=t, model=aq, reference=sq, config=cfg)
                for k in cfg["reference_limits"]:
                    if k in terrain and k in solver:
                        contrasts.append(
                            dict(
                                case=case,
                                dx=dx,
                                method=method,
                                station=station,
                                objective=k,
                                bed_difference=terrain[k],
                                formulation_difference=solver[k],
                                sampled_id=sampled["id"],
                                fixed_id=fixed["id"],
                                fv1_id=first["id"],
                                initial_storage_difference_m3=fixed["d"]["initial_m3"]
                                - sampled["d"]["initial_m3"],
                            )
                        )
    metrics.csv_write(file=target / "bed_contrasts.csv", rows=contrasts)
