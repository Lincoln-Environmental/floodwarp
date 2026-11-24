"""Interpretable nearest-terrain prediction with empirical feasibility and abstention."""

import json
from pathlib import Path

import numpy as np


FEATURES = (
    "mean_slope",
    "slope_p95",
    "local_relief_p50_m",
    "relief_m",
    "valley_fraction",
    "channel_density_m_per_m2",
    "storage_per_area",
    "aggregation_rmse_m",
    "connectivity_change",
    "rain_mm",
    "peak_mm_h",
    "rain_duration_h",
    "forecast_duration_h",
    "rain_centroid_h",
    "spatial_mean_multiplier",
    "manning_n",
    "initial_mean_m",
    "infiltration_mean_mm_h",
    "downstream_stage_m",
    "log_cells",
    "log_area",
)
CANDIDATES = (
    "li",
    "fv1",
    "low_froude_hll",
    "euler_mc",
    "hancock_dense",
    "hancock_adaptive",
    "hancock_auto",
    "adaptive",
)


def load_model(*, path=None):
    location = (
        Path(path) if path is not None else Path(__file__).with_name("model.json")
    )
    if not location.is_file():
        raise FileNotFoundError(
            "A benchmark-trained model.json is required. Run advisor-train or supply its path."
        )
    return json.loads(location.read_text())


def descriptors(*, terrain, hydraulic=None):
    m = terrain["model"]
    c = terrain["comparison"]
    values = {k: float(m[k]) for k in FEATURES[:6]}
    values.update(
        storage_per_area=float(m["depression_storage_m3"] / m["area_m2"]),
        aggregation_rmse_m=float(c["elevation_rmse_m"]),
        connectivity_change=float(c["changed_outlet_fraction"]),
        area_m2=float(m["area_m2"]),
        cells=m["cells"],
        log_cells=float(np.log(m["cells"])),
        log_area=float(np.log(m["area_m2"])),
        critical_connection_lost=(
            bool(c["native_connected"] and not c["represented_connected"])
            if c.get("native_connected") is not None and c.get("represented_connected") is not None else None
        ),
        lost_channel_fraction=c.get("channel_loss_fraction") or 0.0,
        barrier_lowering_m=float(
            c.get("max_local_barrier_lowering_m", c.get("barrier_lowering_p95_m", 0))
        ),
        rows=m["rows"],
        columns=m["columns"],
        resolution=float(m["dx_m"]),
        native_resolution=float(terrain["native"]["dx_m"]),
    )
    if hydraulic:
        regime = hydraulic.get("expected_flow_regime")
        if regime is not None:
            regime = str(regime).casefold()
            if regime == "transcritical":
                regime = "mixed"
            if regime not in ("subcritical", "mixed", "supercritical"):
                raise ValueError(
                    "Expected flow regime must be subcritical, mixed or supercritical"
                )
            values["expected_flow_regime"] = regime
        for key, value in hydraulic.items():
            if (
                key in (*FEATURES, "maximum_barrier_lowering_m")
                and key not in ("log_area", "log_cells")
                and value is not None
            ):
                if not np.isfinite(value):
                    raise ValueError("Hydraulic inputs must be finite")
                values[key] = float(value)
    return values


def scales(*, rows):
    result = {}
    for f in FEATURES:
        v = [r["features"][f] for r in rows if r["features"].get(f) is not None]
        if v:
            a = np.asarray(v)
            result[f] = dict(
                low=float(a.min()),
                high=float(a.max()),
                scale=float(max(np.ptp(a), 1e-12)),
            )
    return result


def distance(*, a, b, scale, slope_only=False):
    subset = ("mean_slope",) if slope_only else FEATURES
    shared = [
        f
        for f in subset
        if f in scale
        and a.get(f) is not None
        and b.get(f) is not None
        and scale[f]["high"] > scale[f]["low"]
    ]
    if not shared:
        return float("inf")
    return float(
        np.sqrt(np.mean([((a[f] - b[f]) / scale[f]["scale"]) ** 2 for f in shared]))
    )


def ordering(pair):
    return pair[0]


def nearest(*, rows, query, model, k, slope_only=False):
    ranked = sorted(
        (
            (
                distance(
                    a=query,
                    b=r["features"],
                    scale=model["scales"],
                    slope_only=slope_only,
                ),
                r,
            )
            for r in rows
        ),
        key=ordering,
    )
    selected = []
    seen = set()
    for d, r in ranked:
        if r["terrain_id"] in seen:
            continue
        seen.add(r["terrain_id"])
        selected.append((d, r))
        if len(selected) >= k:
            break
    return selected


def configuration(method):
    labels = {
        "fv1": ("FV1", "contact HLL", "sparse"),
        "low_froude_hll": ("FV1", "low-Froude HLL", "sparse"),
        "euler_mc": ("MUSCL–Euler", "MC", "dense"),
        "hancock_dense": ("Hancock", "adaptive", "dense"),
        "hancock_auto": ("Hancock", "adaptive", "auto"),
        "hancock_adaptive": ("Hancock", "adaptive", "tiles"),
        "adaptive": ("FV2 RK2", "adaptive", "dense"),
        "li": ("LI", "native", "native"),
    }
    formulation, limiter, execution = labels[method]
    return dict(formulation=formulation, limiter_flux=limiter, execution_mode=execution)


def runtime_projection(*, row, features, cell_power, duration_power):
    """Local timing estimate; zero exponents reproduce the unscaled measured neighbour."""
    value = row["runtime_s"]
    for key, power in (("cells", cell_power), ("forecast_duration_h", duration_power)):
        if power and features.get(key) and row["features"].get(key):
            value *= (features[key] / row["features"][key]) ** power
    return value


def candidate_cost(row):
    return row["runtime_median_s"]


def terrain_requires_refinement(*, features, objective):
    from floodwarp.solver_advisor.fidelity import METRICS

    return bool(
        features.get("critical_connection_lost")
        or (
            features.get("maximum_barrier_lowering_m") is not None
            and features.get("barrier_lowering_m", 0.0)
            > features["maximum_barrier_lowering_m"]
        )
        or features.get("lost_channel_fraction", 0) > 0
        and objective
        in (
            "inundation",
            "discharge",
            "peak_timing",
            "peak_discharge",
            *METRICS,
            "relative_hydrograph_rmse",
            "event_discharge_error",
            "final_inundation_5cm",
            "final_inundation_10cm",
            "maximum_inundation_5cm",
            "maximum_inundation_10cm",
        )
    )


def recommend(
    *,
    model,
    features,
    objective,
    tolerance=None,
    hardware=None,
    memory_mib=None,
    slope_only=False,
):
    """Choose from empirically qualified solver candidates on one requested grid."""
    from floodwarp.solver_advisor.fidelity import METRICS, with_evidence

    objective = {
        "full_hydrograph": "hydrograph_shape",
        "flood_arrival_time": "spatial_arrival",
        "transient_velocity": "transient_momentum",
        "maximum_flood_depth": "maximum_depth",
        "inundation_extent": "final_inundation_5cm",
        "thin_film_extent_1cm": "inundation",
        "outlet_discharge": model.get("outlet_discharge_objective", "discharge"),
        "computational_runtime": "runtime",
    }.get(objective, objective)
    if objective in METRICS and "fidelity_validation" not in model:
        model = with_evidence(model=model)
    practical_extent = objective in (
        "final_inundation_5cm",
        "final_inundation_10cm",
        "maximum_inundation_5cm",
        "maximum_inundation_10cm",
    )
    if objective not in (*model["tolerances"], "runtime") and not practical_extent:
        raise ValueError("Unknown hydraulic objective")
    if tolerance is None and objective != "runtime":
        tolerance = model["tolerances"].get(objective)
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Tolerance must be finite and nonnegative")
    result = dict(
        status="insufficient_evidence",
        method=None,
        resolution_m=features["resolution"],
        objective=objective,
        tolerance=tolerance,
        hardware=model["hardware"],
        candidates=[],
        warnings=[],
        source_revision=model["revision"],
    )
    if objective == "inundation" and "inundation_target" in model:
        result["target"] = model["inundation_target"]
    if terrain_requires_refinement(features=features, objective=objective):
        result.update(
            status="refine_terrain",
            reason="The represented DEM loses a required hydraulic connection or exceeds the specified barrier tolerance.",
        )
        return result
    if ("native_objective_spacings" in model and objective != "runtime"
        and features["resolution"] not in model["native_objective_spacings"].get(objective, ())):
        result["reason"] = "This native-grid objective lacks independent validation at the requested spacing."
        return result
    if practical_extent and (
        tolerance is None
        or features["resolution"]
        not in model.get("extent_validation", {}).get(objective, [])
    ):
        result["reason"] = (
            "This practical inundation threshold and endpoint lack independent validation at the requested spacing."
        )
        return result
    if objective in METRICS and features["resolution"] not in model[
        "fidelity_validation"
    ].get(objective, []):
        result["reason"] = (
            "This hydraulic objective lacks independent validation at the requested spacing."
        )
        return result
    if (
        "validation_spacings" in model
        and features["resolution"] not in model["validation_spacings"]
    ):
        result["warnings"].append(
            "This spacing lacks independent terrain groups for predictive validation. Its measured cost–accuracy comparisons remain exploratory."
        )
        return result
    if hardware and hardware.casefold() not in model["hardware"].casefold():
        result["warnings"].append(
            "No matching timing calibration on the requested GPU."
        )
        return result
    comparable = [
        r
        for r in model["training"]
        if r["features"]["resolution"] == features["resolution"]
    ]
    if not comparable:
        result["warnings"].append("No measured configurations at this grid spacing.")
        return result
    if (
        not min(r["features"]["area_m2"] for r in comparable) * (1 - 1e-10)
        <= features["area_m2"]
        <= max(r["features"]["area_m2"] for r in comparable) * (1 + 1e-10)
    ):
        result["warnings"].append(
            "Domain area lies outside the accuracy-training range at this spacing."
        )
        return result
    for key in ("rows", "columns", "native_resolution"):
        values = {r["features"][key] for r in comparable if key in r["features"]}
        if key in features and values and features[key] not in values:
            result["warnings"].append(
                f"The benchmark does not cover the requested {key}."
            )
            return result
    # Restrict empirical runtimes to measured domain-size neighbours. Never use a
    # 96-cell or metre-grid timing to fill a missing million-cell observation.
    if "size_factor" in model and "cells" in features:
        factor = model["size_factor"]
        comparable = [
            r
            for r in comparable
            if features["cells"] / factor
            <= r["features"]["cells"]
            <= features["cells"] * factor
        ]
        if not comparable:
            result["warnings"].append(
                "No comparable measured cell count at the requested resolution."
            )
            return result
    for key, scale in model["scales"].items():
        if key not in features or scale["high"] != scale["low"]:
            continue
        if not np.isclose(features[key], scale["low"], rtol=1e-9, atol=1e-12):
            result["warnings"].append(
                f"The benchmark contains only one observed value for {key}."
            )
            return result
    missing = [f for f in FEATURES if f not in features]
    if missing:
        result["warnings"].append(
            "Missing hydraulic descriptors broaden uncertainty: " + ", ".join(missing)
        )
    prefix = "slope_" if slope_only and "slope_k" in model else ""
    for method in CANDIDATES:
        pool = [
            r
            for r in comparable
            if r["method"] == method
            and (
                "expected_flow_regime" not in features
                or r.get("sampled_regime") == features["expected_flow_regime"]
            )
        ]
        hydraulic = (
            pool
            if objective == "runtime"
            else [r for r in pool if r["qualified"].get(objective, False)]
        )
        neighbors = nearest(
            rows=hydraulic,
            query=features,
            model=model,
            k=model[prefix + "k"],
            slope_only=slope_only,
        )
        if len(neighbors) < 2 or neighbors[0][0] > model[prefix + "distance_limit"]:
            continue
        ns = [r for _, r in neighbors]
        cost_neighbours = nearest(
            rows=pool,
            query=features,
            model=model,
            k=model[prefix + "k"],
            slope_only=slope_only,
        )
        times = [r for _, r in cost_neighbours]
        runtime = np.array(
            [
                runtime_projection(
                    row=r,
                    features=features,
                    cell_power=model.get("cell_power", 0.0),
                    duration_power=model.get("duration_power", 0.0),
                )
                for r in times
            ]
        )
        memory = float(
            max(
                r["memory_mib"]
                * max(1.0, features.get("cells", 1) / r["features"].get("cells", 1))
                for r in times
            )
        )
        bounds = [None, None]
        value = None
        if objective != "runtime":
            errors = [r["errors"][objective] for r in ns]
            value = float(np.median(errors))
            # Wider bounds for partially specified events. These are empirical envelopes.
            bounds = [
                float(min(errors)),
                float(max(errors))
                if missing
                else float(np.quantile(errors, model["quantile"])),
            ]
            pad = model.get("interval_padding", {}).get(objective, 0.0)
            bounds = [max(0.0, bounds[0] - pad), bounds[1] + pad]
        feasible = (objective == "runtime" or bounds[1] <= tolerance) and (
            memory_mib is None or memory <= memory_mib
        )
        result["candidates"].append(
            dict(
                method=method,
                **configuration(method),
                runtime_median_s=float(np.median(runtime)),
                runtime_range_s=[float(runtime.min()), float(runtime.max())],
                error_median=value,
                error_range=bounds,
                memory_upper_mib=memory,
                feasible=feasible,
                nearest_distance=neighbors[0][0],
                terrain_count=len(ns),
                evidence_ids=[r["run_id"] for r in ns],
                timing_evidence_ids=[r["run_id"] for r in times],
            )
        )
    result["candidates"].sort(key=candidate_cost)
    accepted = [r for r in result["candidates"] if r["feasible"]]
    if accepted:
        selected = accepted[0]
        result.update(
            status="recommended",
            method=selected["method"],
            **configuration(selected["method"]),
            runtime_s=selected["runtime_median_s"],
            error_range=selected["error_range"],
            confidence="empirical neighbourhood; see held-out coverage",
            reason=f"Lowest predicted hydraulic runtime among {len(accepted)} supported candidates satisfying the stated constraints.",
        )
    elif result["candidates"]:
        result.update(
            status="no_feasible_candidate",
            reason="No candidate meets the specified error and memory constraints in this benchmark neighbourhood.",
        )
    else:
        result["warnings"].append(
            "Too few independent qualified terrain neighbours or the nearest case exceeds validation support."
        )
    return result


def recommend_dem(
    *,
    dem,
    resolution,
    objective,
    output,
    model=None,
    tolerance=None,
    hydraulic=None,
    hardware=None,
    memory_mib=None,
    compare_resolutions=None,
    control=None,
):
    from floodwarp.terrain_advisor import analyse_dem

    model = load_model() if model is None else model
    folder = Path(output)
    folder.mkdir(exist_ok=True, parents=True)
    results = []
    if objective == "inundation":
        objective = "final_inundation_5cm"
    grids = list(dict.fromkeys((resolution, *(compare_resolutions or ()))))
    for dx in grids:
        stats = analyse_dem(
            dem=dem, resolution=dx, output=folder / ("terrain_" + str(dx) + "m"), control=control
        )
        if dx == resolution and compare_resolutions is None:
            observed = {
                r["features"]["resolution"]
                for r in model["training"]
                if np.isclose(r["features"]["area_m2"], stats["model"]["area_m2"])
                and r["features"]["resolution"] >= stats["native"]["dx_m"]
            }
            grids.extend(
                sorted(
                    observed.intersection(model.get("validation_spacings", observed))
                    - set(grids)
                )
            )
        rec = recommend(
            model=model,
            features=descriptors(terrain=stats, hydraulic=hydraulic),
            objective=objective,
            tolerance=tolerance,
            hardware=hardware,
            memory_mib=memory_mib,
        )
        change = stats["comparison"]
        rec["terrain_diagnostics"] = change
        if control is None:
            rec["warnings"].append("The DEM has no specified control window. Connectivity at critical openings is unassessed.")
        if change["channel_loss_fraction"] and change["channel_loss_fraction"] > 0:
            rec["warnings"].append(
                f"{100 * change['channel_loss_fraction']:.1f}% of native channel pixels lack a nearby model-grid drainage path. Compare a finer DEM."
            )
        results.append(rec)
    requested = results[0]
    supported = model.get("native_cross_grid_validation", {}).get(
        requested["objective"], []
    )
    feasible = [
        r
        for r in results
        if r["status"] == "recommended"
        and (
            r is requested
            or any(
                resolution in pair and r["resolution_m"] in pair for pair in supported
            )
        )
    ]
    best = min(feasible, key=forecast_time) if feasible else requested
    if len(results) > 1 and not supported:
        requested["warnings"].append(
            "A practical resolution change requires independent native-metre cross-grid validation for this objective."
        )
    out = dict(
        selected=best,
        comparisons=results,
        applied_pilot=False,
        requested_resolution_m=resolution,
        requested_status=requested["status"],
    )
    (folder / "recommendation.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Solver recommendation",
        f"Requested grid **{resolution:g} m**. Assessment **{requested['status']}** for {objective}.",
        f"Decision **{best['status']}** at {best['resolution_m']:g} m for {objective}.",
        best.get("reason", "Benchmark support is insufficient."),
        f"Hardware basis: {model['hardware']}",
    ]
    if best["method"]:
        lines.append(
            f"Selected **{best['method']}**, with predicted hydraulic runtime {best['runtime_s']:.3g} s. The empirical error interval is {best['error_range']}."
        )
    if best["resolution_m"] != resolution:
        lines.append(
            f"The selected {best['resolution_m']:g} m grid is an alternative to the requested {resolution:g} m grid."
        )
        lines.extend(requested["warnings"])
    lines += best["warnings"]
    lines.append(
        "The accompanying GeoTIFFs show terrain aggregation, storage and potential connectivity changes. The comparison list reports every evaluated formulation and tested resolution."
    )
    (folder / "recommendation.md").write_text("\n\n".join(lines) + "\n")
    return out


def forecast_time(row):
    return row["runtime_s"]
