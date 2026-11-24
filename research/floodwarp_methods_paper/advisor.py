"""Train, freeze and evaluate the solver advisor on whole independent terrain groups."""

from collections import defaultdict
import json

from affine import Affine
import numpy as np
import rasterio

from floodwarp.terrain_advisor import analyse_array
from floodwarp.terrain_advisor.raster import write_maps
from floodwarp.solver_advisor.model import (
    CANDIDATES,
    descriptors,
    scales,
    nearest,
    recommend,
)
from research.floodwarp_methods_paper.metrics import (
    OBJECTIVES,
    csv_read,
    csv_write,
    load,
)
from research.floodwarp_methods_paper.runner import (
    write_json,
    archived,
    checksum,
    digest,
)


def key(*, land, seed, dx):
    return f"{land}-{seed}-{float(dx):g}m"


def terrain_features(*, study, config, output):
    features = {}
    for rec, _, folder in load(study=study, stages=("synthetic",)):
        family, seed = rec["spec"]["landform"], rec["spec"]["seed"]
        with np.load(folder / "terrain_1m.npz") as data:
            z = data["elevation_m"][::-1].copy()
        for dx in config["storms"]["practical_grids"]:
            with np.load(folder / f"terrain_{dx:g}m.npz") as represented:
                bed = represented["elevation_m"][::-1].copy()
            stats, native, coarse = analyse_array(
                elevation=z,
                native_resolution=1.0,
                resolution=float(dx),
                represented_elevation=bed,
            )
            features[key(land=family, seed=seed, dx=dx)] = stats
            if seed == config["synthetic"]["seeds"][0]:
                # A local engineering CRS identifies this synthetic grid without attributing a real location.
                crs = rasterio.crs.CRS.from_wkt(
                    'LOCAL_CS["FloodWarp synthetic metre grid",LOCAL_DATUM["Arbitrary datum",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
                )
                transform = Affine.translation(0, z.shape[0]) @ Affine.scale(1, -1)
                folder_out = study / "terrain/diagnostics" / family / f"{dx}m"
                write_maps(
                    maps={
                        k: native[k]
                        for k in (
                            "elevation_change",
                            "lost_channel",
                            "within_cell_std",
                            "saddle_change",
                            "barrier_lowering",
                            "outlet_displacement",
                        )
                    },
                    folder=folder_out,
                    transform=transform,
                    crs=crs,
                )
                write_maps(
                    maps={
                        k: coarse[k]
                        for k in ("slope", "contributing_area", "fill_depth")
                    },
                    folder=folder_out,
                    transform=transform @ Affine.scale(dx),
                    crs=crs,
                )
                if dx == 1:
                    write_maps(
                        maps={"elevation": z},
                        folder=study / "terrain/surfaces" / family,
                        transform=transform,
                        crs=crs,
                    )
    write_json(path=output, data=features)
    return features


def samples(
    *, study, config, features, partitions=("development", "validation", "test")
):
    from research.floodwarp_methods_paper import evidence
    from research.floodwarp_methods_paper.qualification import advisor_data

    family = config["advisor"]["heldout_family"]
    raw = {
        rec["experiment_id"]: study / rec["directory"] / "result.json"
        for rec in evidence.records(study=study, stages=("long-storms", "resolution"))
        if advisor_data.group_is_selected(
            settings=rec["spec"], heldout_family=family, partitions=partitions
        )
    }
    selected = []
    for r in csv_read(study / "processed/metrics/storms.csv"):
        if (
            r["stage"] not in ("long-storms", "resolution")
            or r["reference_setup"] == "True"
            or r["representation"] != "sampled"
            or r["method"] not in CANDIDATES
        ):
            continue
        if r["experiment_id"] not in raw:
            continue
        actual = json.loads(raw[r["experiment_id"]].read_text())
        sampled = [
            frame["supercritical_wet_fraction"]
            for frame in actual["samples"]
            if frame["wet_froude_max"] is not None and frame["wet_froude_max"] > 0
        ]
        regime = "dry or stationary"
        if sampled:
            regime = (
                "subcritical"
                if max(sampled) == 0
                else "supercritical"
                if min(sampled) == 1
                else "mixed"
            )
        forcing = actual["forcing"]
        stats = features[key(land=r["landform"], seed=r["seed"], dx=float(r["dx"]))]
        area = stats["model"]["area_m2"]
        rates = np.asarray(forcing["rain_rates_ms"]) * 3.6e6
        # Physical inputs available before integration. The forcing module integrates the prescribed hyetograph.
        hours = np.asarray(forcing["rain_times_s"]) / 3600
        depth_mm = float(np.trapezoid(rates, hours))
        delta = np.diff(hours)
        centroid = (
            np.sum(
                delta
                * (
                    hours[:-1] * (2 * rates[:-1] + rates[1:])
                    + hours[1:] * (rates[:-1] + 2 * rates[1:])
                )
                / 6
            )
            / depth_mm
        )
        physical = dict(
            forecast_duration_h=float(r["hours"]),
            rain_centroid_h=float(centroid),
            spatial_mean_multiplier=actual["expected_rain_m3"]
            * 1000
            / (area * depth_mm),
            rain_mm=actual["expected_rain_m3"] * 1000 / area,
            peak_mm_h=float(rates.max()),
            rain_duration_h=6,
            manning_n=forcing["roughness"],
            initial_mean_m=actual["initial_m3"] / area,
            infiltration_mean_mm_h=20.0 if r["storm"] == "soil" else 2.0,
            downstream_stage_m=forcing["stage_m"],
        )
        selected.append(
            dict(
                case=r["case"],
                terrain_id=f"{r['landform']}-{r['seed']}",
                landform=r["landform"],
                split=r["split"],
                storm=r["storm"],
                method=r["method"],
                features=descriptors(terrain=stats, hydraulic=physical),
                runtime_s=float(r["runtime_s"]),
                memory_mib=float(r["memory_mib"]),
                errors={
                    k: 1 - float(r["final_iou_1cm"])
                    if k == "inundation"
                    else float(r[v])
                    for k, v in OBJECTIVES.items()
                    if k != "memory"
                },
                qualified={
                    k: r[
                        "qualified_final_inundation"
                        if k == "inundation"
                        else "qualified_" + k
                    ]
                    == "True"
                    and r["reference_self"] == "False"
                    for k in config["advisor"]["tolerances"]
                },
                sampled_regime=regime,
                wet_fraction=float(r["wet_fraction"]),
                run_id=r["experiment_id"],
            )
        )
    return selected + advisor_data.samples(
        study=study, config=config, partitions=partitions
    )


def assess(*, model, rows, rules):
    from floodwarp.solver_advisor.model import terrain_control_status

    groups = defaultdict(list)
    for r in rows:
        groups[r["case"], r["features"]["resolution"]].append(r)
    outcomes = []
    for (case, dx), group in groups.items():
        for objective, tol in model["tolerances"].items():
            actual = {r["method"]: r for r in group if r["qualified"][objective]}
            terrain = group[0]["features"]
            control_status = terrain_control_status(features=terrain, objective=objective)
            admissible = control_status != "refine terrain"
            if "native_objective_spacings" in model and control_status == "unassessed":
                admissible = None
            affordable = [
                r
                for r in actual.values()
                if admissible and r["errors"][objective] <= tol
            ]
            best = min(affordable, key=runtime) if affordable else None
            for rule in rules:
                prediction = None
                if rule == "oracle":
                    method = best["method"] if best else None
                    status = "measured least-cost feasible option"
                elif rule in ("fv1", "hancock_dense", "hancock_adaptive"):
                    method = rule
                    status = "fixed rule"
                else:
                    recommendation = recommend(
                        model=model,
                        features=group[0]["features"],
                        objective=objective,
                        tolerance=tol,
                        slope_only=rule == "slope",
                    )
                    method = recommendation["method"]
                    status = recommendation["status"]
                    prediction = next(
                        (
                            c
                            for c in recommendation["candidates"]
                            if c["method"] == method
                        ),
                        None,
                    )
                choice = actual.get(method)
                feasible = (
                    None
                    if choice is None or admissible is None
                    else admissible and choice["errors"][objective] <= tol
                )
                regret = (
                    choice["runtime_s"] / best["runtime_s"] - 1
                    if best and feasible
                    else None
                )
                outcomes.append(
                    dict(
                        case=case,
                        terrain_id=group[0]["terrain_id"],
                        rain_mm=group[0]["features"]["rain_mm"],
                        family=group[0]["landform"],
                        split=group[0]["split"],
                        storm=group[0]["storm"],
                        dx=dx,
                        cells=terrain["cells"],
                        wet_fraction=group[0].get("wet_fraction"),
                        mean_slope=terrain["mean_slope"],
                        terrain_admissible=admissible,
                        terrain_control_status=control_status,
                        objective=objective,
                        tolerance=tol,
                        rule=rule,
                        status=status,
                        chosen=method,
                        qualified_options=len(actual),
                        optimal=best["method"] if best else None,
                        feasible=feasible,
                        agrees=bool(feasible and choice["runtime_s"] == best["runtime_s"])
                        if best and choice else None,
                        regret=regret,
                        cheaper_option=regret > 1e-12 if regret is not None else None,
                        observed_error=choice["errors"][objective] if choice else None,
                        predicted_error=prediction["error_median"]
                        if prediction
                        else None,
                        interval_low=prediction["error_range"][0]
                        if prediction
                        else None,
                        interval_high=prediction["error_range"][1]
                        if prediction
                        else None,
                        observed_runtime_s=choice["runtime_s"] if choice else None,
                        predicted_runtime_s=prediction["runtime_median_s"]
                        if prediction
                        else None,
                        selected_run=choice["run_id"] if choice else None,
                        optimal_run=best["run_id"] if best else None,
                    )
                )
    return outcomes


def runtime(row):
    return row["runtime_s"]


def summary(*, rows):
    output = []
    for rule in sorted({r["rule"] for r in rows}):
        allrows = [r for r in rows if r["rule"] == rule]
        qualified = [r for r in allrows if r["feasible"] is not None]
        reg = [r["regret"] for r in qualified if r["regret"] is not None]
        predicted = [r for r in qualified if r["predicted_runtime_s"] is not None]
        opportunities = [r for r in allrows if r["optimal"] is not None]
        ties = [r["agrees"] for r in qualified if r["agrees"] is not None]
        output.append(
            dict(
                rule=rule,
                decisions=len(allrows),
                empirical_feasible_opportunities=len(opportunities),
                successful_selections=sum(r["feasible"] is True for r in opportunities),
                feasible_coverage_pct=100
                * sum(r["feasible"] is True for r in opportunities)
                / len(opportunities)
                if opportunities
                else None,
                selected=sum(r["chosen"] is not None for r in allrows),
                unqualified_selections=sum(
                    r["chosen"] is not None and r["feasible"] is None for r in allrows
                ),
                terrain_inadmissible_selections=sum(
                    r["chosen"] is not None and r.get("terrain_admissible") is False for r in allrows
                ),
                qualified_selections=len(qualified),
                false_feasible=sum(r["feasible"] is False for r in qualified),
                met_tolerance_pct=100 * np.mean([r["feasible"] for r in qualified])
                if qualified
                else None,
                agreement_pct=100 * np.mean(ties) if ties else None,
                agreement_all_feasible_pct=100
                * sum(r["agrees"] is True for r in opportunities)
                / len(opportunities)
                if opportunities
                else None,
                false_feasible_per_decision_pct=100
                * sum(r["feasible"] is False for r in allrows)
                / len(allrows),
                abstention_pct=100
                * sum(r["chosen"] is None for r in allrows)
                / len(allrows),
                median_regret_pct=100 * np.median(reg) if reg else None,
                regret_assessable=len(reg),
                interval_predictions_assessable=len(predicted),
                agreement_assessable=len(ties),
                cheaper_acceptable=sum(r["cheaper_option"] is True for r in qualified),
                abstentions=sum(r["chosen"] is None for r in allrows),
                runtime_median_absolute_pct_error=100
                * np.median(
                    [
                        abs(r["predicted_runtime_s"] / r["observed_runtime_s"] - 1)
                        for r in predicted
                    ]
                )
                if predicted
                else None,
                empirical_error_interval_coverage_pct=100
                * np.mean(
                    [
                        r["interval_low"] <= r["observed_error"] <= r["interval_high"]
                        for r in predicted
                    ]
                )
                if predicted
                else None,
            )
        )
    return output


def tune(*, model, training, validation, settings, rule):
    slope_only = rule == "slope"
    prefix = "slope_" if slope_only else ""
    distances = []
    for row in validation:
        peers = [
            r
            for r in training
            if r["method"] == row["method"]
            and r["features"]["resolution"] == row["features"]["resolution"]
        ]
        neighbours = nearest(
            rows=peers, query=row["features"], model=model, k=1, slope_only=slope_only
        )
        if neighbours:
            distances.append(neighbours[0][0])
    model[prefix + "distance_limit"] = float(
        np.quantile(distances, settings["coverage_quantile"])
    )
    criteria = []
    for k in settings["neighbours"]:
        model[prefix + "k"] = k
        predictions = assess(model=model, rows=validation, rules=(rule,))
        false = sum(r["feasible"] is False for r in predictions)
        missed = sum(
            r["optimal"] is not None and r["feasible"] is not True for r in predictions
        )
        regret = sum(r["regret"] for r in predictions if r["regret"] is not None)
        criteria.append((false, missed, regret, k))
    model[prefix + "k"] = min(criteria)[-1]
    model[prefix + "validation_selection"] = [list(row) for row in criteria]


def train(*, study, config, features, context, directory):
    from research.floodwarp_methods_paper.qualification import advisor_data

    advisor_data.verified_partition(
        study=study, config=config, partitions=("development", "validation")
    )
    rows = samples(
        study=study,
        config=config,
        features=features,
        partitions=("development", "validation"),
    )
    settings = config["advisor"]
    from research.floodwarp_methods_paper.qualification import advisor_data

    split = advisor_data.split_groups(
        rows=rows, heldout_family=settings["heldout_family"]
    )
    train = split["development"]
    val = split["validation"]
    if (
        not train
        or not val
        or {r["terrain_id"] for r in train} & {r["terrain_id"] for r in val}
    ):
        raise ValueError("Incomplete or leaking train/validation split")
    model = dict(
        training=train,
        size_factor=2.0,
        validation_spacings=sorted({r["features"]["resolution"] for r in val}),
        scales=scales(rows=train),
        tolerances=settings["tolerances"],
        inundation_target="one minus final 1 cm depth IoU on common physical support",
        hardware=context["gpu"],
        revision=context["git_sha"],
        quantile=settings["quantile"],
    )
    from research.floodwarp_methods_paper.qualification import decisions

    decisions.scale_fit(model=model, training=train, validation=val)
    for rule in ("advisor", "slope"):
        tune(model=model, training=train, validation=val, settings=settings, rule=rule)
    advisor_data.interval_calibration(model=model, rows=val)
    model["training_terrain_ids"] = sorted({r["terrain_id"] for r in train})
    model["validation_terrain_ids"] = sorted({r["terrain_id"] for r in val})
    write_json(path=directory / "model.json", data=model)
    write_json(path=study / "processed/advisor/model.json", data=model)
    write_json(
        path=study.parents[1] / "src/floodwarp/solver_advisor/model.json", data=model
    )
    return dict(
        k=model["k"],
        distance_limit=model["distance_limit"],
        training_rows=len(train),
        training_groups=len(model["training_terrain_ids"]),
        validation_groups=len(model["validation_terrain_ids"]),
    )


def validate(*, study, config, features, directory):
    from research.floodwarp_methods_paper.qualification import advisor_data

    advisor_data.verified_partition(study=study, config=config, partitions=("test",))
    frozen = study / "processed/advisor/model.json"
    model_hash = checksum(frozen)
    fit_records = archived(study=study, stages=("advisor-train",))
    matches = [
        r
        for r in fit_records
        if r["config_sha256"] == digest(config)
        and r["artifacts"].get("model.json") == model_hash
    ]
    if not matches:
        raise ArithmeticError("The model does not match an immutable training receipt")
    if (
        checksum(study.parents[1] / "src/floodwarp/solver_advisor/model.json")
        != model_hash
    ):
        raise ArithmeticError("Packaged advisor differs from the frozen model")
    model = json.loads(frozen.read_text())
    rows = samples(study=study, config=config, features=features, partitions=("test",))
    held = [
        r
        for r in rows
        if r["split"] == "test" or r["landform"] == config["advisor"]["heldout_family"]
    ]
    if {r["terrain_id"] for r in held} & set(
        model["training_terrain_ids"] + model["validation_terrain_ids"]
    ):
        raise ArithmeticError("Held-out group leakage")
    results = assess(
        model=model,
        rows=held,
        rules=("advisor", "fv1", "hancock_dense", "slope", "oracle"),
    )
    from research.floodwarp_methods_paper.qualification import decisions

    decisions.validate(study=study, model=model, held=held, decisions=results)
    totals = summary(rows=results)
    for folder in (directory, study / "processed/advisor"):
        csv_write(file=folder / "heldout.csv", rows=results)
        csv_write(file=folder / "summary.csv", rows=totals)
    return dict(
        total=len(results),
        groups=len({r["terrain_id"] for r in held}),
        rules=totals,
        frozen_model_sha256=model_hash,
    )
