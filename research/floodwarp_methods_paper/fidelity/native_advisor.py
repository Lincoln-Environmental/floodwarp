"""Fit and assess the existing empirical advisor using matched native-metre storm forecasts."""

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

from floodwarp.solver_advisor.fidelity import METRICS
from floodwarp.solver_advisor.model import recommend, scales
from research.floodwarp_methods_paper import advisor, metrics, runner
from research.floodwarp_methods_paper.fidelity import native_matrix, phase_cost
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import advisor_data, decisions


WAVE = {
    **METRICS,
    "relative_hydrograph_rmse": "active_nrmse_peak_pct",
    "event_discharge_error": "integrated_abs_pct",
}


DEPTH = {
    "depth": "depth_rmse_m",
    "maximum_depth": "maximum_depth_rmse_m",
    "final_inundation_5cm": "extent_error_5cm",
    "final_inundation_10cm": "extent_error_10cm",
    "maximum_inundation_5cm": "maximum_extent_error_5cm",
    "maximum_inundation_10cm": "maximum_extent_error_10cm",
}


def conservative_error(*, record, field, margin):
    from research.floodwarp_methods_paper.fidelity import budgets

    if not field or record.get("qualified_" + field) != "True":
        return None
    return budgets.conservative_error(value=record[field], sensitivity=record[margin])


def tolerances(*, pct, depth_m):
    goals = {k: depth_m if "depth" in k else pct / 100 for k in DEPTH}
    goals.update({k: pct for k in WAVE})
    return goals


def sample_rows(*, directory, partitions, require_full, config, design):
    target = directory / "native_results"
    plans = {}
    for spec in native_matrix.specifications(
        study=directory.parent, config=config, design=design
    ):
        split = (
            "test"
            if spec["landform"] == config["advisor"]["heldout_family"]
            else spec["split"]
        )
        if split in partitions and spec["factor"] == 1:
            plans[spec["case"], spec["dx"], spec["requested_method"]] = dict(
                spec, split=split
            )
    features = {
        (r["case"], int(r["dx_m"])): r
        for r in metrics.csv_read(file=directory / "native_features/observations.csv")
    }
    primary = [
        r
        for r in metrics.csv_read(file=target / "measured_windows.csv")
        if r["horizon"] == "complete-native-metre" and float(r["factor"]) == 1
    ]
    current_solver = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    observed = {
        (r["case"], int(r["dx_m"]), r["method"]): r
        for r in primary
        if (r["case"], int(r["dx_m"]), r["method"]) in plans
        and r.get("solver_source") == current_solver
    }
    if require_full and plans.keys() != observed.keys():
        raise ValueError(
            f"Missing {len(plans.keys() - observed.keys())} matched native forecasts in {partitions}"
        )
    hyd = {
        r["experiment_id"]: r
        for r in metrics.csv_read(file=target / "hydraulic_fidelity.csv")
        if r["cohort"] == "complete-native-metre" and r["station"] == "outlet"
    }
    beds = {
        r["id"]: r
        for r in metrics.csv_read(file=target / "recession_depth.csv")
        if r["cohort"] == "complete-native-metre"
    }
    diagnostics = {
        r["id"]: r
        for r in metrics.csv_read(file=target / "whole_storm_diagnostics.csv")
        if r["cohort"] == "complete-native-metre"
    }
    result = []
    for key, row in sorted(observed.items()):
        s = plans[key]
        case, dx, method = key
        feature = features[case, dx]
        fpath = (
            directory
            / "native_features"
            / case
            / f"{dx}m"
            / feature["identity"]
            / "pre_run.json"
        )
        f = json.loads(fpath.read_text())
        if runner.digest(f["identity"])[:24] != feature["identity"]:
            raise ValueError("Altered pre-run descriptor identity")
        physical = json.loads(
            (
                directory
                / "native_terrain"
                / case
                / f["identity"]["physical"]
                / "inputs.json"
            ).read_text()
        )
        if runner.digest(physical["native_sha256"]) != row["native_sha"]:
            raise ValueError(
                "Pre-run descriptors do not match the forecast's physical fields"
            )
        ident = row["source_id"]
        budget = diagnostics[ident]
        obs = hyd[ident]
        errors = {
            o: conservative_error(record=obs, field=m, margin="uncertainty_" + str(m))
            for o, m in WAVE.items()
        }
        errors.update(
            {
                o: conservative_error(
                    record=beds.get(ident, {}), field=m, margin=m + "_sensitivity"
                )
                for o, m in DEPTH.items()
            }
        )
        memory = [
            float(budget[k])
            for k in ("sampled_pool_peak_mib", "source_final_pool_mib")
            if budget.get(k)
        ]
        if not memory:
            raise ValueError("Native forecast has no measured device allocation")
        source = (
            directory / "native_matrix" / case / budget["parent_id"] / "result.json"
        )
        if not source.is_file():
            source = directory / "mean_storms" / budget["parent_id"] / "result.json"
        hardware = json.loads(source.read_text())["hardware"]["gpu"]
        result.append(
            dict(
                case=case,
                terrain_id=f["terrain_id"],
                method=method,
                split=s["split"],
                landform=s["landform"],
                storm=s["storm"],
                run_id=ident,
                descriptor_id=feature["identity"],
                native_sha=row["native_sha"],
                end_h=float(row["end_h"]),
                interval_s=float(row["interval_s"]),
                features=f["features"],
                hardware=hardware,
                runtime_s=float(obs["runtime_s"]),
                memory_mib=max(memory),
                errors=errors,
                qualified={o: value is not None for o, value in errors.items()},
                wet_fraction=float(obs["wet_fraction"])
                if obs.get("wet_fraction")
                else None,
                sampled_regime=obs.get("flow_regime"),
                reference_id=obs.get("reference_id"),
            )
        )
    return result


def support(*, rows, objectives, minimum):
    from floodwarp.solver_advisor.model import terrain_requires_refinement

    independent = defaultdict(set)
    for row in rows:
        for objective in objectives:
            if row["qualified"][objective] and not terrain_requires_refinement(features=row["features"], objective=objective):
                independent[objective, row["features"]["resolution"]].add(
                    row["terrain_id"]
                )
    return {
        objective: sorted(
            dx
            for (obj, dx), groups in independent.items()
            if obj == objective and len(groups) >= minimum
        )
        for objective in objectives
    }


def admissible_training(*, rows):
    """Keep measured costs while excluding numerical-error labels from lost controls."""
    from floodwarp.solver_advisor.model import terrain_requires_refinement

    training = []
    for row in rows:
        errors = {
            objective: None
            if terrain_requires_refinement(features=row["features"], objective=objective)
            else value for objective, value in row["errors"].items()
        }
        qualified = {objective: row["qualified"][objective] and value is not None
                     for objective, value in errors.items()}
        training.append(dict(row, errors=errors, qualified=qualified))
    return training


def train(*, directory, settings, config, design):
    rows = sample_rows(
        directory=directory,
        partitions=("development", "validation"),
        require_full=True,
        config=config,
        design=design,
    )
    groups = advisor_data.split_groups(
        rows=rows, heldout_family=config["advisor"]["heldout_family"]
    )
    train_rows, val = admissible_training(rows=groups["development"]), groups["validation"]
    if not train_rows or not val or groups["test"]:
        raise ValueError(
            "Fitting requires disjoint development and validation terrains"
        )
    goals = tolerances(
        pct=settings["fit_percent"], depth_m=settings["depth_tolerance_m"]
    )
    spacing = support(
        rows=val, objectives=goals, minimum=settings["validation_groups_per_spacing"]
    )
    devices = {r["hardware"] for r in rows}
    if (
        len(devices) != 1
        or settings["hardware"].casefold() not in next(iter(devices)).casefold()
    ):
        raise ValueError("Native advisor costs require a single measured reference GPU")
    model = dict(
        training=train_rows,
        outlet_discharge_objective="relative_hydrograph_rmse",
        objective_units={
            o: ("m" if "depth" in o else "one minus IoU" if o in DEPTH else "percent")
            for o in goals
        },
        scales=scales(rows=train_rows),
        tolerances=goals,
        fidelity_validation={o: spacing[o] for o in WAVE},
        extent_validation={o: spacing[o] for o in DEPTH if "inundation" in o},
        native_objective_spacings=spacing,
        native_cross_grid_validation={},
        hardware=next(iter(devices)),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory.parents[2], text=True
        ).strip(),
        quantile=config["advisor"]["quantile"],
        validation_spacings=sorted({r["features"]["resolution"] for r in val}),
    )
    decisions.scale_fit(model=model, training=train_rows, validation=val)
    for rule in ("advisor", "slope"):
        advisor.tune(
            model=model,
            training=train_rows,
            validation=val,
            settings=config["advisor"],
            rule=rule,
        )
    advisor_data.interval_calibration(model=model, rows=val)
    valid = advisor.assess(model=model, rows=val, rules=("advisor",))
    model["native_cross_grid_validation"] = validated_pairs(
        rows=valid, minimum=settings["validation_groups_per_spacing"]
    )
    model["training_terrain_ids"] = sorted({r["terrain_id"] for r in train_rows})
    model["validation_terrain_ids"] = sorted({r["terrain_id"] for r in val})
    identity = dict(
        development=runner.digest(train_rows),
        validation=runner.digest(val),
        settings=settings,
        fit_source=model_sources(),
        advisor_config=config["advisor"],
    )
    folder = directory / "native_advisor" / runner.digest(identity)[:24]
    path = folder / "model.json"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        runner.write_json(path=path, data=model)
        validation_path = folder / "validation_decisions.csv"
        metrics.csv_write(file=validation_path, rows=valid)
        runner.write_json(
            path=folder / "frozen.json",
            data=dict(identity=identity, sha256=runner.checksum(path),
                      validation_decisions_sha256=runner.checksum(validation_path)),
        )
    frozen = json.loads((folder / "frozen.json").read_text())
    if (frozen["identity"] != identity or frozen["sha256"] != runner.checksum(path)
            or frozen["validation_decisions_sha256"] != runner.checksum(folder / "validation_decisions.csv")):
        raise ValueError("Existing frozen model or fit inputs changed")
    pointer = directory / "native_advisor/frozen_model.json"
    runner.write_json(
        path=pointer,
        data=dict(path=str(path.relative_to(directory)), sha256=runner.checksum(path)),
    )
    return path


def model_sources():
    from floodwarp.solver_advisor import model

    return {
        m.__name__: runner.checksum(Path(m.__file__))
        for m in (advisor, advisor_data, decisions, model, sys.modules[__name__])
    }


def validated_pairs(*, rows, minimum):
    good = defaultdict(set)
    failed = set()
    for r in rows:
        key = r["objective"], r["dx"]
        if r["chosen"] is not None:
            if r["feasible"] is True:
                good[key].add((r["terrain_id"], r["case"]))
            else:
                failed.add(key)
    accepted = defaultdict(list)
    for (objective, dx), groups in sorted(good.items()):
        for (other, res), held in sorted(good.items()):
            if (
                objective == other
                and res > dx
                and not {(objective, dx), (objective, res)} & failed
                and len({terrain for terrain, case in groups & held}) >= minimum
            ):
                accepted[objective].append([dx, res])
    return dict(accepted)


def predictions(*, model, rows):
    by_grid = defaultdict(dict)
    for record in rows:
        by_grid[record["case"], record["features"]["resolution"]][record["method"]] = (
            record
        )
    results = []
    for (case, grid), methods in by_grid.items():
        feature = next(iter(methods.values()))["features"]
        for objective, tol in model["tolerances"].items():
            decision = recommend(
                model=model, features=feature, objective=objective, tolerance=tol
            )
            estimates = {c["method"]: c for c in decision["candidates"]}
            for name, obs in methods.items():
                pred = estimates.get(name)
                actual = obs["errors"][objective]
                bounds = pred["error_range"] if pred else [None, None]
                results.append(
                    dict(
                        case=case,
                        terrain_id=obs["terrain_id"],
                        resolution_m=grid,
                        method=name,
                        objective=objective,
                        observation_id=obs["run_id"],
                        decision=decision["status"],
                        predicted=pred is not None,
                        actual_upper_error=actual,
                        assessable=actual is not None,
                        predicted_error=pred["error_median"] if pred else None,
                        low=bounds[0],
                        high=bounds[1],
                        interval_covers=bounds[0] <= actual <= bounds[1]
                        if pred and actual is not None
                        else None,
                        actual_runtime_s=obs["runtime_s"],
                        predicted_runtime_s=pred["runtime_median_s"] if pred else None,
                        runtime_abs_pct_error=100
                        * abs(pred["runtime_median_s"] / obs["runtime_s"] - 1)
                        if pred
                        else None,
                    )
                )
    return results


def validate(*, directory, settings, config, design):
    pointer = json.loads((directory / "native_advisor/frozen_model.json").read_text())
    path = directory / pointer["path"]
    if runner.checksum(path) != pointer["sha256"]:
        raise ValueError("Frozen advisor model checksum changed")
    model = json.loads(path.read_text())
    held = sample_rows(
        directory=directory,
        partitions=("test",),
        require_full=True,
        config=config,
        design=design,
    )
    seen = set(model["training_terrain_ids"] + model["validation_terrain_ids"])
    if seen & {r["terrain_id"] for r in held}:
        raise ValueError("Held-out terrain used in fitting")
    combined = []
    summary = []
    for pct in settings["assessment_percentages"]:
        m = dict(
            model, tolerances=tolerances(pct=pct, depth_m=settings["depth_tolerance_m"])
        )
        trials = advisor.assess(model=m, rows=held, rules=settings["rules"])
        for r in trials:
            r["wet_band"] = phase_cost.label(
                fraction=r["wet_fraction"], edges=settings["evaluation_wet_edges"]
            )
        combined.extend(dict(r, percentage_scenario=pct) for r in trials)
        summary.extend(
            dict(r, percentage_scenario=pct) for r in advisor.summary(rows=trials)
        )
    from research.floodwarp_methods_paper.fidelity import paired_advisor

    scoring = dict(heldout=runner.digest(held), frozen=pointer["sha256"],
                   paired_analysis=runner.checksum(Path(paired_advisor.__file__)),
                   single_grid_analysis=runner.checksum(Path(advisor.__file__)),
                   native_analysis=runner.checksum(Path(__file__)))
    output = path.parent / ("heldout_" + runner.digest(scoring)[:24])
    output.mkdir(exist_ok=True)
    metrics.csv_write(
        file=output / "candidate_predictions.csv",
        rows=predictions(model=model, rows=held),
    )
    metrics.csv_write(file=output / "decisions.csv", rows=combined)
    metrics.csv_write(file=output / "summary.csv", rows=summary)
    by_subset = []
    paired = []
    for pct in settings["assessment_percentages"]:
        trials = [r for r in combined if r["percentage_scenario"] == pct]
        paired.extend(dict(r, percentage_scenario=pct) for r in paired_advisor.evaluate_pairs(
            model=model, rows=held, trials=trials, spacings=settings["practical_pair_m"]))
        for axis in ("family", "dx", "cells", "terrain_id", "wet_band"):
            for value in sorted({r[axis] for r in trials}):
                for r in advisor.summary(
                    rows=[row for row in trials if row[axis] == value]
                ):
                    by_subset.append(
                        dict(r, percentage_scenario=pct, axis=axis, value=value)
                    )
        final = [
            r for r in trials if r["terrain_id"] in settings["new_final_terrain_ids"]
        ]
        if {r["terrain_id"] for r in final} != set(settings["new_final_terrain_ids"]):
            raise ValueError("Missing prospectively reserved final terrain")
        by_subset.extend(
            dict(
                r,
                percentage_scenario=pct,
                axis="independent_final",
                value="new terrains",
            )
            for r in advisor.summary(rows=final)
        )
    metrics.csv_write(file=output / "strata.csv", rows=by_subset)
    metrics.csv_write(file=output / "joint_decisions.csv", rows=paired)
    paired_summary = []
    for pct in settings["assessment_percentages"]:
        for dx in settings["practical_pair_m"]:
            cohort = [r for r in paired if r["percentage_scenario"] == pct and r["requested_dx"] == dx]
            paired_summary.extend(dict(s,percentage_scenario=pct,requested_dx=dx) for s in advisor.summary(rows=cohort))
    metrics.csv_write(file=output / "joint_summary.csv",rows=paired_summary)
    runner.write_json(
        path=output / "sources.json",
        data=dict(
            frozen_model=pointer,
            scoring=scoring,
            heldout_ids=[r["run_id"] for r in held],
            final_independent_terrain_ids=settings["new_final_terrain_ids"],
            files={
                name: runner.checksum(output / name)
                for name in ("summary.csv", "strata.csv", "decisions.csv", "candidate_predictions.csv",
                             "joint_decisions.csv", "joint_summary.csv")
            },
        ),
    )
    runner.write_json(
        path=directory / "native_advisor/heldout_evaluation.json",
        data=dict(path=str(output.relative_to(directory)), sha256=runner.checksum(output / "sources.json")),
    )
    return output


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    settings = json.loads((directory / "native_advisor_settings.json").read_text())
    if sys.argv[1:] == ["native-advisor-train"]:
        train(directory=directory, settings=settings, config=config, design=design)
    elif sys.argv[1:] == ["native-advisor-validate"]:
        validate(directory=directory, settings=settings, config=config, design=design)
    else:
        raise ValueError("Select native-advisor-train or native-advisor-validate")
