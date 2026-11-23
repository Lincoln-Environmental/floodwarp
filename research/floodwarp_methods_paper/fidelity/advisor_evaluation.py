"""Calibrate hydraulic-objective intervals on separate terrain groups and evaluate held-out decisions."""

from collections import defaultdict
import json
from floodwarp.solver_advisor.fidelity import METRICS, attach
from floodwarp.solver_advisor.model import recommend
from research.floodwarp_methods_paper import advisor, metrics, runner
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification.advisor_data import split_groups


def measured(*, row, observed):
    obs = observed.get(row["run_id"], {})
    return {
        objective: abs(float(obs[field])) + float(obs["uncertainty_" + field])
        for objective, field in METRICS.items()
        if field and obs.get("qualified_" + field) == "True"
    }


def cases(*, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["features"]["resolution"]].append(row)
    return groups


def build(*, directory, default_percent, factors):
    study = directory.parent
    pkg = study.parents[1] / "src/floodwarp/solver_advisor"
    base = json.loads((pkg / "model.json").read_text())
    config = read_config(path=study / "configs/study.json")
    features = json.loads((study / "processed/metrics/terrain.json").read_text())
    rows = advisor.samples(study=study, config=config, features=features)
    split = split_groups(rows=rows, heldout_family=config["advisor"]["heldout_family"])
    from research.floodwarp_methods_paper.fidelity.extended_analysis import available

    observed = available(base=metrics.csv_read(directory / "results/storms.csv"))
    forecasts = {}
    for original in rows:
        row = observed.get(original["run_id"])
        if row:
            forecast = dict(
                runtime_s=float(row["runtime_s"]),
                end_h=float(row["hours"]),
                evidence_id=row["experiment_id"],
            )
            forecasts[original["run_id"]] = forecast
            original["runtime_s"] = forecast["runtime_s"]
            # Duration of a measured recession is an outcome unavailable to a pre-run decision.
    training = {
        r["run_id"]: measured(row=r, observed=observed) for r in split["development"]
    }
    if set(training) != {r["run_id"] for r in base["training"]}:
        raise ValueError("Model training cohort changed")
    tolerances = dict.fromkeys(METRICS, default_percent)
    supplement = dict(
        window="matched_completed_recessions",
        training=training,
        forecasts={k: forecasts[k] for k in training if k in forecasts},
        base_training_ids=sorted(training),
        padding={},
        tolerances=tolerances,
        validation_support={o: base["validation_spacings"] for o in METRICS},
    )
    provisional = attach(model=base, supplement=supplement)
    residuals = defaultdict(dict)
    for (_, dx), group in cases(rows=split["validation"]).items():
        for obj in METRICS:
            pred = recommend(
                model=provisional, features=group[0]["features"], objective=obj
            )
            for p in pred["candidates"]:
                match = next((r for r in group if r["method"] == p["method"]), None)
                if match is None:
                    continue
                truth = measured(row=match, observed=observed).get(obj)
                if truth is not None:
                    ident = match["terrain_id"]
                    gap = max(0.0, truth - p["error_range"][1])
                    residuals[obj, dx][ident] = max(
                        gap, residuals[obj, dx].get(ident, 0)
                    )
    for obj in METRICS:
        supported = [
            dx for dx in base["validation_spacings"] if len(residuals[obj, dx]) >= 2
        ]
        supplement["validation_support"][obj] = supported
        values = [v for dx in supported for v in residuals[obj, dx].values()]
        supplement["padding"][obj] = max(values, default=0.0)
    frozen_id = runner.digest(supplement)
    frozen = attach(model=base, supplement=supplement)
    decisions = []
    baselines = []
    predictions = []
    for (case, dx), group in cases(rows=split["test"]).items():
        f = group[0]["features"]
        for obj in METRICS:
            errors = {
                r["method"]: measured(row=r, observed=observed).get(obj) for r in group
            }
            for factor in factors:
                tol = tolerances[obj] * factor
                decision = recommend(
                    model=frozen, features=f, objective=obj, tolerance=tol
                )
                method = decision["method"]
                slope = recommend(
                    model=frozen,
                    features=f,
                    objective=obj,
                    tolerance=tol,
                    slope_only=True,
                )
                for candidate in decision["candidates"]:
                    name = candidate["method"]
                    found = next(
                        (item for item in group if item["method"] == name), None
                    )
                    actual = errors.get(name)
                    if (
                        factor == 1
                        and found is not None
                        and found["run_id"] in forecasts
                    ):
                        bounds = candidate["error_range"]
                        predictions.append(
                            dict(
                                case=case,
                                dx=dx,
                                objective=obj,
                                cells=f["cells"],
                                method=name,
                                observed_upper_error=actual,
                                predicted_error=candidate["error_median"],
                                low=bounds[0],
                                high=bounds[1],
                                interval_covers=bounds[0] <= actual <= bounds[1]
                                if actual is not None
                                else None,
                                runtime_prediction_s=candidate["runtime_median_s"],
                                runtime_s=found["runtime_s"],
                            )
                        )
                feasible = [
                    r
                    for r in group
                    if errors[r["method"]] is not None
                    and errors[r["method"]] <= tol
                    and decision["status"] != "refine_terrain"
                ]
                oracle = min(feasible, key=runtime) if feasible else None
                selected = next((r for r in group if r["method"] == method), None)
                cost = selected["runtime_s"] if selected else None
                for label, choice in (
                    ("advisor", method),
                    ("slope_only", slope["method"]),
                    ("always_fv1", "fv1"),
                    ("always_hancock_dense", "hancock_dense"),
                    ("oracle", oracle["method"] if oracle else None),
                ):
                    observed_candidate = next(
                        (
                            r
                            for r in group
                            if r["method"] == choice and r["run_id"] in forecasts
                        ),
                        None,
                    )
                    err = errors.get(choice)
                    feasible_choice = (
                        None
                        if err is None or observed_candidate is None
                        else (err <= tol and decision["status"] != "refine_terrain")
                    )
                    baselines.append(
                        dict(
                            case=case,
                            dx=dx,
                            cells=f["cells"],
                            objective=obj,
                            tolerance=tol,
                            rule=label,
                            method=choice,
                            selected=choice is not None,
                            assessable=feasible_choice is not None,
                            meets=feasible_choice,
                            oracle=oracle["method"] if oracle else None,
                            regret_pct=100
                            * (
                                observed_candidate["runtime_s"] / oracle["runtime_s"]
                                - 1
                            )
                            if feasible_choice and oracle
                            else None,
                        )
                    )
                decisions.append(
                    dict(
                        case=case,
                        dx=dx,
                        objective=obj,
                        tolerance=tol,
                        cells=f["cells"],
                        status=decision["status"],
                        method=method,
                        qualified_options=sum(v is not None for v in errors.values()),
                        measured_upper_error=errors.get(method),
                        feasible=errors[method] <= tol
                        if errors.get(method) is not None
                        else None,
                        cheapest=oracle["method"] if oracle else None,
                        regret_pct=100 * (cost / oracle["runtime_s"] - 1)
                        if cost
                        and oracle
                        and errors.get(method) is not None
                        and errors[method] <= tol
                        else None,
                    )
                )
    metrics.csv_write(file=directory / "results/advisor_decisions.csv", rows=decisions)
    metrics.csv_write(file=directory / "results/advisor_baselines.csv", rows=baselines)
    metrics.csv_write(
        file=directory / "results/advisor_predictions.csv", rows=predictions
    )
    if runner.digest(supplement) != frozen_id:
        raise ValueError("The held-out evaluation modified the frozen advisor")
    evidence_file = pkg / "fidelity_evidence.json"
    pending = evidence_file.with_suffix(".pending")
    runner.write_json(path=pending, data=supplement)
    pending.replace(evidence_file)
    return decisions


def runtime(r):
    return r["runtime_s"]
