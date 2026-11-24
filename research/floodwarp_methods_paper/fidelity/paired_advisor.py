"""Score whole-event grid and method choices against the measured two-grid oracle."""

from collections import defaultdict
from math import isfinite

from floodwarp.solver_advisor.model import (
    choose_resolution,
    terrain_requires_refinement,
)


def observed_cost(row):
    return row["runtime_s"]


def qualified_reference(*, observations, objective):
    from research.floodwarp_methods_paper.fidelity.native_advisor import DEPTH

    spatial = objective in DEPTH
    scopes = set()
    for row in observations:
        if not row["qualified"][objective]:
            continue
        reference = row["spatial_reference_id"] if spatial else row["reference_id"]
        support = row["spatial_support_m"] if spatial else None
        if not reference or (spatial and (support is None or not isfinite(support) or support <= 0)):
            raise ValueError("Qualified paired-grid error lacks its reference or spatial support")
        scopes.add((reference, support))
    if len(scopes) > 1:
        raise ValueError("Paired-grid errors use different numerical references or spatial supports")
    return next(iter(scopes), (None, None))


def evaluate_pairs(*, model, rows, trials, spacings):
    observations = defaultdict(dict)
    for row in rows:
        dx = row["features"]["resolution"]
        if dx in spacings:
            observations[row["case"]][dx, row["method"]] = row
    groups = defaultdict(dict)
    for t in trials:
        if t["dx"] in spacings:
            groups[t["case"], t["objective"], t["rule"]][t["dx"]] = t
    output = []
    for (case, objective, rule), choices in groups.items():
        if set(choices) != set(spacings):
            continue
        measured = observations[case]
        signatures = {
            (r["terrain_id"], r["native_sha"], float(r["end_h"]), float(r["interval_s"]))
            for r in measured.values()
        }
        if len(signatures) != 1:
            raise ValueError(
                "Paired validation requires identical terrain, forcing horizon and reporting interval"
            )
        reference, support = qualified_reference(observations=measured.values(), objective=objective)
        tol = choices[spacings[0]]["tolerance"]
        if any(t["tolerance"] != tol for t in choices.values()):
            raise ValueError("Paired-grid choices require the same objective tolerance")
        available = [
            r
            for r in measured.values()
            if r["qualified"][objective] and r["errors"][objective] is not None
            and r["errors"][objective] <= tol
            and not terrain_requires_refinement(
                features=r["features"], objective=objective
            )
        ]
        optimal = min(available, key=observed_cost) if available else None
        for requested in spacings:
            original = choices[requested]
            if rule == "oracle":
                grid = optimal["features"]["resolution"] if optimal else requested
                method = optimal["method"] if optimal else None
                prediction = {}
            elif rule in ("advisor", "slope"):
                decisions = [
                    dict(
                        status=t["status"],
                        objective=objective,
                        method=t["chosen"],
                        resolution_m=dx,
                        represented_area_m2=t["cells"] * dx**2 if "cells" in t else None,
                        runtime_s=t["predicted_runtime_s"],
                        warnings=[],
                    )
                    for dx in (requested, *(x for x in spacings if x != requested))
                    for t in (choices[dx],)
                ]
                decision = choose_resolution(model=model, results=decisions)
                grid, method = decision["resolution_m"], decision["method"]
                prediction = choices[grid]
            else:
                grid, method = requested, original["chosen"]
                prediction = {}
            actual = measured.get((grid, method))
            measured_error = actual["errors"][objective] if actual and actual["qualified"][objective] else None
            feasible = actual in available if measured_error is not None else None
            regret = (
                actual["runtime_s"] / optimal["runtime_s"] - 1
                if feasible and optimal
                else None
            )
            output.append(
                dict(
                    original,
                    requested_dx=requested,
                    reference_id=reference,
                    spatial_support_m=support,
                    chosen_dx=grid if method else None,
                    status=prediction.get("status", original["status"]),
                    chosen=method,
                    optimal=optimal["method"] if optimal else None,
                    optimal_dx=optimal["features"]["resolution"] if optimal else None,
                    optimal_run=optimal["run_id"] if optimal else None,
                    feasible=feasible,
                    regret=regret,
                    qualified_options=sum(
                        r["qualified"][objective] for r in measured.values()
                    ),
                    agrees=(feasible and actual["runtime_s"] == optimal["runtime_s"])
                    if optimal and measured_error is not None
                    else None,
                    cheaper_option=regret > 1.0e-12 if regret is not None else None,
                    selected_run=actual["run_id"] if actual else None,
                    terrain_admissible=not terrain_requires_refinement(features=actual["features"], objective=objective)
                    if actual else original["terrain_admissible"],
                    observed_error=measured_error,
                    observed_runtime_s=actual["runtime_s"] if actual else None,
                    **{
                        k: prediction.get(k)
                        for k in (
                            "predicted_runtime_s",
                            "predicted_error",
                            "interval_low",
                            "interval_high",
                        )
                    },
                )
            )
    return output
