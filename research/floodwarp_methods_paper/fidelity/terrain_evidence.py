"""Associate operational budget decisions with the same pre-run controls used by the advisor."""

import json

from floodwarp.solver_advisor.model import terrain_requires_refinement
from research.floodwarp_methods_paper import metrics, runner


def measured_features(*, directory, index_row, native_sha):
    case, dx, ident = index_row["case"], int(index_row["dx_m"]), index_row["identity"]
    folder = directory / "native_features" / case / f"{dx}m" / ident
    evidence = json.loads((folder / "pre_run.json").read_text())
    if runner.digest(evidence["identity"])[:24] != ident:
        raise ValueError("Altered pre-run descriptor identity")
    for field, observed in evidence["features"].items():
        expected = "" if observed is None else str(observed)
        if index_row[field] != expected:
            raise ValueError(f"Pre-run feature index differs from archived {field}")
    physical = directory / "native_terrain" / case / evidence["identity"]["physical"] / "inputs.json"
    if runner.digest(json.loads(physical.read_text())["native_sha256"]) != native_sha:
        raise ValueError("Pre-run descriptors do not match the forecast's physical fields")
    return evidence


def forecast_controls(*, directory, rows):
    index = directory / "native_features/observations.csv"
    if not index.is_file():
        return {}
    entries = {(r["case"], int(r["dx_m"])): r for r in metrics.csv_read(file=index)}
    controls = {}
    cache = {}
    for r in rows:
        if r["horizon"] != "complete-native-metre":
            continue
        key = r["case"], int(r["dx_m"])
        entry = entries.get(key)
        if entry is None:
            continue
        source = (*key, r["native_sha"])
        if source not in cache:
            evidence = measured_features(directory=directory, index_row=entry, native_sha=r["native_sha"])
            cache[source] = dict(identity=entry["identity"], features=evidence["features"])
        controls[r["source_id"]] = cache[source]
    return controls


def assess(*, rows, controls):
    from research.floodwarp_methods_paper.fidelity.native_advisor import DEPTH, WAVE

    objectives = {metric: objective for objective, metric in (DEPTH | WAVE).items() if metric is not None}
    objectives.update(centroid_delay_pct="hydrograph_propagation", wet_depth_rmse_m="depth",
                      wet_union_depth_rmse_5cm_m="depth", wet_union_depth_rmse_10cm_m="depth",
                      extent_error_1cm="inundation")
    screened, audit = [], []
    for row in rows:
        if row["cohort"] != "complete-native-metre":
            continue
        metric = row["objective"]
        goal = objectives[metric]
        record = controls.get(row["experiment_id"], {})
        features = record.get("features", {})
        connection = features.get("critical_connection_lost")
        if connection is None:
            disposition = "unassessed"
        elif terrain_requires_refinement(features=features, objective=goal):
            disposition = "refine terrain"
        else:
            disposition = "admissible"
        screened.append(dict(row, hydraulic_eligible=row["eligible"], terrain_disposition=disposition,
                             eligible=row["eligible"] and disposition == "admissible"))
        audit.append(dict(case=row["case"], dx_m=row["dx"], experiment_id=row["experiment_id"],
                          station=row["station"], objective=metric, advisor_objective=goal,
                          reference_id=row["reference_id"], hydraulic_eligible=row["eligible"],
                          disposition=disposition, descriptor_id=record.get("identity"),
                          critical_connection_lost=connection, barrier_lowering_m=features.get("barrier_lowering_m"),
                          lost_channel_fraction=features.get("lost_channel_fraction")))
    return screened, audit
