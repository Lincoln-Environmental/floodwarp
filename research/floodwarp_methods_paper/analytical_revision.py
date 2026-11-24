"""Compare the audited and repaired analytical forecasts on identical exact solutions."""

import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import comparisons, evidence, metrics, runner


def field_comparison(*, before, after):
    for name in ("x_m", "t_s", "exact_cell_averages", "exact_gauges"):
        np.testing.assert_array_equal(before[name], after[name], err_msg=f"Reference differs in {name}")
    return {name + "_equal": np.array_equal(before[name], after[name])
            for name in ("model_centreline", "model_gauges", "accepted_dt_s")}


def run(*, study: Path, stages):
    gates = comparisons.exact_check(study=study)
    with evidence.archived_publication(study=study):
        original = metrics.load(study=study, stages=stages)
    original = {m["experiment_id"]: (m, r, path) for m, r, path in original}
    current = metrics.load(study=study, stages=stages)
    if set(original) != {m["experiment_id"] for m, _, _ in current}:
        raise ValueError("Paired analytical cohorts require the same set of physical configurations")
    provenance = json.loads((study / "fidelity/outlet_results/revision_evidence.json").read_text())["revisions"]
    rows = []
    for meta, result, path in sorted(current, key=experiment_key):
        old, baseline, prior = original[meta["experiment_id"]]
        if meta["stage"] in ("analytical", "convergence"):
            for tag, entry in (("audit", old), ("repair", meta)):
                if provenance[tag]["source_sha256"][meta["stage"]] != entry["context"]["source_sha256"]:
                    raise ValueError("Analytical observations differ from the declared solver revision")
        for entry in (old, meta):
            runner.verify_artifacts(record=entry, study=study)
        if "depth_rmse" not in result:
            continue
        if (meta["spec"] != old["spec"] or result["terrain_sha256"] != baseline["terrain_sha256"]
                or meta["config_sha256"] != old["config_sha256"]):
            raise ValueError("Analytical configuration or represented terrain changed between builds")
        with np.load(prior / "fields.npz") as a, np.load(path / "fields.npz") as b:
            agreement = field_comparison(before=a, after=b)
        spec = meta["spec"]
        row = dict(stage=meta["stage"], **spec, experiment_id=meta["experiment_id"],
                   before_execution_id=old["execution_id"], after_execution_id=meta["execution_id"],
                   before_attempt=old["attempt"], after_attempt=meta["attempt"],
                   before_source=old["context"]["source_sha256"], after_source=meta["context"]["source_sha256"],
                   corrected_steps=result["steps"], corrected_retries=result["rejects"], **agreement)
        for field in ("depth_rmse", "maximum_depth_rmse", "hydro_rmse", "peak_error_rmse", "timing_error_rmse"):
            row["before_" + field] = baseline[field]
            row["after_" + field] = result[field]
        rows.append(row)
    output = study / "fidelity/analytical_revision"
    metrics.csv_write(file=output / "comparisons.csv", rows=rows)
    primary = [row for row in rows if row["stage"] == "analytical"]
    if len(primary) != gates["primary_runs"]:
        raise ValueError("Missing matched core exact-solution results")
    unchanged = [row for row in primary if row["model_centreline_equal"] and row["model_gauges_equal"]]
    report = dict(audit_commit=provenance["audit"]["commit"], repair_commit=provenance["repair"]["commit"],
                  gates=gates, matched_field_pairs=len(rows), identical_primary_field_and_gauge_pairs=len(unchanged),
                  identical_primary_histories=sum(int(row["accepted_dt_s_equal"]) for row in primary),
                  comparisons_sha256=runner.checksum(output / "comparisons.csv"))
    runner.write_json(path=output / "verification.json", data=report)
    print(f"{len(unchanged)}/{len(primary)} core exact controls retain identical sampled fields and gauge transport; "
          f"{len(rows)} paired reference archives verified", flush=True)


def experiment_key(observation):
    return observation[0]["experiment_id"]


if __name__ == "__main__":
    study = Path(__file__).resolve().parent
    stages = ("analytical", "convergence", "fixed-dt-space")
    run(study=study, stages=stages)
