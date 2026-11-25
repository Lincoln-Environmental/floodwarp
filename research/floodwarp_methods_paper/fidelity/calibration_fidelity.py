"""Reassess the original bounded FV1 Manning transfers on archived discharge intervals."""

from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np

from research.floodwarp_methods_paper import evidence, metrics, runner
from research.floodwarp_methods_paper.fidelity import assessment, recession


def distinct_fits(*, rows):
    """One pair per pair of executed forecasts, retaining all objectives that selected it."""
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["baseline_id"], row["forecast_id"], row["bound"]].append(row)
    for selected in pairs.values():
        for field in ("case", "dx", "multiplier", "fit_id", "bound", "seed", "landform"):
            if len({r[field] for r in selected}) != 1:
                raise ValueError(f"Conflicting {field} for a calibration pair")
        yield dict(selected[0], fit_objectives=" ".join(sorted({r["objective"] for r in selected})))


def forecast_archive(*, study, needed):
    records = {}
    with evidence.archived_publication(study=study):
        for manifest in evidence.records(study=study, stages=("calibration", "long-storms", "resolution")):
            ident = manifest["experiment_id"]
            if ident not in needed:
                continue
            path = study / manifest["directory"] / "result.json"
            if runner.checksum(path=path) != manifest["artifacts"][path.name]:
                raise ValueError(f"Changed source forecast for calibration {ident}")
            value = assessment.item(spec=manifest["spec"], data=json.loads(path.read_text()),
                                    ident=ident, path=path.parent, cohort="legacy")
            if ident in records:
                raise ValueError(f"Ambiguous original calibration forecast {ident}")
            records[ident] = value
    if needed - records.keys():
        raise ValueError(f"Missing archived calibration forecasts {sorted(needed - records.keys())}")
    return records


def matched_pair(*, baseline, fitted, fit, row):
    for key in ("case", "landform", "seed", "initial", "dx", "dem", "hours", "method"):
        if baseline["s"][key] != fitted["s"][key]:
            raise ValueError(f"Calibration pair differs in {key}")
    if (baseline["s"]["method"] != "fv1" or baseline["s"]["case"] != row["case"]
            or baseline["s"]["dx"] != float(row["dx"]) or baseline["s"]["multiplier"] != 1
            or fitted["s"]["multiplier"] != float(row["multiplier"])
            or fit["s"]["role"] != "fit" or fit["s"]["multiplier"] != fitted["s"]["multiplier"]):
        raise ValueError("Calibration transfer differs from its fitted multiplier or scenario")
    terms = []
    for item in (baseline, fitted):
        forcing = item["d"]["forcing"].copy()
        n = forcing.pop("roughness")
        terms.append(forcing)
        if item is fitted:
            np.testing.assert_allclose(n, baseline["d"]["forcing"]["roughness"] * float(row["multiplier"]),
                                       rtol=1e-13, atol=0)
    if terms[0] != terms[1]:
        raise ValueError("Terrain, initial state, forcing or numerical controls differ in a Manning pair")


def passed_recession(*, item, config):
    data = item["d"]
    passed, _ = recession.settled(frames=data["samples"], storage=data["final_m3"],
                                  supply=data["initial_m3"] + data["rain_m3"], config=config)
    return passed


def reference_index(*, rows):
    index = {}
    for row in rows:
        if row["cohort"] != "legacy":
            continue
        key = row["case"], row["metric"]
        value = row["reference_id"], row["reference_sensitivity_pct"]
        if key in index and index[key] != value:
            raise ValueError("Conflicting reference controls on the original storm window")
        index[key] = value
    return index


def reassess(*, study, directory, target):
    design = json.loads((directory / "design.json").read_text())
    config = json.loads((directory / "recession_settings.json").read_text())
    selected = list(distinct_fits(rows=metrics.csv_read(file=study / "processed/comparisons/transfer.csv")))
    horizons = {r["case"]: float(r["common_end_h"]) for r in
                metrics.csv_read(file=directory / "small_storms/recession_horizons.csv") if r["cohort"] == "legacy"}
    references = reference_index(rows=metrics.csv_read(file=directory / "small_storms/outlet_metrics.csv"))
    needed = {row[key] for row in selected for key in ("baseline_id", "forecast_id", "fit_id")}
    controls = {"hancock": "hancock_adaptive_true_n_id", "rk2": "adaptive_true_n_id"}
    needed.update(row[key] for row in selected for key in controls.values() if row[key])
    needed.update(references[row["case"], metric][0] for row in selected for metric in design["reference_limits"])
    archive = forecast_archive(study=study, needed=needed)
    results = []
    for row in selected:
        case = row["case"]
        fit, baseline, fitted = (archive[row[k]] for k in ("fit_id", "baseline_id", "forecast_id"))
        matched_pair(baseline=baseline, fitted=fitted, fit=fit, row=row)
        by_metric = {k: references[case, k] for k in design["reference_limits"]}
        reference_ids = {v[0] for v in by_metric.values()}
        if len(reference_ids) != 1:
            raise ValueError("The observables have different reference hydrographs")
        ref = archive[reference_ids.pop()]
        if (ref["s"]["case"] != case or ref["s"]["hours"] != fitted["s"]["hours"]
                or ref["s"]["method"] != "adaptive" or ref["s"]["multiplier"] != 1):
            raise ValueError("Calibrated hydrographs require the prescribed true-roughness RK2 reference")
        original, adjusted = [assessment.compare(it=it, ref=ref, station="outlet", cfg=design)
                              for it in (baseline, fitted)]
        baseline_settled, adjusted_settled = [passed_recession(item=it, config=config) for it in (baseline, fitted)]
        high_order = {}
        for method, column in controls.items():
            if not row[column]:
                continue
            peer = archive[row[column]]
            if any(fitted["s"][k] != peer["s"][k] for k in ("case", "dx", "dem", "hours")):
                raise ValueError("True-roughness peer differs in event, spacing or physical horizon")
            physical = [{k: v for k, v in it["d"]["forcing"].items() if k != "numerical"} for it in (baseline, peer)]
            if physical[0] != physical[1]:
                raise ValueError("True-roughness hydraulic inputs differ from baseline FV1")
            high_order[method] = (peer, assessment.compare(it=peer, ref=ref, station="outlet", cfg=design),
                                  passed_recession(item=peer, config=config))
        hours = fitted["s"]["hours"]
        complete = hours == horizons[case] and baseline_settled and adjusted_settled
        for metric, limit in design["reference_limits"].items():
            raw = by_metric[metric][1]
            spread = float(raw) if raw else None
            observable = assessment.limits(row=original, key=metric) and assessment.limits(row=adjusted, key=metric)
            a, b = original.get(metric), adjusted.get(metric)
            change = abs(a) - abs(b) if a is not None and b is not None else None
            qualified = (complete and observable and baseline["s"]["dx"] > ref["s"]["dx"]
                         and spread is not None and spread <= limit)
            comparisons = {}
            for method in controls:
                comparisons[method + "_id"] = row[controls[method]]
                if method not in high_order:
                    continue
                peer, profile, settled = high_order[method]
                comparisons[method + "_true_n_pct"] = profile.get(metric)
                comparisons[method + "_execution"] = peer["d"]["mode"]
                eligible = qualified and settled and assessment.limits(row=profile, key=metric)
                comparisons[method + "_qualified"] = bool(eligible)
                comparisons[method + "_gain_over_fitted_fv1_points"] = abs(b) - abs(profile[metric]) if eligible else None
            results.append(dict(case=case, terrain_id=f"{row['landform']}-{row['seed']}", dx_m=row["dx"],
                                fit_terrain=fit["s"]["landform"] == row["landform"] and fit["s"]["seed"] == int(row["seed"]),
                                applied_n=fitted["d"]["forcing"]["roughness"], true_n=baseline["d"]["forcing"]["roughness"],
                                bound=row["bound"], multiplier=row["multiplier"], fit_objectives=row["fit_objectives"],
                                fit_id=fit["id"], baseline_id=baseline["id"], forecast_id=fitted["id"],
                                reference_id=ref["id"], reference_dx_m=ref["s"]["dx"],
                                end_h=hours, peers_common_end_h=horizons[case], cadence_s=original["interval_max_s"],
                                baseline_receded=baseline_settled, fitted_receded=adjusted_settled,
                                common_recession=complete, metric=metric, resolved_shapes_and_sampling=observable,
                                reference_limit_pct=limit, reference_sensitivity_pct=spread, qualified=qualified,
                                baseline_pct=a, fitted_pct=b, absolute_error_reduction_points=change,
                                beyond_reference_sensitivity=abs(change) > 2 * spread if qualified else None, **comparisons))
    metrics.csv_write(file=target / "transfer_fidelity.csv", rows=results)
    sources = [directory / "design.json", directory / "recession_settings.json",
               directory / "small_storms/recession_horizons.csv", directory / "small_storms/outlet_metrics.csv",
               study / "processed/comparisons/transfer.csv"]
    sources.extend(item["path"] / "result.json" for item in archive.values())
    runner.write_json(path=target / "sources.json", data={
        str(p.relative_to(study)): runner.checksum(path=p) for p in sorted(set(sources))})
    return results


def summarize(*, rows, target):
    groups = defaultdict(list)
    for row in rows:
        if not row["fit_terrain"]:
            groups[row["bound"], row["metric"]].append(row)
    summary = []
    for (bound, metric), peers in sorted(groups.items()):
        qualified = [r for r in peers if r["qualified"]]
        gains = np.asarray([r["absolute_error_reduction_points"] for r in qualified])
        summary.append(dict(bound=bound, metric=metric, forecast_pairs=len(peers),
                            terrains=len({r["terrain_id"] for r in peers}), qualified_pairs=len(qualified),
                            qualified_terrains=len({r["terrain_id"] for r in qualified}),
                            median_reduction_points=float(np.median(gains)) if len(gains) else None,
                            improved=int((gains > 0).sum()), worse=int((gains < 0).sum()),
                            improved_beyond_sensitivity=sum(r["beyond_reference_sensitivity"] and
                                                            r["absolute_error_reduction_points"] > 0 for r in qualified),
                            worse_beyond_sensitivity=sum(r["beyond_reference_sensitivity"] and
                                                         r["absolute_error_reduction_points"] < 0 for r in qualified)))
    metrics.csv_write(file=target / "heldout_fidelity.csv", rows=summary)
    return summary


def generate(*, study, target):
    observations = reassess(study=study, directory=study / "fidelity", target=target)
    summary = summarize(rows=observations, target=target)
    return len(observations), len(summary)


def verify(*, study, target):
    with TemporaryDirectory() as scratch:
        replica = Path(scratch)
        counts = generate(study=study, target=replica)
        for generated in replica.iterdir():
            if runner.checksum(path=target / generated.name) != runner.checksum(path=generated):
                raise ValueError(f"Archived calibration fidelity differs from regeneration in {generated.name}")
    return counts


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    target = study / "supplementary/manning_transfer"
    options = set(sys.argv[1:])
    if options.difference({"--dry-run", "--resume", "--force", "--verify-only"}):
        raise ValueError(f"Unknown calibration fidelity option {options}")
    if "--dry-run" in options:
        print("Reprocess archived calibrated forecasts, true-roughness peers and sensitivity checks on common windows.")
    else:
        action = verify if "--verify-only" in options else generate
        pairs, summaries = action(study=study, target=target)
        print(f"{pairs} paired observable comparisons, {summaries} held-out summaries.")
