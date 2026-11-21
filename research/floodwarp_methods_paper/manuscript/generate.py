"""Populate results and discussion from the frozen experiment outputs."""

from pathlib import Path
from research.floodwarp_methods_paper import synthesis
from statistics import median
from collections import defaultdict

from research.floodwarp_methods_paper.metrics import csv_read
from research.floodwarp_methods_paper.comparisons import exact_check


def fmt(value):
    return "unavailable" if value in (None, "") else f"{float(value):.4g}"


def generate(*, study):
    p = study / "processed"
    raw = csv_read(p / "metrics/storms.csv")
    primary = [
        r
        for r in raw
        if r["stage"] == "long-storms"
        and r["reference_setup"] == "False"
        and r["representation"] == "sampled"
    ]
    matched = defaultdict(dict)
    for r in primary:
        matched[r["case"]][r["method"]] = r
    ratios = [
        float(d["hancock_adaptive"]["runtime_s"]) / float(d["fv1"]["runtime_s"])
        for d in matched.values()
    ]
    dense = [float(d["hancock_dense"]["runtime_s"]) / float(d["fv1"]["runtime_s"]) for d in matched.values()]
    fastest = [min(a, b) for a,b in zip(dense,ratios,strict=True)]
    dense_wins = sum(a<b for a,b in zip(dense,ratios,strict=True))
    error = []
    for metric, screen, unit in (
        ("depth_rmse_m", "depth", "m"),
        ("outlet_rmse_m3s", "discharge", "m³/s"),
    ):
        pairs = [
            d
            for d in matched.values()
            if d["adaptive"]["qualified_" + screen] == "True"
        ]
        vals = [
            median(float(d[m][metric]) for d in pairs) if pairs else None
            for m in ("fv1", "hancock_adaptive", "adaptive")
        ]
        error.append(
            f"| {screen} ({unit}) | {len(pairs)} | " + " | ".join(map(fmt, vals)) + " |"
        )
    gates = exact_check(study=study)
    adv = csv_read(p / "advisor/summary.csv")
    a = next(r for r in adv if r["rule"] == "advisor")
    fits = csv_read(p / "comparisons/calibration.csv")
    transfers = csv_read(p / "comparisons/transfer.csv")
    qual = [r for r in transfers if r["qualified_for_objective"] == "True"]
    worse = sum(float(r["improvement"]) < 0 for r in qual)
    better = sum(float(r["improvement"]) > 0 for r in qual)
    result = (
        f"""# Results

## Exact and manufactured controls

The {gates["primary_runs"]} primary forecasts had a minimum depth of {gates["min_depth_m"]:g} m. The largest relative mass residual was {gates["maximum_relative_residual"]:.3g}. The independent lake-at-rest control had a maximum depth error of {gates["lake_maximum_depth_error_m"]:.3g} m. The refinement table includes spatial, fixed-grid temporal and source-integration sequences. [Analytical errors](../tables/04_analytical_accuracy.md) · [Convergence](../tables/05_convergence.md)

## Storm hydrographs and depth

The {len(matched)} prospective physical events included six hours of rain and a common recession for each matched comparison. The table below gives medians across the events that passed the refinement test for each observable. LI uses its native momentum equation. All methods and the error associated with reference refinement appear in the supporting tables. [Six-hour storm comparisons](../tables/06_storm_accuracy.md) · [Reference uncertainty](../processed/metrics/reference_uncertainty.csv)

| Observable | Qualified events | FV1 | Tiled Hancock | FV2 RK2 |
|---|---:|---:|---:|---:|
"""
        + "\n".join(error)
        + f"""

## Efficiency and terrain representation

On the 96-cell controls, dense Hancock was faster than tiles in {dense_wins} of {len(matched)} complete storms. Relative to sparse FV1 its median runtime was {median(dense):.3f}×, with {sum(r<=1.5 for r in dense)} events meeting 1.5×. Tiled Hancock took {median(ratios):.3f}× and met the target in {sum(r<=1.5 for r in ratios)} events. Selecting the faster qualified execution retrospectively gave {median(fastest):.3f}×. Both schedules reproduced the same Hancock fields. With only 96 cells, marking and compacting tiles adds GPU launches while eliminating little work. The large-domain suite measures how this balance changes with the scheduled footprint. [Measured GPU costs](../tables/07_runtime.md) · [Full performance qualification](../../../validation/hancock_long_storm_review/DECISION.md)

Fixed-geometry and resampled-geometry experiments use the same physical extent. Their objective-specific rankings and measured cost ratios are available for the plain, fan and incised development realisations. An unresolved reference is an explicit missing comparison. [Resolution profiles](../figures/06_resolution.pdf) · [Equal-cost results](../tables/08_cost_accuracy.md) · [Least-cost feasible configurations](../processed/pareto/least_cost.csv)

## Roughness transfer

The {len(fits)} nested training fits selected multipliers between {min(float(r["multiplier"]) for r in fits):.3g} and {max(float(r["multiplier"]) for r in fits):.3g}. Across {len(qual)} reference-qualified transfer rows, {better} scored better than physical roughness and {worse} scored worse. Multiple bounds can select the same roughness. These counts include repeated bounds and objectives on the same terrain. The group intervals preserve that dependence. [Bounded calibration](../tables/09_calibration.md) · [Grouped transfer intervals](../processed/comparisons/calibration_intervals.csv)

## Held-out solver decisions

The advisor made {a["selected"]} selections in {a["decisions"]} held-out objective decisions, including {a["unqualified_selections"]} with unresolved reference evidence. Among {a["qualified_selections"]} assessable selections, {fmt(a["met_tolerance_pct"])}% satisfied the requested accuracy and {a["false_feasible"]} exceeded it. Agreement with the measured least-cost method was {fmt(a["agreement_pct"])}%. The median conditional runtime regret was {fmt(a["median_regret_pct"])}%. The empirical interval covered the observed error in {fmt(a["empirical_error_interval_coverage_pct"])}% of assessable predictions. [Held-out rule comparison](../tables/10_advisor.md) · [Individual decisions](../processed/advisor/heldout.csv)
"""
    )
    discussion = f"""# Discussion

Tiled execution preserves Hancock's conservative update. Differences between Hancock and RK2 depend on the physical observable, timestep history and limiter response. The median storm errors in the results provide evidence at the tested scales. Small discharge peaks and controls that shift on refinement can make percentage comparisons unstable. A solver that attenuates a front may still predict a satisfactory final storage.

The practical outcome depends on the accuracy objective. The Pareto archive identifies measured configurations dominated in both error and runtime within an individual event. A coarse DEM can remove drainage connections or lower a barrier. Comparing fixed represented geometry with continuous terrain resampling quantifies how this changes the error attributable to the hydraulic grid. Flow accumulation on a filled DEM describes potential drainage. Stage control, infiltration and depressional ponding can change the actual flow path.

Bounded Manning fits target integrated transport or depth and extent. A change that compensates for attenuation on one hydrograph can alter recession or peak timing on the next. The grouped transfer results quantify the effect of applying the fitted physical value unchanged. The first- and higher-order solvers at the prescribed roughness remain the comparison controls. [Transfer trials](../processed/comparisons/transfer.csv)

The domain-wide solver advisor abstained on {a["abstentions"]} held-out objectives. Its runtime forecast had median absolute percentage error {fmt(a["runtime_median_absolute_pct_error"])}% for assessable choices. Compare coverage and failed choices together with cost. A low conditional regret can reflect abstention on difficult cases. The original small-domain training cases covered 9,600 m². The expanded splits include independent larger terrain and event realisations. Large-domain accuracy recommendations require a qualified reference for the requested observable, sufficient training support at the requested scale and admissible terrain representation. Additional independent domains, stage controls, storms, GPUs and surveyed hydraulic structures are needed before using forecasts for release decisions.

Retain adaptive Hancock as the development default. Use FV1 or LI when the error for the intended observable is acceptable at lower measured cost. LI also needs a suitable reduced-momentum regime. Publication or operational use of a method recommendation requires a qualified reference and credible predictive coverage on the intended domain. [Rule baselines](../tables/10_advisor.md)
"""
    coverage = [f"{r['rule']} {fmt(r['feasible_coverage_pct'])}%" for r in adv]
    discussion += (
        "\nAcross held-out objectives with at least one empirically feasible option, the proportions receiving a qualified feasible selection were "
        + ", ".join(coverage)
        + ". This denominator includes abstentions. The stratified validation tables identify the resolutions and sizes with independent predictive evidence.\n"
    )
    from research.floodwarp_methods_paper.qualification import paper
    audit,extension=paper.material(study=study)
    (study/"CORRECTIVE_AUDIT.md").write_text(audit)
    result += "\n" + synthesis.case_evidence(study=study) + "\n" + extension + "\n"
    (study / "LIMITATIONS.md").write_text(synthesis.limitations(study=study))
    manuscript = study / "manuscript"
    (manuscript / "results.md").write_text(result)
    (manuscript / "discussion.md").write_text(discussion)
    (study / "RESULTS.md").write_text(
        result.replace("](../", "](").replace(
            "](../../validation", "](../../validation"
        )
    )
    (study / "DISCUSSION.md").write_text(discussion.replace("](../", "]("))
    (manuscript / "conclusions.md").write_text(
        f"# Conclusions\n\nAcross {len(matched)} small-domain storms, the faster qualified Hancock execution met the 1.5× FV1 target in {sum(r<=1.5 for r in fastest)} cases with median {median(fastest):.3f}×. Tiled execution alone had median {median(ratios):.3f}×. A single solver ranking cannot represent the full set of objective-specific Pareto frontiers. Reference uncertainty, terrain representation and held-out feasibility constrain the decision. [Results](results.md) · [Discussion](discussion.md)\n"
    )
    return dict(
        events=len(matched),
        assessed_choices=int(a["qualified_selections"]),
        qualified_transfers=len(qual),
    )


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    generate(study=study)
