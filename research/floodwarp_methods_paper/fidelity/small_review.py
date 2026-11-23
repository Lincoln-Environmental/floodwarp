"""Matched changes in outlet fidelity when all 48 original synthetic storms have receded."""

from collections import Counter
import json
from pathlib import Path
import numpy as np
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment, extended_analysis
from research.floodwarp_methods_paper.plan import read_config


def run(*, directory, target):
    design = json.loads((directory / "design.json").read_text())
    study = read_config(path=directory.parent / "configs/study.json")
    archive = assessment.archive(directory=directory, config=study, partial=True)
    small = {key: items for key, items in archive.items() if key[0] == "legacy"}
    complete = extended_analysis.attach(
        directory=directory, groups=small, target=target
    )
    if len(complete) != 48:
        raise ValueError("The 48-terrain/event matched investigation is incomplete")
    rows, _ = assessment.storm_rows(groups=small | complete, cfg=design)
    evidence = []
    for r in rows:
        if r["station"] != "outlet" or r["dx"] != 10:
            continue
        for metric in design["reference_limits"]:
            evidence.append(
                dict(
                    case=r["case"],
                    landform=r["landform"],
                    seed=r["seed"],
                    cohort=r["cohort"],
                    method=r["method"],
                    hours=r["hours"],
                    metric=metric,
                    qualified=r.get("qualified_" + metric, False),
                    value_pct=r.get(metric),
                    reference_sensitivity_pct=r.get("uncertainty_" + metric),
                    experiment_id=r["experiment_id"],
                    reference_id=r["reference_id"],
                    hydraulic_s=r["runtime_s"],
                )
            )
    metrics.csv_write(file=target / "outlet_metrics.csv", rows=evidence)
    horizons = metrics.csv_read(target / "recession_horizons.csv")
    ends = Counter(float(r["common_end_h"]) for r in horizons)
    added = sum(r["extended"] == "True" for r in horizons)
    report = [
        "# Recession in the 48 small six-hour rainfall events",
        "",
        f"All 48 events completed at 10 m on the same 120 × 80 m physical domain, with 5 m and 2.5 m refinements. {added} needed additional recession after hour 8. Each group uses its slowest passing configuration to determine the common end time.",
        "",
        "| Common end (h) | Events |",
        "|---:|---:|",
    ]
    report.extend(f"| {h:g} | {n} |" for h, n in sorted(ends.items()))
    report.extend(
        [
            "",
            f"Each 10 m forecast is compared with the 2.5 m adaptive FV2 outlet hydrograph for the same physical event. Its numerical sensitivity is the largest observed change against 5 m FV2, 2.5 m FV2 with a halved CFL setting and 2.5 m Hancock. The peak sensitivity limit is {design['reference_limits']['peak_bias_pct']:g}% and the width limit is {design['reference_limits']['width80_bias_pct']:g}%. Each observable must also pass the discharge and temporal sampling checks. Positive centroid bias indicates later passage. Negative peak bias indicates attenuation. For the signed quantities, the final two columns count comparisons whose bias exceeds the reference-sensitivity envelope. Width is the central 80% discharge duration.",
            "",
            "| 10 m method | Observable | Qualified / 48 | Median signed (%) | Median absolute (%) | Below envelope | Above envelope |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "fv1": "FV1 contact-HLL",
        "hancock_dense": "Hancock dense",
        "hancock_adaptive": "Hancock tiled",
        "adaptive": "FV2 RK2",
    }
    measures = {
        "peak_bias_pct": "Peak bias",
        "volume_bias_pct": "Discharge volume",
        "centroid_delay_pct": "Centroid delay",
        "width80_bias_pct": "Central 80% duration",
        "shape_tv_pct": "Redistributed volume",
    }
    for method in ("fv1", "hancock_dense", "adaptive"):
        for k in (
            "peak_bias_pct",
            "volume_bias_pct",
            "centroid_delay_pct",
            "width80_bias_pct",
            "shape_tv_pct",
        ):
            g = [
                r
                for r in evidence
                if r["cohort"] == "complete-legacy"
                and r["method"] == method
                and r["metric"] == k
                and r["qualified"]
            ]
            if g:
                v = np.asarray([r["value_pct"] for r in g])
                u = np.asarray([r["reference_sensitivity_pct"] for r in g])
                report.append(
                    f"| {labels[method]} | {measures[k]} | {len(g)} | {np.median(v):+.2f} | {np.median(abs(v)):.2f} | {(v < -u).sum()} | {(v > u).sum()} |"
                )
            else:
                report.append(
                    f"| {labels[method]} | {measures[k]} | 0 | unresolved | unresolved | — | — |"
                )
    from research.floodwarp_methods_paper.fidelity.statistics import matched

    complete_rows = [
        r for r in rows if r["cohort"] == "complete-legacy" and r["station"] == "outlet"
    ]
    pairs = matched(rows=complete_rows, objectives=design["reference_limits"])
    metrics.csv_write(file=target / "paired_differences.csv", rows=pairs)
    report.extend(
        [
            "",
            "## Paired hydraulic differences",
            "",
            "Positive gain means a reduction in absolute error. The last columns count gains or losses exceeding the sum of the two reference-sensitivity estimates.",
            "",
            "| Comparison | Metric | Matched events | Median gain (points) | Better beyond sensitivity | Worse beyond sensitivity |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for first, second in (("fv1", "hancock_dense"), ("hancock_dense", "adaptive")):
        for metric in ("peak_bias_pct", "shape_tv_pct", "width80_bias_pct"):
            g = [
                r
                for r in pairs
                if (r["first"], r["second"], r["metric"]) == (first, second, metric)
            ]
            if not g:
                continue
            gains = np.array([r["absolute_error_reduction"] for r in g])
            uncertainty = np.array([r["reference_envelope_sum"] for r in g])
            report.append(
                f"| {labels[first]} to {labels[second]} | {measures[metric]} | {len(g)} | {np.median(gains):+.2f} | {(gains > uncertainty).sum()} | {(gains < -uncertainty).sum()} |"
            )
    indexed = {(r["case"], r["method"]): r["runtime_s"] for r in complete_rows}
    cases = sorted({case for case, _ in indexed})
    report.extend(
        [
            "",
            "## Measured runtime",
            "",
            "| Method | Median runtime / FV1 | At or below 1.5× |",
            "|---|---:|---:|",
        ]
    )
    for name in ("hancock_dense", "hancock_adaptive", "adaptive"):
        ratio = np.array([indexed[case, name] / indexed[case, "fv1"] for case in cases])
        report.append(
            f"| {labels[name]} | {np.median(ratio):.3f} | {(ratio <= 1.5).sum()}/{len(cases)} |"
        )
    report.extend(
        [
            "",
            "These grids contain 96 cells at the candidate resolution. The table supports event-scale numerical comparisons on this geometry. The million-cell operational cases have separate measurements.",
            "",
            "[Paired outlet results and reference IDs](outlet_metrics.csv) · [Paired differences](paired_differences.csv) · [Matched horizons](recession_horizons.csv) · [Final two-hour transport and storage](recession_terminal.csv)",
        ]
    )
    from research.floodwarp_methods_paper.fidelity import final_depth, landform_results

    methods = ("fv1", "hancock_dense", "adaptive")
    families = landform_results.collect(
        evidence=evidence, target=target, methods=methods
    )
    report.extend(
        [
            "",
            "## Peak discharge by landform",
            "",
            "Each value is median absolute peak bias over the storms that qualify for all three methods. The counts give the number of matched storms and distinct terrain seeds. Results for a single qualified storm describe that case only.",
            "",
        ]
    )
    report.extend(
        landform_results.table(
            summaries=families, metric="peak_bias_pct", methods=methods
        )
    )
    report.extend(
        [
            "",
            "[Signed distributions and relative runtime for every landform and objective](landform_fidelity.csv)",
        ]
    )

    from research.floodwarp_methods_paper.fidelity import parity

    checks = parity.compare(directory=directory, groups=complete, target=target)
    exact_state = sum(bool(c.get("state")) for c in checks)
    report.extend(
        [
            "",
            f"Dense and tiled Hancock reproduced the same interval transports and accepted step counts in all {len(checks)} paired forecasts. Final conservative depth and both momentum components, cumulative face volumes, soil history and maximum depth had identical FP64 byte hashes in the {exact_state} extended forecasts with retained solver checkpoints. [Exact fields compared in each forecast](recession_execution_parity.csv)",
        ]
    )
    settings = json.loads((directory / "application_settings.json").read_text())
    depth = final_depth.process(
        groups=complete,
        directory=directory,
        target=target,
        settings=settings["endpoint_depth"],
    )
    qualified_depth = {
        r["id"]: r
        for r in depth
        if r["dx"] == 10 and r["method"] == "fv1" and r.get("qualified_depth_rmse_m")
    }
    threshold_m = settings["endpoint_depth"]["depth_floor_m"]
    report.extend(
        [
            "",
            "## Depth and wave errors on the same forecast",
            "",
            f"The table counts how often a reference-qualified FV1 final-depth RMSE of at most {threshold_m * 1000:g} mm accompanied an outlet error above the specified tolerance. A comparison needs independent reference support for both quantities. Each row uses a different hydraulic observable.",
            "",
            "| Observable | Jointly assessable | Depth within threshold | "
            + " | ".join(f"Outlet exceeds {t:g}%" for t in settings["tolerances_pct"])
            + " |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    joint = []
    for k in (
        "peak_bias_pct",
        "shape_tv_pct",
        "width80_bias_pct",
        "centroid_delay_pct",
    ):
        pairs = [
            r
            for r in evidence
            if r["cohort"] == "complete-legacy"
            and r["method"] == "fv1"
            and r["metric"] == k
            and r["qualified"]
            and r["experiment_id"] in qualified_depth
        ]
        close = [
            r
            for r in pairs
            if qualified_depth[r["experiment_id"]]["depth_rmse_m"] <= threshold_m
        ]
        exceed = [
            sum(abs(r["value_pct"]) > t for r in close)
            for t in settings["tolerances_pct"]
        ]
        for tol, count in zip(settings["tolerances_pct"], exceed, strict=True):
            joint.append(
                dict(
                    method="fv1",
                    dx=10,
                    metric=k,
                    jointly_assessable=len(pairs),
                    final_depth_limit_m=threshold_m,
                    within_depth=len(close),
                    outlet_tolerance_pct=tol,
                    exceeds_outlet=count,
                )
            )
        report.append(
            f"| {measures[k]} | {len(pairs)} | {len(close)} | "
            + " | ".join(str(n) for n in exceed)
            + " |"
        )
    report.extend(
        [
            "",
            "[Final depth, maximum depth, connectivity and reference sensitivities](recession_depth.csv) · [Measured depth and extent frontiers](recession_depth_pareto.csv)",
        ]
    )
    metrics.csv_write(file=target / "joint_depth_hydrograph.csv", rows=joint)
    from research.floodwarp_methods_paper.fidelity import recession_figures

    recession_figures.depth_and_wave(
        source=target,
        folder=directory / "figures",
        tolerance_pct=settings["benchmark_tolerance_pct"],
        depth_limit_m=threshold_m,
    )
    report.extend(
        [
            "",
            "[Final depth against outlet peak and shape errors](../figures/final_depth_and_outlet_wave.pdf)",
        ]
    )
    (target / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    target = directory / "small_storms"
    run(directory=directory, target=target)
