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
            "Each entry below is an outlet observation with independent spatial, temporal and cross-formulation checks. Positive centroid bias indicates later passage. Negative peak bias indicates attenuation. For the signed quantities, the final two columns count comparisons whose bias exceeds the reference-sensitivity envelope. Width is the central 80% discharge duration.",
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
    (target / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    target = directory / "small_storms"
    run(directory=directory, target=target)
