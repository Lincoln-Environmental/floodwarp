"""Rebuild the fidelity supplement and paper conclusions from completed independent receipts."""

from pathlib import Path
import json

import numpy as np
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    advisor_evaluation,
    assessment,
    figures,
    spatial,
    statistics,
    terrain,
)


def describe(*, rows, field):
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    return f"{np.median(values):.2f}" if values else "unresolved"


def selected(*, rows, **terms):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in terms.items())]


def summary(*, directory, storms, pulses, decisions):
    result = directory / "results"
    cfg = json.loads((directory / "design.json").read_text())
    run = selected(
        rows=storms, cohort="complete-legacy", method="fv1", station="outlet", dx=10
    )
    lines = [
        "# Hydrograph fidelity per unit hydraulic runtime",
        "",
        "## Six-hour storms",
        "",
        f"The primary matched comparison uses {len(run)} independent event configurations across eight landform families on 96-cell grids. Terrain seeds recur across storms. The table counts each physical event once. Positive timing means later arrival. Negative peak bias means attenuation. Width is the time between 10% and 90% of observed outward discharge volume. Magnitude is relative to the reference. Timing percentages use the reference central 80% duration.",
        "",
        "| FV1 outlet metric | Qualified / all | Negative / positive / zero | Median | IQR | 90th percentile absolute error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in cfg["reference_limits"]:
        v = np.array([r[k] for r in run if r.get("qualified_" + k)])
        if len(v):
            lines.append(
                f"| {k} | {len(v)} / {len(run)} | {(v < 0).sum()} / {(v > 0).sum()} / {(v == 0).sum()} | {np.median(v):+.2f}% | {np.percentile(v, 25):+.2f} to {np.percentile(v, 75):+.2f}% | {np.percentile(abs(v), 90):.2f}% |"
            )
        else:
            lines.append(f"| {k} | 0 / {len(run)} | — | unresolved | — | — |")
    horizons = metrics.csv_read(result / "recession_horizons.csv")
    longer = [r for r in horizons if r["extended"] == "True"]
    endpoints = [float(r["common_end_h"]) for r in horizons]
    lines.extend(
        [
            "",
            f"Across {len(horizons)} matched event/terrain-representation groups, {len(longer)} required additional hydraulic integration. Final common end times ranged from {min(endpoints):g} to {max(endpoints):g} hours. The physical rainfall ends at six hours. Both internal and outlet transport were checked over the final two hours. [Common end times](results/recession_horizons.csv) · [Tail volumes, discharge and remaining storage](results/recession_terminal.csv)",
        ]
    )
    peaks = [r for r in run if r.get("qualified_peak_bias_pct")]
    early = [r for r in peaks if r["peak_bias_pct"] < -r["uncertainty_peak_bias_pct"]]
    high = [r for r in peaks if r["peak_bias_pct"] > r["uncertainty_peak_bias_pct"]]
    centre = [r for r in run if r.get("qualified_centroid_delay_pct")]
    delayed = [
        r
        for r in centre
        if r["centroid_delay_s"] > r["interval_max_s"]
        and r["centroid_delay_pct"] > r["uncertainty_centroid_delay_pct"]
    ]
    advanced = [
        r
        for r in centre
        if r["centroid_delay_s"] < -r["interval_max_s"]
        and -r["centroid_delay_pct"] > r["uncertainty_centroid_delay_pct"]
    ]
    truncated = sum(bool(r.get("reference_truncated")) for r in run)
    flow = sum("reference_truncated" in r for r in run)
    lines += [
        "",
        f"Peak attenuation exceeded reference sensitivity in **{len(early)} of {len(peaks)}** qualified FV1 storms. Amplification exceeded it in **{len(high)}**. Among {len(centre)} qualified centroids, {len(delayed)} were late and {len(advanced)} early beyond both reference sensitivity and one output interval. In {truncated} of {flow} references with outward-flow diagnostics, terminal flow still exceeded 5% of the peak. Their measured-window volumes remain available. Full-event widths and runoff-volume conclusions require a captured rise and recession.",
        "",
        "## Fixed-bed propagation",
        "",
        "A smooth right-travelling wave has an independent pre-breaking characteristic solution. Exact interval means and simulated accepted face volumes cover the same physical section. The following measurements are at 200 m propagation distance under the larger prescribed timestep cap.",
        "",
        "| Grid (m) | Method | Peak bias | Volume bias | Central width bias | Redistributed volume | Centroid delay (s) | Hydraulic time (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pulses:
        if (
            r["case"] == "smooth"
            and r["distance_m"] == 200
            and r["cap_courant"] == max(cfg["pulse"]["cap_courant"])
        ):
            lines.append(
                f"| {r['dx']} | {r['method']} | {r['peak_bias_pct']:+.1f}% | {r['volume_bias_pct']:+.2f}% | {describe(rows=[r], field='width80_bias_pct')}% | {describe(rows=[r], field='shape_tv_pct')}% | {describe(rows=[r], field='centroid_delay_s')} | {r['runtime_s']:.3f} |"
            )
    base = selected(
        rows=pulses,
        case="smooth",
        distance_m=200,
        cap_courant=0.13,
        method="fv1",
        dx=10,
    )[0]
    h = selected(
        rows=pulses,
        case="smooth",
        distance_m=200,
        cap_courant=0.13,
        method="hancock_adaptive",
        dx=10,
    )[0]
    rk = selected(
        rows=pulses,
        case="smooth",
        distance_m=200,
        cap_courant=0.13,
        method="adaptive",
        dx=10,
    )[0]
    lines += [
        "",
        f"On the 10 m grid FV1 redistributed {base['shape_tv_pct']:.1f}% of the volume in time, compared with {h['shape_tv_pct']:.1f}% for Hancock and {rk['shape_tv_pct']:.1f}% for FV2. After aligning centroids these values were {base['aligned_tv_pct']:.1f}%, {h['aligned_tv_pct']:.1f}% and {rk['aligned_tv_pct']:.1f}%. Broadening and peak loss account for much of this fixed-bed difference. The diagnostic alignment preserves the originally reported lag. The 20 m results show that under-resolving a pulse affects every tested formulation.",
        "",
        "The two Courant caps vary the step on each fixed grid. Actual counts and durations are retained. Roe and HLLC are explicit FV1 flux variants. Each uses the production positivity fallback. The smooth longitudinal wave provides little opportunity for a contact-resolving flux to offset first-order spatial diffusion. The sharp and double waves and oblique section crossings have separate results. [Channel records](results/pulses.csv) · [Change between successive downstream sections](results/pulse_distance.csv) · [Orientation controls](results/orientation.csv) · [Measured frontiers](results/pareto.csv)",
        "",
        "## Matched practical benefit",
        "",
        "Positive gain is a reduction in absolute error, measured in percentage points. The intervals resample whole terrain realisations and retain all their matched storms. They express between-terrain sampling variability. Reference sensitivity is reported separately for every pair.",
        "",
        "| Earlier / later method | Metric | Pairs | Terrains | Median gain | Grouped 95% interval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in metrics.csv_read(result / "paired_intervals.csv"):
        if r["metric"] in (
            "peak_bias_pct",
            "t50_delay_pct",
            "shape_tv_pct",
            "width80_bias_pct",
            "volume_bias_pct",
        ):
            lines.append(
                f"| {r['first']} / {r['second']} | {r['metric']} | {r['pairs']} | {r['terrain_groups']} | {float(r['median_gain']):+.2f} | {float(r['grouped_low']):+.2f} to {float(r['grouped_high']):+.2f} |"
            )
    lines += [
        "",
        "## Operational scale and terrain",
        "",
        "The following normal-timestep outlet forecasts used 1,048,576 cells on the reserved mixed catchment. Errors are against a 5 m numerical reference over the same physical domain. The screens compare an 8 m FV2 refinement, a 5 m Hancock solution and a 5 m FV2 temporal refinement. The rolling soil-infiltration event also uses 1,048,576 cells at 5 m, with 4 m and 3.2 m FV2, 3.2 m Hancock and a 3.2 m half-CFL control. Each row reports a measured difference over the common recession horizon. The observable-specific reference screens determine which differences can support an accuracy ranking.",
        "",
        "| Terrain, grid, execution | End (h) | Runtime (s) | Peak bias | Volume bias | Centroid delay (min) | Width bias | Shape / aligned (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mixed = selected(
        rows=storms,
        cohort="complete-expanded",
        case="mixed-708-spatial-10240x10240",
        station="outlet",
        dx=10,
    )
    rolling = selected(
        rows=storms,
        cohort="complete-expanded",
        case="rolling-704-soil-5120x5120",
        station="outlet",
        dx=5,
    )
    for r in mixed + rolling:
        lines.append(
            f"| {r['landform']}, {r['dx']:g} m, {r['formulation']} {r['execution_mode']} | {r['hours']:g} | {r['runtime_s']:.1f} | {describe(rows=[r], field='peak_bias_pct')}% | {describe(rows=[r], field='volume_bias_pct')}% | {r.get('centroid_delay_s', float('nan')) / 60:+.1f} | {describe(rows=[r], field='width80_bias_pct')}% | {describe(rows=[r], field='shape_tv_pct')} / {describe(rows=[r], field='aligned_tv_pct')} |"
        )
    double = selected(
        rows=storms,
        cohort="complete-double",
        case="floodplain-double-rain",
        station="outlet",
        dx=10,
    )
    if double:
        lines.extend(
            [
                "",
                "## Two-pulse rainfall over a fixed floodplain bed",
                "",
                "The 10 m forecasts share the represented floodplain geometry with their finer hydraulic reference. Distinct crests can merge or separate during routing. The table gives signed peak error, volume redistribution and the number of resolved modes from the measured one-minute interval fluxes.",
                "",
                "| Method | End (h) | Model / reference modes | Peak bias | Peak qualified | Redistributed volume | Shape qualified |",
                "|---|---:|---:|---:|---|---:|---|",
            ]
        )
        for item in double:
            peak = describe(rows=[item], field="peak_bias_pct")
            shape = describe(rows=[item], field="shape_tv_pct")
            lines.append(
                f"| {item['method']} | {item['hours']:g} | {item.get('model_modes', '—')} / {item.get('reference_modes', '—')} | {peak}% | {item.get('qualified_peak_bias_pct', False)} | {shape}% | {item.get('qualified_shape_tv_pct', False)} |"
            )
        lines.extend(
            [
                "",
                "[Complete two-pulse hydrograph and cumulative transport](figures/floodplain-double-rain.pdf)",
            ]
        )
    long_pairs = metrics.csv_read(result / "horizon_effects.csv")
    lines.append("")
    for component in ("volume_bias_pct", "width80_bias_pct", "shape_tv_pct"):
        changes = [
            abs(float(r["change"]))
            for r in long_pairs
            if r["method"] == "fv1"
            and r["station"] == "outlet"
            and r["metric"] == component
        ]
        if changes:
            lines.append(
                f"Extending FV1 and its reference changed {component} by a median {np.median(changes):.2f} percentage points across {len(changes)} available event/grid comparisons (maximum {max(changes):.2f})."
            )
    cases = {
        r["case"] for r in storms if r["cells"] > 1000000 and r["station"] == "outlet"
    }
    qualified = {
        r["case"]
        for r in storms
        if r["cohort"] == "complete-expanded"
        and r["cells"] > 1000000
        and r["station"] == "outlet"
        and r.get("qualified_shape_tv_pct")
    }
    lines += [
        "",
        f"The archive contains {len(cases)} operational or reference contexts above one million cells, with {len(qualified)} supporting a qualified full-event outlet shape comparison at that scale. All runs lacking a fine reference or a same-grid temporal check remain labelled. Runtime alone cannot establish fidelity there. [Hydraulic observations](results/storms.csv) · [Reference screens](results/reference_checks.csv) · [Outlet hydrographs](figures/mixed-708-spatial-10240x10240.pdf)",
        "",
    ]
    beds = [
        r
        for r in metrics.csv_read(result / "bed_contrasts_complete.csv")
        if r["station"] == "outlet"
        and r["method"] in ("hancock_dense", "hancock_adaptive")
        and r["metric"] == "shape_tv_pct"
        and r["observable"] == "True"
    ]
    unique = {(r["case"], r["dx"], r["dem"], r["initial_grid"]): r for r in beds}
    beds = list(unique.values())
    major = sum(
        abs(float(r["bed_difference"])) > abs(float(r["formulation_difference"]))
        for r in beds
    )
    lines.append(
        f"At fixed hydraulic spacing, changing the represented bed and its associated initial mapping changed normalised outlet shape more than switching from FV1 to Hancock in {major} of {len(beds)} available contrasts. These paired changes describe sensitivity. They cannot be added as independent error components. The depth maps report arrival only at cells with an observed crossing in both runs and count missed connections separately. [Completed bed controls](results/bed_contrasts_complete.csv) · [Arrival maps](figures/arrival_maps.pdf) · [Original-window spatial differences](results/spatial.csv) · [Recession-end depth, extent and connectivity](results/recession_depth.csv)"
    )
    lines += [
        "",
        "## Solver recommendation",
        "",
        "The advisor now accepts separate runoff-volume, peak, propagation, width and unaligned shape tolerances. It compares error magnitude plus reference sensitivity and uses independent validation terrains for uncertainty padding. Spatial arrival and transient momentum can return insufficient evidence. The held-out decisions below use the same opportunity set at each tolerance.",
        "",
        "| Tolerance (%) | Objective | Decisions | Selected | Assessed | Within tolerance | Measured feasible opportunities |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for tol in sorted({r["tolerance"] for r in decisions}):
        for obj in (
            "relative_peak_discharge",
            "integrated_runoff",
            "outlet_arrival",
            "hydrograph_shape",
            "hydrograph_spreading",
        ):
            d = selected(rows=decisions, tolerance=tol, objective=obj)
            lines.append(
                f"| {tol:g} | {obj} | {len(d)} | {sum(bool(r['method']) for r in d)} | {sum(r['feasible'] is not None for r in d)} | {sum(r['feasible'] is True for r in d)} | {sum(bool(r['cheapest']) for r in d)} |"
            )
    baseline_rows = metrics.csv_read(result / "advisor_baselines.csv")
    predictions = metrics.csv_read(result / "advisor_predictions.csv")
    default = json.loads((directory / "application_settings.json").read_text())[
        "benchmark_tolerance_pct"
    ]
    lines.extend(
        [
            "",
            f"At the {default:g}% application tolerance every rule faces the same terrain, resolution and objective decisions. The table reports failures, unknown outcomes and abstentions separately. Regret compares cost only when the selected configuration met tolerance and a measured feasible benchmark exists.",
            "",
            "| Rule | Decisions | Abstained | Selected, unassessable | Failed tolerance | Met tolerance | Matched cheapest feasible | Median feasible regret |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in (
        "advisor",
        "slope_only",
        "always_fv1",
        "always_hancock_dense",
        "oracle",
    ):
        rows = [
            r
            for r in baseline_rows
            if float(r["tolerance"]) == default and r["rule"] == label
        ]
        abstentions = sum(r["selected"] == "False" for r in rows)
        unknown = sum(
            r["selected"] == "True" and r["assessable"] == "False" for r in rows
        )
        failed = sum(r["meets"] == "False" for r in rows)
        passed = sum(r["meets"] == "True" for r in rows)
        same = sum(r["meets"] == "True" and r["method"] == r["oracle"] for r in rows)
        regret = [float(r["regret_pct"]) for r in rows if r["regret_pct"]]
        display = f"{np.median(regret):.1f}%" if regret else "unresolved"
        lines.append(
            f"| {label} | {len(rows)} | {abstentions} | {unknown} | {failed} | {passed} | {same} | {display} |"
        )
    eligible = [r for r in predictions if r["interval_covers"] in ("True", "False")]
    covered = sum(r["interval_covers"] == "True" for r in eligible)
    lines.extend(
        [
            "",
            f"On held-out candidate predictions the interval contained the measured error plus reference sensitivity in {covered}/{len(eligible)} assessable method–objective combinations. Repeated predictions from the same terrain remain dependent. [Baseline decisions](results/advisor_baselines.csv) · [Prediction intervals and runtimes](results/advisor_predictions.csv)",
        ]
    )
    lines += [
        "",
        "The held-out evidence limits transfer to a new catchment. The terrain associations are exploratory rank correlations of median formulation gains across distinct terrain realisations. Discharge and realised wet fraction are available only after a run. They stratify evaluation and do not select a solver in advance. No universal grid-spacing cutoff follows from these results. [Independent advisor decisions](results/advisor_decisions.csv) · [Conditioned terrain descriptors](results/terrain_descriptors.csv) · [Associations](results/terrain_associations.csv)",
        "",
        "## What changed",
        "",
        "A volume-conserving forecast can substantially attenuate and spread a travelling wave. FV1, Hancock and FV2 are distinguishable on that control even when final-depth and event-volume summaries suggest small differences. The exact and rainfall results support selecting against the hydraulic quantity required by the application. Higher order alone does not establish that a peak, an internal wave or a recession is resolved. FV1 remains a candidate wherever its measured uncertainty-adjusted error meets the specified tolerance. The development solver default stays adaptive Hancock.",
        "",
        "[Metric definitions, censoring and reproduction](README.md) · [Absolute and relative metrics by cohort](results/strata.csv) · [Paired differences](results/pairs.csv) · [Peak sensitivity to output interval](results/cadence.csv) · [Equal-cost decisions](results/equal_cost_fidelity.csv) · [Least-cost feasible methods](results/least_cost_fidelity.csv) · [Cost by recession and wet fraction](results/cost_by_phase.csv)",
    ]
    narrative = (
        f"In the exact smooth pulse after 200 m on a 10 m grid, FV1 peak bias was {base['peak_bias_pct']:.1f}% "
        f"with {base['volume_bias_pct']:+.2f}% volume bias. Central 80% duration increased by {base['width80_bias_pct']:.1f}%, "
        f"compared with {h['width80_bias_pct']:.1f}% for Hancock and {rk['width80_bias_pct']:.1f}% for FV2. "
        f"Across the 48 small six-hour storms, {len(peaks)} FV1 outlet peaks passed the revised reference and sampling screens. "
        f"Attenuation exceeded reference sensitivity in {len(early)} and amplification exceeded it in {len(high)}. "
        f"Only {len(centre)} supported a full-event centroid comparison. Reported frequency depends on the eligible hydraulic objective."
    )
    for r in metrics.csv_read(result / "completed_cost_summary.csv"):
        if r["policy"] == "fastest_over_fv1":
            suite = (
                "small 10 m controls"
                if r["suite"] == "small-48"
                else "million-cell landforms"
            )
            sentence = (
                f"At their verified recession endpoints the faster measured explicit Hancock policy took a median "
                f"{float(r['median']):.2f}× FV1 across the {suite}, meeting 1.5× in {r['n_within_target']}/{r['events']} events."
            )
            narrative += " " + sentence
            lines.extend(
                [
                    "",
                    sentence
                    + " [Per-event timings and run identities](results/completed_costs.csv) · [Mixed-catchment residual storage](storage_tail/README.md)",
                ]
            )
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    (directory / "paper_summary.md").write_text(narrative + "\n")


def run(*, directory, default_percent, factors):
    from research.floodwarp_methods_paper.fidelity import native_horizons, tail_report

    native_horizons.generate(directory=directory)

    tail_report.generate(directory=directory)
    groups, rows = assessment.process(directory=directory)
    from research.floodwarp_methods_paper.fidelity import performance

    performance.costs(directory=directory, rows=rows)
    from research.floodwarp_methods_paper.fidelity import phase_cost

    settings = json.loads((directory / "application_settings.json").read_text())
    from research.floodwarp_methods_paper.fidelity import final_depth, parity

    parity.compare(directory=directory, groups=groups, target=directory / "results")

    final_depth.process(
        groups=groups,
        directory=directory,
        target=directory / "results",
        settings=settings["endpoint_depth"],
    )
    phase_cost.collect(
        groups=groups,
        directory=directory,
        cadence=settings["cost_interval_s"],
        edges=settings["wet_fraction_edges"],
        phase_edges=settings["phase_end_hours"],
    )
    cfg = json.loads((directory / "design.json").read_text())
    pulses = assessment.pulse_rows(directory=directory, cfg=cfg)
    from research.floodwarp_methods_paper.fidelity import pulse_products

    pulse_products.distance_effects(rows=pulses, target=directory / "results")
    from research.floodwarp_methods_paper.fidelity.extended_analysis import (
        horizon_effects,
    )

    horizon_effects(directory=directory, rows=rows, cfg=cfg)
    from research.floodwarp_methods_paper.fidelity.contrasts import compute

    compute(groups=groups, cfg=cfg, directory=directory)
    spatial.run(groups=groups, target=directory / "results")
    statistics.run(
        directory=directory, rows=rows, groups=groups, pulses=pulses, cfg=cfg
    )
    terrain.investigate(directory=directory, cfg=cfg)
    from research.floodwarp_methods_paper.fidelity import bed_recession, transient

    bed_recession.compare_beds(directory=directory, config=cfg)
    transient.run(directory=directory)
    decisions = advisor_evaluation.build(
        directory=directory, default_percent=default_percent, factors=factors
    )
    figures.plots(directory=directory, groups=groups)
    summary(directory=directory, storms=rows, pulses=pulses, decisions=decisions)
    from research.floodwarp_methods_paper.manuscript.generate import generate

    generate(study=directory.parent)
    from research.floodwarp_methods_paper.fidelity import catalogue

    catalogue.database(target=directory / "results")
    hashes = catalogue.manifest(
        directory=directory,
        package=directory.parents[2]
        / "src/floodwarp/solver_advisor/fidelity_evidence.json",
    )
    runner.write_json(path=directory / "outputs.json", data=hashes)
    print(
        f"{len(pulses)} exact pulse sections, {len(rows)} storm sections, {len(decisions)} held-out decisions, {len(hashes)} verified products"
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "application_settings.json").read_text())
    default_percent = settings["benchmark_tolerance_pct"]
    tolerance_factors = tuple(
        tol / default_percent for tol in settings["tolerances_pct"]
    )
    run(directory=directory, default_percent=default_percent, factors=tolerance_factors)
