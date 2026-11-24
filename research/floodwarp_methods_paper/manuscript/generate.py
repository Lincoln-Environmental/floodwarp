"""Populate results and discussion from the frozen experiment outputs."""

from pathlib import Path
from research.floodwarp_methods_paper import synthesis
from statistics import median
from collections import defaultdict

from research.floodwarp_methods_paper.metrics import csv_read
from research.floodwarp_methods_paper.comparisons import exact_check


def fmt(value):
    return "unavailable" if value in (None, "") else f"{float(value):.4g}"


def repaired_budgets(*, directory):
    import json

    balances = directory / "water_differences.csv"
    if not balances.exists():
        return ""
    extent = {r["run_id"]: r for r in csv_read(file=directory / "extent.csv") if float(r["hour"]) == 8}
    lines = []
    provenance = directory / "revision_evidence.json"
    if provenance.exists():
        proof = json.loads(provenance.read_text())
        gpu = sorted({r["gpu"] for r in proof["launch_environment"].values()})
        before, after = (proof["revisions"][tag]["commit"][:12] for tag in ("audit", "repair"))
        lines.append(f"The pre-repair and corrected hydraulic sources match commits `{before}` and `{after}`. "
                     f"The corrected FP64 storm measurements used {' and '.join(gpu)}. "
                     "[Source and workstation receipts](../fidelity/outlet_results/revision_evidence.json)")
    reach = directory / "incised_reach.csv"
    if reach.is_file():
        profiles = {(r["revision"], r["method"], int(r["extension_m"])): float(r["discharge_m3s"])
                    for r in csv_read(file=reach) if int(r["dx_m"]) == 10}
        if all((version, "fv2", offset) in profiles for version in ("audit", "repair") for offset in (0, 100)):
            old_face, old_buffer, new_face, new_buffer = (
                profiles[(version, "fv2", offset)] for version in ("audit", "repair") for offset in (0, 100)
            )
            lines.append(f"At the fixed 120 m-wide incised section on the 10 m grid, adding a 100 m "
                         f"downstream buffer changed mean RK2 discharge from {old_face:.5f} to {old_buffer:.5f} m³/s "
                         f"before the repair and from {new_face:.5f} to {new_buffer:.5f} m³/s after it. "
                         "These accepted volumes between 30 and 60 s measure sensitivity to outlet placement. "
                         "[Fixed-section measurements on the 1, 5 and 10 m grids](../fidelity/outlet_results/incised_reach.csv)")
    for row in csv_read(file=balances):
        if (row["first_method"],row["second_method"],row["extension_m"]) != ("fv1","hancock_adaptive","0"):
            continue
        first, second = extent[row["first"]], extent[row["second"]]
        lines.append(f"At {row['dx_m']} m, FV1 minus Hancock net export was {float(row['gauge_m3']):+,.0f} m³. "
                     f"The corresponding cumulative infiltration differences were {float(row['infiltration_inside_rain_m3']):+,.0f} m³ "
                     f"inside the rainfall footprint and {float(row['infiltration_outside_rain_m3']):+,.0f} m³ outside. "
                     f"The final storage difference was {float(row['footprint_storage_m3']):+,.0f} m³. "
                     f"At hour eight their areas above 5 cm were {float(first['area_above_5cm_m2'])/1e6:.4f} "
                     f"and {float(second['area_above_5cm_m2'])/1e6:.4f} km².")
    regions = directory / "regional_contrasts.csv"
    if regions.exists():
        eight = [r for r in csv_read(file=regions) if float(r["hour"]) == 8 and int(r["dx_m"]) == 5
                 and int(r["extension_m"]) == 0]
        field = "fv1_only_above_5cm_m2"
        total = sum(float(r[field]) for r in eight)
        if total:
            downstream = sum(float(r[field]) for r in eight if r["reach"] != "upstream of internal gauge")
            dry = sum(float(r[field]) for r in eight if r["rainfall_footprint"] == "outside")
            strip = sum(float(r[field]) for r in eight if r["reach"] == "final strip")
            lines.append(f"Of the hour-eight area above 5 cm in FV1 alone on the direct 5 m grid, "
                         f"{100*downstream/total:.2f}% lay downstream of the independent internal gauge "
                         f"and {100*dry/total:.2f}% lay outside the rainfall footprint. The final 40 m contained "
                         f"{100*strip/total:.2f}% of this disagreement. "
                         "[Reach and rainfall-footprint decomposition](../fidelity/outlet_results/regional_contrasts.csv) · "
                         "[Maps of inundation and cumulative infiltration](../supplementary/incised/hour_eight_water.pdf)")
    events = csv_read(file=directory / "events.csv")
    strip_path = directory / "last_40m.csv"
    if strip_path.exists():
        terminal = {(r["method"], r["period"]): r for r in csv_read(file=strip_path)
                    if int(r["dx_m"]) == 10 and int(r["extension_m"]) == 0 and float(r["cfl_factor"]) == 1}
        for method, name in (("hancock_adaptive", "Hancock"), ("adaptive", "RK2")):
            peak = terminal.get((method, "crest_minute"))
            recession = terminal.get((method, "post_rain_recession"))
            if peak and recession:
                lines.append(f"During the corrected 10 m {name} crest minute, "
                             f"{float(peak['net_into_strip_m3']):,.1f} m³ entered the final 40 m, "
                             f"{float(peak['net_export_m3']):,.1f} m³ passed the FREE face, "
                             f"{float(peak['infiltration_m3']):.1f} m³ infiltrated, and surface storage "
                             f"changed by {float(peak['end_storage_m3']) - float(peak['start_storage_m3']):+.1f} m³. "
                             f"Over the post-rain recession, drawdown in that strip after infiltration "
                             f"accounted for {100*float(recession['fraction_from_strip_storage']):.2f}% of outlet export. "
                             "[Accepted transport and strip water accounts](../fidelity/outlet_results/last_40m.csv)")
    temporal = {float(r["cfl_factor"]): r for r in events if r["method"] == "adaptive"
                and int(r["dx_m"]) == 5 and int(r["extension_m"]) == 0}
    if {0.5, 1.0} <= temporal.keys():
        normal, half = temporal[1.0], temporal[0.5]
        lines.append(f"Halving the RK2 CFL on the 5 m grid changed the one-minute peak from "
                     f"{float(normal['minute_peak_m3s']):.3f} to {float(half['minute_peak_m3s']):.3f} m³/s. "
                     f"Net export changed by {float(half['gauge_m3']) - float(normal['gauge_m3']):+,.0f} m³, "
                     f"infiltration by {float(half['footprint_infiltration_m3']) - float(normal['footprint_infiltration_m3']):+,.0f} m³ "
                     f"and final storage by {float(half['footprint_storage_m3']) - float(normal['footprint_storage_m3']):+,.0f} m³. "
                     f"The normal and half-CFL runs accepted {int(normal['steps']):,} and {int(half['steps']):,} timesteps.")
    names = dict(peak_bias_pct="peak magnitude", net_export_bias_pct="net exported volume",
                 infiltration_bias_pct="cumulative infiltration")
    qualified = [names.get(r["metric"], r["metric"]) for r in csv_read(file=directory/"reference_checks.csv")
                 if r["representation"] == "0 m downstream extension" and r["station"] == "outlet" and r["qualified"] == "True"]
    status = ", ".join(qualified) if qualified else "none"
    lines.append(f"The direct-outlet observables passing all three reference checks were {status}. "
                 "These checks support comparisons on the coarser grid. Practical 5 m absolute accuracy remains unresolved. "
                 "[Reference qualification by observable](../fidelity/outlet_results/reference_checks.csv)")
    budgets = csv_read(file=directory / "least_cost_fidelity.csv")
    for metric, label, tolerance in (("peak_bias_pct", "peak discharge", 5),
                                      ("net_export_bias_pct", "net exported volume", 10)):
        selected = [r for r in budgets if r["representation"] == "0 m downstream extension"
                    and r["station"] == "outlet" and r["objective"] == metric
                    and float(r["tolerance"]) == tolerance and r["id"]]
        for decision in selected:
            names = {"fv1": "FV1", "hancock_adaptive": "adaptive Hancock", "adaptive": "FV2 RK2"}
            lines.append(f"With a {tolerance}% tolerance on {label}, {names[decision['method']]} at {decision['dx']} m "
                         f"had the lowest measured hydraulic cost ({float(decision['runtime_s']):.1f} s). "
                         f"Its absolute error against the 5 m RK2 reference was {float(decision['measured_error']):.2f}%, "
                         f"rising to {float(decision['conservative_error']):.2f}% when reference sensitivity was included.")
    lines.append("The two decision tables retain the event endpoint, output cadence and reference for each observable. "
                 "The hydraulic costs use independent timing medians where available. "
                 "[Cost at a stated tolerance](../tables/19_incised_cost_at_tolerance.md) · "
                 "[Error within a measured runtime](../tables/20_incised_error_at_budget.md)")
    return "\n" + "\n\n".join(lines) + "\n"


def widespread_plain(*, study):
    import json

    directory = study/"fidelity/strong_storms"
    path = directory/"completed.csv"
    if not path.exists():
        return ""
    groups = defaultdict(dict)
    receipt = study/"fidelity/outlet_results/sources.json"
    if not receipt.is_file():
        return ""
    current = json.loads(receipt.read_text())["solver"]
    for row in csv_read(file=path):
        source=json.loads((directory/row["id"]/"inputs.json").read_text())
        if source["identity"]["numerical"] == current:
            groups[row["storm"],row["end_h"]][row["method"]] = row
    lines=[]
    for (storm,end), methods in sorted(groups.items()):
        if {"fv1","hancock_dense"} <= methods.keys():
            f, h = methods["fv1"], methods["hancock_dense"]
            lines.append(f"Under {storm} rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at "
                         f"{float(f['peak_m3s']):.3f} and {float(h['peak_m3s']):.3f} m³/s. "
                         f"They exported {float(f['outlet_m3']):,.0f} and {float(h['outlet_m3']):,.0f} m³ "
                         f"through the common {end}-hour endpoint.")
    if not lines:
        return ""
    return "\n" + " ".join(lines) + " Both gauges passed the recession criterion. These CPU controls measure formulation disagreement. " +           "[Six-hour rainfall and complete recession accounts](../fidelity/strong_storms/completed.csv)\n"


def completed_storms(*, study):
    folder = study / "fidelity/small_storms"
    if not (folder / "paired_differences.csv").is_file():
        return ""
    horizons = csv_read(folder / "recession_horizons.csv")
    pairs = csv_read(folder / "paired_differences.csv")
    shape = [
        r
        for r in pairs
        if r["metric"] == "shape_tv_pct"
        and r["first"] == "hancock_dense"
        and r["second"] == "adaptive"
    ]
    resolved = sum(r["exceeds_reference"] == "True" for r in shape)
    joint_file = folder / "joint_depth_hydrograph.csv"
    joint_text = ""
    if joint_file.is_file():
        for r in csv_read(joint_file):
            if (
                r["metric"] == "peak_bias_pct"
                and float(r["outlet_tolerance_pct"]) == 10
            ):
                joint_text = (
                    f" Among {r['within_depth']} FV1 storms with independently assessable peaks and final depth RMSE within "
                    f"{1000 * float(r['final_depth_limit_m']):g} mm, {r['exceeds_outlet']} had an absolute peak bias above {r['outlet_tolerance_pct']}%."
                )
    extended = sum(r["extended"] == "True" for r in horizons)
    maximum = max(float(r["common_end_h"]) for r in horizons)
    return (
        f"Of {len(horizons)} small events with six hours of rain, {extended} required a longer recession. "
        f"The longest matched simulation reached {maximum:g} hours. "
        f"For the {len(shape)} storms with independently qualified normalised outlet shape at 10 m, "
        f"the Hancock–FV2 difference in absolute shape error exceeded the combined reference sensitivity in {resolved} cases. "
        "That comparison concerns the timing distribution of discharged volume. Peak and arrival-time conclusions have separate reference screens."
        + joint_text
    )


def native_incised_crests(*, study):
    path = study / "fidelity/native_tail/crest_windows.csv"
    if not path.is_file():
        return ""
    by_end = defaultdict(dict)
    for row in csv_read(file=path):
        if row["method"] == "adaptive" and row["window"] == "whole_record":
            by_end[int(row["end_s"])][int(row["dx_m"])] = row
    shared = [t for t, group in by_end.items() if {5, 10} <= group.keys()]
    if not shared:
        return ""
    end = max(shared)
    parts = []
    for dx in (10, 5):
        record = by_end[end][dx]
        hour, remaining = divmod(int(record["first_peak_interval_end_s"]), 3600)
        parts.append(f"{float(record['peak_interval_mean_m3s']):.2f} m³/s at {hour} h {remaining // 60:02d} min on the {dx} m grid")
    return (
        f"With the pre-repair FREE boundary, through the matched {end / 3600:g}-hour window, the highest recorded FV2 RK2 outlet "
        f"interval means were {parts[0]} and {parts[1]}. Each interval spans "
        f"{by_end[end][10]['interval_s']} seconds. Disagreement in these two measured maxima "
        "establishes grid sensitivity. The physical storm peak remains unresolved."
    )


def incised_storage_and_infiltration(*, study):
    base = study / "fidelity"
    paragraphs = []
    path = base / "native_tail/crest_minute_budgets.csv"
    if path.is_file():
        rows = [r for r in csv_read(file=path) if int(r["strip_width_m"]) == 40]
        details = [
            f"{float(r['net_storage_share_pct']):.2f}% on the {int(r['dx_m'])} m grid"
            for r in sorted(rows, key=grid_spacing)
        ]
        if details:
            paragraphs.append(
                "The pre-repair replay measured the source of discharge during the highest post-hour-eight minute at each spacing. "
                "Net drawdown in the last 40 m supplied " + " and ".join(details)
                + ". That water may have entered the reach earlier. [Crest-minute budgets](../fidelity/native_tail/CREST.md)"
            )
    path = base / "native_outlet/runoff_partition.csv"
    if path.is_file():
        for row in csv_read(file=path):
            if row["method"] == "hancock_adaptive" and int(row["dx_m"]) == 5:
                paragraphs.append(
                    f"With the pre-repair FREE boundary on the 5 m grid, FV1 exported {float(row['extra_fv1_export_m3']):,.0f} m³ more water than Hancock by hour {float(row['end_h']):g}. "
                    f"Lower cumulative infiltration accounted for {float(row['less_fv1_infiltration_m3']):,.0f} m³ of the difference and lower final storage for "
                    f"{float(row['less_fv1_final_storage_m3']):,.0f} m³. The net infiltration difference arose on cells outside the rainfall footprint. "
                    "These terms close the water balance under the same prescribed infiltration capacity. "
                    "[Runoff and infiltration by grid](../fidelity/native_outlet/runoff_partition.csv)"
                )
    return "\n\n".join(paragraphs)


def grid_spacing(row):
    return int(row["dx_m"])


def generate(*, study):
    from research.floodwarp_methods_paper.evidence import archived_publication

    with archived_publication(study=study):
        return generate_from_records(study=study)


def generate_from_records(*, study):
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
    dense = [
        float(d["hancock_dense"]["runtime_s"]) / float(d["fv1"]["runtime_s"])
        for d in matched.values()
    ]
    fastest = [min(a, b) for a, b in zip(dense, ratios, strict=True)]
    dense_wins = sum(a < b for a, b in zip(dense, ratios, strict=True))
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

The original analytical, synthetic, calibration and advisor tables use their archived numerical revisions. The repaired FREE-outlet measurements are reported separately under [Repaired FREE outlet](#repaired-free-outlet).

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

On the 96-cell controls, dense Hancock was faster than tiles in {dense_wins} of {len(matched)} original fixed-duration forecasts. Relative to sparse FV1 its median runtime was {median(dense):.3f}×. The tiled median was {median(ratios):.3f}×. Selecting the faster qualified execution retrospectively gave {median(fastest):.3f}×. Both schedules reproduced the same Hancock fields. With only 96 cells, marking and compacting tiles adds GPU launches while eliminating little work. The large-domain suite measures how this balance changes with the scheduled footprint. [Measured GPU costs](../tables/07_runtime.md) · [Full performance qualification](../../../validation/hancock_long_storm_review/DECISION.md)

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

    audit, extension = paper.material(study=study)
    discussion += "\n" + paper.operational_discussion(study=study) + "\n"
    result += "\n" + synthesis.case_evidence(study=study) + "\n" + extension + "\n"
    (study / "LIMITATIONS.md").write_text(synthesis.limitations(study=study))
    fidelity = study / "fidelity/paper_summary.md"
    paragraph = completed_storms(study=study)
    link = "[Complete small-storm comparisons](../fidelity/small_storms/README.md)"
    if fidelity.is_file():
        paragraph = fidelity.read_text().strip()
        link = "[Hydrograph fidelity, uncertainty and measured cost](../fidelity/REPORT.md)"
    if (study / "fidelity/NATIVE_RECESSIONS.md").is_file():
        link += " · [Million-cell export after the original stop time](../fidelity/NATIVE_RECESSIONS.md)"
    if paragraph:
        result = result.replace(
            "## Storm hydrographs and depth",
            "## Flood-wave fidelity per unit cost\n\n"
            + paragraph
            + " "
            + link
            + "\n\n## Storm hydrographs and depth",
        )
        discussion += (
            "\n## Flood-wave interpretation\n\n" + paragraph + " " + link + "\n"
        )
        result = result.replace(
            "## Storm hydrographs and depth",
            "## Original fixed-window depth and discharge diagnostics",
        )
        result = result.replace(
            "## Efficiency and terrain representation",
            "## Original-window efficiency and terrain representation",
        )
        audit += (
            "\n## Hydrograph fidelity through completed recessions\n\n"
            + paragraph
            + " "
            + link.replace("](../", "](")
            + "\n"
        )
    if (study / "fidelity/DEM_CORRECTION.md").is_file():
        correction = (
            "\n## DEM aggregation and extent thresholds\n\n"
            "Earlier resolution experiments used cell-centre sampling or repeated parent elevations. "
            "The incised fixed-bed refinement created artificial treads that substantially altered runoff. "
            "These historical experiments do not establish accuracy at a practical resolution using an area-averaged DEM. "
            "The corrected series averages 25 native 1 m elevations per 5 m cell, then four 5 m elevations per 10 m cell. "
            "Initial water, momentum, rainfall footprint and infiltration also come from the native metre source. All practical runs start at time zero or resume their own verified checkpoint. "
            "[Native-input storm windows, budgets and reference qualification](../fidelity/native_results/README.md) "
            "Report practical extent separately at 5 and 10 cm. Historical 1 cm extent is a shallow-water diagnostic. "
            "[DEM construction, numerical controls and corrected outcomes](../fidelity/DEM_CORRECTION.md)\n"
        )
        if (study / "fidelity/native_outlet/README.md").is_file():
            correction += ("\nThe native-grid incised results also show sensitivity to the downstream FREE boundary. "
                           "The steady-sheet experiment checks this effect independently against exact discharge. "
                           "[Outlet placement and terminal storage](../fidelity/native_outlet/README.md)\n")
        if (study / "fidelity/native_tail/README.md").is_file():
            correction += "\n" + native_incised_crests(study=study) + "\n"
            correction += ("\nRunoff travel time and catchment storage allow the outlet maximum to follow rainfall. "
                           "The incised diagnostic compares accepted flow at successive cross-sections with "
                           "storage and infiltration over the intervening reach. The matched finer-grid checks "
                           "test how strongly the outlet timing and magnitude depend on representation. "
                           "[Recession water balance](../fidelity/native_tail/README.md)\n")
        if (study / "fidelity/native_publication/README.md").is_file():
            correction += "\n[Matched native-grid costs and observable-specific reference eligibility](../fidelity/native_publication/README.md)\n"
        detail = incised_storage_and_infiltration(study=study)
        if detail:
            correction += "\n" + detail + "\n"
        repaired = study / "fidelity/outlet_results/events.csv"
        if repaired.exists():
            direct = {r["method"]: r for r in csv_read(file=repaired)
                      if int(r["dx_m"]) == 10 and int(r["extension_m"]) == 0 and float(r["cfl_factor"]) == 1}
            if {"hancock_adaptive", "adaptive"} <= direct.keys():
                correction += (
                    "\n## Repaired FREE outlet\n\n"
                    f"On the same 10 m incised terrain, the corrected 16-hour Hancock and RK2 peaks were "
                    f"{float(direct['hancock_adaptive']['minute_peak_m3s']):.2f} and "
                    f"{float(direct['adaptive']['minute_peak_m3s']):.2f} m³/s. Their earlier late crests "
                    "disappeared. These are one-minute means at the original outlet. "
                    "[Fixed-gauge boundary experiments and water budgets](../fidelity/outlet_results/README.md)\n"
                )
                correction += repaired_budgets(directory=repaired.parent)
                correction += widespread_plain(study=study)
        transfer = study / "fidelity/orton_transfer/transfer.csv"
        if transfer.exists():
            transfers = csv_read(file=transfer)
            correction += (f"\n{len(transfers)} of nine Orton Bradley configurations completed the archived 48-hour HIRDS "
                           "design storm. These measured differences quantify numerical transfer to real terrain under design "
                           "forcing. Errors against an observed event remain unknown. "
                           "[Water balance and inundation](../tables/17_orton_storm.md) · "
                           "[Timesteps, memory and runtime](../tables/18_orton_cost.md) · "
                           "[Fixed-gauge hydrographs](../figures/20_orton_transfer.pdf) · "
                           "[Machine-readable transfer observations](../fidelity/orton_transfer/transfer.csv)\n")
        from research.floodwarp_methods_paper.fidelity.progress import pulse_comparison

        if (study / "fidelity/results/sharp_pulses.csv").exists():
            sharp = next(line for line in pulse_comparison(directory=study / "fidelity")
                         if line.startswith("With quarter-second"))
            correction += "\n" + sharp.replace("](results/", "](../fidelity/results/") + "\n"
        result += correction
        discussion += correction
        audit += correction.replace("](../", "](")
    (study / "CORRECTIVE_AUDIT.md").write_text(audit)
    manuscript = study / "manuscript"
    (manuscript / "results.md").write_text(result)
    (manuscript / "discussion.md").write_text(discussion)
    (study / "RESULTS.md").write_text(
        result.replace("](../", "](").replace(
            "](../../validation", "](../../validation"
        )
    )
    (study / "DISCUSSION.md").write_text(discussion.replace("](../", "]("))
    conclusions = [
        "# Conclusions",
        f"Across the original fixed windows of {len(matched)} small-domain storms, the faster qualified Hancock execution took a median "
        f"{median(fastest):.3f} times the FV1 runtime. "
        f"Tiled execution alone had median {median(ratios):.3f}×.",
        paper.operational_discussion(study=study),
        f"The frozen advisor made a qualified feasible selection in {fmt(a['feasible_coverage_pct'])}% "
        f"of held-out decisions with a measured feasible option. It abstained on {a['abstentions']} of "
        f"{a['decisions']} decisions. Terrain loss and reference uncertainty limit recommendations for "
        "specific sites and observables. [Held-out comparisons](../tables/10_advisor.md)",
        "Retain adaptive Hancock as the development default. Keep explicit execution overrides and "
        "select FV1 when the measured error is acceptable for the intended observable. "
        "[Results](results.md) · [Discussion](discussion.md)",
    ]
    if paragraph:
        conclusions.insert(1, paragraph + " " + link)
    (manuscript / "conclusions.md").write_text(
        "\n\n".join(filter(None, conclusions)) + "\n"
    )
    return dict(
        events=len(matched),
        assessed_choices=int(a["qualified_selections"]),
        qualified_transfers=len(qual),
    )


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    generate(study=study)
