"""Populate results and discussion from the frozen experiment outputs."""

from math import isclose
from pathlib import Path
from operator import itemgetter
from research.floodwarp_methods_paper import synthesis
from statistics import median
from collections import defaultdict

from research.floodwarp_methods_paper.metrics import csv_read
from research.floodwarp_methods_paper.comparisons import exact_check


def fmt(value):
    return "unavailable" if value in (None, "") else f"{float(value):.4g}"


def native_advisor_findings(*, study):
    import json
    from research.floodwarp_methods_paper import runner

    folder = study / "fidelity"
    checklist = folder / "advisor_design/paired.json"
    if not checklist.is_file():
        return ""
    design = json.loads(checklist.read_text())
    paired = {r["split"]: r for r in design["pairs"] if r["dx"] == [5, 10] and r["objective"] == "depth"}
    lines = ["## Practical grid selection on native terrain"]
    for partition in ("validation", "test"):
        row = paired[partition]
        lines.append(f"The {partition} design includes {len(row['possible'])} independent terrains with an admissible "
                     f"5 and 10 m pair and all scheduled depth reference controls ({', '.join(row['possible'])}). "
                     f"Completed runs currently qualify {len(row['observed_qualified'])} terrain pairs for depth. "
                     "[Design and measured eligibility by observable](../fidelity/advisor_design/paired.json)")
    footprints = {(*r["candidate_cells"], r["reference_dx_m"], r["reference_cells"])
                  for partition in ("validation", "test") for r in paired[partition].get("opportunity_domains", [])}
    for low_cells, high_cells, fine, reference_cells in sorted(footprints):
        lines.append(f"The paired depth design tests forecasts of {low_cells:,} cells at 5 m and {high_cells:,} "
                     f"cells at 10 m against a {reference_cells:,}-cell {fine:g} m numerical reference on the same footprint. "
                     "Independent validation of larger-domain grid choices requires admissible comparisons at those scales. "
                     "[Candidate and reference grids, CFL controls and measured endpoints](../fidelity/advisor_design/scheduled.csv)")
    cli_path = folder / "advisor_design/installed_frozen_cli.csv"
    if not cli_path.exists():
        cli_path = folder / "advisor_design/installed_cli.csv"
    if cli_path.is_file():
        controls = [r for r in csv_read(file=cli_path) if r["terrain_id"] == "floodplain-8103"]
        for r in sorted(controls, key=cli_spacing):
            lines.append(f"On floodplain-8103 at {r['dx_m']} m, the installed DEM reader measured maximum local "
                         f"barrier lowering of {float(r['max_lowering_m']):.3f} m. Its solver decision was "
                         f"`{r['status']}`. [Installed CLI and archived features](../fidelity/advisor_design/{cli_path.name})")
    pointer = folder / "native_advisor/heldout_evaluation.json"
    if pointer.exists():
        active = json.loads(pointer.read_text())
        archive = folder / active["path"]
        evidence = json.loads((archive / "sources.json").read_text())
        data = archive / "joint_summary.csv"
        if (runner.checksum(archive / "sources.json") != active["sha256"]
                or runner.checksum(data) != evidence["files"][data.name]):
            raise ValueError("Unverified held-out joint decision summary")
        setting = json.loads((folder / "native_advisor_settings.json").read_text())["fit_percent"]
        for r in csv_read(file=data):
            if (r["rule"] != "advisor" or r.get("subset") != "new final terrains"
                    or float(r["percentage_scenario"]) != setting):
                continue
            regret = fmt(r["median_regret_pct"]) + "%" if r["median_regret_pct"] else "unassessable"
            lines.append(f"With {r['requested_dx']} m requested on the prospectively held-out final terrains "
                         f"({setting}% scenario), joint method and grid selection covered {r['successful_selections']} "
                         f"of {r['empirical_feasible_opportunities']} feasible opportunities. Of {r['decisions']} "
                         f"decisions, {r['false_feasible']} were false-feasible and {r['abstentions']} abstained. "
                         f"Median runtime regret was {regret} across {r['regret_assessable']} "
                         f"assessable choices. [Joint rules and measured oracle](../fidelity/{active['path']}/joint_summary.csv)")
    else:
        lines.append("Joint method and grid selection on the new final terrains awaits the held-out evaluation.")
    return "\n" + "\n\n".join(lines) + "\n"


def cli_spacing(row):
    return float(row["dx_m"])


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
    analytical = directory.parent / "analytical_revision/verification.json"
    if analytical.is_file():
        from research.floodwarp_methods_paper import runner

        verified = json.loads(analytical.read_text())
        if runner.checksum(analytical.with_name("comparisons.csv")) != verified["comparisons_sha256"]:
            raise ValueError("Analytical revision evidence differs from its verification receipt")
        gates = verified["gates"]
        lines.append(f"Of {gates['primary_runs']} core exact-solution controls, {verified['identical_primary_field_and_gauge_pairs']} "
                     f"reproduced the pre-repair sampled fields and gauge transport exactly. "
                     f"{verified['identical_primary_histories']} retained the same ordered accepted timesteps. "
                     f"The maximum relative mass-balance residual was {gates['maximum_relative_residual']:.2e} "
                     f"and the lake-at-rest depth error was {gates['lake_maximum_depth_error_m']:.2e} m. "
                     "Measuring the actual steep FREE face exposed a defect that these sampled controls did not detect. "
                     "[Paired exact references, results and run identities](../fidelity/analytical_revision/comparisons.csv)")
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
    phase_file = directory / "phase_volume_differences.csv"
    if phase_file.is_file():
        phases = {float(r["start_h"]): r for r in csv_read(file=phase_file)
                  if r["first_method"] == "fv1" and r["second_method"] == "hancock_adaptive"
                  and int(r["extension_m"]) == 0 and int(r["dx_m"]) == 5}
        if len(phases) == 2:
            rain, recession = (phases[k] for k in sorted(phases))
            lines.append(f"At rain cessation ({float(rain['end_h']):g} h), the direct 5 m FV1-minus-Hancock "
                         f"differences were {float(rain['gauge_m3']):+,.0f} m³ in cumulative export and "
                         f"{float(rain['footprint_storage_m3']):+,.0f} m³ in remaining storage. "
                         f"During the common recession alone FV1 exported {float(recession['gauge_m3']):+,.0f} m³ "
                         f"relative to Hancock. [Rain and recession water accounts](../fidelity/outlet_results/phase_volume_differences.csv)")
    maxima = directory / "peak_extent.csv"
    if maxima.is_file():
        direct = {r["method"]: r for r in csv_read(file=maxima) if int(r["dx_m"]) == 5
                  and int(r["extension_m"]) == 0 and int(r["threshold_cm"]) == 5 and float(r["cfl_factor"]) == 1}
        order = ("fv1", "hancock_adaptive", "adaptive")
        if set(order) <= direct.keys():
            areas = [float(direct[key]["largest_sampled_area_m2"]) / 1e6 for key in order]
            interval = float(direct["fv1"]["snapshot_interval_s"]) / 60
            lines.append(f"Across {direct['fv1']['samples']} snapshots at {interval:g}-minute intervals, the "
                         f"largest simultaneous areas above 5 cm were {areas[0]:.4f}, {areas[1]:.4f} and "
                         f"{areas[2]:.4f} km² for 5 m FV1, Hancock and RK2. These measurements cover the full "
                         "direct-outlet storm and recession. Cumulative inundation also includes cells that "
                         "flooded at different times. [Sampled maxima and cumulative extent](../fidelity/outlet_results/peak_extent.csv)")
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
    measured_timings = directory / "independent_timings.csv"
    if measured_timings.is_file():
        records = csv_read(file=measured_timings)
        costs = {(r["method"], int(r["dx_m"])): r for r in records
                 if int(r["extension_m"]) == 0 and r["primary"] == r["id"] and int(r["end_s"]) == 57600}
        for dx in sorted({spacing for _, spacing in costs}, reverse=True):
            fv1, hancock = costs.get(("fv1", dx)), costs.get(("hancock_adaptive", dx))
            if fv1 and hancock:
                time_ratio = float(hancock["hydraulic_median_s"]) / float(fv1["hydraulic_median_s"])
                step_ratio = int(hancock["steps"]) / int(fv1["steps"])
                counts = [len({r["id"] for r in records if r["primary"] == peer["id"] and r["end_s"] == peer["end_s"]})
                          for peer in (hancock, fv1)]
                if min(counts) < 3:
                    raise ValueError("A close storm cost comparison requires three independent timings per method")
                lines.append(f"On the {dx} m direct-outlet grid, Hancock required {int(hancock['steps']):,} "
                             f"accepted steps versus {int(fv1['steps']):,} for FV1. Across {counts[0]} and {counts[1]} "
                             f"warmed timings respectively, the median Hancock runtime was {time_ratio / step_ratio:.2f}× as much "
                             f"per accepted step and {time_ratio:.3f}× as much over the full 16 hours. "
                             "[Independent timings and execution policies](../fidelity/outlet_results/independent_timings.csv)")
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
    window_path = directory / "fixed_window_shape.csv"
    if window_path.is_file():
        at_outlet = [r for r in csv_read(file=window_path) if r["representation"] == "0 m downstream extension"
                     and r["station"] == "outlet" and r["shape_tv_pct"]]
        coarse = {r["method"]: r for r in at_outlet if int(r["grid_m"]) == 10}
        controls = {r["reference_role"]: r for r in at_outlet}
        if {"fv1", "hancock_adaptive"} <= coarse.keys() and {"space", "cross", "time"} <= controls.keys():
            lines.append(f"For all outward discharge over the fixed 16-hour window, the fraction of volume "
                         f"redistributed in time relative to the normal 5 m RK2 forecast was "
                         f"{float(coarse['fv1']['shape_tv_pct']):.2f}% for 10 m FV1 and "
                         f"{float(coarse['hancock_adaptive']['shape_tv_pct']):.2f}% for 10 m Hancock. "
                         f"Their centroid shifts were {float(coarse['fv1']['centroid_delay_s'])/60:+.1f} and "
                         f"{float(coarse['hancock_adaptive']['centroid_delay_s'])/60:+.1f} minutes. "
                         "Applying the same shape diagnostic to the spatial, fine-formulation and temporal controls "
                         f"gave {float(controls['space']['shape_tv_pct']):.2f}%, "
                         f"{float(controls['cross']['shape_tv_pct']):.2f}% and {float(controls['time']['shape_tv_pct']):.2f}%. "
                         "The initial moving channel contributes to these distributions. Rainfall-only hydrograph shape "
                         "remains unresolved. [Fixed-window comparisons and reference controls](../fidelity/outlet_results/fixed_window_shape.csv)")
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


def native_operational_balances(*, study, minimum_cells):
    import json

    folder = study / "fidelity/native_results"
    budgets = folder / "measured_windows.csv"
    pairs = folder / "policy_cost_events.csv"
    if not budgets.is_file() or not pairs.is_file():
        return ""
    source = json.loads((study / "fidelity/outlet_results/sources.json").read_text())["solver"]
    completed = {r["source_id"]: r for r in csv_read(file=budgets)
                 if r["horizon"] == "complete-native-metre" and r["solver_source"] == source}
    times = {ident: float(r["hydraulic_median_s"])
             for r in csv_read(file=folder / "timing_distribution.csv")
             if r["horizon"] == "completed recession"
             for ident in r["run_ids"].split()}
    verified = {(r["fv1_id"], r["dense_id"], r["tiled_id"]) for r in csv_read(file=pairs)
                if r["cohort"] == "complete-native-metre"}
    groups = defaultdict(dict)
    for r in completed.values():
        if int(r["cells"]) >= minimum_cells and float(r["factor"]) == 1:
            key = r["case"], r["native_sha"], int(r["dx_m"]), float(r["interval_s"]), float(r["end_h"]), r["hardware"]
            if r["method"] in groups[key]:
                raise ValueError("Duplicate primary method in matched native costs")
            groups[key][r["method"]] = r
    rows = []
    storage_rows = []
    station_rows = []
    flow = None
    for (case, _, dx, cadence, end_h, _), methods in sorted(groups.items()):
        fv = methods.get("fv1")
        options = [methods[m] for m in ("hancock_dense", "hancock_adaptive") if m in methods]
        if fv is None or not options:
            continue
        if len(options) == 2 and (fv["source_id"], options[0]["source_id"], options[1]["source_id"]) not in verified:
            raise ValueError("Native costs lack hydraulic parity for the compared Hancock executions")
        best = min(times[r["source_id"]] for r in options)
        winners = [r for r in options if times[r["source_id"]] == best]
        h = winners[0]
        policy = " / ".join(r["execution_mode"] for r in winners)
        if len(options) == 1:
            policy += " only measured"
        cells = int(fv["cells"])
        delta_export = float(fv["net_outlet_m3"]) - float(h["net_outlet_m3"])
        delta_soil = float(fv["infiltration_m3"]) - float(h["infiltration_m3"])
        ratio = best / times[fv["source_id"]]
        rows.append(f"| {case} | {cells:,} | {cells * dx**2 / 1e6:.3f} | {dx} | {end_h:g} | {cadence:g} | "
                    f"{float(fv['outlet_peak_m3s']):.5g} | {float(h['outlet_peak_m3s']):.5g} | "
                    f"{delta_export:+,.0f} | {delta_soil:+,.0f} | {policy} | {ratio:.3f} |")
        if all(r.get(key) not in ("", None) for r in (fv, h)
               for key in ("section_x_m", "section_peak_m3s", "net_section_m3")):
            if flow is None:
                flow = json.loads((study / "fidelity/design.json").read_text())
            if float(fv["section_x_m"]) != float(h["section_x_m"]):
                raise ValueError("Internal gauge coordinates differ in a matched storm comparison")
            station_cells = []
            for station, volume in (("section", "net_section_m3"), ("outlet", "net_outlet_m3")):
                peak_f, peak_h = [float(r[station + "_peak_m3s"]) for r in (fv, h)]
                discharge_difference = "unresolved"
                if peak_h >= flow["minimum_peak_m3s"] and float(h[volume]) >= flow["minimum_volume_m3"]:
                    discharge_difference = f"{100 * (peak_f / peak_h - 1):+.2f}"
                station_cells.extend((f"{peak_f:.5g}", f"{peak_h:.5g}", discharge_difference))
            station_rows.append(f"| {case} | {dx} | {end_h:g} | {float(h['section_x_m']):g} | "
                                + " | ".join(station_cells) + " |")
        else:
            station_rows.append(f"| {case} | {dx} | {end_h:g} | unmeasured | — | — | — | "
                                f"{float(fv['outlet_peak_m3s']):.5g} | {float(h['outlet_peak_m3s']):.5g} | — |")
        rain = float(fv["rain_m3"])
        for key in ("rain_m3", "initial_m3"):
            if not isclose(float(fv[key]), float(h[key]), rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError(f"Compared storm inputs disagree in {key} for {case}")
        fv_storage, h_storage = float(fv["storage_m3"]), float(h["storage_m3"])
        storage_rows.append(
            f"| {case} | {dx} | {end_h:g} | {rain:,.0f} | {1000 * rain / (cells * dx**2):.2f} | "
            f"{fv_storage:,.0f} | {h_storage:,.0f} | "
            f"{int(fv['model_grid_wet_area_5cm_m2']):,} | {int(h['model_grid_wet_area_5cm_m2']):,} | "
            f"{int(fv['model_grid_wet_area_10cm_m2']):,} | {int(h['model_grid_wet_area_10cm_m2']):,} |"
        )
    if not rows:
        return ""
    return "\n" + "\n".join([
        f"## Complete storms on native-metre terrain at {minimum_cells:,} cells and above", "",
        "Each row compares solvers on the same event and physical footprint. Domain area varies between rows. "
        "Peaks are interval means of accepted outlet transport. The volume columns give FV1 minus Hancock. "
        "Costs use the fastest measured explicit Hancock policy through the common recession endpoint "
        "and the median hydraulic time where repeats exist.", "",
        "| Storm | Cells | Area (km²) | Grid (m) | End (h) | Interval (s) | FV1 peak (m³/s) | Hancock peak (m³/s) | "
        "Δ net export (m³) | Δ infiltration (m³) | Measured Hancock policy | Hancock / FV1 time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|", *rows, "",
        "### Internal section and outlet peaks", "",
        "Each section spans the domain width at the stated distance from its upstream grid edge. "
        "The signed percentage is 100 × (FV1 peak / Hancock peak − 1). Hancock supplies the comparison scale. "
        "Negligible outward flow gives an unresolved percentage. The peaks at the two gauges may occur at different times. "
        "Rainfall, infiltration and storage between the gauges also affect the outlet response.", "",
        "| Storm | Grid (m) | End (h) | Section x (m) | Section FV1 (m³/s) | Section Hancock (m³/s) | Difference (%) | "
        "Outlet FV1 (m³/s) | Outlet Hancock (m³/s) | Difference (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *station_rows, "",
        "### Storm input and remaining inundation", "",
        "Catchment-mean rainfall divides the integrated six-hour rainfall volume by the whole modelled area. "
        "A spatially concentrated storm can deliver a much greater depth inside its rainfall footprint. "
        "The areas count original hydraulic cells at the common end time. Maximum extent during the storm is a separate observable.", "",
        "| Storm | Grid (m) | End (h) | Rain (m³) | Catchment mean (mm) | FV1 storage (m³) | Hancock storage (m³) | "
        "FV1 ≥5 cm (m²) | Hancock ≥5 cm (m²) | FV1 ≥10 cm (m²) | Hancock ≥10 cm (m²) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *storage_rows, "",
        "The differences measure sensitivity to formulation on the same represented terrain. Accuracy decisions "
        "use the independently screened reference for each observable. "
        "[Water budgets and internal gauges](../fidelity/native_results/measured_windows.csv) · "
        "[Policy-specific observations and reference eligibility](../fidelity/native_publication/README.md)", "",
    ])


def plain_grid_fidelity(*, study, cases):
    file = study / "fidelity/native_results/measured_windows.csv"
    if not file.is_file():
        return ""
    groups = defaultdict(dict)
    required = {("fv1", 1), ("hancock_dense", 1), ("hancock_dense", 5), ("hancock_dense", 10)}
    for r in csv_read(file=file):
        key = r["method"], int(r["dx_m"])
        if (r["case"] not in cases or key not in required or float(r["factor"]) != 1 or int(r["timing_repeat"] or 0)
                or r["horizon"] not in ("original_window", "complete-native-metre")):
            continue
        group = (r["case"], float(r["end_h"]), r["horizon"], float(r["interval_s"]), r["native_sha"], r["solver_source"])
        if key in groups[group]:
            raise ValueError("Duplicate primary run in a matched plain grid comparison")
        groups[group][key] = r
    rows = []
    for (case, end, horizon, cadence, _, _), peers in sorted(groups.items()):
        if peers.keys() != required:
            continue
        stage = "Initial window" if horizon == "original_window" else "Common recession"
        inputs = [peers[("fv1", 1)], *[peers["hancock_dense", dx] for dx in (1, 5, 10)]]
        for metric, unit in (("outlet_peak_m3s", "Peak (m³/s)"), ("net_outlet_m3", "Export (m³)")):
            values = [float(r[metric]) for r in inputs]
            digits = ".4g" if metric == "outlet_peak_m3s" else ",.0f"
            rows.append(f"| {cases[case]} | {stage} | {end:g} | {cadence:g} | {unit} | "
                        + " | ".join(format(v, digits) for v in values) + " |")
    if not rows:
        return ""
    return "\n".join(["", "### Hydraulic grid and formulation on the same plain", "",
                      "All grids cover the same 1 km². The comparison across grid spacings includes changes to "
                      "both the represented bed and the hydraulic discretisation. The FV1 and Hancock columns at "
                      "1 m use identical terrain and forcing. Initial-window peaks are the largest recorded through "
                      "hour eight. Rain stops at hour six, and a later crest can exceed the initial-window peak.", "",
                      "| Storm | Window | End (h) | Interval (s) | Observable | FV1, 1 m | Hancock, 1 m | Hancock, 5 m | Hancock, 10 m |",
                      "|---|---|---:|---:|---|---:|---:|---:|---:|", *rows, "",
                      "[Per-configuration observations and physical input identities](../fidelity/native_results/measured_windows.csv)", ""])


def plain_embankment(*, study, cases, face_m):
    file = study / "fidelity/native_results/embankment_transport.csv"
    if not file.is_file():
        return ""
    rows = [r for r in csv_read(file=file) if r["case"] in cases and r["method"] == "hancock_dense"
            and float(r["factor"]) == 1 and float(r["face_x_m"]) == face_m]
    if not rows:
        return ""
    bounds = {(r["band_south_m"], r["band_north_m"], r["crest_strip_west_m"], r["crest_strip_east_m"]) for r in rows}
    if len(bounds) != 1:
        raise ValueError("Embankment comparisons require a fixed physical window")
    south, north, west, east = bounds.pop()
    order = sorted((r["case"], float(r["end_h"]), int(r["dx_m"]), r["horizon"], r) for r in rows)
    lines = ["", "### Flow across the transverse embankment", "",
             f"The embankment crest is centred at x = 875 m. At x = {face_m:g} m the table partitions accepted "
             f"eastward transport into the opening band (northings {south}–{north} m) and the rest of the face. "
             f"The bed diagnostic takes the highest elevation between eastings {west} and {east} m on each "
             "east–west transect. Its reported shift is the median represented-minus-native crest elevation "
             "outside the opening band. Each FV1, Hancock and RK2 profile retains signed volumes, including any reversal.", "",
             "| Storm | Window | End (h) | Hancock grid (m) | In opening band (m³) | Elsewhere (m³) | "
             "Median crest shift elsewhere (cm) |", "|---|---|---:|---:|---:|---:|---:|"]
    for case, end, dx, horizon, r in order:
        stage = "Initial" if horizon == "original_window" else "Common recession"
        lines.append(f"| {cases[case]} | {stage} | {end:g} | {dx} | {float(r['net_in_band_m3']):,.0f} | "
                     f"{float(r['net_outside_band_m3']):,.0f} | {float(r['outside_median_crest_shift_cm']):+.2f} |")
    initial = {(r["case"], int(r["dx_m"])): r for r in rows if r["horizon"] == "original_window"}
    for case, label in cases.items():
        if (case, 1) not in initial or (case, 5) not in initial:
            continue
        fine, coarse = initial[case, 1], initial[case, 5]
        outside = [float(r["net_outside_band_m3"]) for r in (fine, coarse)]
        total = [float(r["net_section_m3"]) for r in (fine, coarse)]
        if all(t > 0 and 0 <= v <= t for t, v in zip(total, outside, strict=True)):
            lines.extend(["", f"Through hour {float(fine['end_h']):g} of the {label.lower()} storm, the portion of net "
                          f"eastward transport outside the band was {100 * outside[0] / total[0]:.1f}% on the 1 m "
                          f"grid and {100 * outside[1] / total[1]:.1f}% on the 5 m grid."])
    lines.extend(["", "This exploratory fixed-section measurement locates changes in net routing during the storm. "
                  "The opening band includes adjacent land, and grid refinement changes both the represented crest and "
                  "the hydraulic solution. [All formulations, crest elevations and accepted face-volume identities]"
                  "(../fidelity/native_results/embankment_transport.csv)", ""])
    figure = "supplementary/operational_storms/plain_709_embankment_profiles.pdf"
    if (study / figure).is_file():
        lines.extend([f"[Transverse profiles of crest elevation and transport](../{figure}). Shading marks the opening band.", ""])
    return "\n".join(lines)


def native_reach_contrasts(*, study):
    file = study / "fidelity/native_results/regional_water.csv"
    if not file.exists():
        return ""
    groups = defaultdict(dict)
    for row in csv_read(file=file):
        if float(row["factor"]) != 1:
            continue
        key = (row["case"], float(row["dx_m"]), float(row["end_h"]), float(row["interval_s"]),
               float(row["section_x_m"]), row["physical_sha"])
        reaches = groups[key].setdefault(row["method"], {})
        if row["reach"] in reaches:
            raise ValueError("Duplicate water account for one matched reach")
        reaches[row["reach"]] = row
    rows = []
    for (case, dx, end, cadence, section, _), peers in sorted(groups.items()):
        if "fv1" not in peers or "hancock_dense" not in peers:
            continue
        for reach in ("upstream", "downstream"):
            fv, h = (peers[m][reach] for m in ("fv1", "hancock_dense"))
            values = [float(fv[k]) - float(h[k]) for k in ("east_eastward_m3", "infiltration_m3", "storage_m3")]
            rows.append(f"| {case} | {dx:g} | {end:g} | {cadence:g} | {section:g} | {reach} | "
                        + " | ".join(f"{v:+,.0f}" for v in values) + " |")
    if not rows:
        return ""
    return "\n".join(["", "### Where export differences develop", "",
                      "The table gives FV1 minus dense Hancock on either side of the internal gauge. "
                      "The upstream east face is the gauge and the downstream east face is the outlet. "
                      "Infiltration and retained storage come from separate cell fields. Each reach closes "
                      "against its initial water, rain and signed transport on all four sides.", "",
                      "| Storm | Grid (m) | End (h) | Cadence (s) | Gauge x (m) | Reach | Δ eastward export (m³) | "
                      "Δ infiltration (m³) | Δ retained storage (m³) |",
                      "|---|---:|---:|---:|---:|---|---:|---:|---:|", *rows, "",
                      "[Per-formulation water accounts and field identities](../fidelity/native_results/regional_water.csv)", ""])


def native_grid_field_comparison(*, study):
    file = study / "fidelity/native_results/native_grid_field_differences.csv"
    rows = csv_read(file=file) if file.exists() else []
    if not rows:
        return ""
    text = ["## Native hydraulic cell comparisons", "",
            "Each row compares fields at the stated physical time on identical hydraulic cells against ordinary-CFL RK2 for the same "
            "storm. Depth RMSE covers all cells. Extent uses thresholds on individual cells. "
            "The half-CFL RK2 rows measure temporal sensitivity. Absolute practical accuracy at the native "
            "terrain floor remains unresolved.", "",
            "| Storm | End (h) | Grid (m) | Compared formulation | CFL factor | Depth RMSE (mm) | "
            "Maximum-depth RMSE (mm) | 5 cm IoU | 10 cm IoU |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=itemgetter("case", "method", "cfl_factor")):
        name = {"hancock_dense": "Hancock", "fv1": "FV1", "adaptive": "RK2"}[r["method"]]
        ious = [format(float(r[k]), ".5f") if r[k] else "empty union" for k in ("iou_5cm", "iou_10cm")]
        text.append(f"| {r['case']} | {r['end_h']} | {float(r['dx_m']):g} | {name} | {r['cfl_factor']} | "
                    f"{1000 * float(r['depth_rmse_m']):.5g} | {1000 * float(r['maximum_depth_rmse_m']):.5g} | "
                    f"{ious[0]} | {ious[1]} |")
    text.extend(["", "[Wet-union errors, connectivity and map identities](../fidelity/native_results/native_grid_field_differences.csv)"])
    maps = study / "supplementary/operational_storms"
    links = [f"[{dx} m hydraulic grid](../supplementary/operational_storms/plain_709_extent_10cm_{dx}m.pdf)"
             for dx in (1, 5, 10) if (maps / f"plain_709_extent_10cm_{dx}m.pdf").is_file()]
    if links:
        text.extend(["", "The maps show the same plain two hours after rainfall ended for the moderate and intense "
                     "storms. Shading preserves the fraction of hydraulic cells reaching 10 cm in each 10 m reporting "
                     "cell. The dashed line marks the internal gauge. " + " · ".join(links)])

    return "\n" + "\n".join(text) + "\n"


def native_timing_correction(*, study):
    file = study / "fidelity/native_results/timing_provenance.csv"
    if not file.exists():
        return ""
    rows = [r for r in csv_read(file=file) if r["cohort"] == "complete-native-metre"]
    changed = [r for r in rows if int(r["changed_timing_intervals"]) > 0]
    if not changed:
        return ""
    cells = sum(int(r["initial_intervals"]) for r in rows)
    return (f"\nAcross {len(rows)} completed native storm configurations, {len(changed)} restart records "
            f"inherited initial-window timings from a different execution of the same hydraulics. "
            f"The revised costs use each configuration's own measured initial and recession intervals. "
            f"Both gauges and accepted step counts agreed at all {cells:,} initial intervals checked. "
            "All operational costs, budget decisions and advisor inputs use this attribution. "
            "[Individual timing corrections and run identities](../fidelity/native_results/timing_provenance.csv)\n")


def native_choice_table(*, file, references, objectives):
    groups = defaultdict(dict)
    for r in csv_read(file=file):
        scope = r["station"], r["objective"]
        if (r["cohort"] != "complete-native-metre" or (r["case"], r["reference_id"]) not in references
                or scope not in objectives or float(r["tolerance"]) != objectives[scope][1]):
            continue
        key = (r["case"], float(r["end_h"]), float(r["interval_s"]), r["reference_id"], r["representation"], r["cap"])
        values = groups[key].setdefault(scope, [])
        if not r["id"]:
            if int(r["assessable"]):
                status = "Outside tolerance"
            elif int(r.get("terrain_unassessed") or 0):
                status = "No assessed admissible grid"
            elif int(r.get("terrain_requires_refinement") or 0):
                status = "Refine terrain"
            else:
                status = "Unresolved"
            values.append(status)
        else:
            method = {"fv1": "FV1", "adaptive": "RK2", "hancock_auto": "Hancock auto",
                      "hancock_dense": "Hancock dense", "hancock_adaptive": "Hancock tiles"}.get(r["method"], r["method"])
            values.append(f"{method} {r['dx']} m ({float(r['runtime_s']):.3g} s)")
    lines = ["| Storm | End (h) | Interval (s) | " + " | ".join(v[0] for v in objectives.values()) + " |",
             "|---|---:|---:|" + "---|" * len(objectives)]
    for (case, hours, cadence, _, _, _), decisions in sorted(groups.items()):
        lines.append(f"| {case} | {hours:g} | {cadence:g} | "
                     + " | ".join(" / ".join(decisions.get(scope, ["Unresolved"])) for scope in objectives) + " |")
    return lines


def native_budget_evidence(*, study, tolerance):
    folder = study / "fidelity/native_results"
    file = folder / "least_cost_terrain_fidelity.csv"
    if not file.is_file():
        return ""
    checks = [r for r in csv_read(file=folder / "reference_checks.csv")
              if r["cohort"] == "complete-native-metre" and r["station"] == "outlet"
              and r["metric"] == "peak_bias_pct" and all(r[k] for k in ("space_id", "cross_id", "time_id"))]
    references = {(r["case"], r["reference_id"]) for r in checks}
    if not references:
        return ""
    scopes = (("outlet", "peak_bias_pct"), ("outlet", "shape_tv_pct"), ("outlet", "net_export_bias_pct"),
              ("outlet", "infiltration_bias_pct"), ("outlet", "width80_bias_pct"),
              ("section", "peak_bias_pct"), ("section", "shape_tv_pct"))
    labels = ("Outlet peak", "Outlet shape", "Net export", "Infiltration", "Outlet width", "Internal peak", "Internal shape")
    hydraulic = {scope: (label, tolerance) for scope, label in zip(scopes, labels, strict=True)}
    text = ["", "## Fixed-area storms and the hydraulic objective", "",
            f"The table gives the cheapest terrain-admissible measured configuration at a {tolerance:g}% tolerance on each "
            "observable. Compliance includes the measured reference sensitivity. Each row fixes the storm, "
            "reference, reporting interval and recession endpoint. Shape measures redistribution of normalised "
            "discharge volume, and width measures the central 80% of that volume.", "",
            *native_choice_table(file=file, references=references, objectives=hydraulic)]
    spatial_file = folder / "least_cost_terrain_depth_extent.csv"
    if spatial_file.is_file():
        import json

        settings = json.loads((study / "fidelity/native_advisor_settings.json").read_text())
        depth = settings["depth_tolerance_m"]
        spatial = {("domain", "depth_rmse_m"): ("Final depth", depth),
                   ("domain", "maximum_depth_rmse_m"): ("Maximum-depth map", depth),
                   ("domain", "wet_union_depth_rmse_5cm_m"): ("Depth on 5 cm wet union", depth),
                   ("domain", "wet_union_depth_rmse_10cm_m"): ("Depth on 10 cm wet union", depth),
                   ("domain", "maximum_extent_error_5cm"): ("Maximum 5 cm extent", tolerance / 100),
                   ("domain", "maximum_extent_error_10cm"): ("Maximum 10 cm extent", tolerance / 100)}
        text.extend(["", f"Depth decisions use an RMSE tolerance of {100 * depth:g} cm. Maximum extent uses the "
                     f"cellwise maximum-depth map and a {tolerance:g}% tolerance on one minus inundation IoU.", "",
                     *native_choice_table(file=spatial_file, references=references, objectives=spatial)])
    text.extend(["", "Unresolved marks observations without an eligible hydraulic reference. Outside tolerance "
                 "marks a qualified comparison in which every terrain-admissible candidate exceeds the error bound. "
                 "Refine terrain marks a hydraulically assessable comparison where each candidate grid failed "
                 "the pre-run terrain check. The accompanying tables retain the hydraulic-only and terrain-admissible decisions.", "",
                 "| Storm | RK2 spatial peak difference (%) | Fine formulation difference (%) | Half-CFL difference (%) | Peak reference |",
                 "|---|---:|---:|---:|---|"])
    for r in sorted(checks, key=case_name):
        values = [fmt(r[k]) for k in ("space", "cross", "time")]
        outcome = "Qualified" if r["qualified"] == "True" else "Unresolved"
        text.append(f"| {r['case']} | " + " | ".join(values) + f" | {outcome} |")
    screen = folder / "terrain_screen_fidelity.csv"
    if screen.is_file():
        import json

        controls = {(r["case"], int(r["dx_m"])): r for r in csv_read(file=screen)
                    if (r["case"], r["reference_id"]) in references and r["hydraulic_eligible"] == "True"
                    and r["critical_connection_lost"] == "True"}
        probe = json.loads((study / "fidelity/native_feature_settings.json").read_text())
        if controls:
            text.extend(["", f"The static connection test uses an {probe['control_window_m']} m square "
                         f"around the prescribed opening and a water level {100 * probe['spill_head_m']:g} cm above "
                         "the native spill elevation. These grids lose the connection even where a hydraulic observable "
                         "meets tolerance.", "", "| Storm | Grid (m) | Maximum local barrier lowering (cm) |",
                         "|---|---:|---:|"])
            for (case, spacing), row in sorted(controls.items()):
                text.append(f"| {case} | {spacing} | {100 * float(row['barrier_lowering_m']):.2f} |")
            text.extend(["", "[Control maps and spill-height sensitivity](../fidelity/native_publication/control_stage_sensitivity.csv)"])
    for r in csv_read(file=folder / "recession_depth.csv"):
        if (r["cohort"] != "complete-native-metre" or r["id"] != r["reference_id"]
                or (r["case"], r["reference_id"]) not in references):
            continue
        support = float(r["support_m"])
        area_final = float(r["reference_wet_area_5cm_m2"])
        area_max = float(r["maximum_reference_wet_area_10cm_m2"])
        text.extend(["", f"On the {support:g} m assessment grid for {r['case']}, the fine reference retained "
                     f"{area_final / support**2:g} cells ({area_final:g} m²) at or above 5 cm at the endpoint. "
                     f"The area-averaged cell maxima reached 10 cm on {area_max / support**2:g} cells ({area_max:g} m²)."])
    text.extend(["", "[Cost and error bounds at a stated tolerance](../tables/21_native_cost_at_tolerance.md) · "
                 "[Error within an FV1 runtime budget](../tables/22_native_error_at_budget.md) · "
                 "[Timing repeats and spread](../fidelity/native_results/timing_distribution.csv) · "
                 "[Depth and wet-area support](../fidelity/native_results/recession_depth.csv)", ""])
    return "\n".join(text)


def case_name(row):
    return row["case"]


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
                correction += native_timing_correction(study=study)
                correction += native_budget_evidence(study=study, tolerance=5)
                correction += native_operational_balances(study=study, minimum_cells=250_000)
                for name, label in (("fv1", "FV1"), ("hancock", "Hancock")):
                    temporal = study / f"fidelity/native_results/{name}_temporal_sensitivity.csv"
                    if temporal.is_file():
                        for row in csv_read(file=temporal):
                            correction += (
                                f"\nOn {row['case']} at {row['dx_m']} m, reducing {label} CFL from "
                                f"{float(row['normal_cfl']):g} to {float(row['reduced_cfl']):g} changed the "
                                f"one-minute outlet peak by {float(row['pct_change_peak_m3s']):+.3f}% "
                                f"and net export by {float(row['change_net_export_m3']):+,.1f} m³ through "
                                f"{float(row['end_h']):g} h. CPU runs used the same initial fields and forcing. "
                                "[Time refinement, both gauge records and water accounts]"
                                f"(../fidelity/native_results/{temporal.name})\n"
                            )
                correction += plain_grid_fidelity(study=study, cases={"plain-709-moderate-1000x1000": "Moderate",
                                                                     "plain-709-intense-1000x1000": "Intense"})
                correction += plain_embankment(study=study, cases={"plain-709-moderate-1000x1000": "Moderate",
                                                                  "plain-709-intense-1000x1000": "Intense"}, face_m=880)
                correction += native_reach_contrasts(study=study)
                correction += native_grid_field_comparison(study=study)
                correction += native_advisor_findings(study=study)
        transfer = study / "fidelity/orton_transfer/transfer.csv"
        if transfer.exists():
            transfers = csv_read(file=transfer)
            correction += (f"\n{len(transfers)} of nine Orton Bradley configurations completed the archived 48-hour HIRDS "
                           "design storm. These measured differences quantify numerical transfer to real terrain under design "
                           "forcing. The footprint area added by aggregation, represented stage width and greatest outlet-face "
                           "displacement accompany each grid comparison. Errors against an observed event remain unknown. "
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
    revisions_file = study / "fidelity/outlet_results/revision_evidence.json"
    if revisions_file.is_file():
        import json

        revisions = json.loads(revisions_file.read_text())["revisions"]
        lead = (f"The steep FREE-outlet investigation compares audited revision "
                f"`{revisions['audit']['commit'][:12]}` with repair `{revisions['repair']['commit'][:12]}`. "
                "The original and extended 104.86 km² incised forecasts, observable-specific reference "
                "checks and measured costs are in [Repaired FREE outlet](#repaired-free-outlet). "
                "The earlier operational timings and advisor scores below describe their archived implementations. "
                "[Current native-grid qualification](fidelity/native_publication/README.md)")
        heading, body = audit.split("\n", 1)
        audit = heading + "\n\n" + lead + "\n" + body
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
        f"The historical frozen advisor made a qualified feasible selection in {fmt(a['feasible_coverage_pct'])}% "
        f"of held-out decisions with a measured feasible option. It abstained on {a['abstentions']} of "
        f"{a['decisions']} decisions. Terrain loss and reference uncertainty limit recommendations for "
        "specific sites and observables. [Held-out comparisons](../tables/10_advisor.md)",
        "Retain adaptive Hancock as the development default. Keep explicit execution overrides and "
        "select FV1 when the measured error is acceptable for the intended observable. "
        "[Results](results.md) · [Discussion](discussion.md)",
    ]
    if (study / "fidelity/advisor_design/paired.json").exists():
        conclusions.insert(-1, "The native-grid advisor uses whole-terrain held-out tests. "
                           + ("The frozen evaluation is available."
                              if (study / "fidelity/native_advisor/heldout_evaluation.json").exists()
                              else "Its independent validation is pending.")
                           + " [Independent evaluation and available evidence](../fidelity/native_publication/README.md#held-out-solver-advisor-evaluation)")
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
