"""Build the work-to-date summary from completed hydraulic records without starting a simulation."""

import json
from pathlib import Path
from statistics import median

import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment


def pulse_comparison(*, directory):
    records = metrics.csv_read(file=directory / "results/pulses.csv")
    folder = directory / "long_pulses"
    archive = folder / (folder / "current.txt").read_text().strip()
    pulse = json.loads((archive / "inputs.json").read_text())["cfg"]
    chosen = {
        (int(row["dx"]), row["method"]): row
        for row in records
        if row["case"] == "smooth"
        and float(row["distance_m"]) == 200
        and float(row["cap_courant"]) == 0.13
    }
    ref = chosen[10, "fv1"]
    text = [
        "## Travelling flood pulse",
        "",
        f"At the section 200 m from the initial pulse centre, the analytical peak averaged over one second is {float(ref['reference_peak_m3s']):.3f} m³/s. The central 80% of discharge passes in {float(ref['reference_width80_s']):.2f} s. The table gives signed bias relative to this solution.",
        "",
        "| Hydraulic grid | FV1 peak | Hancock peak | FV2 peak | FV1 width | Hancock width | FV2 width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spacing in (1, 5, 10):
        values = [
            float(chosen[spacing, method][metric])
            for metric in ("peak_bias_pct", "width80_bias_pct")
            for method in ("fv1", "hancock_adaptive", "adaptive")
        ]
        text.append(
            f"| {spacing} m | " + " | ".join(f"{v:+.2f}%" for v in values) + " |"
        )
    text.extend(
        [
            "",
            "### Measured 5 m Hancock and 1 m FV1 costs",
            "",
            f"The following runs cover the same {pulse['length']:,} × {pulse['width']:g} m channel and {pulse['duration']:g} s of propagation, with the same Courant cap and {pulse['interval']:g} s output. Hydraulic times exclude compilation, graph capture and field extraction. The signed biases use the exact wave at 200 m. A single timing per configuration supports this channel comparison.",
            "",
            "| Pulse | Method | Hydraulic time (s) | Peak bias | Central 80% width bias |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for profile in sorted({r["case"] for r in records}):
        for method, spacing, name in (
            ("fv1", 1, "FV1 sparse"),
            ("hancock_adaptive", 5, "Hancock dense"),
        ):
            selected = [
                r
                for r in records
                if r["case"] == profile
                and r["method"] == method
                and int(r["dx"]) == spacing
                and float(r["distance_m"]) == 200
                and float(r["cap_courant"]) == 0.13
            ]
            (row,) = selected
            values = [
                f"{float(row[key]):+.2f}%"
                if row["qualified_" + key] == "True"
                else "unresolved"
                for key in ("peak_bias_pct", "width80_bias_pct")
            ]
            text.append(
                f"| {profile.capitalize()} | {name}, {spacing} m | {float(row['runtime_s']):.3f} | "
                + " | ".join(values)
                + " |"
            )
    sharp = {
        (r["method"], int(r["dx"])): r for r in metrics.csv_read(file=directory / "results/sharp_pulses.csv")
        if float(r["cap_courant"]) == .13 and float(r["distance_m"]) == 200 and r["qualified_peak_bias_pct"] == "True"
    }
    coarse, fine, fv1 = sharp["hancock_adaptive", 5], sharp["hancock_adaptive", 1], sharp["fv1", 1]
    text += [
        "",
        f"With quarter-second accepted transport at 200 m, 5 m Hancock took {float(coarse['runtime_s']):.3f} s "
        f"and 1 m FV1 took {float(fv1['runtime_s']):.3f} s ({float(fv1['runtime_s'])/float(coarse['runtime_s']):.2f}×). "
        f"Their respective sharp-pulse peak biases were {float(coarse['peak_bias_pct']):+.2f}% and "
        f"{float(fv1['peak_bias_pct']):+.2f}%. Refining Hancock to 1 m reduced its peak bias to "
        f"{float(fine['peak_bias_pct']):+.2f}%. [Quarter-second analytical comparisons](results/sharp_pulses.csv)",
        "",
        "[Measured physical and relative quantities](results/pulses.csv) · [Independent integrations of the reference](results/pulse_reference_checks.csv)",
    ]
    return text


def storm_comparison(*, directory):
    records = metrics.csv_read(file=directory / "small_storms/outlet_metrics.csv")
    horizons = metrics.csv_read(file=directory / "small_storms/recession_horizons.csv")
    text = [
        "## Six-hour rainfall on the small domains",
        "",
        f"All {len(horizons)} events have a common checked recession. {sum(float(r['common_end_h']) > 8 for r in horizons)} needed longer than eight hours. The longest reached {max(float(r['common_end_h']) for r in horizons):g} h. Each 10 m forecast is compared to its own 2.5 m adaptive FV2 numerical reference. A 5 m spatial control, a 2.5 m halved-CFL control and a 2.5 m Hancock control test its sensitivity.",
        "",
        "| Median absolute outlet error | Matched eligible events | FV1 | Hancock dense | FV2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in (
        ("peak_bias_pct", "Peak discharge"),
        ("width80_bias_pct", "Central 80% duration"),
        ("shape_tv_pct", "Volume redistribution"),
    ):
        values = {}
        for method in ("fv1", "hancock_dense", "adaptive"):
            values[method] = {
                r["case"]: abs(float(r["value_pct"]))
                for r in records
                if r["cohort"] == "complete-legacy"
                and r["qualified"] == "True"
                and r["method"] == method
                and r["metric"] == metric
            }
        shared = set.intersection(*(set(v) for v in values.values()))
        if shared:
            cells = [
                f"{median(group[case] for case in shared):.2f}%"
                for group in values.values()
            ]
        else:
            cells = ["unresolved"] * len(values)
        text.append(f"| {label} | {len(shared)} | " + " | ".join(cells) + " |")
    text += [
        "",
        "[Results by landform, signed effects and uncertainty](small_storms/README.md)",
    ]
    return text


def incised_windows(*, directory, design):
    end_h = design["diagnostic_start_h"]
    settings = json.loads((directory / "recession_settings.json").read_text())
    window_s = settings["window_s"]
    folder = directory / "incised_crest"
    text = [
        "",
        f"### Incised channel through hour {end_h:g}",
        "",
        f"These historical runs repeat each {design['bed_m']:g} m parent elevation on its children. They diagnose artificial terrain steps. Outlet peaks are accepted face transport averaged over the reporting interval. Late export is the fraction of outward volume passing in the final {window_s / 3600:g} h of this window. Stored water can contribute to the later hydrograph.",
        "",
        "| Method | Grid | CFL | Peak (m³/s) | Outlet volume (m³) | Late export | End storage (m³) | Hydraulic time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runs = {}
    for path in folder.glob("*/result.json"):
        record = json.loads(path.read_text())
        inputs = record["inputs"]
        spec = inputs["spec"]
        if spec["case"] != design["case"] or inputs["design"] != design:
            continue
        key = spec["dx"], spec["method"], inputs["factor"]
        if key in runs:
            raise ValueError(f"Multiple incised runs for control {key}")
        runs[key] = record["data"]
    labels = {"adaptive": "FV2 RK2", "hancock_adaptive": "Hancock tiled"}
    for dx, method, factor in design["controls"]:
        data = runs.get((dx, method, factor))
        if data is None:
            continue
        t, q = assessment.curve(it={"d": data}, station="outlet")
        if t[-1] != end_h * 3600 or t[-1] - window_s not in t:
            raise ValueError("Incomplete reporting window for the incised control")
        outward = np.maximum(q, 0) * np.diff(t)
        tail = outward[t[:-1] >= t[-1] - window_s].sum()
        volume = outward.sum()
        fraction = (
            f"{100 * tail / volume:.2f}%"
            if volume >= settings["minimum_volume_m3"]
            else "negligible"
        )
        cfl = data["numerical"]["cfl"]
        text.append(
            f"| {labels[method]} | {dx:g} m | {cfl:g} | {q.max():.3f} | "
            f"{volume:,.0f} | {fraction} | {data['final_storage_m3']:,.0f} | {data['hydraulic_s']:.1f} |"
        )
    storage = folder / "initial_window_storage.csv"
    if storage.is_file():
        fine, coarse = sorted(metrics.csv_read(file=storage), key=grid_spacing)
        text.extend(
            [
                "",
                f"At hour {float(fine['end_h']):g}, outward boundary transport on the finer FV2 grid was {float(coarse['outflow_m3']) - float(fine['outflow_m3']):,.0f} m³ lower. It held {float(fine['storage_m3']) - float(coarse['storage_m3']):,.0f} m³ more water and had infiltrated {float(coarse['infiltration_m3']) - float(fine['infiltration_m3']):,.0f} m³ less. At depths ≥5 cm the inundated area was {float(fine['inundated_5cm_km2']):.3f} versus {float(coarse['inundated_5cm_km2']):.3f} km² (5 m versus 10 m grid). At depths ≥10 cm the corresponding areas were {float(fine['inundated_10cm_km2']):.3f} versus {float(coarse['inundated_10cm_km2']):.3f} km². [Storage and inundation maps](incised_crest/initial_window_storage.pdf) · [Water balance and run identities](incised_crest/initial_window_storage.csv)",
            ]
        )
    return text


def grid_spacing(row):
    return float(row["dx_m"])


def terrain_and_calibration(*, study):
    processed = study / "processed"
    terrain = metrics.csv_read(file=study / "fidelity/mean_windows/controls.csv")
    connectivity = []
    for spacing in sorted({float(r["dx_m"]) for r in terrain}):
        cases = [r for r in terrain if float(r["dx_m"]) == spacing]
        lost = sum(r["connected"] == "False" for r in cases)
        connectivity.append(f"{lost}/{len(cases)} at {spacing:g} m")
    transfers = metrics.csv_read(file=processed / "comparisons/transfer.csv")
    scores = [
        float(r["improvement"])
        for r in transfers
        if r["qualified_for_objective"] == "True"
    ]
    large = [
        r
        for r in metrics.csv_read(file=processed / "advisor/eligibility_breakdown.csv")
        if int(r["cells"]) >= 1_000_000
    ]
    passes = [r for r in large if int(r["within_hydraulic_tolerance"]) > 0]
    admissible = sum(r["terrain_admissible"] == "True" for r in passes)
    return [
        "## Terrain, calibration and solver selection",
        "",
        "With bottom-up area averaging, the prescribed 8 m opening lost its native four-neighbour connection in "
        + ", ".join(connectivity)
        + ". The probe stage was 1 cm above the native spill elevation. [Aligned native and aggregated terrain controls](mean_windows/controls.csv)",
        "",
        f"Bounded FV1 Manning fits improved {sum(x > 0 for x in scores)}/{len(scores)} reference-qualified transfer scores and worsened {sum(x < 0 for x in scores)}. Several scores come from different objectives and calibration bounds on the same terrain. [Calibration and transfer](../tables/09_calibration.md)",
        "",
        f"In the original independent million-cell advisor evaluation, {len(passes)}/{len(large)} decisions had a measured configuration within hydraulic tolerance. Of those, {admissible} also met the terrain requirement. This test supplies no feasible operational cost comparison for the advisor. [Held-out decisions and abstentions](../tables/13_advisor_scale.md)",
    ]


def repaired_incised(*, directory):
    file = directory / "outlet_results/events.csv"
    if not file.is_file():
        return []
    events = metrics.csv_read(file=file)
    pairs = {}
    for row in events:
        if float(row["cfl_factor"]) != 1:
            continue
        key = (int(row["dx_m"]), row["method"], float(row["end_h"]))
        reaches = pairs.setdefault(key, {})
        length = int(row["extension_m"])
        if length in reaches:
            raise ValueError("Duplicate incised reach under the same physical comparison")
        reaches[length] = row
    lines = [
        "## Repaired FREE outlet on the 104.86 km² incised catchment", "",
        "Each pair reports discharge through the same physical section over six hours of rainfall "
        "and the common recession. Export integrates accepted face transport. Moving the boundary "
        "downstream retains that section as an internal gauge.", "",
        "| Grid (m) | Method | End (h) | Extension (m) | Peak at FREE face (m³/s) | "
        "Peak at buffered gauge (m³/s) | Unbuffered export (m³) | Buffered gauge volume (m³) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {"adaptive": "FV2 RK2", "hancock_adaptive": "Hancock tiled", "fv1": "FV1 sparse"}
    for (dx, method, hours), reaches in sorted(pairs.items()):
        if 0 not in reaches:
            continue
        face = reaches[0]
        for extension, gauge in sorted(reaches.items()):
            if extension > 0:
                lines.append(f"| {dx} | {names[method]} | {hours:g} | {extension} | "
                             f"{float(face['minute_peak_m3s']):.3f} | {float(gauge['minute_peak_m3s']):.3f} | "
                             f"{float(face['gauge_m3']):,.0f} | {float(gauge['gauge_m3']):,.0f} |")
    lines.extend(["", "[Repaired outlet, water budgets and refinement checks](outlet_results/README.md) · "
                  "[Current native-grid storm measurements and budget decisions](native_publication/README.md)", ""])
    return lines


def generate(*, directory):
    rows = metrics.csv_read(file=directory / "results/native_horizons.csv")
    longest = {}
    for r in rows:
        if r["station"] == "outlet" and r["method"] != "fv1":
            if r["case"] not in longest or float(r["end_h"]) > float(
                longest[r["case"]]["end_h"]
            ):
                longest[r["case"]] = r
    ratios = [float(r["ratio_over_fv1"]) for r in longest.values()]
    text = [
        "# FloodWarp results to date",
        "",
        f"In the original centre-sampled archive, paired sparse FV1 and adaptive Hancock forecasts are complete on {len(longest)} 1,048,576-cell landforms. Each received six hours of rain. The verified endpoints range from {min(float(r['end_h']) for r in longest.values()):g} to {max(float(r['end_h']) for r in longest.values()):g} h. Taking the longest measured pair once per landform, Hancock's median hydraulic runtime is {median(ratios):.2f}× FV1 (range {min(ratios):.2f}–{max(ratios):.2f}×). Execution policies here were selected from original-window measurements.",
        "",
        "[Landform timings, storage, omitted export and accepted timesteps](NATIVE_RECESSIONS.md) · [Common-timestep GPU qualification](../../../validation/hancock_long_storm_review/DECISION.md) · [Hancock structures and verification](../../../docs/HANCOCK_STRUCTURES.md)",
        "",
        "The 5 m incised runs that replicated 10 m elevations are a staircase diagnostic. Practical terrain comparisons use bottom-up native cell averages. [Cause, verification and corrected runs](DEM_CORRECTION.md)",
        "",
        *pulse_comparison(directory=directory),
        "",
        *storm_comparison(directory=directory),
        "",
        *terrain_and_calibration(study=directory.parent),
        "",
        "## Operational reference controls",
        "",
        "The following counts describe the original forecast windows. A separate matched-recession receipt records the common end used for hydraulic assessment.",
        "",
        "| Investigation | Original windows completed | Verified matched end |",
        "|---|---:|---|",
    ]
    incised_design = json.loads((directory / "incised_settings.json").read_text())
    designs = [
        (
            "Rolling terrain, independently refined SWE",
            "rolling-704-soil-5120x5120",
            "expanded",
            directory / "storm_runs",
            4,
        ),
        (
            "Incised late crest, fixed 10 m bed",
            "incised-707-spatial-10240x10240",
            "incised-crest",
            directory / "incised_crest",
            len(incised_design["controls"]),
        ),
    ]
    groups = [
        json.loads(p.read_text())
        for p in (directory / "extended_runs").glob("group_*.json")
    ]
    for name, case, cohort, folder, planned in designs:
        count = 0
        for p in folder.glob("*/result.json"):
            record = json.loads(p.read_text())
            spec = (
                record["inputs"]["spec"]
                if cohort == "incised-crest"
                else record["spec"]
            )
            if spec["case"] == case:
                count += 1
        verified = [
            g["end_s"] / 3600
            for g in groups
            if g["case"] == case
            and g["cohort"] == cohort
            and len(g["series"]) >= planned
        ]
        common = f"{max(verified):g} h" if verified else "pending"
        text.append(f"| {name} | {count}/{planned} | {common} |")
    native_file = directory / "mean_storms/complete_events.csv"
    if native_file.is_file():
        completed = metrics.csv_read(file=native_file)
        practical = [
            "## Archived incised storm before the FREE-outlet repair", "",
            "The 10 m bed averages four 5 m elevations. Each 5 m elevation averages 25 native metre cells. "
            "The six pre-repair forecasts used a common rainfall field and native initial water. Peak discharge is a one-minute mean of accepted face volumes.",
            "", "| Grid (m) | Method | End (h) | Peak (m³/s) | Outlet volume (m³) | Infiltration (m³) | Hydraulic time (s) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        names = {"adaptive": "FV2 RK2", "hancock_adaptive": "Hancock tiled", "fv1": "FV1 sparse"}
        for row in completed:
            practical.append(f"| {row['dx_m']} | {names[row['method']]} | {float(row['end_h']):g} | "
                             f"{float(row['outlet_peak_m3s']):.2f} | {float(row['net_outlet_m3']):,.0f} | "
                             f"{float(row['infiltration_m3']):,.0f} | {float(row['hydraulic_s']):.1f} |")
        practical.extend(["", "Runoff can peak after rainfall ends. The two FV2 grids disagree about the largest one-minute peak. "
                          "Repeated measurements at the outlet and upstream sections locate the water released during that interval. "
                          "[Hydrographs](native_tail/README.md) · [Peak-minute budget](native_tail/CREST.md) · [Runoff partition](native_outlet/runoff_partition.csv)", ""])
        text[2:2] = practical
    text[2:2] = repaired_incised(directory=directory)
    text += incised_windows(directory=directory, design=incised_design)
    text += [
        "",
        "The incised diagnostic was defined after observing a late crest. It remains separate from advisor training and test decisions. Practical grid comparisons require the correctly aggregated DEMs and matched recession.",
        "",
        "[Incised one-minute and 15-minute hydrographs](figures/incised_native_cadence.pdf) · [Original terrain, calibration and advisor results](../CORRECTIVE_AUDIT.md)",
    ]
    (directory / "WORK_TO_DATE.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    generate(directory=directory)
