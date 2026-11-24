"""Build the work-to-date summary from completed hydraulic records without starting a simulation."""

import json
from pathlib import Path
from statistics import median

from research.floodwarp_methods_paper import metrics


def pulse_comparison(*, directory):
    records = metrics.csv_read(directory / "results/pulses.csv")
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
    text += [
        "",
        "[Measured physical and relative quantities](results/pulses.csv) · [Independent integrations of the reference](results/pulse_reference_checks.csv)",
    ]
    return text


def storm_comparison(*, directory):
    records = metrics.csv_read(directory / "small_storms/outlet_metrics.csv")
    horizons = metrics.csv_read(directory / "small_storms/recession_horizons.csv")
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


def generate(*, directory):
    rows = metrics.csv_read(directory / "results/native_horizons.csv")
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
        f"Paired sparse FV1 and adaptive Hancock forecasts are complete on {len(longest)} native 1,048,576-cell landforms. Each received six hours of rain. The verified endpoints range from {min(float(r['end_h']) for r in longest.values()):g} to {max(float(r['end_h']) for r in longest.values()):g} h. Taking the longest measured pair once per landform, Hancock's median hydraulic runtime is {median(ratios):.2f}× FV1, with {sum(v <= 1.5 for v in ratios)}/{len(ratios)} within 1.5×. Execution policies here were selected from original-window measurements.",
        "",
        "[Landform timings, storage, omitted export and accepted timesteps](NATIVE_RECESSIONS.md) · [Common-timestep GPU qualification](../../../validation/hancock_long_storm_review/DECISION.md) · [Hancock structures and verification](../../../docs/HANCOCK_STRUCTURES.md)",
        "",
        *pulse_comparison(directory=directory),
        "",
        *storm_comparison(directory=directory),
        "",
        "## Operational reference controls",
        "",
        "The following counts describe the original forecast windows. A separate matched-recession receipt records the common end used for hydraulic assessment.",
        "",
        "| Investigation | Original windows completed | Verified matched end |",
        "|---|---:|---|",
    ]
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
            len(
                json.loads((directory / "incised_settings.json").read_text())[
                    "controls"
                ]
            ),
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
    text += [
        "",
        "The incised diagnostic was defined after observing a late crest. It remains separate from advisor training and test decisions. Each control needs the same recession coverage before its hydrograph can support a solver ranking.",
        "",
        "[Incised one-minute and 15-minute hydrographs](figures/incised_native_cadence.pdf) · [Original terrain, calibration and advisor results](../CORRECTIVE_AUDIT.md)",
    ]
    (directory / "WORK_TO_DATE.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    generate(directory=directory)
