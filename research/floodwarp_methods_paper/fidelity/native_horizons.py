"""Measured influence of ending the million-cell forecasts at their original horizon."""

import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import receipts


def station_rows(*, data, minimum_volume, original_s):
    frames = data["frames"]
    time = np.asarray([f["t_s"] for f in frames])
    dt = np.diff(np.r_[0, time])
    original = time <= original_s
    if (
        not original.any()
        or time[original][-1] != original_s
        or time[-1] != data["end_s"]
    ):
        raise ValueError("Missing a measured endpoint")
    result = []
    for station, check in data["checks"].items():
        if not check["passed"]:
            raise ValueError(
                "The matched forecast still has significant late transport"
            )
        q = np.array([f[station + "_m3s"] for f in frames])
        outward = np.maximum(q, 0) * dt
        total = float(outward.sum())
        late = float(outward[~original].sum())
        result.append(
            dict(
                station=station,
                outward_m3=total,
                inward_m3=float(np.maximum(-q, 0) @ dt),
                original_outward_m3=total - late,
                omitted_outward_m3=late,
                omitted_pct=100 * late / total if total >= minimum_volume else None,
                original_peak_out_m3s=float(np.max(q[original], initial=0)),
                final_peak_out_m3s=float(np.max(q, initial=0)),
                last_two_h_over_total=check["recent_to_volume"],
            )
        )
    return result


def collect(*, directory, cases, minimum_volume):
    root = directory / "extended_runs"
    groups = []
    for file in sorted(root.glob("group_*.json")):
        g = json.loads(file.read_text())
        if (
            g["case"] in cases
            and g["cohort"] == "expanded"
            and g["representation"] == "sampled"
            and len(g["series"]) == 2
        ):
            groups.append((g["case"], g["end_s"], g["series"], file, "two-hour check"))
    for file in sorted((directory / "storage_tail").glob("*.json")):
        g = json.loads(file.read_text())
        groups.append(
            (
                g["options"]["case"],
                g["common_end_s"],
                g["series"],
                file,
                "12-hour sensitivity",
            )
        )
    rows = []
    for case, end, ids, file, design in sorted(groups):
        pair = [receipts.read(root=root, ident=i, end=end) for i in ids]
        methods = {r["spec"]["method"] for r in pair}
        if methods not in ({"fv1", "hancock_dense"}, {"fv1", "hancock_adaptive"}):
            continue
        fv1 = next(r for r in pair if r["spec"]["method"] == "fv1")
        fv1_steps = sum(f["steps"] for f in fv1["frames"])
        for r in pair:
            s = r["spec"]
            steps = sum(f["steps"] for f in r["frames"])
            time_ratio = r["hydraulic_s"] / fv1["hydraulic_s"]
            if s["case"] != case or r["end_s"] != end or not r["settled"]:
                raise ValueError("Incomplete native comparison for " + case)
            if any(length / s["dx"] != 1024 for length in s["dimensions"]):
                raise ValueError("Unexpected native comparison grid for " + case)
            for station in station_rows(
                data=r, minimum_volume=minimum_volume, original_s=3600 * s["hours"]
            ):
                rows.append(
                    dict(
                        case=case,
                        landform=s["landform"],
                        method=s["method"],
                        dx_m=s["dx"],
                        design=design,
                        original_h=s["hours"],
                        end_h=end / 3600,
                        series_id=r["id"],
                        original_id=r["parent_id"],
                        receipt=str(file.relative_to(directory)),
                        hydraulic_s=r["hydraulic_s"],
                        ratio_over_fv1=time_ratio,
                        steps=steps,
                        fv1_steps=fv1_steps,
                        steps_over_fv1=steps / fv1_steps,
                        ms_per_accepted_step=1000 * r["hydraulic_s"] / steps,
                        per_step_over_fv1=time_ratio * fv1_steps / steps,
                        storage_m3=r["storage_m3"],
                        infiltration_m3=r["infiltration_m3"],
                        residual_m3=r["residual_m3"],
                        **station,
                    )
                )
    return rows


def generate(*, directory):
    cases = json.loads((directory / "operational_settings.json").read_text())[
        "primary_cases"
    ]
    settings = json.loads((directory / "recession_settings.json").read_text())
    rows = collect(
        directory=directory, cases=cases, minimum_volume=settings["minimum_volume_m3"]
    )
    target = directory / "results" / "native_horizons.csv"
    metrics.csv_write(file=target, rows=rows)
    outlets = [r for r in rows if r["station"] == "outlet"]
    from research.floodwarp_methods_paper.fidelity import statistics

    cfg = json.loads((directory / "design.json").read_text())
    cadences = []
    for r in outlets:
        d = receipts.read(
            root=directory / "extended_runs",
            ident=r["series_id"],
            end=round(r["end_h"] * 3600),
        )
        item = dict(id=r["series_id"], d=d, s=d["spec"])
        group = {(r["design"], r["case"], "sampled"): [item]}
        cadences.extend(
            dict(c, end_h=r["end_h"]) for c in statistics.cadence(groups=group, cfg=cfg)
        )
    metrics.csv_write(file=target.with_name("native_cadence.csv"), rows=cadences)
    count = len({r["case"] for r in outlets})
    text = [
        "# Million-cell discharge after the original end time",
        "",
        f"Paired native-grid recession measurements are available for {count} of the {len(cases)} primary landforms. Each pair received six hours of rainfall. The original end time includes the modelled recession available in the earlier archive. The omitted share is the additional outward volume divided by total outward volume through the paired end time.",
        "",
        "| Landform | Method | Original end (h) | Paired end (h) | Outlet volume after original end (%) | Total hydraulic time (s) | Runtime / FV1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "fv1": "FV1 sparse",
        "hancock_dense": "Hancock dense",
        "hancock_adaptive": "Hancock tiled",
    }
    for r in outlets:
        share = (
            "negligible export"
            if r["omitted_pct"] is None
            else f"{r['omitted_pct']:.2f}"
        )
        text.append(
            f"| {r['landform'].capitalize()} | {labels[r['method']]} | {r['original_h']:g} | {r['end_h']:g} | {share} | {r['hydraulic_s']:.2f} | {r['ratio_over_fv1']:.2f} |"
        )
    longest = {}
    for r in outlets:
        if r["method"] == "fv1":
            continue
        if r["case"] not in longest or r["end_h"] > longest[r["case"]]["end_h"]:
            longest[r["case"]] = r
    if longest:
        ratios = [r["ratio_over_fv1"] for r in longest.values()]
        text.extend(
            [
                "",
                "## Accepted timesteps and whole-event cost",
                "",
                f"Using the longest paired native horizon once per landform, Hancock's median runtime was {np.median(ratios):.2f}× FV1. It met the 1.5× target on {sum(v <= 1.5 for v in ratios)} of {len(ratios)} landforms. Time per accepted step includes time spent in rejected trials.",
                "",
                "| Landform | Hancock steps | FV1 steps | Step count ratio | Time per step ratio |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        from research.floodwarp_methods_paper.fidelity import recession_figures

        costs = []
        for r in longest.values():
            baseline = next(
                f
                for f in outlets
                if f["receipt"] == r["receipt"] and f["method"] == "fv1"
            )
            costs.append(
                dict(r, hancock_s=r["hydraulic_s"], fv1_s=baseline["hydraulic_s"])
            )
            text.append(
                f"| {r['landform'].capitalize()} | {r['steps']:,} | {r['fv1_steps']:,} | {r['steps_over_fv1']:.3f} | {r['per_step_over_fv1']:.3f} |"
            )
        recession_figures.native_cost(directory=directory, rows=costs)
        text.extend(
            [
                "",
                "[Measured runtimes and contributions from timestep count and cost per step](figures/native_runtime_components.pdf)",
            ]
        )
    text += [
        "",
        "The mixed catchment also has a 12-hour residual-storage check at a later endpoint. Repeated horizons follow the same trajectories. Each timing covers both rainfall and recession. The Hancock policy in this comparison was selected from the original-window measurements. Independent spatial and temporal refinements determine eligibility for accuracy comparisons and can require a later shared endpoint.",
        "",
        "[Outlet and internal section volumes, peak discharges, storage, timing and run identities](results/native_horizons.csv) · [Effect of 15-minute reporting on peak magnitude and the interval containing the maximum](results/native_cadence.csv)",
    ]
    from research.floodwarp_methods_paper.fidelity import crest_report

    design = json.loads((directory / "incised_settings.json").read_text())
    if crest_report.native_panel(
        directory=directory, case=design["case"], records=outlets
    ):
        text.extend(
            [
                "",
                "[One-minute and 15-minute hydrographs for the incised catchment](figures/incised_native_cadence.pdf)",
            ]
        )
    (directory / "NATIVE_RECESSIONS.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    generate(directory=directory)
