"""Render measured practical comparisons from archived CSV files without running hydraulics."""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_control_report
from research.floodwarp_methods_paper.fidelity.assessment import numerical_identity
from research.floodwarp_methods_paper.fidelity.figures import save


OBJECTIVES = {
    "peak_bias_pct": ("Peak discharge error", "%"),
    "peak_delay_pct": ("Peak timing error", "% of reference duration"),
    "shape_tv_pct": ("Hydrograph shape error", "%"),
    "active_nrmse_peak_pct": ("Active-window RMSE", "% of reference peak"),
    "net_export_bias_pct": ("Net exported volume error", "%"),
    "infiltration_bias_pct": ("Cumulative infiltration error", "%"),
    "depth_rmse_m": ("Final depth RMSE", "m"),
    "extent_error_5cm": ("Final extent error at 5 cm", "1 − IoU"),
    "maximum_extent_error_10cm": ("Maximum extent error at 10 cm", "1 − IoU"),
}


def measured_fronts(*, directory, output):
    source = directory / "native_results"
    groups = defaultdict(list)
    counts = []
    for file in ("hydrograph_pareto.csv", "recession_depth_pareto.csv"):
        for row in metrics.csv_read(file=source / file):
            if (
                row["station"] in ("outlet", "domain")
                and row["objective"] in OBJECTIVES
            ):
                key = (
                    row["cohort"], row["case"], row["representation"],
                    row["reference_id"], float(row["interval_s"]), float(row["end_h"]),
                )
                groups[key].append(row)
    for key, records in sorted(groups.items()):
        cohort, case, representation, reference, interval, end_h = key
        panels = defaultdict(list)
        for row in records:
            panels[row["objective"]].append(row)
        for metric, rows in panels.items():
            permitted = [r for r in rows if r["eligible"] == "True"]
            counts.append(
                dict(
                    cohort=cohort,
                    case=case,
                    representation=representation,
                    end_h=end_h,
                    reference_id=reference,
                    interval_s=interval,
                    metric=metric,
                    measurements=len(rows),
                    eligible=len(permitted),
                    nondominated=sum(r["nondominated"] == "True" for r in permitted),
                )
            )
        if not any(r["eligible"] == "True" for r in records):
            continue
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), layout="constrained")
        for ax, (metric, (title, unit)) in zip(
            axes.flat, OBJECTIVES.items(), strict=True
        ):
            rows = [r for r in panels[metric] if r["eligible"] == "True"]
            ax.set(title=title, xlabel="Hydraulic time (s)", ylabel=unit)
            if not rows:
                ax.text(
                    0.5,
                    0.5,
                    "Reference unresolved",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue
            front = sorted(
                (float(r["runtime_s"]), float(r["absolute_error"]))
                for r in rows
                if r["nondominated"] == "True"
            )
            if front:
                ax.plot(
                    *zip(*front), color="#555555", linewidth=0.8, drawstyle="steps-post"
                )
            for r in rows:
                x, y = float(r["runtime_s"]), float(r["absolute_error"])
                ax.scatter(
                    x,
                    y,
                    s=18,
                    facecolors="#266582" if r["nondominated"] == "True" else "white",
                    edgecolors="#266582",
                )
                ax.annotate(
                    f"{r['formulation']} {r['execution_mode']} {r['dx']} m",
                    (x, y),
                    xytext=(2, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        fig.suptitle(f"{case} · {representation} · {end_h:g} h")
        save(
            fig=fig,
            folder=output,
            name=f"{case}_{cohort}_{runner.digest(representation)[:10]}_{reference}_{interval}_{end_h:g}h_pareto",
        )
    metrics.csv_write(file=output / "reference_eligibility.csv", rows=counts)
    return counts


def advisor_report(*, directory, output):
    pointer = directory / "native_advisor/heldout_evaluation.json"
    if not pointer.is_file():
        return ["The held-out native-terrain evaluation is pending."], []
    active = json.loads(pointer.read_text())
    archive = directory / active["path"]
    receipt = archive / "sources.json"
    if runner.checksum(receipt) != active["sha256"]:
        raise ValueError("Held-out evaluation receipt checksum changed")
    evidence = json.loads(receipt.read_text())
    frozen = directory / "native_advisor/frozen_model.json"
    model = json.loads(frozen.read_text())
    if model != evidence["frozen_model"] or runner.checksum(directory / model["path"]) != model["sha256"]:
        raise ValueError("Held-out evaluation does not match the frozen model")
    sources = [pointer, receipt, frozen, directory / model["path"]]
    for name, checksum in evidence["files"].items():
        path = archive / name
        if runner.checksum(path) != checksum:
            raise ValueError(f"Changed held-out evidence in {name}")
        sources.append(path)
    rows = metrics.csv_read(file=archive / "summary.csv")
    label = {"advisor": "Advisor", "slope": "Slope only", "fv1": "FV1",
             "hancock_dense": "Dense Hancock", "oracle": "Measured oracle"}
    panels = (
        ("feasible_coverage_pct", "Feasible opportunities covered (%)"),
        ("abstention_pct", "Abstentions / decisions (%)"),
        ("false_feasible_per_decision_pct", "False feasible / decisions (%)"),
        ("median_regret_pct", "Median assessed runtime regret (%)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout="constrained")
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for rule, name in label.items():
            series = sorted((float(r["percentage_scenario"]), float(r[metric]))
                            for r in rows if r["rule"] == rule and r[metric] != "")
            if series:
                ax.plot(*zip(*series), marker="o", label=name)
        ax.set(xlabel="Hydrograph / extent tolerance scenario (%)", ylabel=title,
               xticks=sorted({float(r["percentage_scenario"]) for r in rows}))
    handles, names = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="outside upper center", ncol=3)
    save(fig=fig, folder=output, name="heldout_advisor")
    table = ["| Scenario (%) | Rule | Decisions | Feasible opportunities | Successful selections | False feasible | Abstained |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['percentage_scenario']} | {label[r['rule']]} | {r['decisions']} | "
                     f"{r['empirical_feasible_opportunities']} | {r['successful_selections']} | "
                     f"{r['false_feasible']} | {r['abstentions']} |")
    relative = "../" + archive.relative_to(directory).as_posix()
    return [*table, "", "Depth tolerances are fixed across the percentage scenarios. "
            "Runtime regret requires an assessable selection and a measured feasible alternative.",
            "", f"[Validation figure](heldout_advisor.pdf) · [All decisions]({relative}/decisions.csv) · "
            f"[Independent final terrain and subgroup results]({relative}/strata.csv) · "
            f"[Joint method and resolution decisions, by requested spacing]({relative}/joint_summary.csv)"], sources


def policy_report(*, source, output):
    window_file = source / "policy_cost_windows.csv"
    event_file = source / "policy_cost_events.csv"
    if not window_file.is_file() or not event_file.is_file():
        return ["Matched execution timings are pending."], []
    events = metrics.csv_read(file=event_file)
    windows = metrics.csv_read(file=window_file)
    lines = []
    for cohort in sorted({r["cohort"] for r in events}):
        group = [r for r in events if r["cohort"] == cohort]
        chunks = [r for r in windows if r["cohort"] == cohort]
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout="constrained")
        names = sorted({r["case"].split("-")[0] for r in group})
        colours = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(names)}
        for case in sorted({r["case"] for r in group}):
            for ax, rows, x, xlabel, factor in (
                (axes[0, 0], group, "cells", "Cells (millions)", 1e-6),
                (axes[0, 1], group, "hancock_steps", "Accepted steps (thousands)", .001),
                (axes[1, 0], chunks, "wet_fraction", "Wet cells at interval end (%)", 100.),
                (axes[1, 1], chunks, "tiled_scheduled_fraction", "Scheduled cells at interval end (%)", 100.),
            ):
                sample = [r for r in rows if r["case"] == case and r[x] != ""]
                if sample:
                    ax.scatter([factor * float(r[x]) for r in sample],
                               [float(r["tiled_over_dense"]) for r in sample],
                               s=16, alpha=.7, color=colours[case.split("-")[0]],
                               label=case.split("-")[0].capitalize())
                ax.set(xlabel=xlabel, ylabel="Tiled / dense hydraulic time")
        for ax in axes.flat:
            ax.axhline(1, color=".4", linestyle=":", linewidth=.8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        families = dict(zip(labels, handles, strict=True))
        if families:
            fig.legend(families.values(), families.keys(), loc="outside upper center", ncol=4)
        name = "policy_cost_" + cohort.replace("-", "_")
        save(fig=fig, folder=output, name=name)
        lines.append(f"[{cohort}, {len(group)} matched grid/event comparisons]({name}.pdf)")
    durations = ", ".join(f"{seconds / 60:g}" for seconds in sorted({float(r["window_s"]) for r in windows}))
    lines.extend(["", "Each upper-panel point represents a measured forecast window. Each lower-panel point "
                  f"represents a recorded timing interval (minutes: {durations}). Footprints are sampled at interval end. "
                  "A ratio below one favours tiles. The same trajectory can contribute to both the initial "
                  "window and the matched recession products.",
                  "", "[Policy run identities and timings](../native_results/policy_cost_events.csv) · "
                  "[Individual timing intervals](../native_results/policy_cost_windows.csv)"])
    measurements = source / "measured_windows.csv"
    distribution = source / "timing_distribution.csv"
    summaries = selection_costs(events=events, windows=metrics.csv_read(file=measurements),
                                timing=metrics.csv_read(file=distribution))
    metrics.csv_write(file=output / "execution_choice_cost.csv", rows=summaries)
    if summaries:
        lines.extend(["", "### Automatic policy over the complete event", "",
                      "| Case | Grid (m) | End (h) | Auto chose | Fastest explicit | Chosen explicit / fastest explicit | Actual auto / chosen explicit |",
                      "|---|---:|---:|---|---|---:|---:|"])
        for r in summaries:
            lines.append(f"| {r['case']} | {r['dx_m']} | {r['end_h']:g} | {r['chosen']} | {r['fastest_explicit']} | "
                         f"{r['chosen_over_fastest']:.3f} | {r['auto_over_chosen']:.3f} |")
        lines.extend(["", "The first ratio measures the cost of selecting a schedule using the two explicit runs. "
                      "The second compares the measured automatic run with its selected explicit counterpart and includes timing variability. "
                      "Each cost uses its available hydraulic timing median. "
                      "[Run identities, sample sizes and timing ranges](execution_choice_cost.csv)"])
    return lines, [window_file, event_file, measurements, distribution]


def selection_costs(*, events, windows, timing):
    modes = {r["source_id"]: r["resolved_execution_mode"] for r in windows
             if r["horizon"] == "complete-native-metre" and r["method"] == "hancock_auto"}
    samples = {ident: r for r in timing if r["horizon"] == "completed recession"
               for ident in r["run_ids"].split()}
    out = []
    for event in events:
        if event["cohort"] != "complete-native-metre" or not event["auto_id"]:
            continue
        chosen = modes[event["auto_id"]]
        if chosen not in ("dense", "tiles"):
            raise ValueError("Automatic policy has no measured explicit schedule")
        records = {policy: samples[event[key]] for policy, key in
                   (("dense", "dense_id"), ("tiles", "tiled_id"), ("auto", "auto_id"))}
        times = {key: float(r["hydraulic_median_s"]) for key, r in records.items()}
        best = min(times["dense"], times["tiles"])
        row = dict(case=event["case"], dx_m=event["dx_m"], end_h=float(event["end_h"]), chosen=chosen,
                   fastest_explicit=", ".join(p for p in ("dense", "tiles") if times[p] == best),
                   chosen_over_fastest=times[chosen]/best, auto_over_chosen=times["auto"]/times[chosen])
        for policy, r in records.items():
            for field in ("run_ids", "independent_timings", "hydraulic_median_s", "hydraulic_min_s", "hydraulic_max_s"):
                row[policy + "_" + field] = r[field]
        out.append(row)
    return out


def run(*, directory):
    study = directory.parent
    source = directory / "native_results"
    output = directory / "native_publication"
    output.mkdir(exist_ok=True)
    rows = [
        r
        for r in metrics.csv_read(file=source / "measured_windows.csv")
        if r["horizon"].startswith("complete-") and not r["horizon"].startswith("complete-native-repeat-")
    ]
    counts = measured_fronts(directory=directory, output=output)
    current = runner.source_identity(root=directory.parents[2], study=study, stage="long-storms")
    text = [
        "# Native terrain, complete events and GPU cost",
        "",
        "All positions share the same native metre grid origin. Each 10 m elevation averages its four 5 m children. "
        "Each row reports the cost of its recorded method, execution policy and common event endpoint. "
        "The normalized-error Pareto tables use independently screened references for each hydraulic observable.",
        "",
        (f"The primary event rows use the current numerical source `{current[:12]}`."
         if rows and all(r["solver_source"] == current for r in rows) else
         "The post-repair wide-matrix qualification is pending. Historical window identities and numerical sources are retained in the linked data. "
         "[Repaired incised comparison](../outlet_results/README.md)"),
        "",
        "The internal gauge spans the middle column of x faces. Both peaks use interval-mean discharge from "
        "accepted face volumes. Positive net transport points downstream.",
        "",
        "| Case | Grid (m) | Method | Execution | CFL factor | End (h) | Cadence (s) | Internal peak (m³/s) | Outlet peak (m³/s) | Net export (m³) | Primary hydraulic time (s) | Steps |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        method = numerical_identity(method=r["method"])["formulation"]
        policy = r.get("resolved_execution_mode") or r["execution_mode"]
        if r["execution_mode"] == "auto":
            policy = "auto → " + policy
        text.append(
            f"| {r['case']} | {r['dx_m']} | {method} | {policy} | "
            f"{r['factor']} | {float(r['end_h']):g} | {float(r['interval_s']):g} | {float(r['section_peak_m3s']):.3f} | "
            f"{float(r['outlet_peak_m3s']):.3f} | {float(r['net_outlet_m3']):,.0f} | {float(r['hydraulic_s']):.2f} | {int(r['steps']):,} |"
        )
    text.extend(
        [
            "",
            f"{sum(r['eligible'] for r in counts)} of {sum(r['measurements'] for r in counts)} assessed objective/configuration rows have an eligible numerical reference. "
            "Multiple objectives on the same simulation share a physical realisation. "
            "A frontier can be estimated only among eligible observations for the same case, common time window and reference.",
            "",
            "[Eligibility by objective](reference_eligibility.csv) · [Measured windows and full water budgets](../native_results/measured_windows.csv) · "
            "[Peak, timing and shape comparisons](../native_results/hydrograph_pareto.csv) · [Depth and extent comparisons](../native_results/recession_depth_pareto.csv)",
        ]
    )
    policy_text, policy_sources = policy_report(source=source, output=output)
    text.extend(["", "## Dense and tiled execution", "", *policy_text])
    histories = directory / "native_step_history/parity.csv"
    if histories.is_file():
        traces = metrics.csv_read(file=histories)
        ids = {r["source_id"] for r in rows}
        confirmed = [r for r in traces if {r["dense_id"], r["peer_id"]} <= ids]
        if confirmed:
            text.extend(["", f"{len(confirmed)} complete policy comparisons have independent replays "
                         "with identical ordered FP64 timesteps, interval transport and final fields. "
                         "Instrumented replays are excluded from GPU timing samples. "
                         "[Trace hashes and raw replay identities](../native_step_history/parity.csv)"])
            policy_sources = [*policy_sources, histories]
    distribution = source / "timing_distribution.csv"
    if distribution.exists():
        repeats = metrics.csv_read(file=distribution)
        measured = [r for r in repeats if r["horizon"] == "completed recession"]
        complete = [r for r in measured if r["has_three_timings"] == "True"]
        text.extend(["",f"{len(complete)} of {len(measured)} matched full-recession configurations have at least three "
                     "independent hydraulic timings. Cost comparisons use their median. Other configurations retain their measured single-run costs "
                     "where available. End-to-end timing includes setup and recorded continuation work.",
                     "", "[Timing identities, quartiles and range](../native_results/timing_distribution.csv)"])
        policy_sources = [*policy_sources,distribution]
    settings_file = directory / "application_settings.json"
    settings = json.loads(settings_file.read_text())
    terrain_text, terrain_sources = native_control_report.run(
        directory=directory, output=output, heads_cm=settings["control_levels_above_native_spill_cm"],
    )
    text.extend(["", "## Terrain control elevations and connectivity", "", *terrain_text])
    advisor_text, advisor_sources = advisor_report(directory=directory, output=output)
    text.extend(["", "## Held-out Solver Advisor evaluation", "", *advisor_text])
    (output / "README.md").write_text("\n".join(text) + "\n")
    runner.write_json(
        path=output / "sources.json",
        data={
            str(p.relative_to(study)): runner.checksum(p)
            for p in (
                source / "measured_windows.csv",
                source / "hydrograph_pareto.csv",
                source / "recession_depth_pareto.csv",
                Path(__file__),
                *policy_sources,
                *terrain_sources,
                settings_file,
                Path(native_control_report.__file__),
                *advisor_sources,
            )
        },
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
