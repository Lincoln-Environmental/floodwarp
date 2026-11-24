"""Render measured practical comparisons from archived CSV files without running hydraulics."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity.figures import save


OBJECTIVES = {
    "peak_bias_pct": ("Peak discharge error", "%"),
    "peak_delay_pct": ("Peak timing error", "% of reference duration"),
    "shape_tv_pct": ("Hydrograph shape error", "%"),
    "active_nrmse_peak_pct": ("Active-window RMSE", "% of reference peak"),
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
                key = row["cohort"], row["case"], row["reference_id"], row["interval_s"]
                groups[key].append(row)
    for key, records in sorted(groups.items()):
        cohort, case, reference, interval = key
        panels = defaultdict(list)
        for row in records:
            panels[row["objective"]].append(row)
        for metric, rows in panels.items():
            permitted = [r for r in rows if r["eligible"] == "True"]
            counts.append(
                dict(
                    cohort=cohort,
                    case=case,
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
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), layout="constrained")
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
                    f"{r['method']} {r['dx']} m",
                    (x, y),
                    xytext=(2, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        fig.suptitle(case)
        save(
            fig=fig,
            folder=output,
            name=f"{case}_{cohort}_{reference}_{interval}_pareto",
        )
    metrics.csv_write(file=output / "reference_eligibility.csv", rows=counts)
    return counts


def run(*, directory):
    study = directory.parent
    source = directory / "native_results"
    output = directory / "native_publication"
    output.mkdir(exist_ok=True)
    rows = [
        r
        for r in metrics.csv_read(file=source / "measured_windows.csv")
        if r["horizon"].startswith("complete-")
    ]
    counts = measured_fronts(directory=directory, output=output)
    text = [
        "# Native terrain, complete events and GPU cost",
        "",
        "All positions share the same native metre grid origin. Each 10 m elevation averages its four 5 m children. "
        "Each row reports the cost of its recorded method, execution policy and common event endpoint. "
        "The normalized-error Pareto tables use independently screened references for each hydraulic observable.",
        "",
        "| Case | Grid (m) | Method | CFL factor | End (h) | Peak (m³/s) | Export (m³) | Hydraulic time (s) | Steps |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        text.append(
            f"| {r['case']} | {r['dx_m']} | {r['method']} | {r['factor']} | {float(r['end_h']):g} | "
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
            )
        },
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
