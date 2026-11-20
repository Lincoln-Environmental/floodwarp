"""Produce physical-error, hydrograph, convergence and fixed-budget comparisons."""

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from validation.fv1_riemann import report_json


def load_rows(*, directory):
    rows = []
    for p in sorted((directory / "matrix").glob("*.json")):
        if p.name != "configuration.json":
            rows.extend(json.loads(p.read_text()))
    return rows


def extent(row):
    return 1 - row["extent"]["0.01"]["iou"]


def physical(row, key):
    return extent(row) if key == "extent" else row[key]


def comparison(*, rows, output, targets, budgets):
    records = []
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        subset = [r for r in rows if r["case"] == case and r["policy"] == "production"]
        for dx in budgets:
            ref = next(r for r in subset if r["dx"] == dx and r["method"] == "fv1")
            budget = ref["median_wall_s"]
            for key in targets:
                for method in sorted({r["method"] for r in subset}):
                    within = [r for r in subset if r["method"] == method and r["median_wall_s"] <= budget]
                    if within:
                        best = min((physical(r, key), r["median_wall_s"], r["dx"]) for r in within)
                        records.append(
                            dict(
                                case=case,
                                reference_dx=dx,
                                budget_s=budget,
                                metric=key,
                                method=method,
                                error=best[0],
                                runtime_s=best[1],
                                dx=best[2],
                                fv1_error=physical(ref, key),
                            )
                        )
        for key, tol in targets.items():
            for method in sorted({r["method"] for r in subset}):
                ok = [r for r in subset if r["method"] == method and physical(r, key) <= tol]
                if ok:
                    cost, err, dx = min((r["median_wall_s"], physical(r, key), r["dx"]) for r in ok)
                    records.append(
                        dict(
                            case=case,
                            tolerance=tol,
                            metric=key,
                            method=method,
                            error=err,
                            runtime_s=cost,
                            dx=dx,
                        )
                    )
    report_json(path=output, content=records)


def cost_plot(*, directory, rows, methods, cases):
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3 * len(cases)), squeeze=False)
    for i, case in enumerate(cases):
        for j, (key, label) in enumerate(
            (
                ("depth_rmse", "Depth RMSE (m)"),
                ("extent", "1 − wet IoU (1 cm)"),
                ("hydro_rmse", "Discharge RMSE (m²/s)"),
            )
        ):
            ax = axes[i, j]
            for m in methods:
                subset = sorted(
                    (
                        r
                        for r in rows
                        if r["method"] == m and r["case"] == case and r["policy"] == "production"
                    ),
                    key=grid_order,
                )
                ax.plot(
                    [r["median_wall_s"] * 1e3 for r in subset],
                    [physical(r, key) for r in subset],
                    "o-",
                    markersize=3,
                    label=m,
                )
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_ylabel(label)
            ax.set_xlabel("Warmed hydraulic wall time (ms)")
            ax.set_title(case.replace("_", " "))
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "accuracy_cost.png", dpi=155)
    plt.close(fig)


def grid_order(r):
    return -r["dx"]


def figures(*, directory, rows, methods):
    history = json.loads((directory / "pulse_histories.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="row")
    for i, case in enumerate(("shear_pulse", "gravity_pulse")):
        for j, dx in enumerate((10.0, 5.0, 1.0)):
            ax = axes[i, j]
            sub = [r for r in history if r["case"] == case and r["dx"] == dx]
            for r in sub:
                t = np.array(r["interval_end_s"])
                t -= t[0] / 2
                ax.plot(t, np.array(r["model_m2_s"])[:, 1], label=r["method"], lw=1.5)
            ax.plot(t, np.array(sub[0]["exact_m2_s"])[:, 1], "k--", label="analytical", lw=2)
            ax.set_title(f"{case.replace('_', ' ')} · {dx:g} m")
            ax.set(xlabel="Time (s)", ylabel="Section discharge (m²/s)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "hydrographs.png", dpi=155)
    plt.close(fig)
    mms = json.loads((directory / "manufactured.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for key, v in mms.items():
        for i, study in enumerate(("space", "time")):
            sub = [r for r in v if r["study"] == study]
            for j, label in enumerate(("Depth RMSE (m)", "x momentum RMSE (m²/s)", "y momentum RMSE (m²/s)")):
                ax = axes[i, j]
                ax.loglog(
                    [8 / r["n"] if study == "space" else r["dt"] for r in sub],
                    [r["rmse"][j] for r in sub],
                    "o-",
                    label=key,
                )
                ax.set_xlabel("Grid spacing (m)" if study == "space" else "Timestep (s)")
                ax.set_ylabel(label)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(directory / "convergence.png", dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in methods:
        sub = [r for r in rows if r["dx"] == 5 and r["method"] == m and r["policy"] == "common"]
        for ax, x, label in zip(
            axes,
            ([r["mean_abs_grade"] for r in sub], [r["fr_wet_median"] for r in sub]),
            ("Bed slope (m/m)", "Median exact wet Froude number"),
            strict=True,
        ):
            ax.scatter(x, [r["hydro_rmse"] for r in sub], label=m, s=17)
            ax.set_xscale("symlog", linthresh=1e-5)
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set(xlabel=label, ylabel="Section discharge RMSE (m²/s)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "regimes.png", dpi=155)
    plt.close(fig)


def export(*, directory, rows):
    with (directory / "primary_matrix.csv").open("w", newline="") as f:
        writer = None
        for r in rows:
            row = {
                k: r[k]
                for k in (
                    "case",
                    "dx",
                    "policy",
                    "method",
                    "model_n",
                    "steps",
                    "dt_mean_s",
                    "dt_min_s",
                    "depth_rmse",
                    "wet_depth_rmse",
                    "depth_linf",
                    "maximum_depth_rmse",
                    "momentum_x_rmse",
                    "momentum_y_rmse",
                    "fr_wet_median",
                    "fr_wet_max",
                    "hydro_rmse",
                    "peak_error_rmse",
                    "timing_error_rmse",
                    "volume_error_rmse",
                    "residual_m3",
                    "min_depth_m",
                    "median_wall_s",
                    "cell_steps_s",
                )
            }
            row.update(r["extent"]["0.01"])
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row, lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    rows = load_rows(directory=directory)
    methods = ("fv1", "euler_mc", "hancock_mc", "low_froude_hll", "mc", "li")
    cost_cases = ("front_plain", "front_steep", "gravity_pulse")
    targets = dict(depth_rmse=0.005, extent=0.05, hydro_rmse=0.001)
    comparison(rows=rows, output=directory / "budgets.json", targets=targets, budgets=(1.0, 5.0, 10.0))
    export(directory=directory, rows=rows)
    cost_plot(directory=directory, rows=rows, methods=methods, cases=cost_cases)
    figures(directory=directory, rows=rows, methods=methods)
