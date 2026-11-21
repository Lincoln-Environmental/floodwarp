"""Draw the study figures without invoking a hydraulic model."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import rasterio

from research.floodwarp_methods_paper.metrics import csv_read, load
from research.floodwarp_methods_paper.tables.generate import LABELS


COLOURS = dict(
    zip(
        LABELS,
        ("#0072B2", "#56B4E9", "#CC79A7", "#D55E00", "#E69F00", "#009E73", "#595959"),
        strict=True,
    )
)
UNITS = {
    "depth": "Depth RMSE (m)",
    "maximum_depth": "Maximum-depth RMSE (m)",
    "inundation": "1 − inundation IoU",
    "discharge": "Outlet RMSE (m³/s)",
    "peak_discharge": "Peak error (m³/s)",
    "peak_timing": "Peak timing error (s)",
    "memory": "Live pool (MiB)",
}


def save(*, fig, folder, name):
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    folder.mkdir(exist_ok=True, parents=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(folder / (name + "." + extension), dpi=350, bbox_inches="tight")
    plt.close(fig)


def legend(*, fig):
    handles = [
        Line2D([], [], color=c, marker="o", linestyle="", label=LABELS[m], markersize=4)
        for m, c in COLOURS.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)


def workflow(*, out):
    fig, ax = plt.subplots(figsize=(7.1, 3.1), layout="constrained")
    ax.set(xlim=(0, 3), ylim=(0, 3))
    ax.axis("off")
    groups = [
        [
            "Independent exact solutions",
            "Six full-event rain forcings",
            "24 terrain realisations",
        ],
        ["Six formulations", "1, 5 and 10 m grids", "Bounded roughness fits"],
        [
            "Physical error and GPU cost",
            "Qualified Pareto sets",
            "Held-out solver decisions",
        ],
    ]
    for x, col in enumerate(groups):
        for y, entry in enumerate(col):
            ax.text(
                x + 0.5,
                2.55 - y,
                entry,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=.55",
                    facecolor=("#e4eff5", "#fff1da", "#e6f0e7")[x],
                    edgecolor="#707070",
                ),
            )
            if x < 2:
                ax.annotate(
                    "",
                    xy=(x + 1.09, 2.55 - y),
                    xytext=(x + 0.92, 2.55 - y),
                    arrowprops=dict(arrowstyle="->", color="#333"),
                )
    ax.text(
        1.5,
        -0.02,
        "FV1 contact / low-Froude   ·   MUSCL–Euler   ·   Hancock   ·   FV2 RK2   ·   Local inertia",
        ha="center",
        fontsize=8,
    )
    save(fig=fig, folder=out, name="01_design")


def exact_profiles(*, study, out):
    raw = load(study=study, stages=("analytical",))
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), layout="constrained")
    hydro, haxes = plt.subplots(1, 3, figsize=(8, 2.9), layout="constrained")
    for i, case in enumerate(("front_steep", "gravity_pulse", "shear_pulse")):
        ax = axes[i]
        hx = haxes[i]
        first = True
        for rec, row, p in raw:
            s = rec["spec"]
            if s["case"] != case or s["dx"] != 5:
                continue
            method = s["method"]
            with np.load(p / "fields.npz") as f:
                k = 2 if case == "shear_pulse" else 0
                t = f["t_s"]
                x = f["x_m"]
                gauge = f["exact_gauges"].shape[1] // 2
                if first:
                    ax.plot(
                        x,
                        f["exact_cell_averages"][-1, k],
                        color="black",
                        lw=1.6,
                        label="Exact",
                    )
                    hx.plot(
                        t[1:],
                        f["exact_gauges"][:, gauge],
                        color="black",
                        lw=1.6,
                        label="Exact",
                    )
                    first = False
                ax.plot(
                    x,
                    f["model_centreline"][-1, k],
                    color=COLOURS[method],
                    lw=0.9,
                    ls="--",
                )
                hx.plot(
                    t[1:],
                    f["model_gauges"][:, gauge],
                    color=COLOURS[method],
                    lw=0.9,
                    ls="--",
                )
        ax.set(
            xlabel="Distance (m)",
            ylabel="Transverse momentum (m²/s)"
            if case == "shear_pulse"
            else "Depth (m)",
            title=case.replace("_", " "),
        )
        hx.set(
            xlabel="Interval end (s)",
            ylabel="Face discharge (m²/s)",
            title=case.replace("_", " "),
        )
    for f, n in ((fig, "02_exact_profiles"), (hydro, "04_hydrograph_diffusion")):
        legend(fig=f)
        save(fig=f, folder=out, name=n)


def convergence(*, study, out):
    raw = csv_read(study / "processed/metrics/analytical.csv")
    temporal = csv_read(study / "processed/convergence/temporal.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
    for ax, case, key, y in zip(
        axes[:2],
        ("gravity_pulse", "shear_pulse"),
        ("depth_rmse_m", "momentum_rmse_m2s"),
        ("Depth RMSE (m)", "Momentum RMSE (m²/s)"),
        strict=True,
    ):
        orders = []
        for m, c in COLOURS.items():
            rows = sorted(
                (float(r["dx"]), float(r[key]))
                for r in raw
                if r["case"] == case
                and r["stage"] == "convergence"
                and r["method"] == m
                and float(r[key]) > 0
            )
            if rows:
                ax.loglog(*zip(*rows), ".-", color=c)
                if m in ("hancock_adaptive", "adaptive") and len(rows) > 1:
                    p = np.log(rows[1][1] / rows[0][1]) / np.log(
                        rows[1][0] / rows[0][0]
                    )
                    orders.append(f"{LABELS[m]} {p:.2f}")
        ax.text(
            0.03,
            0.98,
            "Finest-pair order\n" + "\n".join(orders),
            transform=ax.transAxes,
            va="top",
            fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )
        ax.set(xlabel="Spacing (m)", ylabel=y, title=case.replace("_", " "))
    for m, c in COLOURS.items():
        vals = sorted(
            (float(r["spacing"]), float(r["error"]))
            for r in temporal
            if r["method"] == m
            and r["case"] == "accelerating"
            and r["metric"] == "momentum"
        )
        if vals:
            axes[2].loglog(*zip(*vals), ".-", color=c)
    axes[2].set(
        xlabel="Timestep (s)",
        ylabel="Momentum error (m²/s)",
        title="Frictional acceleration",
    )
    legend(fig=fig)
    save(fig=fig, folder=out, name="03_convergence")


def terrains(*, study, out):
    surfaces = sorted((study / "terrain/surfaces").glob("*/elevation.tif"))
    fig, axes = plt.subplots(4, 6, figsize=(9, 6.2), layout="constrained")
    for i, p in enumerate(surfaces):
        land = p.parent.name
        paths = (
            p,
            study / "terrain/diagnostics" / land / "1m/slope.tif",
            study / "terrain/diagnostics" / land / "1m/contributing_area.tif",
        )
        for j, (raster, name) in enumerate(
            zip(paths, ("Elevation (m)", "Slope (m/m)", "Area (log₁₀ m²)"), strict=True)
        ):
            ax = axes[i // 2, (i % 2) * 3 + j]
            with rasterio.open(raster) as src:
                field = src.read(1, masked=True)
            img = ax.imshow(
                np.log10(field) if j == 2 else field,
                cmap=("terrain", "magma", "viridis")[j],
            )
            ax.set_title(land + "\n" + name, fontsize=7)
            ax.set_axis_off()
            fig.colorbar(img, ax=ax, orientation="horizontal", shrink=0.9, aspect=16)
    save(fig=fig, folder=out, name="05_terrain")


def resolution(*, study, out):
    rows = csv_read(study / "processed/comparisons/terrain_attribution.csv")
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8), layout="constrained")
    for col, land in enumerate(("plain", "fan", "incised")):
        for ax, observable, units in zip(
            axes[:, col],
            ("depth", "discharge"),
            ("Depth RMSE (m)", "Outlet RMSE (m³/s)"),
            strict=True,
        ):
            for field, style in (
                ("fixed_geometry_numerical_rmse", "-"),
                ("matched_initial_bed_effect_rmse", "--"),
            ):
                for method in ("fv1", "hancock_adaptive", "adaptive"):
                    pts = sorted(
                        (float(r["dx"]), float(r[field]))
                        for r in rows
                        if (r["family"], r["observable"], r["method"])
                        == (land, observable, method)
                    )
                    ax.plot(
                        *zip(*pts),
                        linestyle=style,
                        marker="o",
                        ms=3,
                        color=COLOURS[method],
                    )
            ax.set(
                xlabel="Hydraulic spacing (m)",
                ylabel=units,
                xticks=(1, 5, 10),
                title=land,
            )
    fig.suptitle(
        "Fixed-bed refinement (solid) and bed change at identical initial state (dashed)",
        fontsize=9,
    )
    legend(fig=fig)
    save(fig=fig, folder=out, name="06_resolution")


def frontier(*, ax, rows, obj, reference=None, repeats=None):
    chosen = [r for r in rows if r["objective"] == obj]
    pts = []
    for r in chosen:
        x, y = float(r["runtime_s"]), float(r["error"])
        if repeats and r["experiment_id"] in repeats:
            lo, hi = repeats[r["experiment_id"]]
            ax.plot([lo, hi], [y, y], lw=0.7, color=COLOURS[r["method"]], alpha=0.7)
        ax.scatter(x, y, s=14, c=COLOURS[r["method"]], alpha=0.8)
        if r["nondominated"] == "True":
            pts.append((x, y))
    if pts:
        ax.step(*zip(*sorted(pts)), where="post", color="#444", lw=0.85)
    if reference is not None and obj != "memory":
        ax.axhline(float(reference[obj]), color="#777", ls=":", lw=1)
    if not chosen:
        ax.text(0.5, 0.5, "Reference unresolved", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Hydraulic runtime (s)", ylabel=UNITS[obj])
    if chosen:
        ax.set_xscale("log")


def costs(*, study, out):
    data = csv_read(study / "processed/pareto/storms.csv")
    unc = csv_read(study / "processed/metrics/reference_uncertainty.csv")
    refs = {r["case"]: r for r in unc if r["representation"] == "sampled"}
    primary = [r for r in data if r["representation"] == "sampled"]
    repeat_rows = csv_read(study / "processed/metrics/operational_repeats.csv")
    repeat_case = repeat_rows[0]["physical_case"]
    by_method = {
        r["method"]: (float(r["minimum_s"]), float(r["maximum_s"])) for r in repeat_rows
    }
    timings = {
        r["experiment_id"]: by_method[r["method"]]
        for r in primary
        if r["case"] == repeat_case and float(r["dx"]) == 10
    }
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8), layout="constrained")
    for axs, case in zip(axes, ("plain-18217-moderate", repeat_case), strict=True):
        for ax, obj in zip(axs, ("depth", "discharge", "inundation"), strict=True):
            frontier(
                ax=ax,
                rows=[r for r in primary if r["case"] == case],
                obj=obj,
                reference=refs[case],
                repeats=timings,
            )
            ax.set_title(case, fontsize=8)
    fig.suptitle(
        "Dotted lines show refinement differences. Horizontal bars show repeated GPU timings.",
        fontsize=9,
    )
    legend(fig=fig)
    save(fig=fig, folder=out, name="07_pareto")
    # Each panel is an individual physical experiment. Curves never join different storms.
    for family in sorted({r["landform"] for r in primary}):
        cases = sorted(c for c in refs if c.startswith(family + "-"))
        fig, axes = plt.subplots(
            len(cases),
            len(UNITS),
            figsize=(15, 2 * len(cases)),
            layout="constrained",
            squeeze=False,
        )
        for axs, c in zip(axes, cases, strict=True):
            for ax, obj in zip(axs, UNITS, strict=True):
                frontier(
                    ax=ax,
                    rows=[r for r in primary if r["case"] == c],
                    obj=obj,
                    reference=refs[c],
                )
            axs[0].set_title(c, fontsize=8)
        legend(fig=fig)
        save(fig=fig, folder=study / "supplementary/pareto", name=family)
    exact = csv_read(study / "processed/pareto/analytical.csv")
    repeats = {
        r["experiment_id"]: (float(r["timing_min_s"]), float(r["timing_max_s"]))
        for r in csv_read(study / "processed/metrics/analytical.csv")
    }
    for i, c in enumerate(sorted({r["case"] for r in exact})):
        fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), layout="constrained")
        for ax, obj in zip(axes.flat, UNITS):
            frontier(
                ax=ax,
                rows=[r for r in exact if r["case"] == c],
                obj=obj,
                repeats=repeats,
            )
            if obj in ("discharge", "peak_discharge"):
                ax.set_ylabel("Unit discharge error (m²/s)")
        axes.flat[-1].axis("off")
        fig.suptitle(c, fontsize=10)
        legend(fig=fig)
        save(fig=fig, folder=study / "supplementary/pareto/exact", name=c)


def calibration(*, study, out):
    archive = csv_read(study / "processed/comparisons/archival_analytical_transfer.csv")
    transfer = csv_read(study / "processed/comparisons/transfer.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), layout="constrained")
    for ax, obj in zip(axes, sorted({r["objective"] for r in archive}), strict=True):
        for bound in sorted({r["bound"] for r in archive}):
            values = sorted(
                {
                    (float(r["training_dx"]), 100 * (float(r["multiplier"]) - 1))
                    for r in archive
                    if r["bound"] == bound and r["objective"] == obj
                }
            )
            if values:
                ax.scatter(*zip(*values), s=12, alpha=0.6, label=bound)
        ax.axhline(0, color="black", lw=0.6)
        ax.set(
            title=obj, xlabel="Calibration spacing (m)", ylabel="Roughness change (%)"
        )
    axes[-1].legend(title="Bound", fontsize=6)
    save(fig=fig, folder=out, name="08_roughness")
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
    for ax, obj in zip(axes, sorted({r["objective"] for r in transfer}), strict=True):
        for i, land in enumerate(sorted({r["landform"] for r in transfer})):
            for status, marker, colour in (
                ("True", "o", "#0072B2"),
                ("False", "x", "#777"),
            ):
                vals = [
                    float(r["improvement"])
                    for r in transfer
                    if r["landform"] == land
                    and r["objective"] == obj
                    and r["qualified_for_objective"] == status
                ]
                if vals:
                    ax.scatter(
                        np.full(len(vals), i),
                        vals,
                        s=12,
                        alpha=0.5,
                        color=colour,
                        marker=marker,
                    )
            ax.set_xticks(
                range(len({r["landform"] for r in transfer})),
                sorted({r["landform"] for r in transfer}),
                rotation=35,
            )
        ax.axhline(0, color="black", lw=0.7)
        ax.set(title=obj, ylabel="Held-out score improvement")
    fig.suptitle(
        "Circles show qualified reference comparisons. Crosses show unresolved references.",
        fontsize=9,
    )
    save(fig=fig, folder=out, name="09_transfer")


def storm(*, study, out):
    raw = [
        (rec, row)
        for rec, row, p in load(study=study, stages=("long-storms",))
        if rec["spec"]["hours"] == 24 and rec["spec"]["reference"] is False
    ]
    fig, axes = plt.subplots(
        4, 1, figsize=(6.8, 6.1), sharex=True, layout="constrained"
    )
    for i, (rec, row) in enumerate(raw):
        f = row["forcing"]
        m = rec["spec"]["method"]
        s = row["samples"]
        t = np.array([a["t_s"] for a in s]) / 3600
        if not i:
            axes[0].plot(
                np.array(f["rain_times_s"]) / 3600,
                np.array(f["rain_rates_ms"]) * 3.6e6,
                color="#0072B2",
            )
        for ax, key in zip(
            axes[1:], ("outlet_m3s", "wet_fraction", "hydraulic_wall_s"), strict=True
        ):
            ax.plot(t, [a[key] for a in s], color=COLOURS[m], lw=1, alpha=0.85)
    for ax, label in zip(
        axes,
        (
            "Rain (mm/h)",
            "Outlet (m³/s)",
            "Wet fraction",
            "GPU loop wall time\n(s / 15 min interval)",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.axvline(6, color="#555", ls=":")
    axes[-1].set_xlabel("Elapsed physical time (h)")
    legend(fig=fig)
    save(fig=fig, folder=out, name="10_long_storm")


def advisor(*, study, out):
    rows = csv_read(study / "processed/advisor/heldout.csv")
    terrains = json.loads((study / "processed/metrics/terrain.json").read_text())
    selected = [r for r in rows if r["rule"] == "advisor"]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.5), layout="constrained")
    for ax, obj in zip(
        axes.flat, sorted({r["objective"] for r in selected}), strict=True
    ):
        for r in selected:
            if r["objective"] != obj:
                continue
            terrain = terrains[f"{r['terrain_id']}-{float(r['dx']):g}m"]
            ax.scatter(
                terrain["model"]["mean_slope"],
                float(r["rain_mm"]),
                c=COLOURS.get(r["chosen"], "#aaa"),
                marker="o" if r["chosen"] else "x",
                s=18,
            )
        ax.set(title=obj, xlabel="Mean slope (m/m)", ylabel="Area-mean storm rain (mm)")
    legend(fig=fig)
    save(fig=fig, folder=out, name="11_selection_regimes")
    fig, grid = plt.subplots(2, 2, figsize=(7.3, 6.5), layout="constrained")
    axes = grid.ravel()
    choices = [m for m in LABELS if m != "hancock_dense"] + ["abstain"]
    matrix = np.zeros((len(choices), len(choices)))
    for r in selected:
        matrix[
            choices.index(r["optimal"] or "abstain"),
            choices.index(r["chosen"] or "abstain"),
        ] += 1
    im = axes[0].imshow(matrix, cmap="Blues")
    labels = [LABELS.get(m, m) for m in choices]
    axes[0].set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=[*labels[:-1], "No qualified feasible option"],
        xlabel="Predicted",
        ylabel="Measured least cost",
    )
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)
    axes[0].tick_params(axis="y", labelsize=6)
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    for i, rule in enumerate(sorted({r["rule"] for r in rows})):
        v = [
            100 * float(r["regret"]) for r in rows if r["rule"] == rule and r["regret"]
        ]
        if v:
            axes[1].scatter(np.full(len(v), i), v, s=9, alpha=0.5)
    axes[1].set(
        xticks=range(4),
        xticklabels=sorted({r["rule"] for r in rows}),
        ylabel="Feasible runtime regret (%)",
    )
    axes[1].tick_params(axis="x", rotation=65, labelsize=7)
    with rasterio.open(
        study / "terrain/diagnostics/mixed/10m/outlet_displacement.tif"
    ) as src:
        im = axes[2].imshow(src.read(1, masked=True), cmap="magma")
    axes[2].set_title("Mixed terrain, 10 m", fontsize=8)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="Outlet displacement (m)")
    q = [
        r
        for r in selected
        if r["objective"] == "discharge"
        and r["predicted_error"]
        and r["observed_error"]
    ]
    ax = axes[3]
    for r in q:
        x = float(r["observed_error"])
        y = float(r["predicted_error"])
        ax.plot(
            [x, x],
            [float(r["interval_low"]), float(r["interval_high"])],
            color=COLOURS[r["chosen"]],
            lw=1,
        )
        ax.scatter(x, y, s=12, color=COLOURS[r["chosen"]])
    if q:
        top = max(max(float(r["observed_error"]), float(r["interval_high"])) for r in q)
        ax.plot([0, top], [0, top], color="#555", ls="--", lw=0.8)
    else:
        ax.text(
            0.5, 0.5, "No assessable predictions", transform=ax.transAxes, ha="center"
        )
    ax.set(
        xlabel="Measured outlet RMSE (m³/s)",
        ylabel="Predicted outlet RMSE (m³/s)",
        title="Empirical prediction intervals",
    )
    save(fig=fig, folder=out, name="12_advisor_validation")


def generate(*, study):
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "floodwarp-methods-paper",
        }
    )
    out = study / "figures"
    workflow(out=out)
    exact_profiles(study=study, out=out)
    for render in (
        convergence,
        terrains,
        resolution,
        costs,
        calibration,
        storm,
        advisor,
    ):
        render(study=study, out=out)
    return dict(
        main_figures=len(list(out.glob("*.pdf"))),
        supplementary_figures=len(
            list((study / "supplementary/pareto").rglob("*.pdf"))
        ),
    )


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    generate(study=study)
