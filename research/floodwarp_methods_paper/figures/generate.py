"""Draw the study figures without invoking a hydraulic model."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
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
GRID_MARKERS = {1: "^", 5: "s", 10: "o"}
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


def legend(*, fig, grids=False):
    handles = [
        Line2D([], [], color=c, marker="o", linestyle="", label=LABELS[m], markersize=4)
        for m, c in COLOURS.items()
    ]
    if grids:
        for spacing, symbol in GRID_MARKERS.items():
            handles.append(
                Line2D(
                    [],
                    [],
                    marker=symbol,
                    linestyle="",
                    color="#555",
                    markerfacecolor="none",
                    label=f"{spacing:g} m",
                    markersize=4,
                )
            )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5 if grids else 4,
        frameon=False,
    )


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
        ax.scatter(
            x,
            y,
            s=14,
            c=COLOURS[r["method"]],
            alpha=0.8,
            marker=GRID_MARKERS.get(float(r["dx"]), "D"),
        )
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
        low, high = ax.get_xlim()
        subs = (1, 2, 5) if high / low < 60 else (1,)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())


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
    legend(fig=fig, grids=True)
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
        legend(fig=fig, grids=True)
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
        legend(fig=fig, grids=True)
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
            slope = float(r["mean_slope"])
            ax.scatter(
                slope,
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
    choices = [*LABELS, "hancock_auto", "abstain"]
    matrix = np.zeros((len(choices), len(choices)))
    for r in selected:
        matrix[
            choices.index(r["optimal"] or "abstain"),
            choices.index(r["chosen"] or "abstain"),
        ] += 1
    im = axes[0].imshow(matrix, cmap="Blues")
    labels = [LABELS.get(m, "Hancock auto" if m=="hancock_auto" else m) for m in choices]
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
        xticks=range(len({r["rule"] for r in rows})),
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
            color=COLOURS.get(r["chosen"], "#704d9f"),
            lw=1,
        )
        ax.scatter(x, y, s=12, color=COLOURS.get(r["chosen"], "#704d9f"))
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


def paper_style():
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


def generate(*, study):
    from research.floodwarp_methods_paper.evidence import archived_publication

    with archived_publication(study=study):
        return from_records(study=study)


def outlet_repair(*, study, out):
    folder = study / "fidelity"
    measured = folder / "outlet_results/events.csv"
    if not measured.exists():
        return
    before = {(r["method"], r["dx_m"]): r for r in csv_read(file=folder / "mean_storms/complete_events.csv")}
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 4.9), layout="constrained")
    for row in csv_read(file=measured):
        name, dx = row["method"], row["dx_m"]
        if name not in ("hancock_adaptive", "adaptive") or float(row["cfl_factor"]) != 1:
            continue
        ax = axes[int(dx == "5"), int(name == "adaptive")]
        run = folder / "outlet_storms" / row["run_id"]
        frames = []
        for end in range(4*3600, int(float(row["end_h"])*3600)+1, 4*3600):
            frames.extend(json.loads((run / f"{end}.json").read_text())["frames"])
        original = row["extension_m"] == "0"
        if original:
            path = folder / "extended_runs" / before[name,dx]["series_id"] / "series.json"
            old = json.loads(path.read_text())["frames"]
            ax.plot([f["t_s"]/3600 for f in old], [f["outlet_m3s"] for f in old], color="0.5",
                    lw=1.0, label="Pre-repair outlet")
        ax.plot([f["t_s"]/3600 for f in frames], [f["outlet_m3s"] for f in frames],
                lw=1.15, ls="-" if original else "--", color="#126482" if original else "#ab591d",
                label="Repaired outlet" if original else "400 m extension")
    for i, dx in enumerate((10,5)):
        for j, method in enumerate(("Hancock", "RK2")):
            ax = axes[i,j]
            ax.set_title(f"{method}, {dx} m")
            ax.axvline(6, color="0.6", lw=.8, ls=":")
            if ax.lines and not ax.get_legend_handles_labels()[0]:
                ax.text(.5,.5,"Trajectory pending",ha="center",transform=ax.transAxes)
    for ax in axes[:,0]:
        ax.set_ylabel("Discharge (m³/s)")
    for ax in axes[-1,:]:
        ax.set_xlabel("Time (hours)")
    axes[0,0].legend(fontsize=7)
    save(fig=fig,folder=out,name="19_free_outlet")


def orton_transfer(*, study, out):
    folder = study / "fidelity/orton_transfer"
    table = folder / "transfer.csv"
    if not table.is_file():
        return
    rows = csv_read(file=table)
    grids = sorted({int(r["dx_m"]) for r in rows}, reverse=True)
    settings = json.loads((folder.parent / "orton_transfer_settings.json").read_text())
    gauges = settings["gauges_m"]
    fig, axes = plt.subplots(1 + len(gauges), len(grids), sharex=True, sharey="row", squeeze=False,
                             figsize=(2.8 * len(grids), 2.0 * (len(gauges) + 1)), layout="constrained")
    colours = {"fv1": "#666666", "hancock": "#126482", "rk2": "#ab591d"}
    for row in rows:
        run = folder / row["id"]
        metadata = json.loads((run / "inputs.json").read_text())
        if gauges != metadata["gauges_above_southern_origin_m"]:
            raise ValueError("Different physical Orton gauge locations")
        t_end = int(row["duration_h"]) * 3600
        cadence = int(row["interval_s"])
        frames = [r for h in range(settings["checkpoint_h"], int(row["duration_h"]) + 1, settings["checkpoint_h"])
                  for r in json.loads((run / f"hour_{h}.json").read_text())["frames"]]
        edges = np.arange(0, t_end + 1, cadence)
        np.testing.assert_array_equal([r["t_s"] for r in frames], edges[1:])
        series = [[r["net_outlet_m3s"] for r in frames]]
        series.extend([r["gauge_m3s"][i] for r in frames] for i in range(len(gauges)))
        if not np.isclose(max(series[0]), float(row["peak_m3s"]), rtol=1e-13, atol=1e-13):
            raise ValueError("Archived Orton peak and accepted interval transport disagree")
        column = grids.index(int(row["dx_m"]))
        for ax, q in zip(axes[:, column], series, strict=True):
            ax.stairs(q, edges / 3600, baseline=None, color=colours[row["method"]],
                      label=row["method"].upper(), linewidth=0.95)
    for i, grid in enumerate(grids):
        axes[0, i].set_title(f"{grid} m")
        axes[-1, i].set_xlabel("Time (h)")
    axes[0, 0].set_ylabel("Net outlet (m³/s)")
    for ax, location in zip(axes[1:, 0], gauges, strict=True):
        ax.set_ylabel(f"y = {location:,} m\nDischarge (m³/s)")
    handles = {}
    for ax in axes[0]:
        artists, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, artists, strict=True))
    fig.legend(handles.values(), handles.keys(), loc="outside upper center", ncol=len(colours))
    save(fig=fig, folder=out, name="20_orton_transfer")


def incised_spatial_contrasts(*, study):
    from research.floodwarp_methods_paper import runner

    folder = study / "fidelity/outlet_results"
    manifest = folder / "hour_eight_contrasts.json"
    if not manifest.exists():
        return
    info = json.loads(manifest.read_text())
    path = manifest.with_suffix(".npz")
    if runner.checksum(path) != info["fields_sha256"]:
        raise ValueError("Changed spatial contrast fields")
    ny, nx = info["shape"]
    dx = info["dx_m"]
    extent = [0, nx * dx / 1000, 0, ny * dx / 1000]
    with np.load(path) as fields:
        rain = fields["rain_fraction"]
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharex=True, sharey=True, layout="constrained")
        for ax, cm in zip(axes[:2], (5, 10), strict=True):
            fraction = fields[f"fv1_only_{cm}cm"]
            image = ax.imshow(100 * fraction, origin="lower", extent=extent, vmin=0, vmax=100,
                              cmap="Blues", interpolation="nearest")
            ax.set_title(f"Above {cm} cm in FV1 alone")
        fig.colorbar(image, ax=axes[:2], location="bottom", label=f"Area of {info['bin_m']} m reporting bin (%)")
        delta = fields["infiltration_difference_m"] * 1000
        limit = max(float(np.abs(delta).max()), np.finfo(float).eps)
        image = axes[2].imshow(delta, origin="lower", extent=extent, vmin=-limit, vmax=limit,
                               cmap="BrBG", interpolation="nearest")
        axes[2].set_title("FV1 minus Hancock infiltration")
        fig.colorbar(image, ax=axes[2], location="bottom", label="Cumulative difference (mm)")
        for ax in axes:
            if rain.min() < .5 < rain.max():
                ax.contour(rain, levels=[.5], extent=extent, origin="lower", colors="black", linewidths=.6)
            ax.axvline(info["common_section_x_m"] / 1000, ls="--", lw=.8, color="#444")
            ax.set_xlabel("Distance east (km)")
        axes[0].set_ylabel("Distance north (km)")
        fig.suptitle(f"Incised catchment at {info['hour']:g} h on the {dx} m grid\n"
                     "Solid line shows rainfall footprint, dashed line shows internal gauge")
        save(fig=fig, folder=study / "supplementary/incised", name="hour_eight_water")


def incised_extent_history(*, study):
    table = study / "fidelity/outlet_results/extent.csv"
    if not table.is_file():
        return
    rows = csv_read(file=table)
    direct = [row for row in rows if int(row["extension_m"]) == 0 and float(row["cfl_factor"]) == 1]
    if not any(float(r["hour"]) == 0 for r in direct):
        return
    grids = sorted({int(r["dx_m"]) for r in direct}, reverse=True)
    fig, axes = plt.subplots(2, len(grids), squeeze=False, sharex=True, sharey="row",
                             figsize=(3.5 * len(grids), 4.9), layout="constrained")
    methods = {"fv1": ("FV1", "#666666", "-"), "hancock_adaptive": ("Hancock", "#126482", "-"),
               "adaptive": ("RK2", "#b95c22", "--")}
    for col, dx in enumerate(grids):
        for method, (label, colour, style) in methods.items():
            points = {float(r["hour"]): r for r in direct if int(r["dx_m"]) == dx and r["method"] == method}
            if not points:
                continue
            times = sorted(points)
            for ax, cm in zip(axes[:, col], (5, 10), strict=True):
                area = [float(points[t][f"area_above_{cm}cm_m2"]) / 1e6 for t in times]
                ax.plot(times, area, color=colour, linestyle=style, linewidth=1.1, label=label)
        axes[0, col].set_title(f"{dx} m hydraulic grid")
        axes[1, col].set_xlabel("Time since storm start (h)")
    for ax, cm in zip(axes[:, 0], (5, 10), strict=True):
        ax.set_ylabel(f"Area above {cm} cm (km²)")
    for ax in axes.flat:
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_yticks([0, .01, .1, 1, 4], labels=["0", "0.01", "0.1", "1", "4"])
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    save(fig=fig, folder=study / "supplementary/incised", name="simultaneous_extent")


def plain_extent_field(*, directory, row, source, threshold_cm):
    from research.floodwarp_methods_paper import runner

    base = directory / "native_matrix" / row["case"] / row["source_id"]
    if (source["case"] != row["case"] or float(source["end_s"]) != float(row["end_h"]) * 3600
            or runner.checksum(path=base / "spatial.npz") != source["spatial_sha256"]
            or runner.checksum(path=base / "result.json") != source["result_sha256"]):
        raise ValueError("Plain map differs from its completed forecast")
    raw = json.loads((base / "result.json").read_text())
    spec = raw["inputs"]["spec"]
    width, height = spec["dimensions"]
    support = spec["support"]
    gauge = float(row["section_x_m"])
    grid = (width, height, support, gauge, raw["data"]["forcing"]["terrain_sha256"], float(row["interval_s"]))
    if spec["dx"] != float(row["dx_m"]) or spec["interval"] != float(row["interval_s"]):
        raise ValueError("Plain map uses a different hydraulic spacing or reporting interval")
    with np.load(base / "spatial.npz", allow_pickle=False) as fields:
        wet = fields[f"wet_fraction_{threshold_cm}cm"]
    area = float(row[f"model_grid_wet_area_{threshold_cm}cm_m2"])
    if (wet.shape != (height // support, width // support) or np.any((wet < 0) | (wet > 1))
            or not np.isclose(wet.sum() * support**2, area, rtol=0, atol=1e-6)):
        raise ValueError("Mapped fractions disagree with the native hydraulic wet area")
    return wet, area, grid, raw["inputs"]["physical_sha"]


def plain_extent_panels(*, study, cases, methods, spacings, hours, threshold_cm):
    directory = study / "fidelity"
    product = directory / "native_results"
    if not (product / "measured_windows.csv").exists():
        return []
    receipts = {r["id"]: r for r in csv_read(file=product / "verified_archives.csv")
                if r["evidence"] == "completed initial window"}
    windows = {}
    for r in csv_read(file=product / "measured_windows.csv"):
        if (r["case"] in cases and r["method"] in methods and r["horizon"] == "original_window"
                and float(r["end_h"]) == hours and float(r["factor"]) == 1 and int(r["timing_repeat"] or 0) == 0):
            key = (r["case"], r["method"], float(r["dx_m"]))
            if key in windows:
                raise ValueError("Multiple primary runs for one plain map")
            windows[key] = r
    files = []
    for dx in spacings:
        if any((case, method, dx) not in windows for case in cases for method in methods):
            continue
        fig, axes = plt.subplots(len(cases), len(methods), sharex=True, sharey=True, squeeze=False,
                                 figsize=(7.1, 5.4), layout="constrained")
        matched = set()
        for row_index, (case, storm) in enumerate(cases.items()):
            forcings = set()
            for col, (method, label) in enumerate(methods.items()):
                r = windows[case, method, dx]
                wet, area, grid, physical = plain_extent_field(
                    directory=directory, row=r, source=receipts[r["source_id"]], threshold_cm=threshold_cm,
                )
                width, height, support, gauge = grid[:4]
                matched.add(grid)
                forcings.add(physical)
                ax = axes[row_index, col]
                im = ax.imshow(wet * 100, origin="lower", extent=(0, width, 0, height), interpolation="nearest",
                               cmap="Blues", vmin=0, vmax=100)
                ax.axvline(gauge, ls="--", color="#6e4a39", linewidth=.7)
                ax.set_title(f"{label}   {area / 1e6:.3f} km²")
                if col == 0:
                    rain_mm = float(r["rain_m3"]) / (width * height) * 1000
                    ax.set_ylabel(f"{storm} ({rain_mm:g} mm)\nNorthing (m)")
                if row_index == len(cases) - 1:
                    ax.set_xlabel("Easting (m)")
            if len(forcings) != 1:
                raise ValueError("Physical inputs differ between solvers in a plain map")
        if len(matched) != 1:
            raise ValueError("Plain map panels have different terrain, support, gauge or cadence")
        fig.suptitle(f"Extent ≥{threshold_cm} cm at hour {hours:g} on the {dx:g} m hydraulic grid")
        fig.colorbar(im, ax=axes, location="bottom", shrink=.66, pad=.04,
                     label=f"Area of each {support:g} m reporting cell at or above the depth threshold (%)")
        name = f"plain_709_extent_{threshold_cm}cm_{dx:g}m"
        save(fig=fig, folder=study / "supplementary/operational_storms", name=name)
        files.append(name)
    return files


def embankment_profiles(*, study, cases, spacings, face_m, hours):
    from research.floodwarp_methods_paper import runner

    directory = study / "fidelity"
    table = directory / "native_results/embankment_transport.csv"
    if not table.is_file():
        return
    records = {}
    for row in csv_read(file=table):
        dx = int(row["dx_m"])
        if (row["case"] in cases and dx in spacings and row["method"] == "hancock_dense"
                and row["horizon"] == "original_window" and float(row["factor"]) == 1
                and float(row["face_x_m"]) == face_m and float(row["end_h"]) == hours):
            records[row["case"], dx] = row
    if any((case, dx) not in records for case in cases for dx in spacings):
        return
    fig, axes = plt.subplots(1, 1 + len(cases), sharex=True, figsize=(8, 3.3), layout="constrained")
    bounds = set()
    for dx, (colour, style) in spacings.items():
        previous = None
        for col, (case, label) in enumerate(cases.items(), start=1):
            row = records[case, dx]
            ident = row["receipt_id"]
            raw = json.loads((directory / "embankment_transport" / ident / "profile.json").read_text())
            if runner.digest(raw["identity"])[:24] != ident or runner.digest(raw["profile"]) != raw["profile_sha"]:
                raise ValueError("Changed archived crest or transport profile")
            profile = raw["profile"]
            z = np.asarray(profile["crest_m"])
            bounds.add((float(row["band_south_m"]), float(row["band_north_m"]), z.size * dx))
            edges = np.arange(z.size + 1) * dx
            if previous is None:
                outside = (edges[1:] <= float(row["band_south_m"])) | (edges[:-1] >= float(row["band_north_m"]))
                axes[0].stairs(np.where(outside, z, np.nan), edges, baseline=None, color=colour,
                               linestyle=style, linewidth=1, label=f"{dx} m")
                previous = z
            elif not np.array_equal(z, previous):
                raise ValueError("Storm profiles refer to different represented embankments")
            ax = axes[col]
            cumulative = np.r_[0., np.cumsum(profile["accepted_eastward_m3"])] / 1000
            ax.plot(edges, cumulative, color=colour, linestyle=style, linewidth=1)
            ax.set_title(label)
            ax.set_ylabel("Net volume summed from south (10³ m³)")
    if len(bounds) != 1:
        raise ValueError("Embankment profiles use different transverse extents")
    south, north, width = bounds.pop()
    for ax in axes:
        ax.axvspan(south, north, color="#99763f", alpha=.14, linewidth=0)
        ax.set(xlim=(0, width), xlabel="Northing (m)")
        ax.grid(axis="y", alpha=.18)
    axes[0].set(title="Crest outside opening band", ylabel="Maximum bed elevation in control strip (m)")
    axes[0].legend(title="Hydraulic grid", loc="lower right")
    fig.suptitle(f"Plain embankment and Hancock transport at x = {face_m:g} m through hour {hours:g}")
    save(fig=fig, folder=study / "supplementary/operational_storms", name="plain_709_embankment_profiles")


def from_records(*, study):
    paper_style()
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
    from research.floodwarp_methods_paper.qualification import plots
    plots.generate(study=study)
    outlet_repair(study=study,out=out)
    incised_spatial_contrasts(study=study)
    incised_extent_history(study=study)
    plain_extent_panels(study=study, cases={"plain-709-moderate-1000x1000": "Moderate",
                                           "plain-709-intense-1000x1000": "Intense"},
                        methods={"fv1": "FV1", "hancock_dense": "Hancock", "adaptive": "RK2"},
                        spacings=(1, 5, 10), hours=8, threshold_cm=10)
    embankment_profiles(study=study, cases={"plain-709-moderate-1000x1000": "Moderate rain",
                                            "plain-709-intense-1000x1000": "Intense rain"},
                        spacings={1: ("#0072B2", "-"), 5: ("#D55E00", "--"), 10: ("#333333", ":")},
                        face_m=880, hours=8)
    from research.floodwarp_methods_paper.figures import native_extent
    native_extent.generate(study=study, thresholds=(5, 10), save=save,
                           methods={"fv1": ("FV1", "#0072B2", "o"), "hancock_dense": ("Hancock", "#D55E00", "s"),
                                    "adaptive": ("RK2", "#009E73", "^")})
    orton_transfer(study=study, out=out)
    return dict(
        main_figures=len(list(out.glob("*.pdf"))),
        supplementary_figures=len(
            list((study / "supplementary").rglob("*.pdf"))
        ),
    )


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    generate(study=study)
