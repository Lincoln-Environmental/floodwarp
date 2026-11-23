"""Publication panels for observed interval means and matched, measured errors."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity.assessment import curve, references

LABELS = {
    "fv1": "FV1 contact-HLL",
    "roe": "FV1 Roe",
    "hllc": "FV1 HLLC",
    "hancock_adaptive": "Hancock",
    "hancock_dense": "Hancock",
    "adaptive": "FV2 RK2",
}
COLOURS = {
    "fv1": "#ba4b19",
    "roe": "#a33f7b",
    "hllc": "#a3830b",
    "hancock_adaptive": "#006f79",
    "hancock_dense": "#006f79",
    "adaptive": "#5368a4",
}
STYLES = {
    "fv1": "-",
    "roe": "--",
    "hllc": ":",
    "hancock_adaptive": "-",
    "hancock_dense": "-",
    "adaptive": "--",
}


def save(*, fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg", "png"):
        fig.savefig(folder / (name + "." + fmt), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw(*, axes, t, q, label, colour, style):
    dt = np.diff(t)
    volume = q @ dt
    axes[0].stairs(q, t, label=label, color=colour, linestyle=style)
    if volume > 0 and np.min(q) >= -0.01 * q.max():
        outward = np.maximum(q, 0)
        v = outward @ dt
        axes[1].stairs(outward / v, t, color=colour, linestyle=style)
        axes[2].plot(
            t, np.r_[0.0, np.cumsum(outward * dt)] / v, color=colour, linestyle=style
        )


def overlay(*, folder, series, name, xlabel, time_unit, title):
    fig, ax = plt.subplots(3, 1, figsize=(6.5, 6.2), sharex=True, layout="constrained")
    for label, t, q, c, s in series:
        draw(axes=ax, t=t, q=q, label=label, colour=c, style=s)
    ax[0].set_title(title)
    ax[0].set_ylabel("Discharge (m³/s)")
    ax[1].set_ylabel("Volume density (" + time_unit + "⁻¹)")
    ax[2].set_ylabel("Fraction passed")
    ax[2].set_xlabel(xlabel)
    ax[0].legend(ncol=3, fontsize=8)
    save(fig=fig, folder=folder, name=name)


def plots(*, directory, groups):
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    plt.rcParams["svg.hashsalt"] = "floodwarp-hydrograph-fidelity"
    folder = directory / "figures"
    target = directory / "results"
    cfg = json.loads((directory / "design.json").read_text())["pulse"]
    archive = directory / "long_pulses"
    source = archive / (archive / "current.txt").read_text().strip()
    q = []
    for method in cfg["methods"]:
        with np.load(source / f"smooth_10_0.13_{method}.npz") as f:
            if not q:
                q.append(("Exact", f["edges"], f["truth"][:, -1], "#242424", "-"))
            q.append(
                (
                    LABELS[method],
                    f["edges"],
                    f["flow"][:, -1],
                    COLOURS[method],
                    STYLES[method],
                )
            )
    overlay(
        folder=folder,
        series=q,
        name="channel_hydrographs",
        xlabel="Time (s)",
        time_unit="s",
        title="Smooth pulse, 10 m grid, 200 m downstream",
    )
    for case, cohort, rep, dx in (
        ("floodplain-double-rain", "complete-double", "10m bed", 10),
        ("mixed-708-spatial-10240x10240", "complete-expanded", "sampled", 10),
        ("rolling-704-soil-5120x5120", "complete-expanded", "sampled", 5),
        ("floodplain-18217-front", "complete-legacy", "sampled", 10),
    ):
        items = groups[cohort, case, rep]
        fine, _ = references(items=items)
        t, q = curve(it=fine, station="outlet")
        seq = [(f"FV2 {fine['s']['dx']:g} m reference", t / 3600, q, "#242424", "-")]
        seen = set()
        for i in items:
            method = i["s"]["method"]
            if (
                i["factor"] != 1
                or i["s"]["dx"] != dx
                or method not in LABELS
                or LABELS[method] in seen
            ):
                continue
            t, q = curve(it=i, station="outlet")
            seen.add(LABELS[method])
            seq.append((LABELS[method], t / 3600, q, COLOURS[method], STYLES[method]))
        overlay(
            folder=folder,
            series=seq,
            name=case,
            xlabel="Time since storm start (h)",
            time_unit="h",
            title=case.split("-")[0].capitalize() + f" outlet, {dx} m forecasts",
        )
    rows = metrics.csv_read(target / "pulses.csv")
    fig, ax = plt.subplots(2, 3, figsize=(9, 5.6), layout="constrained")
    for col, (metric, title) in enumerate(
        (
            ("peak_bias_pct", "Peak bias (%)"),
            ("centroid_delay_s", "Centroid delay (s)"),
            ("width80_bias_pct", "Central 80% width bias (%)"),
        )
    ):
        for idx, variable in enumerate(("dx", "distance_m")):
            for method in cfg["methods"]:
                pts = sorted(
                    (float(r[variable]), float(r[metric]))
                    for r in rows
                    if r["case"] == "smooth"
                    and r["method"] == method
                    and float(r["cap_courant"]) == 0.13
                    and r.get(metric)
                    and (
                        float(r["distance_m"]) == 200
                        if idx == 0
                        else float(r["dx"]) == 10
                    )
                )
                if pts:
                    x, y = zip(*pts, strict=True)
                    ax[idx, col].plot(
                        x,
                        y,
                        STYLES[method],
                        color=COLOURS[method],
                        marker=".",
                        label=LABELS[method],
                    )
            ax[idx, col].axhline(0, color=".7", lw=0.7)
            ax[idx, col].set_ylabel(title)
            ax[idx, col].set_xlabel(
                "Grid spacing (m)" if idx == 0 else "Propagation distance (m)"
            )
    ax[0, 0].legend(fontsize=7)
    save(fig=fig, folder=folder, name="resolution_distance")
    # Draw interval bounds for broad peaks; a dot requires a supported point estimate.
    fig, ax = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
    for index, method in enumerate(cfg["methods"]):
        matched = [
            r
            for r in rows
            if r["case"] == "smooth"
            and r["method"] == method
            and r["distance_m"] == "200"
            and float(r["cap_courant"]) == 0.13
            and r["peak_delay_lower_s"]
        ]
        for j, r in enumerate(matched):
            # Slight horizontal dodge only for legibility, at the labelled grid spacing.
            x = cfg["dx"].index(int(r["dx"])) + 0.075 * (index - 2)
            ax.vlines(
                x,
                float(r["peak_delay_lower_s"]),
                float(r["peak_delay_upper_s"]),
                color=COLOURS[method],
                lw=2,
                label=LABELS[method] if j == 0 else None,
            )
            if r["peak_delay_s"]:
                ax.scatter(x, float(r["peak_delay_s"]), s=12, color=COLOURS[method])
    ax.set_xticks(range(len(cfg["dx"])), cfg["dx"])
    ax.axhline(0, color=".7", lw=0.7)
    ax.set(
        xlabel="Grid spacing (m)",
        ylabel="Peak delay (s)",
        title="Peak interval uncertainty, 200 m downstream",
    )
    ax.legend(fontsize=7)
    save(fig=fig, folder=folder, name="peak_delay")
    pairs = [
        r
        for r in metrics.csv_read(target / "pairs.csv")
        if r["station"] == "outlet"
        and r["first"] == "fv1"
        and r["second"] == "hancock_fastest"
        and r["cohort"] == "complete-legacy"
    ]
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 3.2), layout="constrained")
    for k, colour in (("peak_bias_pct", "#ba4b19"), ("shape_tv_pct", "#006f79")):
        use = [r for r in pairs if r["metric"] == k]
        for a, key in zip(ax, ("gradient", "runtime_ratio"), strict=True):
            a.scatter(
                [float(r[key]) for r in use],
                [float(r["absolute_error_reduction"]) for r in use],
                s=14,
                color=colour,
                alpha=0.75,
                label="Peak" if k == "peak_bias_pct" else "Shape",
            )
            a.axhline(0, color=".7", lw=0.6)
    ax[0].set(
        xscale="log",
        xlabel="Terrain generator grade (m/m)",
        ylabel="FV1 error minus Hancock (points)",
    )
    ax[1].set_xlabel("Hancock / FV1 runtime")
    ax[0].legend()
    save(fig=fig, folder=folder, name="grade_and_cost")
    fronts = metrics.csv_read(target / "pareto.csv")
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2), layout="constrained")
    for axis, metric in zip(
        ax, ("peak_bias_pct", "t50_delay_pct", "shape_tv_pct"), strict=True
    ):
        use = [
            r
            for r in fronts
            if r["case"] == "smooth"
            and r["station"] == "40"
            and r["cap"] == "0.13"
            and r["objective"] == metric
            and r["eligible"] == "True"
        ]
        edge = sorted(
            (float(r["runtime_s"]), float(r["absolute_error"]))
            for r in use
            if r["nondominated"] == "True"
        )
        if edge:
            axis.plot(*zip(*edge, strict=True), color=".4", lw=1)
        for r in use:
            axis.scatter(
                float(r["runtime_s"]),
                float(r["absolute_error"]),
                s=18,
                color=COLOURS[r["method"]],
            )
        axis.set(
            xlabel="Hydraulic runtime (s)",
            ylabel={
                "peak_bias_pct": "Absolute peak bias (%)",
                "t50_delay_pct": "Absolute t50 delay (% duration)",
                "shape_tv_pct": "Redistributed volume (%)",
            }[metric],
        )
    save(fig=fig, folder=folder, name="pareto")
    with np.load(target / "arrival_maps.npz") as maps:
        cases = (
            "incised-18217-moderate",
            "floodplain-18217-intense",
            "rolling-18217-back",
        )
        methods = ("fv1", "hancock_dense", "adaptive")
        fig, ax = plt.subplots(
            3, 3, figsize=(7, 5.4), sharex=True, sharey=True, layout="constrained"
        )
        bound = max(
            1,
            max(
                np.nanmax(abs(maps[k]))
                for k in maps.files
                if np.isfinite(maps[k]).any()
            )
            / 60,
        )
        for i, case in enumerate(cases):
            for j, method in enumerate(methods):
                axis = ax[i, j]
                axis.set_facecolor(".85")
                im = axis.imshow(
                    maps[case + "_" + method] / 60,
                    origin="lower",
                    extent=(0, 120, 0, 80),
                    cmap="RdBu_r",
                    vmin=-bound,
                    vmax=bound,
                    interpolation="nearest",
                )
                if i == 0:
                    axis.set_title(LABELS[method])
                if j == 0:
                    axis.set_ylabel(case.split("-")[0] + "\ny (m)")
                if i == 2:
                    axis.set_xlabel("x (m)")
        fig.colorbar(im, ax=ax, label="First 1 cm arrival delay (min)")
        save(fig=fig, folder=folder, name="arrival_maps")

    from research.floodwarp_methods_paper.fidelity import recession_figures

    recession_figures.operational(
        groups=groups,
        directory=directory,
        cases=(
            ("mixed-708-spatial-10240x10240", 10),
            ("rolling-704-soil-5120x5120", 5),
        ),
    )
    recession_figures.spatial(directory=directory)
