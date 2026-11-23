"""Figures for measured operational recessions, observer cadence and spatial propagation."""

import numpy as np
import matplotlib.pyplot as plt
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import figures


def operational(*, groups, directory, cases):
    folder = directory / "figures"
    for case, dx in cases:
        cohort = groups["complete-expanded", case, "sampled"]
        rows = [
            i
            for i in cohort
            if i["factor"] == 1
            and i["s"]["dx"] == dx
            and i["s"]["method"] in ("fv1", "hancock_adaptive", "hancock_dense")
        ]
        if not rows:
            continue
        fig, axes = plt.subplots(
            4, 1, figsize=(6.8, 7.5), sharex=True, layout="constrained"
        )
        f = rows[0]["d"]["forcing"]
        end = rows[0]["s"]["hours"]
        t = np.asarray(f["rain_times_s"]) / 3600
        rates = np.asarray(f["rain_rates_ms"]) * 3.6e6
        rain_x = np.r_[t[t < end], end]
        axes[0].fill_between(rain_x, np.interp(rain_x, t, rates), color="#356b91")
        axes[0].set_ylabel("Rain (mm/h)")
        axes[0].set_title(
            f"{case.split('-')[0].capitalize()}, {dx:g} m, {end:g} simulated hours"
        )
        for row in rows:
            method = row["s"]["method"]
            d = row["d"]
            frames = d.get("frames", d.get("samples"))
            edges = np.r_[0.0, [f["t_s"] for f in frames]] / 3600
            colour = figures.COLOURS[method]
            style = "--" if method == "hancock_dense" else "-"
            label = f"{figures.LABELS[method]} ({'sparse' if method == 'fv1' else 'tiles' if method == 'hancock_adaptive' else 'dense'})"
            axes[1].stairs(
                [f["outlet_m3s"] for f in frames],
                edges,
                color=colour,
                linestyle=style,
                label=label,
            )
            spatial = d.get("samples", d.get("spatial", []))
            if spatial:
                axes[2].plot(
                    [f["t_s"] / 3600 for f in spatial],
                    [f["wet_fraction"] * 100 for f in spatial],
                    color=colour,
                    linestyle=style,
                )
            elapsed = np.r_[
                0.0,
                np.cumsum(
                    [f.get("hydraulic_s", f.get("hydraulic_wall_s")) for f in frames]
                ),
            ]
            axes[3].plot(edges, elapsed, color=colour, linestyle=style)
        axes[1].set_ylabel("Outlet Q (m³/s)")
        axes[1].legend(fontsize=8)
        axes[2].set_ylabel("Wet cells (%)")
        axes[3].set(
            ylabel="Hydraulic time (s)",
            xlabel="Time since start of rainfall (h)",
            xlim=(0, end),
        )
        figures.save(fig=fig, folder=folder, name=case + "_cost")


def spatial(*, directory):
    table = metrics.csv_read(directory / "results/transient_arrival.csv")
    folder = directory / "figures"
    with np.load(directory / "results/transient_arrival.npz") as maps:
        for case in sorted({r["case"] for r in table}):
            records = [
                r for r in table if r["case"] == case and r["experiment_id"] in maps
            ]
            if not records:
                continue
            spacing = max(float(r["dx"]) for r in records if r["method"] == "fv1")
            show = [
                r
                for r in records
                if float(r["dx"]) == spacing
                and r["method"] in ("fv1", "hancock_dense", "hancock_adaptive")
            ]
            if not show:
                continue
            finite = []
            for r in show:
                a = maps[r["experiment_id"]]
                if np.isfinite(a).any():
                    finite.append(abs(a[np.isfinite(a)]) / 60)
            limit = (
                max(15, float(np.quantile(np.concatenate(finite), 0.98)))
                if finite
                else 15
            )
            fig, axes = plt.subplots(
                1, len(show), squeeze=False, figsize=(6.8, 3.8), layout="constrained"
            )
            for axis, r in zip(axes[0], show, strict=True):
                arr = maps[r["experiment_id"]] / 60
                support = float(r["support_m"])
                ny, nx = arr.shape
                axis.set_facecolor(".85")
                image = axis.imshow(
                    arr,
                    origin="lower",
                    extent=(0, nx * support, 0, ny * support),
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="nearest",
                )
                axis.set(
                    title=figures.LABELS[r["method"]],
                    xlabel="Easting (m)",
                    ylabel="Northing (m)",
                )
            fig.colorbar(
                image, ax=axes[0], label="1 cm arrival delay (min)", extend="both"
            )
            figures.save(fig=fig, folder=folder, name=case + "_arrival")


def paired_examples(*, directory, groups):
    from research.floodwarp_methods_paper.fidelity.assessment import curve, references

    records = metrics.csv_read(directory / "results/storms.csv")
    eligible = [
        r
        for r in records
        if r["cohort"] == "complete-legacy"
        and r["method"] == "fv1"
        and r["station"] == "outlet"
        and r["qualified_peak_bias_pct"] == "True"
    ]
    if len(eligible) < 2:
        return
    errors = np.array([abs(float(r["peak_bias_pct"])) for r in eligible])
    order = np.argsort(errors, kind="stable")
    for label, fraction in (("lower_quartile", 0.25), ("upper_quartile", 0.75)):
        chosen = eligible[order[round((len(order) - 1) * fraction)]]
        case = chosen["case"]
        group = groups["complete-legacy", case, "sampled"]
        ref, _ = references(items=group)
        t, q = curve(it=ref, station="outlet")
        series = [("FV2 reference", t / 3600, q, "#242424", "-")]
        for i in group:
            method = i["s"]["method"]
            if (
                method in ("fv1", "hancock_dense", "adaptive")
                and i["s"]["dx"] == 10
                and i["factor"] == 1
            ):
                t, q = curve(it=i, station="outlet")
                series.append(
                    (
                        figures.LABELS[method],
                        t / 3600,
                        q,
                        figures.COLOURS[method],
                        figures.STYLES[method],
                    )
                )
        figures.overlay(
            folder=directory / "figures",
            series=series,
            name="peak_" + label,
            xlabel="Hours since rainfall began",
            time_unit="h",
            title=case
            + " • FV1 peak bias "
            + format(float(chosen["peak_bias_pct"]), "+.1f")
            + "%",
        )


def depth_and_wave(*, source, folder, tolerance_pct, depth_limit_m):
    depths = {
        r["id"]: r
        for r in metrics.csv_read(source / "recession_depth.csv")
        if r.get("qualified_depth_rmse_m") == "True"
    }
    evidence = metrics.csv_read(source / "outlet_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.45), layout="constrained", sharex=True)
    for axis, (metric, title) in zip(
        axes,
        (
            ("peak_bias_pct", "Signed peak bias (%)"),
            ("shape_tv_pct", "Redistributed volume (%)"),
        ),
        strict=True,
    ):
        axis.axvline(1000 * depth_limit_m, color="0.65", ls=":", zorder=0)
        axis.axhline(tolerance_pct, color="0.65", ls=":", zorder=0)
        if metric == "peak_bias_pct":
            axis.axhline(0, color="0.75", lw=0.7, zorder=0)
            axis.axhline(-tolerance_pct, color="0.65", ls=":", zorder=0)
        for method, marker in (("fv1", "o"), ("hancock_dense", "^"), ("adaptive", "s")):
            pts = [
                (
                    1000 * float(depths[r["experiment_id"]]["depth_rmse_m"]),
                    float(r["value_pct"]),
                )
                for r in evidence
                if r["experiment_id"] in depths
                and r["metric"] == metric
                and r["cohort"] == "complete-legacy"
                and r["method"] == method
                and r["qualified"] == "True"
            ]
            if pts:
                x, y = zip(*pts)
                axis.scatter(
                    x,
                    y,
                    s=23,
                    facecolors="none",
                    edgecolors=figures.COLOURS[method],
                    marker=marker,
                    label=figures.LABELS[method],
                    linewidths=0.9,
                )
        axis.set(xlabel="Final depth RMSE (mm)", ylabel=title)
        axis.set_xscale("symlog", linthresh=0.1, linscale=0.6)
    axes[1].legend(fontsize=8, frameon=False)
    figures.save(fig=fig, folder=folder, name="final_depth_and_outlet_wave")
