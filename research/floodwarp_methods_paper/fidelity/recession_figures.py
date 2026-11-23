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
            label = f"{figures.LABELS[method]} ({'sparse' if method == 'fv1' else 'tiles' if method == 'hancock_adaptive' else 'dense'})"
            axes[1].stairs(
                [f["outlet_m3s"] for f in frames], edges, color=colour, label=label
            )
            spatial = d.get("samples", d.get("spatial", []))
            if spatial:
                axes[2].plot(
                    [f["t_s"] / 3600 for f in spatial],
                    [f["wet_fraction"] * 100 for f in spatial],
                    color=colour,
                )
            elapsed = np.r_[0.0, np.cumsum([f["hydraulic_s"] for f in frames])]
            axes[3].plot(edges, elapsed, color=colour)
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
