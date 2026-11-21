"""Export measured full-event hydrographs, inundation, convergence and GPU costs."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from validation.hancock_long_storm_review.landforms import physical_fields


def finish(*, fig, path, plt):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def resolution_figure(*, records, review, plt, colours):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained", squeeze=False)
    for column, land in enumerate(("plain", "fan", "incised")):
        for row_index, representation in enumerate(("sampled", "fixed")):
            ax = axes[row_index, column]
            panel = [
                r
                for r in records
                if r["landform"] == land
                and r["storm"] == "moderate"
                and r["representation"] == representation
                and r["group"] in ("primary", "refinement")
            ]
            for method, colour in colours.items():
                runs = [
                    r
                    for r in panel
                    if r["method"] == method
                    and (method != "hancock" or r["mode"] == "tiles")
                    and r["run"] != r["reference"]
                ]
                runs.sort(key=spacing)
                if runs:
                    ax.plot(
                        [float(r["dx"]) for r in runs],
                        [float(r["outlet_rmse_m3s"]) for r in runs],
                        marker="o",
                        color=colour,
                        label=method,
                    )
            ax.set(
                xscale="log",
                yscale="symlog",
                xticks=[1, 5, 10],
                xticklabels=["1", "5", "10"],
                xlabel="Hydraulic spacing (m)",
                ylabel="Outlet RMSE (m³/s)",
                title=land.capitalize()
                + (
                    " · sampled DEM"
                    if representation == "sampled"
                    else " · fixed 10 m DEM"
                ),
            )
    axes[0, 0].legend()
    finish(fig=fig, path=review / "resolution_discharge.png", plt=plt)


def interval_figure(*, review, plt):
    with (review / "occupancy_events.csv").open() as f:
        data = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    bins = ["0–1%", "1–5%", "5–20%", "20–50%", "50–100%"]
    lookup = {0.0: 0, 0.01: 1, 0.05: 2, 0.2: 3, 0.5: 4}
    for ax, measure in zip(axes, ("positive depth", "scheduled cells"), strict=True):
        for offset, (group, title) in enumerate(
            (("primary", "384 cells"), ("large", "250k–2M"), ("moving", "Moving storm"))
        ):
            subset = [
                r for r in data if r["group"] == group and r["measure"] == measure
            ]
            ax.bar(
                [lookup[float(r["lower"])] + 0.24 * (offset - 1) for r in subset],
                [float(r["ratio"]) for r in subset],
                width=0.23,
                label=title,
            )
        ax.axhline(1.5, color=".3", ls=":")
        ax.set(
            xticks=range(len(bins)),
            xticklabels=bins,
            title=measure.capitalize(),
            xlabel="End-of-interval fraction",
        )
    axes[0].set_ylabel("Tiled Hancock / FV1 hydraulic runtime")
    axes[1].legend()
    finish(fig=fig, path=review / "storm_occupancy.png", plt=plt)


def hydrograph_figure(*, plt, compare, event_choice, colours, file):
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for r in compare:
        name = r["method"]
        tt = np.r_[0.0, [s["t_s"] for s in r["samples"]]] / 3600.0
        t = tt[1:]
        if r is compare[0]:
            a = np.prod(r["dimensions_m"])
            rate = (
                np.diff([0.0, *[s["rain_m3"] for s in r["samples"]]])
                / (np.diff(tt) * a)
                * 1000.0
            )
            axes[0].stairs(rate, tt, fill=True, color="#407eb3")
        axes[1].stairs(
            [s["outlet_m3s"] for s in r["samples"]], tt, label=name, color=colours[name]
        )
        axes[2].plot(
            t, [s["wet_fraction"] * 100 for s in r["samples"]], color=colours[name]
        )
        axes[3].stairs(
            [s["hydraulic_wall_s"] for s in r["samples"]], tt, color=colours[name]
        )
    axes[0].set_title(" • ".join(event_choice).capitalize())
    axes[1].legend(ncol=4)
    for ax, lab in zip(
        axes,
        (
            "Mean rain (mm/h)",
            "Outlet (m³/s)",
            "Positive depth (%)",
            "Hydraulic cost (s)",
        ),
        strict=True,
    ):
        ax.set_ylabel(lab)
        ax.axvline(6, color=".5", ls=":", lw=1)
    axes[-1].set_xlabel("Elapsed time (h)")
    finish(fig=fig, path=file, plt=plt)


def render(*, review, fields, event_choice, colours):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads((review / "cases.json").read_text())
    rows = [
        json.loads(p.read_text()) for p in sorted((review / "events").glob("*.json"))
    ]
    primary = [r for r in rows if r["group"] == "primary"]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5), layout="constrained")
    for ax, (land, spec) in zip(axs.flat, config["landforms"].items(), strict=True):
        z, *_ = physical_fields(
            landform=land,
            spec=spec,
            seed=config["seed"],
            length=240,
            width=160,
            dx=1,
            terrain_dx=1,
        )
        im = ax.imshow(z, extent=(0, 240, 0, 160), origin="lower", cmap="terrain")
        ax.set(title=land.capitalize(), xlabel="East (m)", ylabel="North (m)")
        fig.colorbar(im, ax=ax, shrink=0.72, label="Bed (m)")
    finish(fig=fig, path=review / "landforms.png", plt=plt)
    compare = [
        r
        for r in primary
        if (r["landform"], r["storm"]) == event_choice
        and (r["method"] != "hancock" or r["mode"] == "tiles")
    ]
    hydrograph_figure(
        plt=plt,
        compare=compare,
        event_choice=event_choice,
        colours=colours,
        file=review / "event.png",
    )
    long_event = [
        r
        for r in primary
        if r["hours"] == 24 and (r["method"] != "hancock" or r["mode"] == "tiles")
    ]
    hydrograph_figure(
        plt=plt,
        compare=long_event,
        event_choice=("plain", "moderate · 24 hours"),
        colours=colours,
        file=review / "plain_24h.png",
    )
    occ = json.loads((review / "occupancy.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for mode in ("tiles", "dense"):
        runs = [r for r in occ if r["mode"] == mode]
        fv = [r for r in occ if r["method"] == "contact_hll"]
        axes[0].plot(
            [r["final_fraction"] * 100 for r in runs],
            [r["graph_s"] / b["graph_s"] for r, b in zip(runs, fv, strict=True)],
            "o-",
            label=mode,
        )
    axes[0].axhline(1.5, color="k", ls=":", label="1.5× target")
    axes[0].set(
        xlabel="Final positive-depth cells (%)",
        ylabel="Hancock / FV1 hydraulic runtime",
    )
    axes[0].legend()
    t = [r for r in occ if r["mode"] == "tiles"]
    d = [r for r in occ if r["mode"] == "dense"]
    axes[1].plot(
        [r["active"]["evaluated_cells"] / 655360 * 100 for r in t],
        [r["graph_s"] / b["graph_s"] for r, b in zip(t, d, strict=True)],
        "o-",
    )
    axes[1].axhline(1, color="k", ls=":")
    axes[1].set(xlabel="Scheduled cells (%)", ylabel="Tiled / dense Hancock runtime")
    finish(fig=fig, path=review / "occupancy.png", plt=plt)
    lands = list(config["landforms"])
    storms = list(config["storms_mm_h"])
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    arr = np.full((len(lands), len(storms)), np.nan)
    for r in primary:
        if r["mode"] == "tiles":
            f = next(
                s
                for s in primary
                if s["landform"] == r["landform"]
                and s["storm"] == r["storm"]
                and s["method"] == "fv1"
            )
            arr[lands.index(r["landform"]), storms.index(r["storm"])] = (
                r["hydraulic_s"] / f["hydraulic_s"]
            )
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=1.0)
    for (j, i), v in np.ndenumerate(arr):
        ax.text(
            i,
            j,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if v > 2.3 else "black",
        )
    ax.set_xticks(range(len(storms)), storms)
    ax.set_yticks(range(len(lands)), lands)
    fig.colorbar(im, ax=ax, label="Tiled Hancock / FV1 hydraulic runtime")
    finish(fig=fig, path=review / "primary_cost.png", plt=plt)
    if (review / "accuracy.csv").exists():
        with (review / "accuracy.csv").open() as stream:
            records = list(csv.DictReader(stream))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, land in zip(axes, ("plain", "fan", "incised"), strict=True):
            for m in colours:
                selected = [
                    r
                    for r in records
                    if r["method"] == m
                    and r["mode"] in ("tiles", "sparse", "native", "dense")
                    and (m != "hancock" or r["mode"] == "tiles")
                    and r["landform"] == land
                    and r["storm"] == "moderate"
                    and r["group"] in ("primary", "refinement")
                    and r["representation"] == "sampled"
                    and r["run"] != r["reference"]
                ]
                selected.sort(key=spacing)
                if selected:
                    ax.plot(
                        [float(r["runtime_s"]) for r in selected],
                        [float(r["depth_rmse_m"]) * 1000 for r in selected],
                        "o-",
                        label=m,
                        color=colours[m],
                    )
                    for r in selected:
                        ax.annotate(
                            r["dx"] + " m",
                            (float(r["runtime_s"]), float(r["depth_rmse_m"]) * 1000),
                            fontsize=7,
                        )
            ax.set(
                title=land.capitalize(),
                xlabel="Hydraulic runtime (s)",
                ylabel="Final-depth RMSE (mm)",
                xscale="log",
            )
        axes[0].legend(fontsize=8)
        finish(fig=fig, path=review / "accuracy_cost.png", plt=plt)
        resolution_figure(records=records, review=review, plt=plt, colours=colours)
    if (review / "occupancy_events.csv").exists():
        interval_figure(review=review, plt=plt)
    if compare:
        fig, axes = plt.subplots(
            1,
            len(compare),
            figsize=(3 * len(compare), 3.2),
            layout="constrained",
            squeeze=False,
        )
        arrays = []
        for r in compare:
            with np.load(fields / (r["name"] + ".npz")) as a:
                arrays.append(a["maximum"])
        for ax, r, z in zip(axes.flat, compare, arrays, strict=True):
            im = ax.imshow(
                z,
                extent=(0, 240, 0, 160),
                origin="lower",
                vmin=0.0,
                vmax=max(x.max() for x in arrays),
                cmap="Blues",
            )
            ax.set(title=r["method"], xlabel="East (m)", ylabel="North (m)")
        fig.colorbar(
            im, ax=axes.ravel().tolist(), shrink=0.65, label="Maximum depth (m)"
        )
        finish(fig=fig, path=review / "maximum_depth.png", plt=plt)


def spacing(row):
    return float(row["dx"])


if __name__ == "__main__":
    review = Path(__file__).parent
    fields = Path("outputs/hancock_long_storm_review/fields")
    event_choice = ("floodplain", "spatial")
    colours = dict(fv1="#777777", hancock="#ba442f", fv2="#3976a4", li="#408857")
    render(review=review, fields=fields, event_choice=event_choice, colours=colours)
