"""Measure the largest outward crest before and after the original eight-hour forecast."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import figures, receipts


def peak(*, edges, flow, start, end):
    k = np.flatnonzero((edges[:-1] >= start) & (edges[1:] <= end))
    if not len(k):
        raise ValueError("No complete observed bin inside the specified window")
    selected = k[np.argmax(flow[k])]
    return float(flow[selected]), int(edges[selected]), int(edges[selected + 1])


def native_panel(*, directory, case, records):
    pair = [
        r
        for r in records
        if r["case"] == case
        and r["station"] == "outlet"
        and r["design"] == "two-hour check"
    ]
    if not pair:
        return False
    if len(pair) != 2 or len({r["end_h"] for r in pair}) != 1:
        raise ValueError("The cadence plot requires a complete native pair")
    fig, axes = plt.subplots(
        2, 1, figsize=(6.4, 4.9), sharex=True, sharey=True, layout="constrained"
    )
    for ax, row in zip(axes, sorted(pair, key=method_name), strict=True):
        data = receipts.read(
            root=directory / "extended_runs",
            ident=row["series_id"],
            end=round(row["end_h"] * 3600),
        )
        frames = data["frames"]
        t = np.r_[0, [f["t_s"] for f in frames]]
        q = np.asarray([f["outlet_m3s"] for f in frames])
        coarse_interval = 900
        interval = data["spec"]["interval"]
        if (
            coarse_interval % interval
            or t[-1] % coarse_interval
            or np.any(np.diff(t) != interval)
        ):
            raise ValueError("Cadence plot requires nested observed intervals")
        n = coarse_interval // interval
        ax.stairs(
            q,
            t / 3600,
            color=figures.COLOURS[row["method"]],
            linewidth=1,
            label=f"{interval} s",
        )
        ax.stairs(
            q.reshape(-1, n).mean(axis=1),
            t[::n] / 3600,
            color="#252525",
            linestyle="--",
            linewidth=1.2,
            label="15 min",
        )
        ax.axvline(row["original_h"], color="0.4", linestyle=":", label="Original end")
        ax.set(
            ylabel="Outlet Q (m³/s)",
            title=figures.LABELS[row["method"]],
            xlim=(0, row["end_h"]),
        )
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time since rainfall began (h)")
    figures.save(fig=fig, folder=directory / "figures", name="incised_native_cadence")
    return True


def method_name(row):
    return row["method"]


def prefix_storage(*, directory, design):
    """Locate the change in storage on a common spatial support at the original endpoint."""
    folder = directory / "incised_crest"
    controls = {}
    for path in folder.glob("*/result.json"):
        record = json.loads(path.read_text())
        inputs = record["inputs"]
        spec = inputs["spec"]
        if (
            inputs["design"] != design
            or spec["method"] != "adaptive"
            or inputs["factor"] != 1
        ):
            continue
        if spec["dx"] in controls:
            raise ValueError("Ambiguous spatial comparison of incised controls")
        controls[spec["dx"]] = (path, spec, record["data"])
    spacings = sorted(
        dx
        for dx, method, factor in design["controls"]
        if method == "adaptive" and factor == 1
    )
    if len(spacings) != 2 or set(controls) != set(spacings):
        raise ValueError("Two completed fixed-bed FV2 refinements are required")
    fine, coarse = spacings
    support = controls[coarse][1]["support"]
    extent = controls[coarse][1]["dimensions"]
    thresholds = json.loads((directory / "application_settings.json").read_text())[
        "endpoint_depth"
    ]["thresholds_m"]
    levels = [round(t * 100) for t in thresholds]
    arrays = {}
    rows = []
    for dx in spacings:
        path, spec, data = controls[dx]
        if (
            data["hours_completed"] != design["diagnostic_start_h"]
            or spec["support"] != support
            or spec["dimensions"] != extent
            or spec["dem"] != design["bed_m"]
        ):
            raise ValueError(
                "Storage maps require matched physical extent, support, bed and time"
            )
        with np.load(path.with_name("spatial.npz")) as archive:
            depth = archive["depth"]
            wet = {cm: archive[f"wet_fraction_{cm}cm"] for cm in levels}
        if depth.shape != (extent[1] // support, extent[0] // support) or any(
            a.shape != depth.shape for a in wet.values()
        ):
            raise ValueError("Unexpected common-support map dimensions")
        storage = float(depth.sum() * support**2)
        np.testing.assert_allclose(
            storage, data["final_storage_m3"], rtol=1e-12, atol=0
        )
        arrays[dx] = {"depth": depth, **wet}
        rows.append(
            dict(
                run_id=path.parent.name,
                dx_m=dx,
                bed_m=spec["dem"],
                support_m=support,
                end_h=data["hours_completed"],
                initial_m3=data["initial_m3"],
                rain_m3=data["rain_m3"],
                inflow_m3=data["inflow_m3"],
                infiltration_m3=data["infiltration_m3"],
                outflow_m3=data["outflow_m3"],
                storage_m3=storage,
                residual_m3=data["residual_m3"],
                **{
                    f"inundated_{cm}cm_km2": float(a.sum() * support**2 / 1e6)
                    for cm, a in wet.items()
                },
            )
        )
    metrics.csv_write(file=folder / "initial_window_storage.csv", rows=rows)
    fig, axes = plt.subplots(
        1, 3, figsize=(9, 3.6), layout="constrained", sharex=True, sharey=True
    )
    for ax, key, label in zip(
        axes,
        ("depth", 5, 10),
        (
            "Mean depth (cm)",
            "Area ≥5 cm (percentage points)",
            "Area ≥10 cm (percentage points)",
        ),
        strict=True,
    ):
        field = 100 * (arrays[fine][key] - arrays[coarse][key])
        limit = float(np.abs(field).max())
        image = ax.imshow(
            field,
            origin="lower",
            extent=(0, extent[0] / 1000, 0, extent[1] / 1000),
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xlabel("Easting (km)")
        fig.colorbar(image, ax=ax, orientation="horizontal", label=label, shrink=0.94)
    axes[0].set_ylabel("Northing (km)")
    fig.suptitle(
        f"Incised catchment at {design['diagnostic_start_h']:g} h, FV2 at {fine:g} m minus FV2 at {coarse:g} m"
    )
    figures.save(fig=fig, folder=folder, name="initial_window_storage")
    return rows


def generate(*, directory, design):
    prefix_storage(directory=directory, design=design)
    folder = directory / "incised_crest"
    root = directory / "extended_runs"
    groups = [
        g
        for f in root.glob("group_*.json")
        if (g := json.loads(f.read_text()))["cohort"] == "incised-crest"
    ]
    if len(groups) != 1:
        raise ValueError("A single completed fixed-bed sensitivity group is required")
    group = groups[0]
    rows = []
    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 6.2), sharex=True, layout="constrained"
    )
    for ident in group["series"]:
        d = receipts.read(root=root, ident=ident, end=group["end_s"])
        s = d["spec"]
        frames = d["frames"]
        t = np.r_[0, [f["t_s"] for f in frames]]
        label = (
            f"{figures.LABELS[s['method']]} {s['dx']:g} m, CFL {d['factor'] * 0.4:g}"
        )
        for ax, station in zip(axes, ("outlet", "section"), strict=True):
            q = np.asarray([f[station + "_m3s"] for f in frames])
            ax.stairs(q, t / 3600, label=label, linewidth=1)
            for cadence in (s["interval"], 900):
                if (
                    np.any(np.diff(t) != s["interval"])
                    or cadence % s["interval"]
                    or t[-1] % cadence
                ):
                    raise ValueError("Incompatible late-crest output cadence")
                bins = cadence // s["interval"]
                series = q.reshape(-1, bins).mean(axis=1)
                edges = t[::bins]
                for window, left, right in (
                    ("original", 0, design["diagnostic_start_h"] * 3600),
                    ("recession", design["diagnostic_start_h"] * 3600, t[-1]),
                ):
                    discharge, lo, hi = peak(
                        edges=edges, flow=series, start=left, end=right
                    )
                    rows.append(
                        dict(
                            case=s["case"],
                            station=station,
                            method=s["method"],
                            dx=s["dx"],
                            dem=s["dem"],
                            cfl=0.4 * d["factor"],
                            interval_s=cadence,
                            window=window,
                            run_id=ident,
                            common_end_h=t[-1] / 3600,
                            peak_m3s=discharge,
                            bin_start_s=lo,
                            bin_end_s=hi,
                        )
                    )
    metrics.csv_write(file=folder / "peak_windows.csv", rows=rows)
    axes[0].set_ylabel("Outlet discharge (m³/s)")
    axes[1].set_ylabel("Internal section (m³/s)")
    for ax in axes:
        ax.axvline(
            design["diagnostic_start_h"], color="0.4", linewidth=0.8, linestyle=":"
        )
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("Time since start of rainfall (h)")
    figures.save(fig=fig, folder=folder, name="incised_crest")
    text = [
        "# Incised catchment late-crest sensitivity",
        "",
        "This follow-up was defined after the native 10 m Hancock forecast produced a larger one-minute crest during recession than during the original eight-hour forecast. It contributes no training or validation decisions to the Solver Advisor. Six hours of spatial rain, roughness, the 10 m represented bed, initial channel flow and prescribed boundaries are held fixed. The 5 m hydraulic grid subdivides the same physical inputs. The diagnostic compares the highest measured interval means in the two windows. All runs share the same verified end time.",
        "",
        "| Method | Grid (m) | CFL | Interval (s) | Original peak (m³/s) | Late peak (m³/s) | Late peak interval ending (h) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for early, late in zip(rows[::2], rows[1::2], strict=True):
        if early["station"] == "outlet":
            text.append(
                f"| {figures.LABELS[early['method']]} | {early['dx']:g} | {early['cfl']:g} | {early['interval_s']} | {early['peak_m3s']:.3f} | {late['peak_m3s']:.3f} | {late['bin_end_s'] / 3600:.3f} |"
            )
    text.extend(
        [
            "",
            "The maximum within each window can occur at a different time across methods and cadences. Agreement across models alone cannot establish a physical crest. [Measured 60 and 900 s peaks](peak_windows.csv) · [Hydrographs at the outlet and internal section](incised_crest.pdf) · [Recession checks](recession_terminal.csv) · [Water balance and inundation on the fixed bed at hour eight](initial_window_storage.csv) · [Spatial storage difference at hour eight](initial_window_storage.pdf)",
        ]
    )
    (folder / "README.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    design = json.loads((directory / "incised_settings.json").read_text())
    if sys.argv[1:] == ["incised-storage"]:
        prefix_storage(directory=directory, design=design)
    else:
        generate(directory=directory, design=design)
