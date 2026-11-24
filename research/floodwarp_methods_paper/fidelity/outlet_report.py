"""Generate the free-outlet qualification from analytical controls and accepted storm states."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from research.floodwarp_methods_paper import metrics


def draw(*, target, records, common_support):
    maps = []
    for r in records:
        with np.load(
            target.parent / "mean_storms" / r["source_id"] / "spatial.npz"
        ) as state:
            maps.append(state["depth"])
    wet = np.maximum(*maps) >= 0.05
    j, i = np.where(wet)
    west = max(0, i.min() - 3)
    south = max(0, j.min() - 4)
    north = min(wet.shape[0], j.max() + 5)
    bounds = np.array([west, wet.shape[1], south, north]) * common_support
    fig, axes = plt.subplots(
        1, 3, figsize=(7.4, 4.3), sharey=True, layout="constrained"
    )
    fields = [a[south:north, west:] for a in maps]
    for ax, r, h in zip(axes[:2], records, fields, strict=True):
        im = ax.imshow(
            h,
            origin="lower",
            extent=bounds,
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=max(f.max() for f in fields),
        )
        ax.set_title(f"{r['dx_m']} m FV2")
    fig.colorbar(im, ax=list(axes[:2]), location="bottom", label="Mean depth (m)")
    delta = fields[0] - fields[1]
    maxabs = abs(delta).max()
    im = axes[2].imshow(
        delta,
        origin="lower",
        extent=bounds,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-maxabs,
        vmax=maxabs,
    )
    axes[2].set_title("10 m minus 5 m")
    fig.colorbar(im, ax=axes[2], location="bottom", label="Difference (m)")
    axes[0].set_ylabel("Northing (m)")
    for ax in axes:
        ax.set_xlabel("Easting (m)")
        ax.ticklabel_format(useOffset=False)
    with plt.rc_context({"svg.hashsalt": "native-incised-outlet"}):
        for suffix, tags in (
            ("pdf", {"CreationDate": None}),
            ("svg", {"Date": None}),
            ("png", {}),
        ):
            path = target / f"outlet_depth.{suffix}"
            fig.savefig(path, dpi=300, metadata=tags)
            if suffix == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
    plt.close(fig)


def run(*, directory):
    target = directory / "native_outlet"
    windows = metrics.csv_read(directory / "native_results/measured_windows.csv")
    settings = json.loads((directory / "mean_settings.json").read_text())
    forecasts = sorted(
        (
            r
            for r in windows
            if r["case"] == settings["case"]
            and r["method"] == "adaptive"
            and r["horizon"] == "original_window"
        ),
        key=spacing,
        reverse=True,
    )
    if len(forecasts) != 2:
        raise ValueError("Both original-window native FV2 refinements are required")
    comparisons = metrics.csv_read(
        directory / "native_results/incised_state_disagreement.csv"
    )
    five = next(
        r
        for r in comparisons
        if r["left_method"] == r["right_method"] == "adaptive"
        and r["wet_union_cm"] == "5"
    )
    strips = {r["id"]: r for r in metrics.csv_read(target / "strip_storage.csv")}
    plane = metrics.csv_read(directory / "free_plane/discharge.csv")
    controls = json.loads((directory / "free_plane_settings.json").read_text())
    draw(target=target, records=forecasts, common_support=int(five["common_support_m"]))
    text = [
        "# Free downstream boundary on a steep slope",
        "",
        f"In the matched eight-hour incised FV2 forecasts, {five['eastern_edge_blocks']} of the {five['blocks']} common blocks exceeding 5 cm mean depth in either forecast lie in the last {five['common_support_m']} m of the catchment. The eastern strip accounts for {100 * float(five['eastern_strip_fraction_squared_difference']):.3f}% of the full-domain squared depth difference. The open boundary was checked face by face in the saved state. Each station flux integral agrees with its recorded boundary export.",
        "",
        "| Grid (m) | Peak (m³/s) | Export (m³) | Storage in last 40 m (m³) | Storage upstream (m³) | Deepest terminal-strip cell at hour 8 (m) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for f in forecasts:
        r = strips[f["source_id"]]
        text.append(
            f"| {f['dx_m']} | {float(f['outlet_peak_m3s']):.3f} | {float(f['net_outlet_m3']):,.0f} | {float(r['strip_storage_m3']):,.0f} | {float(r['interior_storage_m3']):,.0f} | {float(r['strip_maximum_final_depth_m']):.3f} |"
        )
    text.extend(
        [
            "",
            "[Native checkpoint strip budget](strip_storage.csv) · [Spatial depth comparison](outlet_depth.pdf)",
            "",
            "## Independent steady sheet",
            "",
            f"Steady uniform full-SWE sheet flow on a plane has discharge Q = Wh⁵ᐟ³√S/n. Here W = {controls['width_m']} m, h = {controls['depth_m']} m, S = {controls['grade']} and n = {controls['manning']}. Both characteristics travel downstream. The upstream boundary disturbance cannot reach the measuring section within the test window. A pair of runs measures flux at the same physical section, first with a FREE boundary there and then with another 400 m of the same native plane downstream. Mean fluxes in the table cover seconds 30–60.",
            "",
            "| Formulation | Timestep cap (s) | 10 m free-face bias | 10 m interior bias |",
            "|---|---:|---:|---:|",
        ]
    )
    for m in controls["methods"]:
        for dt in (controls["maximum_dt_s"], controls["fine_dt_s"]):
            points = {
                int(r["downstream_extension_m"]): r
                for r in plane
                if r["method"] == m
                and int(r["dx_m"]) == 10
                and float(r["time_s"]) == max(controls["times_s"])
                and float(r["maximum_dt_s"]) == dt
                and r["face"] == "east"
            }
            text.append(
                f"| {m} | {dt:g} | {float(points[0]['discharge_bias_pct']):+.2f}% | {float(points[400]['discharge_bias_pct']):+.2f}% |"
            )
    text.extend(
        [
            "",
            "All four outlet orientations are included in the data. The MUSCL reconstruction sets the normal slope to zero at a missing exterior neighbour. FREE copies the interior trace into the ghost state and suppresses inward flux. The normal surface-slope source therefore vanishes in the terminal cell. Moving the boundary restores a two-sided stencil at the measured section. This test establishes sensitivity to boundary placement on a known supercritical sheet. The incised storm also contains lateral convergence, changing storage and friction. Its final depth error against an independent physical reference remains unresolved.",
            "",
            "[Analytical discharge controls](../free_plane/discharge.csv) · [Implementation and source identity](../free_plane/identity.json)",
        ]
    )
    reach_files = list((directory / "free_reach").glob("*/comparison.csv"))
    if len(reach_files) == 1:
        curves = metrics.csv_read(reach_files[0])
        last = max(float(row["end_s"]) for row in curves)
        rows = [r for r in curves if float(r["end_s"]) == last]
        ends = sorted({int(r["extension_m"]) for r in rows})
        text.extend(
            [
                "",
                "## Same physical section in the incised catchment",
                "",
                "The section was selected after localising the depth difference. Each run starts with the native metre bed, depth, momentum, rainfall and infiltration fields on an aligned subdomain of the original catchment. The coarser subsets agree cell for cell with the full-domain inputs. The table reports the final observed time interval, in m³/s. The two longer reaches test sensitivity to the new downstream boundary. This reach has no exact analytical solution.",
                "",
                "| Grid (m) | " + " | ".join(f"{d} m downstream" for d in ends) + " |",
                "|---:" + "|---:" * len(ends) + "|",
            ]
        )
        for dx in sorted({int(r["grid_m"]) for r in rows}):
            by = {int(r["extension_m"]): r for r in rows if int(r["grid_m"]) == dx}
            text.append(
                f"| {dx} | "
                + " | ".join(format(float(by[d]["section_m3s"]), ".6f") for d in ends)
                + " |"
            )
        text += [
            "",
            f"[Reach histories and water balance](../free_reach/{reach_files[0].parent.name}/comparison.csv)",
        ]
    (target / "README.md").write_text("\n".join(text) + "\n")


def spacing(row):
    return float(row["dx_m"])


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
