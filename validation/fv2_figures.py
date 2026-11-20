"""Generate the numerical tables and accuracy/cost figures from measured GPU results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(*, folder, name):
    return json.loads((folder / (name + ".json")).read_text())


def rows_table(*, title, columns, rows, note):
    text = ["## " + title, "", note, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = [format(v, ".5g") if isinstance(v, (np.floating, float)) else str(v) for v in row]
        text.append("| " + " | ".join(values) + " |")
    return "\n".join(text) + "\n"


def verification(*, data, analytical, equilibria, dx):
    sections = []
    samples = []
    for study in ("space", "time"):
        previous = None
        for row in data:
            if row["study"] != study:
                continue
            e = np.array(row["rmse"])
            orders = ["—"] * 3 if previous is None else list(np.log2(previous / e))
            samples.append([study, row["n"], row["dt"], *e, *orders])
            previous = e
    sections.append(
        rows_table(
            title="Smooth manufactured solution",
            columns=[
                "Study",
                "Cells/axis",
                "dt (s)",
                "h RMSE (m)",
                "hu RMSE (m²/s)",
                "hv RMSE (m²/s)",
                "p(h)",
                "p(hu)",
                "p(hv)",
            ],
            rows=samples,
            note="The spatial study compares cell averages with the exact field. The temporal study fixes the grid and refines time against the same spatial operator with a further 16× smaller step.",
        )
    )
    methods = ("fv1", "minmod", "mc")
    combined = analytical + equilibria["rows"]
    names = list(dict.fromkeys(r["case"] for r in combined))
    output = []
    for name in names:
        group = [
            next(r for r in combined if r["method"] == m and r["case"] == name and r["dx"] == dx)
            for m in methods
        ]
        output.append(
            [
                name,
                *[r["depth_rmse_m"] for r in group],
                *group[0]["momentum_rmse_m2_s"],
                *group[-1]["momentum_rmse_m2_s"],
                max(abs(r["balance"]["residual"]) for r in group),
            ]
        )
    sections.append(
        rows_table(
            title=f"Analytical controls at {dx:g} m",
            columns=[
                "Case",
                "FV1 h",
                "Minmod h",
                "MC h",
                "FV1 hu",
                "FV1 hv",
                "MC hu",
                "MC hv",
                "Max ledger residual (m³)",
            ],
            rows=output,
            note="RMSE units are metres for depth and m²/s for momentum. All methods use contact-HLL. The oblique, smooth-shear and steady-channel norms use interior windows before reflected boundary information arrives.",
        )
    )
    output = []
    for r in analytical:
        if r["dx"] != dx or r["method"] == "minmod":
            continue
        for key, v in r.get("features", {}).items():
            output.append(
                [
                    r["case"],
                    r["method"],
                    key,
                    v["error_m"],
                    v.get("propagation_error_m", "—"),
                    v.get("numerical_10_to_90_width_m", "—"),
                    v.get("exact_cell_average_10_to_90_width_m", "—"),
                ]
            )
    sections.append(
        rows_table(
            title="Wave positions and smearing",
            columns=[
                "Case",
                "Method",
                "Feature",
                "Position error (m)",
                "Displacement error, second half (m)",
                "10–90% width (m)",
                "Exact cell-average width (m)",
            ],
            rows=output,
            note="Shorelines use a matched 1 cm depth threshold. The Ritter head uses a matched 1.9 m threshold. An absent width means the corresponding crossing was unavailable or inapplicable.",
        )
    )
    samples = []
    last = None
    for row in equilibria["coupled_source"]:
        e = np.array(row["errors"])
        orders = ["—"] * 4 if last is None else list(np.log2(last / e))
        samples.append([row["dt"], *e, *orders])
        last = e
    sections.append(
        rows_table(
            title="Coupled rain, exponential infiltration and Manning drag",
            columns=[
                "dt (s)",
                "h error (m)",
                "hu error",
                "hv error",
                "Soil error (m)",
                "p(h)",
                "p(hu)",
                "p(hv)",
                "p(soil)",
            ],
            rows=samples,
            note="Absolute error at 2 s in the uniform interior versus an independently integrated DOP853 ODE (relative tolerance 1e-13).",
        )
    )
    return sections


def terrain_tables(*, synthetic, sites):
    fixed = [r for r in synthetic if r.get("terrain_fixed_5m")]
    samples = []
    for r in fixed:
        ref = r["refined"]["mc"]
        samples.append(
            [
                r["case"],
                r["method"],
                ref["depth_rmse_m"],
                r["refined"]["fv1"]["depth_rmse_m"],
                r["reference_gap"]["depth_rmse_m"],
                ref["wet_iou"],
                ref["peak_envelope_rmse_m"],
                ref["export_m3_difference"],
            ]
        )
    sections = [
        rows_table(
            title="Truncation error on identical represented 5 m terrain",
            columns=[
                "Case",
                "Method",
                "h vs 1.25 m MC (m)",
                "h vs 1.25 m FV1 (m)",
                "2.5→1.25 m gap (m)",
                "Wet IoU",
                "Envelope RMSE (m)",
                "Export difference (m³)",
            ],
            rows=samples,
            note="Every reference subcell inherits its 5 m parent's bed, forcing, roughness and initial conserved state. The gap compares the two reference grids using the listed method. IoU and export differences use the 1.25 m MC reference. The JSON also records resolution-dependent sampling of the continuous terrain and the 0.625 m timestep check.",
        )
    ]
    for data in sites:
        samples = []
        for r in data["rows"]:
            a = r["common_1m"]
            samples.append(
                [
                    r["terrain_dx"],
                    r["hydraulic_dx"],
                    r["method"],
                    r.get("bed_rmse_m", "—"),
                    a["depth_rmse_m"] * 1000,
                    a["peak_envelope_rmse_m"] * 1000,
                    a["wet_iou"],
                    a["export_m3_difference"],
                    a["wet_component_difference"],
                ]
            )
        sections.append(
            rows_table(
                title=data["site"].title() + " native terrain control",
                columns=[
                    "DEM (m)",
                    "Hydraulic grid (m)",
                    "Method",
                    "Bed RMSE (m)",
                    "h RMSE (mm)",
                    "Envelope RMSE (mm)",
                    "Wet IoU",
                    "Export difference (m³)",
                    "Wet component difference",
                ],
                rows=samples,
                note=f"Tile forcing lasts {data['duration']:g} s. Fields are compared on identical 1 m support against the finest MC solution on the represented 1 m DEM. The independent finest FV1 alternative and timestep study quantify sensitivity.",
            )
        )
    return sections


def cost_figure(*, folder, rows, colors):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for i, (case, label) in enumerate((("stoker", "Stoker dam break"), ("orton_crop", "Orton terrain tile"))):
        for method, color in zip(("fv1", "mc"), colors, strict=True):
            values = [r for r in rows if r["case"] == case and r["mode"] == "dense" and r["method"] == method]
            values.sort(key=grid_spacing)
            y = [r["errors"]["depth_rmse_m"] if case == "stoker" else r["common_1m_error_m"] for r in values]
            for j, (key, xlabel) in enumerate(
                (("dx", "Grid spacing (m)"), ("wall_s", "Median GPU hydraulic time (s)"))
            ):
                x = [r[key] for r in values]
                ax = axes[i, j]
                ax.loglog(x, y, "o-", color=color, label="FV1" if method == "fv1" else "FV2 MC", ms=5)
                if j == 1:
                    for xx, yy, r in zip(x, y, values, strict=True):
                        ax.annotate(
                            f"{r['dx']:g} m", (xx, yy), xytext=(1, 5), textcoords="offset points", fontsize=7
                        )
                ax.set(title=label, xlabel=xlabel, ylabel="Depth RMSE (m)")
                ax.grid(alpha=0.2, which="both")
        axes[i, 0].legend(frameon=False)
    fig.savefig(folder / "accuracy_cost.png", dpi=170)
    plt.close(fig)


def grid_spacing(row):
    return row["dx"]


def native_figure(*, folder, sites, colors):
    fig, axes = plt.subplots(1, len(sites), figsize=(9, 4), layout="constrained")
    for ax, data in zip(axes, sites, strict=True):
        for method, color in zip(("fv1", "mc"), colors, strict=True):
            subset = [
                r for r in data["rows"] if r["method"] == method and r["hydraulic_dx"] == r["terrain_dx"]
            ]
            x = [r["terrain_dx"] for r in subset]
            y = [1000 * r["common_1m"]["depth_rmse_m"] for r in subset]
            ax.plot(x, y, "o-", color=color, label="FV1" if method == "fv1" else "FV2 MC")
        gap = next(r for r in data["rows"] if r["method"] == "mc" and r["hydraulic_dx"] == 0.25)
        ax.axhline(
            1000 * gap["common_1m"]["depth_rmse_m"], ls=":", color="black", label="Last MC refinement gap"
        )
        ax.set(
            title=data["site"].title(),
            xlabel="Terrain and solver spacing (m)",
            ylabel="Depth RMSE on 1 m support (mm)",
            xticks=x,
            ylim=(0, None),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(folder / "native_terrain.png", dpi=170)
    plt.close(fig)


def publish(*, folder, dx, colors):
    data = read(folder=folder, name="manufactured")
    analytical = read(folder=folder, name="analytical")
    equilibria = read(folder=folder, name="equilibria")
    sections = verification(data=data, analytical=analytical, equilibria=equilibria, dx=dx)
    sites = [read(folder=folder, name=site + "_terrain") for site in ("orton", "purau")]
    sections.extend(terrain_tables(synthetic=read(folder=folder, name="synthetic"), sites=sites))
    rows = read(folder=folder, name="cost")
    samples = [
        [
            r["case"],
            r["dx"],
            r["method"],
            r["mode"],
            r["steps"],
            r["min_dt"],
            r["mean_dt"],
            1000 * r["wall_s"],
            r["cell_steps_s"] / 1e6,
            r["device_array_bytes"] / 1e6,
            1000 * r["graph_capture_s"],
        ]
        for r in rows
    ]
    sections.append(
        rows_table(
            title="Replicated GPU cost",
            columns=[
                "Case",
                "dx",
                "Method",
                "Execution",
                "Steps",
                "Min dt (s)",
                "Mean dt (s)",
                "Median (ms)",
                "M cell-steps/s",
                "Device arrays (MB)",
                "First graph capture (ms)",
            ],
            rows=samples,
            note="RTX 4070 Ti SUPER, FP64, five synchronised warm repeats. Timing excludes input/output, figure creation and first graph capture. The JSON also contains all individual replicates and per-kernel CUDA timings.",
        )
    )
    (folder / "TABLES.md").write_text("# Measured FV2 verification results\n\n" + "\n".join(sections))
    cost_figure(folder=folder, rows=rows, colors=colors)
    native_figure(folder=folder, sites=sites, colors=colors)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    spacing = 5.0
    palette = ("#2166ac", "#11865a")
    publish(folder=directory, dx=spacing, colors=palette)
