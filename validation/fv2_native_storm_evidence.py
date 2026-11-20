"""Compare complete coastal-tile HIRDS storms on common survey support."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from validation.fv2_figures import rows_table
from validation.fv2_native_storm import support
from validation.fv1_riemann import report_json


def export_hydrographs(*, work, output):
    columns = []
    names = []
    endpoints = None
    for path in sorted(work.glob("dem*_factor*_*.npz")):
        with np.load(path) as archive:
            record = json.loads(str(archive["record"]))
        values = np.array(record["hydrograph_m3_s"])
        times = record["sample_s"] * np.arange(1, len(values) + 1)
        if times[-1] != record["time_s"] or record["time_s"] != 48 * 3600:
            raise ValueError(f"Incomplete design storm at {path}")
        np.testing.assert_allclose(
            sum(values) * record["sample_s"], record["ledger"]["boundary_outflow"], rtol=2e-12
        )
        if endpoints is None:
            endpoints = times
            columns.append(times)
            names.append("interval_end_s")
        else:
            np.testing.assert_array_equal(times, endpoints)
        columns.append(values)
        names.append(path.stem + "_m3_s")
    if endpoints is None:
        raise ValueError(f"No complete HIRDS coastal tile records in {work}")
    np.savetxt(
        output, np.column_stack(columns), delimiter=",", fmt="%.17g", header=",".join(names), comments=""
    )


def connectivity(*, source, work, spacings, wet, factor):
    row, col, height, width = source["north_first_5m_window"]
    with np.load(source["source"]) as archive:
        coarse = archive["active_mask"][row : row + height, col : col + width][::-1]
        mask = np.repeat(np.repeat(coarse, 5, axis=0), 5, axis=1).astype(bool)
    with np.load(work / f"dem1_factor{factor}_mc_fine.npz") as archive:
        expected = support(field=archive["maximum"], dx=1 / factor)
    truth = (expected > wet) & mask
    results = []
    for dx in spacings:
        for m in ("mc", "fv1"):
            with np.load(work / f"dem{dx}_factor1_{m}_base.npz") as archive:
                peak = support(field=archive["maximum"], dx=dx)
            wet_cells = (peak > wet) & mask
            union = np.count_nonzero(wet_cells | truth)
            results.append(
                dict(
                    dx_m=dx,
                    method=m,
                    event_wet_iou=np.count_nonzero(wet_cells & truth) / union if union else 1.0,
                    event_wet_area_change_m2=int(wet_cells.sum() - truth.sum()),
                    event_components=int(label(wet_cells)[1]),
                    reference_components=int(label(truth)[1]),
                )
            )
    return results


def compare_storms(*, directory, work, wet, spacings, reference_factors):
    export_hydrographs(work=work, output=directory / "native_tile_hydrographs.csv")
    source = json.loads((directory / "native_hirds48.json").read_text())
    records = source["rows"]

    def pick(*, dx, hydraulic, method, half=False):
        matches = [
            r
            for r in records
            if r["terrain_dx_m"] == dx and r["hydraulic_dx_m"] == hydraulic and r["method"] == method
        ]
        return matches[-1] if half else matches[0]

    rows = []
    comparisons = []
    for dx in spacings:
        first = pick(dx=dx, hydraulic=dx, method="fv1")
        second = pick(dx=dx, hydraulic=dx, method="mc")
        a = first["vs_fine_mc"]
        b = second["vs_fine_mc"]
        improvements = {}
        for metric in (
            "depth_rmse_m",
            "wet_depth_rmse_m",
            "peak_envelope_rmse_m",
            "hydrograph_rmse_m3_s",
        ):
            candidates = [
                first[k][metric] - second[k][metric] for k in ("vs_fine_mc", "vs_fine_fv1", "vs_half_dt")
            ]
            improvements[metric] = dict(
                improvement=candidates[0],
                reference_range=[min(candidates), max(candidates)],
                percent_of_coarse_fv1_error=100 * candidates[0] / a[metric] if a[metric] else None,
            )
        comparisons.append(dict(dx_m=dx, improvements=improvements))
        for r, m in ((first, a), (second, b)):
            rows.append(
                [
                    dx,
                    r["method"],
                    m["depth_rmse_m"] * 1000,
                    m["peak_envelope_rmse_m"] * 1000,
                    m["wet_iou"],
                    r["peak_m3_s"] * 1000,
                    r["peak_interval_end_s"] / 3600,
                    r["ledger"]["boundary_outflow"],
                    m["hydrograph_rmse_m3_s"] * 1000,
                ]
            )
    fine = 1 / reference_factors[-1]
    mc = pick(dx=1, hydraulic=fine, method="mc")
    last = pick(dx=1, hydraulic=1 / reference_factors[-2], method="mc")["vs_fine_mc"]
    other = pick(dx=1, hydraulic=fine, method="fv1")["vs_fine_mc"]
    half = pick(dx=1, hydraulic=fine, method="mc", half=True)
    if half["cap_s"] != 0.5 * mc["cap_s"]:
        raise ValueError("Reference timestep caps do not form a halving pair")
    time = mc["vs_half_dt"]
    report_json(
        path=directory / "native_storm_improvement.json",
        content=dict(
            common_support_m=1,
            comparisons=comparisons,
            reference_last_mc_refinement=last,
            reference_cross_formulation=other,
            reference_dt_halving=time,
            timestep_check=dict(
                base_cap_s=mc["cap_s"],
                half_cap_s=half["cap_s"],
                base_steps=mc["steps"],
                half_cap_steps=half["steps"],
                base_mean_accepted_dt_s=mc["time_s"] / mc["steps"],
                half_cap_mean_accepted_dt_s=half["time_s"] / half["steps"],
            ),
            connectivity=connectivity(
                source=source,
                work=work,
                spacings=spacings,
                wet=wet,
                factor=reference_factors[-1],
            ),
        ),
    )
    table = rows_table(
        title="Full 48-hour HIRDS forcing on the native coastal tile",
        columns=[
            "Grid (m)",
            "Method",
            "Final depth RMSE (mm)",
            "Maximum depth RMSE (mm)",
            "Final wet IoU",
            "Peak outlet (L/s)",
            "Peak interval end (h)",
            "Export (m³)",
            "Hydrograph RMSE (L/s)",
        ],
        rows=rows,
        note=f"The surveyed tile receives {source['tile_rain_volume_m3']:.3f} m³ of the original spatially scaled Orton rain. Cut edges are closed except mapped stage-zero coastal faces. The tile has {source['active_area_m2']} m² of original active footprint. Outlet samples average 15 minutes. All spatial errors use common active 1 m support relative to the finest {fine:g} m MC hydraulic grid on the fixed 1 m DEM.",
    )
    checks = rows_table(
        title="Sensitivity of the full-event coastal-tile numerical reference",
        columns=[
            "Check",
            "Final depth RMSE (mm)",
            "Maximum-depth RMSE (mm)",
            "Outlet RMSE (L/s)",
        ],
        rows=[
            [
                name,
                1000 * data["depth_rmse_m"],
                1000 * data["peak_envelope_rmse_m"],
                1000 * data["hydrograph_rmse_m3_s"],
            ]
            for name, data in [
                ("Last MC refinement", last),
                ("Finest FV1 vs MC", other),
                ("Finest MC cap halving", time),
            ]
        ],
        note=f"All fields use common active 1 m support. The original and halved timestep caps were {mc['cap_s']:g} and {half['cap_s']:g} s, with {mc['steps']:,} and {half['steps']:,} accepted steps. The device wave-speed restriction can set an even smaller step in wet regions.",
    )
    (directory / "NATIVE_STORM.md").write_text(table + "\n" + checks)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = {"fv1": "#2166ac", "mc": "#11865a"}
    for m in colors:
        values = [pick(dx=dx, hydraulic=dx, method=m) for dx in spacings]
        axes[0].plot(
            spacings,
            [1000 * r["vs_fine_mc"]["peak_envelope_rmse_m"] for r in values],
            "o-",
            label=m,
            color=colors[m],
        )
        for r in values:
            if r["hydraulic_dx_m"] in (1, 5, 10):
                hydro = np.array(r["hydrograph_m3_s"]) * 1000
                x = np.arange(1, len(hydro) + 1) * r["sample_s"] / 3600
                style = {1: "-", 5: "--", 10: ":"}[r["hydraulic_dx_m"]]
                axes[1].plot(
                    x,
                    hydro,
                    color=colors[m],
                    linestyle=style,
                    label=f"{m} {r['hydraulic_dx_m']:g} m",
                )
    axes[0].axhline(
        1000 * last["peak_envelope_rmse_m"],
        color="gray",
        ls=":",
        label="Last MC refinement",
    )
    axes[0].set(
        xlabel="DEM and hydraulic spacing (m)",
        ylabel="Maximum-depth RMSE (mm)",
        xticks=spacings,
    )
    axes[1].set(xlabel="Storm time (hours)", ylabel="Coastal tile outlet (L/s)", xlim=(0, 48))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(directory / "native_storm.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    work = Path("outputs/fv2_hirds_native_tile")
    spacings = (1, 2, 5, 10, 20)
    reference_factors = (2, 4)
    compare_storms(
        directory=directory,
        work=work,
        wet=0.01,
        spacings=spacings,
        reference_factors=reference_factors,
    )
