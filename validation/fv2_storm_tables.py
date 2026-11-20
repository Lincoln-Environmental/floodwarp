"""Publish the outlet, inundation, reference and runtime comparisons from complete storm records."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_figures import rows_table
from validation.fv2_storm_evidence import fallback


def external_hydrograph(*, directory, archived_csv, grids, methods):
    series = np.genfromtxt(archived_csv, delimiter=",", names=True)
    final_s = 48 * 3600
    event = series[series["interval_end_s"] <= final_s]
    ends = event["interval_end_s"]
    interval = ends - event["interval_start_s"]
    ref = event["lisflood_fv1_mean_net_outlet_m3_s"]
    ref_volume = float(np.dot(ref, interval))
    ref_peak = float(ref.max())
    peak_end = float(ends[np.argmax(ref)])
    rows = []
    for dx in grids:
        for method in methods:
            record = json.loads((directory / f"{dx}m_{method}.json").read_text())
            if record["ledger"]["boundary_inflow"] != 0:
                raise ValueError("Gross and net coastal flows differ in this event")
            np.testing.assert_array_equal(ends, [x["t"] for x in record["hydrograph"]])
            hydro = np.array([x["outflow_m3_s"] for x in record["hydrograph"]])
            rows.append(
                dict(
                    dx=dx,
                    method=method,
                    difference_rmse_m3_s=float(np.sqrt(np.mean((hydro - ref) ** 2))),
                    peak_difference_m3_s=float(hydro.max() - ref_peak),
                    peak_time_difference_min=float(ends[np.argmax(hydro)] - peak_end)
                    / 60,
                    net_export_difference_m3=float(np.dot(hydro - ref, interval)),
                )
            )
    report_json(
        path=directory / "lisflood_comparison.json",
        content=dict(
            source=str(archived_csv),
            source_sha256=hashlib.sha256(archived_csv.read_bytes()).hexdigest(),
            series="lisflood_fv1_mean_net_outlet_m3_s",
            minutes=len(ends),
            peak_m3_s=ref_peak,
            peak_interval_end_s=peak_end,
            export_m3=ref_volume,
            rows=rows,
        ),
    )
    return rows_table(
        title="Outlet disagreement with the archived 5 m LISFLOOD-FP FV1 simulation",
        columns=[
            "Grid (m)",
            "Scheme",
            "Hydrograph RMSE (m³/s)",
            "Peak change (m³/s)",
            "Peak time change (min)",
            "Export change (m³)",
        ],
        rows=[list(r.values()) for r in rows],
        note=f"The independent model used the same design event. Its first 48 hours exported {ref_volume:,.1f} m³ and peaked at {ref_peak:.3f} m³/s. These differences quantify agreement between models. There is no measured storm hydrograph.",
    )


def plot_cost(*, directory, common, timings, methods):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ("#2166ac", "#d89000", "#9a4c97", "#11865a")
    cost = {(r["dx"], r["method"]): 1000 * r["median_wall_s"] for r in timings}
    for ax, key, scale, ylabel in zip(
        axes,
        ("peak_envelope_rmse_m", "hydrograph_rmse_m3_s"),
        (1000, 1),
        ("Maximum-depth difference (mm RMSE)", "Outlet difference (m³/s RMSE)"),
        strict=True,
    ):
        for method, color in zip(methods, colors, strict=True):
            points = sorted(
                (r["dx"], cost[r["dx"], method], scale * r["vs_5m_contact"][key])
                for r in common
                if r["method"] == method
            )
            ax.plot(
                [x for _, x, _ in points],
                [y for _, _, y in points],
                "o-",
                color=color,
                label=method,
            )
            for dx, x, y in points:
                ax.annotate(
                    f"{dx:g}",
                    (x, y),
                    xytext=(2, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.set(
            xscale="log", xlabel="Warm 12-second wet replay on GPU (ms)", ylabel=ylabel
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(directory / "storm_cost.png", dpi=165)
    plt.close(fig)


def publish(*, directory, grids, methods, archived_csv):
    rows = []
    shares = []
    for dx in grids:
        for method in methods:
            record = json.loads((directory / f"{dx}m_{method}.json").read_text())
            if record["time_s"] != 48 * 3600 or len(record["hydrograph"]) != 2880:
                raise ValueError("Incomplete storm")
            ledger = record["ledger"]
            rows.append(
                [
                    dx,
                    method,
                    record["peak_outflow_m3_s"],
                    record["peak_interval_end_s"] / 3600,
                    ledger["boundary_outflow"],
                    ledger["infiltration"],
                    ledger["storage"],
                    record["max_envelope_wet_area_m2"] / 1.0e6,
                    record["wet_speed_p95_m_s"],
                    record["steps"],
                    sum(x["seconds"] for x in record["hydrograph"]),
                    ledger["residual"],
                ]
            )
            for sample in fallback(row=record):
                groups = sample["fallback_fraction_by_counts_volume_impulse"]
                if sample["hour"] == 24:
                    shares.append(
                        [
                            dx,
                            method,
                            *[
                                100 * f if f is not None else "—"
                                for f in (
                                    *groups["all"],
                                    groups["jointly_wet"][1],
                                    groups["channel_proxy"][1],
                                    groups["jump_proxy"][1],
                                )
                            ],
                        ]
                    )
    sections = [
        rows_table(
            title="Whole-domain 48-hour Orton Bradley HIRDS storm",
            columns=[
                "Grid (m)",
                "Solver",
                "Peak (m³/s)",
                "Peak interval end (h)",
                "Export (m³)",
                "Infiltration (m³)",
                "Storage at 48 h (m³)",
                "Envelope >1 cm (km²)",
                "Wet speed p95 at 48 h (m/s)",
                "Steps",
                "Recorded loop time (s)",
                "Ledger residual (m³)",
            ],
            rows=rows,
            note="Each run received 3,809,237.012 m³ of rainfall from identical HIRDS time nodes. Export is gross outward boundary transport integrated from minute zero. Recorded loop time uses each event's actual output windows. GPU sharing and archive dates differ across event records. Use the exclusive warm repeats below to compare implementation cost.",
        )
    ]
    common = json.loads((directory / "common_support.json").read_text())
    fields = []
    for r in common:
        m = r["vs_5m_contact"]
        fields.append(
            [
                r["dx"],
                r["method"],
                m["depth_rmse_m"],
                m["wet_depth_rmse_m"],
                m["peak_envelope_rmse_m"],
                m["final_wet_iou"],
                m["envelope_wet_iou"],
                m["hydrograph_rmse_m3_s"],
                m["arrival_mae_h"],
                m["peak_time_delta_min"],
            ]
        )
    sections.append(
        rows_table(
            title="Difference from 5 m FV1 contact-HLL on its original active footprint",
            columns=[
                "Grid (m)",
                "Solver",
                "Final h RMSE (m)",
                "Wet h RMSE (m)",
                "Envelope RMSE (m)",
                "Final IoU",
                "Envelope IoU",
                "Hydrograph RMSE (m³/s)",
                "Arrival MAE (h)",
                "Peak time difference (min)",
            ],
            rows=fields,
            note="All model fields are transferred conservatively to common 5 m support. Wet area uses >1 cm. Arrival is bracketed by 2-hour maximum-depth envelopes and compared only where both runs flood. The 5 m simulation supplies a model comparator. It has no claim to a converged native-resolution solution or an observed hydrograph.",
        )
    )
    sections.append(
        rows_table(
            title="HLL fallback during the minute ending at storm hour 24",
            columns=[
                "Grid (m)",
                "FV1 scheme",
                "Faces (%)",
                "Abs. discharge (%)",
                "Momentum impulse (%)",
                "Jointly wet discharge (%)",
                "Channel proxy discharge (%)",
                "Jump proxy discharge (%)",
            ],
            rows=shares,
            note="Each fraction sums shallow and inadmissible HLL paths. Jointly wet means both reconstructed depths ≥ 0.01 m. The fixed channel proxy at each sample start requires h ≥ 0.1 m and |q| ≥ 0.05 m²/s. The jump proxy uses a convergent supercritical-to-subcritical transition. The JSON retains all five depth bands, the individual reasons, and four event samples.",
        )
    )
    diag = json.loads((directory / "muscl_fallback.json").read_text())
    sparse = []
    for r in diag:
        a = np.array(r["bands"])
        fractions = np.divide(
            a[:, 1:].sum(axis=(0, 1)),
            a.sum(axis=(0, 1)),
            out=np.zeros(2),
            where=a.sum(axis=(0, 1)) > 0,
        )
        if r["start_h"] == 24:
            sparse.append(
                [
                    r["dx"],
                    *list(100 * fractions),
                    *list(100 * a.sum(axis=0)[1:, 1] / a[:, :, 1].sum()),
                ]
            )
    sections.append(
        rows_table(
            title="FV2 reconstruction during the minute beginning at storm hour 24",
            columns=[
                "Grid (m)",
                "Fallback face evaluations (%)",
                "Fallback absolute volume (%)",
                "Boundary volume (%)",
                "Positivity volume (%)",
                "Dry-stencil volume (%)",
            ],
            rows=sparse,
            note="Classification uses both adjacent stencils and accumulated donor-limited absolute flux, half-weighted at each RK stage. A slope set to zero by ordinary limiter extrema is included in the unrestricted-stencil category.",
        )
    )
    cost = []
    timings = json.loads((directory / "wet_cost.json").read_text())
    for r in timings:
        cost.append(
            [
                r["dx"],
                r["method"],
                1000 * r["median_wall_s"],
                r["steps"],
                r["min_dt_s"],
                r["mean_dt_s"],
                r["cell_steps_s"] / 1.0e6,
                r["pool_used_bytes"] / 1.0e6,
                1000 * r["graph_capture_s"],
            ]
        )
    sections.append(
        rows_table(
            title="Independent GPU timing, full footprint at storm hour 25",
            columns=[
                "Grid (m)",
                "Solver",
                "12 s hydraulic replay (ms)",
                "Steps",
                "Minimum dt (s)",
                "Mean dt (s)",
                "M cell-steps/s",
                "CUDA pool used (MB)",
                "First capture (ms)",
            ],
            rows=cost,
            note="All methods start from the same conservative state for their resolution, on the same RTX 4070 Ti SUPER in dense mode. Five synchronised graph runs follow warm-up. Every reset reproduced its first final state bitwise. The raw file includes per-kernel CUDA timings.",
        )
    )
    plot_cost(directory=directory, common=common, timings=timings, methods=methods)
    sections.append(
        external_hydrograph(
            directory=directory, archived_csv=archived_csv, grids=grids, methods=methods
        )
    )
    (directory / "RESULTS.md").write_text(
        "# Orton Bradley design-storm results\n\n" + "\n".join(sections)
    )


if __name__ == "__main__":
    directory = Path("validation/fv2_review/hirds48")
    grids = (5, 10, 20)
    methods = ("contact_hll", "roe", "hllc", "fv2_mc")
    archived_csv = Path("validation/sparse_modified_event_review/hydrographs_72h.csv")
    publish(
        directory=directory, grids=grids, methods=methods, archived_csv=archived_csv
    )
