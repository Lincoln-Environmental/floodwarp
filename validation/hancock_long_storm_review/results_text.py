"""Build the numerical comparison tables from matched full-duration runs."""

import csv
import json
from pathlib import Path

import numpy as np


def read(*, file):
    with file.open() as stream:
        return list(csv.DictReader(stream))


def table(*, headers, rows):
    text = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    return "\n".join(text + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def hancock_fv2(*, accuracy, uncertainty):
    primary = {
        (r["landform"], r["storm"], r["method"]): r
        for r in accuracy
        if r["group"] == "primary"
        and r["representation"] == "sampled"
        and (
            r["method"] == "fv2" or (r["method"] == "hancock" and r["mode"] == "tiles")
        )
    }
    pairs = [
        (primary[land, storm, "hancock"], row)
        for (land, storm, method), row in primary.items()
        if method == "fv2"
    ]
    references = {
        (r["landform"], r["storm"]): r
        for r in uncertainty
        if r["representation"] == "sampled"
    }
    rows = []
    for label, metric, delta, resolved in (
        ("Final depth (mm)", "depth_rmse_m", "depth_difference_m", "depth_resolved"),
        (
            "Outlet discharge (m³/s)",
            "outlet_rmse_m3s",
            "outlet_difference_m3s",
            "discharge_resolved",
        ),
    ):
        counts = dict(hancock=0, fv2=0, inside=0, unresolved=0)
        errors = []
        for h, f in pairs:
            ref = references[h["landform"], h["storm"]]
            eh, ef = float(h[metric]), float(f[metric])
            errors.append((eh, ef))
            band = 2.0 * float(ref[delta])
            if ref[resolved] != "True":
                counts["unresolved"] += 1
            elif eh + band < ef:
                counts["hancock"] += 1
            elif ef + band < eh:
                counts["fv2"] += 1
            else:
                counts["inside"] += 1
        scale = 1000.0 if metric == "depth_rmse_m" else 1.0
        medians = np.median(errors, axis=0) * scale
        rows.append(
            [
                label,
                len(pairs),
                f"{medians[0]:.4g}",
                f"{medians[1]:.4g}",
                counts["hancock"],
                counts["fv2"],
                counts["inside"],
                counts["unresolved"],
            ]
        )
    return table(
        headers=[
            "Metric",
            "Pairs",
            "Hancock median RMSE",
            "FV2 median RMSE",
            "Hancock lower",
            "FV2 lower",
            "Within refinement margin",
            "Reference unresolved",
        ],
        rows=rows,
    )


def comparisons(*, review):
    land = read(file=review / "landform_tradeoffs.csv")
    accuracy = read(file=review / "accuracy.csv")
    uncertainty = read(file=review / "reference_uncertainty.csv")
    trades = read(file=review / "solver_tradeoffs.csv")
    lines = [
        "# Full-event method and grid comparisons",
        "The small tests use a common 240 × 160 m domain. All errors use area-mean depths on a common 10 m support and interval-mean discharge. A 1 m forecast compared against a 1 m FV2 reference tests formulation differences at that resolution. Channels narrower than the evaluation support can have greater local errors than these averages indicate.",
        "## Landform comparisons",
        "The entries are medians over the six rain events at 10 m. Depth and outlet columns are absolute differences from the finest available FV2 solution for the same physical case. Reference uncertainty varies by event.",
    ]
    rows = []
    for r in land:
        rows.append(
            [
                r["landform"].capitalize(),
                f"{float(r['runtime_ratio']):.2f}",
                *(
                    f"{float(r[k]):.2f}"
                    for k in ("fv1_depth_rmse_mm", "hancock_depth_rmse_mm")
                ),
                *(
                    f"{float(r[k]):.4f}"
                    for k in ("fv1_outlet_rmse_m3s", "hancock_outlet_rmse_m3s")
                ),
                f"{float(r['sampled_iou_1cm']):.2f}",
            ]
        )
    lines.append(
        table(
            headers=[
                "Landform",
                "Hancock/FV1 time",
                "FV1 depth (mm)",
                "Hancock depth (mm)",
                "FV1 outlet (m³/s)",
                "Hancock outlet (m³/s)",
                "Hancock 1 cm space–time IoU",
            ],
            rows=rows,
        )
    )
    lines += [
        "## Resolution of the reference",
        "For a depth comparison, successive refinements must differ by at most 1 mm RMSE or 2% relative L2. For discharge they must differ by at most 5% of the finer peak over the whole hydrograph, with an exact match when the reference has zero flow. These are engineering convergence screens. They do not bound a shared formulation or boundary error. The storm comparisons use FV2 grid refinement, plus independent formulations at 1 m on selected cases. The analytical controls have separate exact references.",
    ]
    for rep in ("sampled", "fixed"):
        subset = [r for r in uncertainty if r["representation"] == rep]
        if subset:
            both = sum(r["both_resolved"] == "True" for r in subset)
            lines.append(
                f"{rep.capitalize()} terrain passed both screens on {both} of {len(subset)} cases."
            )
    lines.append(
        "For a pairwise advantage we additionally require the difference of the two forecast error norms to exceed twice the observed fine-grid refinement difference. The triangle inequality motivates this margin. A resolution difference remains an estimator of uncertainty. It cannot guarantee proximity to the continuum solution."
    )
    rows = []
    for m in ("hancock", "fv2", "li"):
        sample = [r for r in trades if r["method"] == m]
        if sample:
            z = []
            for variable, gate in (
                ("depth_m_advantage", "ref_depth_resolved"),
                ("hydrograph_m3s_advantage", "ref_discharge_resolved"),
            ):
                z.extend(
                    sum(r[variable] == x and r[gate] == "True" for r in sample)
                    for x in ("method", "fv1")
                )
            rows.append(
                [
                    m,
                    len(sample),
                    f"{np.median([float(r['runtime_ratio_to_fv1']) for r in sample]):.2f}",
                    *z,
                ]
            )
    lines.append(
        table(
            headers=[
                "Method",
                "Pairs vs FV1",
                "Median runtime ratio",
                "Method lower depth error",
                "FV1 lower depth error",
                "Method lower outlet error",
                "FV1 lower outlet error",
            ],
            rows=rows,
        )
    )
    lines += [
        "## Direct comparison of adaptive Hancock and adaptive FV2",
        "The selected 10 m cases have equal terrain, forcing and output cadence. Both methods are assessed against the same finer numerical reference. A lower error is credited only after the corresponding reference passes its refinement screen and the gap exceeds twice the observed refinement difference. Small differences can also arise because the two solvers share their spatial reconstruction and face flux.",
        hancock_fv2(accuracy=accuracy, uncertainty=uncertainty),
    ]
    lines += [
        "## Hydraulic grid and DEM representation",
        "The moderate storm uses the same domain and end time at each spacing. Each entry gives hydraulic runtime in seconds, final-depth RMSE in millimetres and outlet hydrograph RMSE in m³/s. Runtime includes the production timestep policy. The final-depth RMSE can be small even where peak depth or connectivity differs. Both are also available in the full metric table.",
    ]
    for rep in ("sampled", "fixed"):
        rows = []
        for case in ("plain", "fan", "incised"):
            for method in ("fv1", "hancock", "fv2", "li"):
                trio = []
                for dx in ("1", "5", "10"):
                    hit = next(
                        (
                            r
                            for r in accuracy
                            if r["group"] in ("primary", "refinement")
                            and r["representation"] == rep
                            and r["method"] == method
                            and r["landform"] == case
                            and r["storm"] == "moderate"
                            and r["dx"] == dx
                            and (method != "hancock" or r["mode"] == "tiles")
                        ),
                        None,
                    )
                    if hit:
                        error = (
                            "ref"
                            if hit["run"] == hit["reference"]
                            else f"{1000 * float(hit['depth_rmse_m']):.3g} / {float(hit['outlet_rmse_m3s']):.3g}"
                        )
                        trio.append(f"{float(hit['runtime_s']):.2f} / {error}")
                    else:
                        trio.append("—")
                if trio != ["—"] * 3:
                    rows.append([case.capitalize(), method, *trio])
        lines.append(
            table(
                headers=[rep.capitalize() + " terrain", "Method", "1 m", "5 m", "10 m"],
                rows=rows,
            )
        )
    lines += [
        "## Independent analytical rainfall and recession",
        "Over a frictional uniform sheet, exact depth is the time integral of rain plus initial depth. The reciprocal discharge follows the integral of Manning drag divided by the depth to the 7/3 power. A central section stays outside the boundary signal cone. The analytical comparison integrates interval transport per unit width across six hours of rain and two further hours. The kinematic conditions give both the SWE and LI the same governing interior solution.",
    ]
    sheet = json.loads((review / "sheet_cycle.json").read_text())
    lines.append(
        table(
            headers=[
                "Method",
                "Depth RMSE (m)",
                "Section-discharge RMSE (m²/s)",
                "Final transport error (m²)",
            ],
            rows=[
                [
                    r["method"],
                    f"{r['depth_rmse_m']:.3g}",
                    f"{r['section_discharge_rmse_m2s']:.3g}",
                    f"{r['final_volume_error_m2']:.3g}",
                ]
                for r in sheet
            ],
        )
    )
    lines.append(
        "Maxima, wet-area departures, arrival and recession, threshold-specific IoU, connected components, internal sections, outlet volume and peak timing are available for every grid pair. A missing threshold crossing has an empty arrival/recession RMSE. An empty inundation union has IoU one. Peak timing is the end of the largest output interval. [Full accuracy table](accuracy.csv) · [Reference differences](reference_uncertainty.csv) · [Pairwise comparisons](solver_tradeoffs.csv) · [Runtime and depth](accuracy_cost.png) · [Discharge at fixed and resampled terrain](resolution_discharge.png)"
    )
    (review / "COMPARISONS.md").write_text("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    comparisons(review=review)
