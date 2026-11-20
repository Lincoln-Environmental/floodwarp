"""Measure separate solver and grid sensitivities of the complete Orton design storm."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_figures import rows_table
from validation.fv2_storm_evidence import on_footprint, read_snapshot


def paired_effects(*, directory, grids, methods, output):
    base = min(grids)
    baseline_path = directory / f"{base}m_contact_hll.json"
    baseline = json.loads(baseline_path.read_text())
    _, _, mask = read_snapshot(path=Path(baseline["origin"]))
    inputs = {}
    for dx in grids:
        for method in methods:
            entry = json.loads((directory / f"{dx}m_{method}.json").read_text())
            if entry["time_s"] != 48 * 3600:
                raise ValueError(f"Incomplete storm for {method} at {dx} m")
            origin = Path(entry["origin"])
            if method == "fv2_mc":
                with np.load(origin) as archive:
                    configuration = json.loads(str(archive["metadata"]))["configuration"]
                if configuration["reconstruction"] != "muscl_ordered_bed_velocity_v2":
                    raise ValueError(f"Superseded reconstruction at {origin}")
            q, maximum, _ = read_snapshot(path=origin)
            hydrograph = np.array([r["outflow_m3_s"] for r in entry["hydrograph"]])
            endpoints = np.array([r["t"] for r in entry["hydrograph"]])
            np.testing.assert_array_equal(endpoints, np.arange(60, entry["time_s"] + 1, 60))
            inputs[dx, method] = dict(
                peak=entry["peak_outflow_m3_s"],
                hydrograph=hydrograph,
                depth=on_footprint(field=q[..., 0], dx=dx, base=base, shape=mask.shape)[mask],
                envelope=on_footprint(field=maximum, dx=dx, base=base, shape=mask.shape)[mask],
            )
    rows = []
    for dx in grids:
        for left, right in combinations(methods, 2):
            rows.append(change(first=(dx, left), second=(dx, right), inputs=inputs))
    for left, right in combinations(grids, 2):
        for method in methods:
            rows.append(change(first=(left, method), second=(right, method), inputs=inputs))
    report_json(path=output, content=rows)
    table = []
    for row in rows:
        a, b = row["first"], row["second"]
        table.append(
            [
                f"{a[0]} m {a[1]}",
                f"{b[0]} m {b[1]}",
                row["peak_change_m3_s"],
                row["hydrograph_rmse_m3_s"],
                1000 * row["depth_rmse_m"],
                1000 * row["envelope_rmse_m"],
            ]
        )
    return rows_table(
        title="Sensitivity to FV formulation and grid spacing in the full Orton storm",
        columns=[
            "First run",
            "Second run",
            "Peak change (m³/s)",
            "Hydrograph RMSE (m³/s)",
            "End depth RMSE (mm)",
            "Maximum depth RMSE (mm)",
        ],
        rows=table,
        note="Peak change is second minus first. Discharge RMSE uses all 2,880 one-minute intervals. Spatial RMSE uses the original active 5 m footprint and piecewise-constant transfer of coarser predictions. Fixed-grid comparisons share all represented inputs. Changing grid also changes represented terrain, mask, rain distribution and coastal openings. Each value measures disagreement between simulations.",
    )


def change(*, first, second, inputs):
    a, b = inputs[first], inputs[second]
    return dict(
        first=first,
        second=second,
        peak_change_m3_s=b["peak"] - a["peak"],
        hydrograph_rmse_m3_s=rms_difference(a=a["hydrograph"], b=b["hydrograph"]),
        depth_rmse_m=rms_difference(a=a["depth"], b=b["depth"]),
        envelope_rmse_m=rms_difference(a=a["envelope"], b=b["envelope"]),
    )


def rms_difference(*, a, b):
    return float(np.sqrt(np.mean(np.square(a - b))))


if __name__ == "__main__":
    directory = Path("validation/fv2_review/hirds48")
    grids = (5, 10, 20)
    methods = ("contact_hll", "roe", "hllc", "fv2_mc")
    table = paired_effects(
        directory=directory,
        grids=grids,
        methods=methods,
        output=directory / "resolution_effects.json",
    )
    (directory / "RESOLUTION_EFFECTS.md").write_text(table)
