"""Compare transfer forecasts with the release's measured design-storm hydrographs."""

import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import metrics, runner


def release_curves(*, file: Path, end_s: int, cadence_s: int):
    rows = [row for row in metrics.csv_read(file=file) if float(row["interval_end_s"]) <= end_s]
    ends = np.arange(cadence_s, end_s + 1, cadence_s)
    np.testing.assert_array_equal([float(row["interval_end_s"]) for row in rows], ends)
    np.testing.assert_array_equal([float(row["interval_start_s"]) for row in rows], ends - cadence_s)
    curves = {}
    for method in ("floodwarp_fv1", "lisflood_fv1", "floodwarp_li"):
        q = np.array([float(row[method + "_mean_net_outlet_m3_s"]) for row in rows])
        volume = np.array([float(row[method + "_cumulative_net_export_m3"]) for row in rows])
        if not np.all(np.isfinite(q)):
            raise ValueError("Nonfinite discharge in archived release control")
        np.testing.assert_allclose(np.cumsum(q) * cadence_s, volume, rtol=1e-10, atol=1e-7)
        curves[method] = q
    return ends, curves


def compare(*, release: Path, destination: Path, settings, rows):
    file = release / "outputs/orton_bradley_hydrographs_72h.csv"
    digest = runner.checksum(file)
    contrasts = []
    for row in rows:
        cadence = int(row["interval_s"])
        if cadence != settings["cadence_s"]:
            raise ValueError("Transfer cadence differs from the prescribed release comparison")
        ends, curves = release_curves(file=file, end_s=int(row["duration_h"] * 3600), cadence_s=cadence)
        frames = []
        for hour in range(settings["checkpoint_h"], row["duration_h"] + 1, settings["checkpoint_h"]):
            receipt = destination / row["id"] / f"hour_{hour}.json"
            frames.extend(json.loads(receipt.read_text())["frames"])
        np.testing.assert_array_equal([r["t_s"] for r in frames], ends)
        model = np.array([r["net_outlet_m3s"] for r in frames])
        np.testing.assert_allclose(model.sum() * cadence, row["boundary_outflow"] - row["boundary_inflow"],
                                   rtol=1e-10, atol=1e-7)
        for method, control in curves.items():
            peak = float(control.max())
            contrasts.append(dict(
                run_id=row["id"], method=row["method"], dx_m=row["dx_m"], interval_s=cadence,
                end_h=row["duration_h"], historical_method=method, historical_dx_m=5,
                release_hydrographs_sha256=digest, historical_peak_m3s=peak,
                historical_peak_end_s=int(ends[np.argmax(control)]),
                historical_net_export_m3=float(control.sum() * cadence),
                signed_peak_difference_pct=float(100 * (model.max() / peak - 1)) if peak > 0 else None,
                peak_end_difference_s=int(ends[np.argmax(model)] - ends[np.argmax(control)]),
                net_export_difference_m3=float(np.sum(model - control) * cadence),
                hydrograph_rms_difference_m3s=float(np.sqrt(np.mean((model - control)**2))),
            ))
    metrics.csv_write(file=destination / "historical_transfer.csv", rows=contrasts)
