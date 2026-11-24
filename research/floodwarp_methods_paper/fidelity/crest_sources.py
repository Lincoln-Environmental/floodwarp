"""Locate the discharge released from storage during a measured late crest."""

from pathlib import Path
import json

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity.figures import save


def budget(*, rows, peak_s, widths):
    selected = [
        r
        for r in rows
        if float(r["end_s"]) == peak_s and int(r["strip_width_m"]) in widths
    ]
    if {int(r["strip_width_m"]) for r in selected} != set(widths) or len(
        selected
    ) != len(widths):
        raise ValueError("Missing or duplicated crest-minute strip")
    output = []
    for row in selected:
        seconds = float(row["end_s"]) - float(row["start_s"])
        export = float(row["net_export_m3"])
        upstream = float(row["net_into_strip_m3"])
        local = float(row["net_storage_contribution_m3"])
        np.testing.assert_allclose(export, upstream + local, rtol=1e-9, atol=1e-8)
        output.append(
            dict(
                strip_width_m=int(row["strip_width_m"]),
                peak_interval_end_s=peak_s,
                interval_s=seconds,
                export_m3=export,
                outlet_m3s=export / seconds,
                net_upstream_m3=upstream,
                net_upstream_m3s=upstream / seconds,
                net_storage_supply_m3=local,
                net_storage_supply_m3s=local / seconds,
                net_storage_share_pct=100 * local / export if export > 0 else None,
                rain_m3=float(row["rain_m3"]),
                infiltration_m3=float(row["infiltration_m3"]),
                residual_m3=float(row["residual_m3"]),
            )
        )
    return output


def plot(*, folder, rows, peak_s, width, dx, window_min):
    import matplotlib.pyplot as plt

    band = [
        r
        for r in rows
        if int(r["strip_width_m"]) == width
        and abs(float(r["end_s"]) - peak_s) <= window_min * 60
    ]
    edges = np.r_[float(band[0]["start_s"]), [float(r["end_s"]) for r in band]]
    dt = np.diff(edges)
    fig, ax = plt.subplots(figsize=(6.8, 3.3), layout="constrained")
    for variable, name, colour in (
        ("net_export_m3", "Outlet", "#2a3d78"),
        ("net_into_strip_m3", f"Across face {width} m upstream", "#00775d"),
        ("net_storage_contribution_m3", "Net release from reach storage", "#ad5b18"),
    ):
        q = np.array([float(r[variable]) for r in band]) / dt
        ax.stairs(q, (edges - peak_s) / 60, color=colour, label=name, baseline=None)
    ax.set(
        xlabel="Minutes relative to end of peak interval",
        ylabel="Discharge (m³/s)",
        title=f"FV2 RK2 on the {dx} m grid",
    )
    ax.legend(fontsize=8)
    save(fig=fig, folder=folder, name=f"crest_budget_{dx}m")


def run(*, directory, settings):
    dest = directory / "native_tail"
    aggregate = []
    description = [
        "# Source of the late outlet crest",
        "",
        "These measurements cover the highest one-minute outlet discharge after hour eight in each replay. "
        "The upstream face spans the full catchment width. Water released from storage may have travelled from far upstream in earlier intervals.",
        "",
        "| Grid (m) | Peak interval ends (h:min) | Width of reach (m) | Outlet (m³/s) | Net inflow to reach (m³/s) | Net local release (m³/s) | Local share (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for receipt in sorted(dest.glob("*/replay.json")):
        data = json.loads(receipt.read_text())
        source = receipt.with_name("minute_strips.csv")
        if data["result_sha256"] != runner.checksum(source):
            raise ValueError("Archived minute sections failed checksum verification")
        rows = metrics.csv_read(file=source)
        peak = data["highest_post_window_outlet_interval_end_s"]
        dx = data["dx_m"]
        for r in budget(
            rows=rows, peak_s=peak, widths=settings["crest_report_widths_m"]
        ):
            aggregate.append(
                dict(
                    r,
                    dx_m=dx,
                    method=data["method"],
                    series_id=data["series"],
                    replay_id=receipt.parent.name,
                )
            )
            hours, rem = divmod(int(peak), 3600)
            description.append(
                f"| {dx} | {hours}:{rem // 60:02d} | {r['strip_width_m']} | {r['outlet_m3s']:.3f} | "
                f"{r['net_upstream_m3s']:.3f} | {r['net_storage_supply_m3s']:.3f} | {r['net_storage_share_pct']:.2f} |"
            )
        plot(
            folder=dest,
            rows=rows,
            peak_s=peak,
            width=settings["plot_width_m"],
            dx=dx,
            window_min=settings["minutes_either_side"],
        )
    metrics.csv_write(file=dest / "crest_minute_budgets.csv", rows=aggregate)
    description.extend(
        [
            "",
            "Local release is the decrease in reach storage plus rain less infiltration. "
            "The replay verified the original accepted step counts and the two original section hydrographs every minute. "
            "This budget locates the water in the numerical solution immediately before export. "
            "Outlet placement and terrain sensitivity determine how confidently that release represents the physical catchment.",
            "",
            "[Minute budgets](crest_minute_budgets.csv) · [Full recession budgets](README.md) · [Independent boundary controls](../native_outlet/README.md)",
        ]
    )
    (dest / "CREST.md").write_text("\n".join(description) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "crest_source_settings.json").read_text())
    run(directory=directory, settings=settings)
