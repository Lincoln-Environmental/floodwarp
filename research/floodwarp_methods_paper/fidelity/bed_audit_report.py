"""Summarise the verified checkpoints, plane experiment and nested DEM resolution controls."""

import json
from pathlib import Path
from research.floodwarp_methods_paper import metrics


def run(*, directory):
    folder = directory / "incised_crest"
    dem = json.loads((folder / "mean_dem_inputs.json").read_text())
    reach = metrics.csv_read(file=folder / "reach_replay.csv")
    margin = max(float(r["margin_m"]) for r in reach)
    plane = metrics.csv_read(file=folder / "plane_controls.csv")
    sections = metrics.csv_read(file=folder / "checkpoint_spatial_differences.csv")
    text = [
        "# Incised catchment DEM correction",
        "",
        "The earlier refinement copied a 10 m centre elevation into its four 5 m children. The limiter can reconstruct a nonzero slope between adjacent parent centres on the 10 m grid. On the child grid each pair contains a flat tread and a jump. The experiment changes the bed gradient seen by the flux and source operators. Its depths, storage and discharge describe a staircase sensitivity experiment. We withdraw its use for choosing practical grid resolution or ranking solver accuracy.",
        "",
        f"The corrected {dem['inputs']['dimensions_m'][0]:,} × {dem['inputs']['dimensions_m'][1]:,} m terrain uses {dem['native_1m_cells']:,} native metre elevations. Each 5 m elevation is the mean of 25 native cells. Each 10 m elevation is the mean of the four 5 m cells covering that footprint. Every 10 m elevation was independently checked against its 100 native cells. Maximum absolute difference was {dem['max_difference_direct_1_to_10_m']:.3g} m. Only the short reach replay holds the original forcing support fixed to isolate the effect of bed construction. [DEM identities](incised_crest/mean_dem_inputs.json)",
        "",
        "## Numerical mechanism",
        "",
        "The plane test starts a 3 cm uniform sheet from rest on a 10% slope with Manning roughness 0.035. The infinite smooth plane has constant depth and obeys du/dt = gS − g n² u² / h^(4/3). Values below are mass-weighted interior velocities after two seconds. The smooth-plane solution is a reference for the smooth geometry. The staircase has a different bed source term.",
        "",
        "| Method | Grid (m) | Bed spacing (m) | Velocity (m/s) | Smooth-plane exact (m/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in plane:
        text.append(
            f"| {r['method']} | {r['grid_m']} | {r['represented_bed_m']} | {float(r['interior_velocity_ms']):.4f} | {float(r['smooth_plane_exact_ms']):.4f} |"
        )
    text.extend(
        [
            "",
            "## Replay on the incised terrain",
            "",
            f"The same 400 m internal cross-section and first 60 seconds of the storm were replayed with FV2 on CPU using the actual initial water and momentum, identical forcing and a common timestep cap. The table uses a {margin:g} m buffer between the measured section and each external reach boundary. The archive includes both buffer sizes. This short replay tests the cause of the initial discrepancy.",
            "",
            "| Bed construction | Hydraulic grid (m) | Mean discharge (m³/s) |",
            "|---|---:|---:|",
        ]
    )
    for r in reach:
        if float(r["margin_m"]) == margin:
            text.append(
                f"| {r['representation']} | {r['dx_m']} | {float(r['interval_mean_section_m3s']):.4f} |"
            )
    text += [
        "",
        "[Matched reach controls](incised_crest/reach_replay.csv) · [Plane controls](incised_crest/plane_controls.csv)",
        "",
        "## Historical eight-hour states",
        "",
        "Direct native depth threshold counts reproduce every 1, 5 and 10 cm area. The 1 cm measure is a thin-water diagnostic. Accepted face volumes reproduce the outlet and internal hydrograph integrals. The closed faces have zero accepted cumulative transport. Bed, active mask, roughness, rain multiplier and infiltration match across every four-child block in the historical replicated-bed study.",
        "",
        "| Comparison | Common wet-area criterion (cm) | Compared 40 m blocks | Depth RMSE (cm) |",
        "|---|---:|---:|---:|",
    ]
    for r in sections:
        text.append(
            f"| {r['comparison']} | {r['any_run_depth_cm']} | {r['blocks']} | {100 * float(r['rmse_m']):.3f} |"
        )
    text.extend(
        [
            "",
            "Masks use the union across the three compared states at hour eight. A threshold of zero includes the full grid. Differences between forecasts quantify their disagreement. [Native checksums and areas](incised_crest/checkpoint_audit.csv) · [Common-support differences](incised_crest/checkpoint_spatial_differences.csv) · [Station hydrographs](incised_crest/checkpoint_station_histories.csv)",
            "",
            "## Corrected complete storms",
            "",
        ]
    )
    native = directory / "mean_storms/terrain.json"
    if native.is_file():
        bundle = json.loads(native.read_text())
        text.extend([
            f"Whole-catchment comparisons start at time zero from {bundle['native_cells']:,} native metre cells. Bed elevation, initial water depth, initial x momentum, rainfall multiplier and infiltration capacity each have an independent native checksum. Each coarser input is its aligned block mean. At the prescribed initial condition, channel water deeper than {100*bundle['inputs']['moving_channel_min_depth_m']:g} cm flows at 0.12 m/s. Shallower margins retain their water and start at rest. All methods use this same physical state. A continuation must match the new inputs and its hydraulic checkpoint identity. [Native input identities and block checks](mean_storms/terrain.json)",
            "",
        ])
    full = directory / "mean_storms/complete_events.csv"
    if full.is_file():
        text.extend(
            [
                "| Method | Grid (m) | Common end (h) | Peak (m³/s) | Net outlet (m³) | Storage (m³) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for r in metrics.csv_read(file=full):
            text.append(
                f"| {r['method']} | {r['dx_m']} | {float(r['end_h']):g} | {float(r['outlet_peak_m3s']):.3f} | {float(r['net_outlet_m3']):,.0f} | {float(r['storage_m3']):,.0f} |"
            )
        text.extend(
            [
                "",
                "[Measured area-mean storms through matched recession](mean_storms/complete_events.csv)",
            ]
        )
    else:
        text.append(
            "The corrected whole-event comparison is pending. The original eight-hour staircase result cannot supply it."
        )
    (directory / "DEM_CORRECTION.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
