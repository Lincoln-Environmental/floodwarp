"""Separate terrain and hydraulic-grid effects in the 48-hour coastal-tile HIRDS experiment."""

import json
from pathlib import Path

import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_figures import rows_table
from validation.fv2_native_storm import step_event, support, tile_inputs


def differences(*, coarse, refined, native, mask):
    transport = (coarse - refined)[mask]
    terrain = (refined - native)[mask]
    total = (coarse - native)[mask]
    transport_squared = float(np.mean(transport**2))
    terrain_squared = float(np.mean(terrain**2))
    cross = float(2 * np.mean(transport * terrain))
    total_squared = float(np.mean(total**2))
    np.testing.assert_allclose(
        transport_squared + terrain_squared + cross,
        total_squared,
        rtol=2e-13,
        atol=1e-22,
    )
    return dict(
        total_rmse=float(np.sqrt(total_squared)),
        fixed_bed_grid_rmse=float(np.sqrt(transport_squared)),
        terrain_at_fixed_grid_rmse=float(np.sqrt(terrain_squared)),
        interaction_in_squared_error=cross,
    )


def compare(*, tile, spacings, methods, cap, sample, work, device, wet, output):
    metrics = []
    summary = []
    mask = tile.fields["active"]
    for m in methods:

        def run(*, dx, factor, suffix):
            record, q, maximum = step_event(
                tile=tile,
                terrain_dx=dx,
                factor=factor,
                method=m,
                cap=cap,
                sample=sample,
                work=work,
                device=device,
                wet=wet,
                suffix=suffix,
            )
            return dict(
                end_depth_m=support(field=q[..., 0], dx=dx / factor),
                max_depth_m=support(field=maximum, dx=dx / factor),
                hydrograph_m3_s=np.array(record["hydrograph_m3_s"]),
            ), record

        native, _ = run(dx=1, factor=1, suffix="base")
        for dx in spacings:
            coarse, _ = run(dx=dx, factor=1, suffix="base")
            fine, record = run(dx=dx, factor=dx, suffix="fixed_bed")
            result = {
                key: differences(
                    coarse=coarse[key],
                    refined=fine[key],
                    native=native[key],
                    mask=mask if key != "hydrograph_m3_s" else slice(None),
                )
                for key in native
            }
            metrics.append(
                dict(
                    terrain_dx_m=dx,
                    method=m,
                    hydraulic_refinement_m=1,
                    refined_ledger=record["ledger"],
                    steps=record["steps"],
                    refined_input_sha256=record["input_sha256"],
                    metrics=result,
                )
            )
            summary.append(
                [
                    dx,
                    m,
                    *[
                        1000 * result[key][quantity]
                        for key in native
                        for quantity in (
                            "total_rmse",
                            "fixed_bed_grid_rmse",
                            "terrain_at_fixed_grid_rmse",
                        )
                    ],
                ]
            )
            report_json(path=output, content=metrics)
            print(
                f"{dx} m terrain, 1 m hydraulic grid, {m}, completed matched 48 h storm",
                flush=True,
            )
    table = rows_table(
        title="Separating represented terrain from hydraulic spacing in the coastal-tile storm",
        columns=[
            "Terrain (m)",
            "Method",
            "End depth: combined (mm)",
            "Grid (mm)",
            "Terrain (mm)",
            "Max depth: combined (mm)",
            "Grid (mm)",
            "Terrain (mm)",
            "Hydrograph: combined (L/s)",
            "Grid (L/s)",
            "Terrain (L/s)",
        ],
        rows=summary,
        note="Each row uses the same solver throughout. Combined compares the coarse run to its 1 m terrain/1 m hydraulic counterpart. Grid refines the hydraulic spacing to 1 m with every coarse input held fixed. Terrain compares the two terrain/forcing representations at identical 1 m hydraulic spacing. Both groups include the mask and coastal openings belonging to the represented domain. Depth uses the original active 1 m footprint. The stored signed interaction makes the squared-error decomposition exact. The 1 m runs are model comparators with finite truncation error.",
    )
    output.with_suffix(".md").write_text(table)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    work = Path("outputs/fv2_hirds_native_ordered_bed")
    source = Path(
        "../floodfm_data/examples/purau_point_1575902_5166615_catchment_model/model_inputs.npz"
    )
    checkpoint = Path(
        json.loads(
            Path("validation/fv1_riemann_review/corrected/catchment.json").read_text()
        )["checkpoint"]
    )
    tile = tile_inputs(
        checkpoint=checkpoint, native_file=source, row=112, col=172, width=40
    )
    compare(
        tile=tile,
        spacings=(5, 10),
        methods=("fv1", "mc"),
        cap=1.0,
        sample=900,
        work=work,
        device="cuda:0",
        wet=0.01,
        output=directory / "storm_geometry.json",
    )
