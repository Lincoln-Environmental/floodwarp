"""Test control-window connectivity on aligned area means of the archived native metre DEM."""

import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from floodwarp.terrain_advisor.diagnostics import analyse_array
from floodwarp.terrain_advisor.raster import write_maps
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification.observations import current
from research.floodwarp_methods_paper.qualification.terrain import connection
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def measure(*, record, study, output):
    parent = study / record["directory"]
    native_file = parent / "control_1m/represented/elevation.tif"
    if (
        runner.checksum(native_file)
        != record["artifacts"]["control_1m/represented/elevation.tif"]
    ):
        raise ValueError(f"Changed native terrain input {native_file}")
    archive = json.loads((parent / "result.json").read_text())
    reference = archive["window"]["1"]["comparison"]
    stage = reference["probe_stage_m"]
    with rasterio.open(native_file) as src:
        native = src.read(1, masked=True)
        if native.mask.any() or src.res != (1, 1) or src.units != ("m",):
            raise ValueError("Control windows must contain complete metre cells")
        native = np.asarray(native)
        transform, crs = src.transform, src.crs
    if not connection(elevation=native, level=stage):
        raise ValueError("Archived stage fails to connect its native control window")
    five = mean_cells(values=native, factor=5)
    ten = mean_cells(values=five, factor=2)
    results = []
    for dx, bed in ((1, native), (5, five), (10, ten)):
        stats, maps, coarse = analyse_array(
            elevation=native,
            native_resolution=1,
            resolution=dx,
            represented_elevation=bed,
        )
        legacy = archive["window"][str(dx)]["comparison"]
        valid = connection(elevation=bed, level=stage)
        case = record["spec"]["case"]
        folder = output / case / f"{dx}m"
        write_maps(
            maps={
                k: maps[k]
                for k in (
                    "lost_channel",
                    "barrier_lowering",
                    "saddle_change",
                    "outlet_displacement",
                    "elevation_change",
                )
            },
            folder=folder,
            transform=transform,
            crs=crs,
        )
        write_maps(
            maps={
                k: coarse[k]
                for k in ("elevation", "contributing_area", "channel", "fill_depth")
            },
            folder=folder / "represented",
            transform=transform @ Affine.scale(dx),
            crs=crs,
        )
        results.append(
            dict(
                case=case,
                landform=record["spec"]["landform"],
                dx_m=dx,
                probe_stage_m=stage,
                opening_width_cells=legacy["opening_width_cells"],
                connected=valid,
                centre_sampled_connected=legacy["represented_connected"],
                max_local_barrier_lowering_m=float(np.max(maps["barrier_lowering"])),
                native_input=str(native_file.relative_to(study)),
                native_sha256=runner.checksum(native_file),
                bed_sha256=runner.checksum(folder / "represented/elevation.tif"),
                cropped_window_m=archive["control_window_m"],
                **stats["comparison"],
            )
        )
    return results


def run(*, directory):
    study = directory.parent
    output = directory / "mean_windows"
    records = current(
        study=study,
        stage="million-terrain",
        config=read_config(path=study / "configs/study.json"),
    )
    if not records:
        raise ValueError("Missing qualified native control windows")
    rows = []
    seen = {}
    for record in records:
        case = record["spec"]["case"]
        sha = record["artifacts"]["control_1m/represented/elevation.tif"]
        if case in seen:
            if seen[case] != sha:
                raise ValueError(
                    f"Source metre grid changed between experiments for {case}"
                )
            continue
        rows.extend(measure(record=record, study=study, output=output))
        seen[case] = sha
    metrics.csv_write(file=output / "controls.csv", rows=rows)
    print(
        f"Measured {len(rows)} connectivity controls from {len(seen)} independent physical windows"
    )
    for dx in (1, 5, 10):
        subset = [row for row in rows if row["dx_m"] == dx]
        print(
            f"{dx} m: {sum(not row['connected'] for row in subset)} of {len(subset)} lost the native connection"
        )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
