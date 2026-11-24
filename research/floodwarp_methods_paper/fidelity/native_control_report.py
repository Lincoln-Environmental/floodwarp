"""Verify represented control rasters and test the same physical opening across grid sizes."""

from collections import defaultdict
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile

from floodwarp.terrain_advisor.raster import UNITS
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_features
from research.floodwarp_methods_paper.fidelity.figures import save
from research.floodwarp_methods_paper.qualification.terrain import connection
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def geotiff_crs(*, wkt):
    """GDAL omits the user-defined LOCAL_DATUM node when encoding GeoTIFF keys."""
    with MemoryFile() as buffer:
        with buffer.open(
            driver="GTiff",
            width=1,
            height=1,
            count=1,
            dtype="uint8",
            transform=Affine.translation(0, 1) @ Affine.scale(1, -1),
            crs=rasterio.crs.CRS.from_wkt(wkt),
        ):
            pass
        with buffer.open() as reopened:
            return reopened.crs


def verify_maps(*, record, folder, window, dx, represented, crs):
    west, south, size, _ = window
    for relative, sha in record["maps"].items():
        path = folder / relative
        if runner.checksum(path) != sha:
            raise ValueError(f"Map checksum mismatch in {path}")
        spacing = dx if path.parent.name == "represented" else 1
        with rasterio.open(path) as src:
            if (
                src.crs != crs
                or src.res != (spacing, spacing)
                or src.nodata != -9999.0
                or tuple(src.bounds) != (west, south, west + size, south + size)
                or src.shape != (size // spacing, size // spacing)
                or src.units != (UNITS.get(path.stem, "m"),)
            ):
                raise ValueError(
                    f"Invalid control map georeferencing or units in {path}"
                )
            if relative == "represented/elevation.tif":
                np.testing.assert_array_equal(src.read(1), represented[::-1])


def control_levels(*, native, heads_cm):
    five = mean_cells(values=native, factor=5)
    ten = mean_cells(values=five, factor=2)
    levels = []
    base = native_features.spill_level(bed=native)
    beds = {1: native, 5: five, 10: ten}
    for dx, bed in beds.items():
        spill = native_features.spill_level(bed=bed)
        for head in heads_cm:
            stage = base + head / 100
            open_path = connection(elevation=bed, level=stage)
            if open_path != (stage >= spill):
                raise ArithmeticError("Spill and connected-component tests disagree")
            levels.append(
                dict(
                    dx_m=dx,
                    head_above_native_spill_cm=head,
                    stage_m=stage,
                    native_spill_m=base,
                    represented_spill_m=spill,
                    shift_cm=100 * (spill - base),
                    connected=open_path,
                )
            )
    return beds, levels


def run(*, directory, output, heads_cm):
    inventory = directory / "native_features/observations.csv"
    if not inventory.is_file():
        return ["Native control descriptors are pending."], []
    options_file = directory / "native_feature_settings.json"
    options = json.loads(options_file.read_text())
    crs = geotiff_crs(wkt=options["local_crs"])
    groups = defaultdict(list)
    for row in metrics.csv_read(file=inventory):
        groups[row["case"]].append(row)
    comparisons, maps = [], []
    sources = [inventory, options_file]
    for case, records in sorted(groups.items()):
        physical_id = None
        for row in records:
            dx = int(row["dx_m"])
            folder = directory / "native_features" / case / f"{dx}m" / row["identity"]
            file = folder / "pre_run.json"
            rec = json.loads(file.read_text())
            identity = rec["identity"]
            if runner.digest(identity)[:24] != row["identity"]:
                raise ValueError("Altered terrain descriptor identity")
            sources.append(file)
            if physical_id is None:
                physical_id = identity["physical"]
                inputs = (
                    directory / "native_terrain" / case / physical_id / "inputs.json"
                )
                sources.append(inputs)
                receipt = json.loads(inputs.read_text())
                if runner.digest(receipt) != physical_id:
                    raise ValueError("Physical terrain checksum changed")
                west, south, native = native_features.window(
                    receipt=receipt, options=options
                )
                beds, levels = control_levels(native=native, heads_cm=heads_cm)
            elif physical_id != identity["physical"]:
                raise ValueError("Grid comparison uses different physical controls")
            window = [west, south, *native.shape]
            if rec["control_window_m"] != window:
                raise ValueError("Control window moved between representations")
            verify_maps(
                record=rec,
                folder=folder,
                window=window,
                dx=dx,
                represented=beds[dx],
                crs=crs,
            )
            maps.append(
                dict(
                    case=case,
                    dx_m=dx,
                    descriptor_id=row["identity"],
                    maps_checked=len(rec["maps"]),
                    west_m=west,
                    south_m=south,
                    width_m=native.shape[1],
                    physical_id=physical_id,
                )
            )
        hydraulic_grids = {int(r["dx_m"]) for r in records}
        for level in levels:
            comparisons.append(
                dict(
                    case=case,
                    physical_id=physical_id,
                    family=rec["landform"],
                    native_bed_sha256=receipt["native_sha256"]["bed"],
                    hydraulic_grid_in_suite=level["dx_m"] in hydraulic_grids,
                    **level,
                )
            )
    metrics.csv_write(file=output / "control_stage_sensitivity.csv", rows=comparisons)
    metrics.csv_write(file=output / "control_map_checks.csv", rows=maps)
    independent = {}
    for r in comparisons:
        key = r["native_bed_sha256"], r["dx_m"], r["head_above_native_spill_cm"]
        if key in independent and independent[key]["shift_cm"] != r["shift_cm"]:
            raise ValueError(
                "Identical terrain has inconsistent spill levels across rainfall cases"
            )
        independent[key] = r
    terrain_count = len({s[0] for s in independent})
    rows = []
    for head in heads_cm:
        losses = [
            sum(
                not r["connected"]
                for r in independent.values()
                if r["dx_m"] == dx and r["head_above_native_spill_cm"] == head
            )
            for dx in (1, 5, 10)
        ]
        rows.append([head, *losses])
    figure, ax = plt.subplots(figsize=(5.4, 3.3), layout="constrained")
    data = np.asarray(rows)
    for col, dx in enumerate((1, 5, 10), start=1):
        spills = np.array(
            [
                r["shift_cm"]
                for r in independent.values()
                if r["dx_m"] == dx and r["head_above_native_spill_cm"] == heads_cm[0]
            ]
        )
        x = np.unique(
            np.r_[
                heads_cm, spills[(spills >= min(heads_cm)) & (spills <= max(heads_cm))]
            ]
        )
        (line,) = ax.step(
            x, [np.count_nonzero(spills > h) for h in x], where="post", label=f"{dx} m"
        )
        ax.scatter(data[:, 0], data[:, col], color=line.get_color(), s=18)
    ax.set(
        xlabel="Water level above native spill (cm)",
        ylabel="Disconnected control windows",
        ylim=(-0.5, terrain_count + 0.5),
        xticks=heads_cm,
    )
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend()
    save(fig=figure, folder=output, name="control_stage_sensitivity")
    lines = [
        f"A {options['control_window_m']} m window surrounds the prescribed opening on each of {terrain_count} distinct native beds. "
        "Repeated storms on a bed count once. "
        "A connection requires a continuous path along shared cell faces from west to east. "
        "A single water-surface elevation is prescribed across each window. "
        "Its height above the native spill controls which paths are available.",
        "",
        "| Level above native spill (cm) | Disconnected at 1 m | At 5 m | At 10 m |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in r) + " |" for r in rows)
    lines.extend(
        [
            "",
            "All grid comparisons start from native metre elevation. The control-stage test is static. "
            "Hydraulic peak and arrival errors need separate matched storm observations.",
            "",
            "[Spill elevations by terrain](control_stage_sensitivity.csv) · "
            "[Map read-back checks](control_map_checks.csv) · [Connectivity figure](control_stage_sensitivity.pdf)",
        ]
    )
    return lines, sources
