"""Import and export georeferenced terrain evidence."""

import json
import math
from pathlib import Path
import re

import numpy as np
from affine import Affine
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)

from floodwarp.terrain_advisor.diagnostics import analyse_array
from floodwarp.terrain_advisor.controls import extract_control, nested_grid


UNITS = {
    "slope": "m/m",
    "plan_curvature": "1/m",
    "profile_curvature": "1/m",
    "contributing_area": "m2",
    "outlet_id": "index",
    "channel": "indicator",
    "saddle": "indicator",
    "lost_channel": "indicator",
    "valley_position": "fraction",
}


def metric_raster(*, dem):
    with rasterio.open(dem) as src:
        if src.crs is None:
            raise ValueError("A DEM with a declared horizontal CRS is required")
        z = src.read(1, masked=True).astype("float64").filled(np.nan)
        units = (src.units[0] or "m").lower()
        if units in ("ft", "feet", "foot", "us survey foot"):
            z *= 1200 / 3937 if units == "us survey foot" else 0.3048
        elif units not in ("m", "metre", "meter", "metres", "meters"):
            raise ValueError("Elevation units must be metres or declared feet")
        trans = src.transform
        crs = src.crs
        local = crs.to_wkt().startswith("LOCAL_CS[")
        metric = crs.is_projected and abs(crs.linear_units_factor[1] - 1) < 1e-10
        if local:
            unit = re.search(r'UNIT\["[^"]+",([0-9.eE+-]+)', crs.to_wkt())
            metric = bool(unit and abs(float(unit.group(1)) - 1) < 1e-10)
        if (
            metric
            and trans.a > 0
            and trans.e < 0
            and max(abs(trans.b), abs(trans.d), abs(trans.a + trans.e)) < 1e-8
        ):
            return z, trans, crs
        if local:
            raise ValueError(
                "A local engineering CRS requires a north-up square metric grid"
            )
        west, south, east, north = transform_bounds(crs, "EPSG:4326", *src.bounds)
        longitude = (west + east) / 2
        latitude = (south + north) / 2
        if abs(latitude) > 84 or east - west > 12:
            raise ValueError(
                "Reproject this large or polar DEM to an appropriate metric grid"
            )
        utm = (32600 if latitude >= 0 else 32700) + min(
            60, max(1, math.floor((longitude + 180) / 6) + 1)
        )
        target = rasterio.crs.CRS.from_epsg(utm)
        dest, width, height = calculate_default_transform(
            crs, target, src.width, src.height, *src.bounds
        )
        grid = np.full((height, width), np.nan)
        reproject(
            z,
            grid,
            src_transform=trans,
            src_crs=crs,
            src_nodata=np.nan,
            dst_crs=target,
            dst_transform=dest,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return grid, dest, target


def write_maps(*, maps, folder, transform, crs):
    folder.mkdir(exist_ok=True, parents=True)
    for key, arr in maps.items():
        file = folder / (key + ".tif")
        with rasterio.open(
            file,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="float64",
            transform=transform,
            crs=crs,
            nodata=-9999.0,
            compress="deflate",
        ) as dest:
            dest.write(np.where(np.isfinite(arr), arr, -9999.0), 1)
            dest.set_band_description(1, key)
            dest.set_band_unit(1, UNITS.get(key, "m"))
            dest.update_tags(
                algorithm="FloodWarp terrain diagnostics, priority flood and D8",
                measurement=key,
            )


def plot_maps(*, maps, output):
    import matplotlib.pyplot as plt

    keys = (
        "elevation",
        "slope",
        "contributing_area",
        "fill_depth",
        "elevation_change",
        "outlet_displacement",
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2), layout="constrained")
    for ax, key in zip(axes.flat, keys, strict=True):
        im = ax.imshow(
            maps[key], cmap="viridis" if key != "elevation_change" else "RdBu_r"
        )
        ax.set_title(key.replace("_", " "), fontsize=9)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.67, label=UNITS.get(key, "m"))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyse_dem(*, dem, resolution, output=None, window_m=30.0, channel_area_m2=None, control=None):
    """Analyse one raster in metres and optionally write GeoTIFFs, a map and a report."""
    z, transform, crs = metric_raster(dem=Path(dem))
    grid = nested_grid(north_up=z, native_dx=transform.a, dx=resolution) if control is not None else None
    stats, maps, coarse = analyse_array(
        elevation=z,
        native_resolution=transform.a,
        resolution=resolution,
        window_m=window_m,
        channel_area_m2=channel_area_m2,
        represented_elevation=grid,
    )
    if control is not None:
        stats["domain_comparison"] = stats["comparison"]
        stats["control_geometry"] = control
        stats["comparison"] = extract_control(native=z, represented=coarse["elevation"], transform=transform,
                                              dx=resolution, control=control, window_m=window_m)
    stats.update(
        crs=crs.to_string(),
        native_resolution_m=transform.a,
        model_resolution_m=resolution,
        routing="D8 steepest descent on the priority-filled surface; deterministic priority-flood parent across flats. Valid edges and nodata-adjacent cells are outlets.",
        width_interpretation="Width proxy from a relative-elevation valley mask. Surveyed channel widths and hydraulic connectivity may differ.",
    )
    if output is not None:
        folder = Path(output)
        folder.mkdir(exist_ok=True, parents=True)
        write_maps(maps=maps, folder=folder / "native", transform=transform, crs=crs)
        write_maps(
            maps=coarse,
            folder=folder / "model",
            transform=transform @ Affine.scale(resolution / transform.a),
            crs=crs,
        )
        (folder / "terrain.json").write_text(
            json.dumps(stats, indent=2, allow_nan=False) + "\n"
        )
        c = stats["comparison"]
        m = stats["model"]
        (folder / "terrain.md").write_text(
            f"# Terrain diagnostics\n\nThe {resolution:g} m grid has {m['cells']:,} active cells and covers {m['area_m2']:.1f} m². Its median slope is {m['median_slope']:.4g} m/m. Aggregation changes depression storage by {c['storage_change_m3']:.3g} m³ and elevation by {c['elevation_rmse_m']:.3g} m RMSE. The 95th-percentile outlet displacement is {c['outlet_displacement_p95_m']:.2f} m.\n\n{stats['routing']}\n\n{stats['width_interpretation']}\n"
        )
        plot_maps(maps=maps, output=folder / "terrain.png")
    return stats
