"""Evaluate a surveyed spill connection in an aligned, explicitly located DEM window."""

import numpy as np
from scipy.ndimage import label

from floodwarp.terrain_advisor.diagnostics import analyse_array


def connected(*, bed, level):
    groups, _ = label(bed <= level)
    west = np.unique(groups[:, 0])
    return bool(np.intersect1d(west[west > 0], groups[:, -1]).size)


def spill_level(*, bed):
    if not np.isfinite(bed).all():
        raise ValueError("Control geometry requires complete finite terrain")
    levels = np.unique(bed)
    low, high = 0, len(levels)-1
    while low < high:
        mid = (low + high)//2
        if connected(bed=bed, level=levels[mid]):
            high = mid
        else:
            low = mid+1
    return float(levels[low])


def assess_window(*, native, represented, native_dx, dx, head, axis, window_m):
    if not np.isfinite(head) or head < 0 or axis not in ("west_east", "south_north"):
        raise ValueError("A nonnegative probe head and opposing control faces are required")
    stats, maps, coarse_maps = analyse_array(elevation=native, native_resolution=native_dx,
                                             represented_elevation=represented, resolution=dx, window_m=window_m)
    a = native if axis == "west_east" else native.T
    b = represented if axis == "west_east" else represented.T
    spill = spill_level(bed=a)
    probe = spill + head
    comparison = dict(stats["comparison"], native_connected=connected(bed=a, level=probe),
                      represented_connected=connected(bed=b, level=probe), probe_stage_m=probe,
                      native_spill_m=spill, represented_spill_m=spill_level(bed=b),
                      max_local_barrier_lowering_m=float(maps["barrier_lowering"].max()))
    return comparison, maps, coarse_maps


def nested_grid(*, north_up, native_dx, dx):
    """Reproduce the south-to-north 1 → 5 → 10 m complete-cell averages."""
    ratio = dx/native_dx
    if not ratio.is_integer() or not np.isfinite(north_up).all():
        return None
    factors = (5, 2) if native_dx == 1 and dx == 10 else (int(ratio),)
    south_up = north_up[::-1]
    for factor in factors:
        ny, nx = south_up.shape
        if ny % factor or nx % factor:
            return None
        south_up = south_up.reshape(ny//factor, factor, nx//factor, factor).mean(axis=(1, 3), dtype=np.float64)
    return south_up[::-1]


def extract_control(*, native, represented, transform, dx, control, window_m):
    """Window is [west, south, width, height] in the analysis grid's metric CRS."""
    west, south, width, height = control["window_m"]
    x, y = ~transform * (west, south+height)
    edges = np.array([y, y+height/transform.a, x, x+width/transform.a])
    coarse_edges = edges * transform.a / dx
    if (min(width, height) <= 0 or not np.allclose(coarse_edges, np.rint(coarse_edges), rtol=0, atol=1e-8)
            or not np.allclose(edges, np.rint(edges), rtol=0, atol=1e-8)):
        raise ValueError("Control bounds must align with both grids")
    i, j, k, l = np.rint(edges).astype(int)
    if min(i, k) < 0 or j > native.shape[0] or l > native.shape[1]:
        raise ValueError("Control bounds outside the terrain")
    a, b, c, d = np.rint(coarse_edges).astype(int)
    stats, _, _ = assess_window(native=native[i:j, k:l], represented=represented[a:b, c:d],
                                native_dx=transform.a, dx=dx, head=control["spill_head_m"],
                                axis=control["axis"], window_m=window_m)
    return stats
