"""D8 priority flooding, drainage and physical-support raster aggregation."""

from heapq import heappop, heappush
import math

import numpy as np
from numba import njit
from scipy import ndimage as nd
from scipy.sparse import csr_matrix


@njit(cache=True)
def drainage(z, valid, cell_area, spacing):
    ny, nx = z.shape
    size = ny * nx
    surface = z.copy()
    seen = np.zeros((ny, nx), np.uint8)
    border = np.zeros(size, np.uint8)
    parent = np.full(size, -1, np.int64)
    order = np.empty(size, np.int64)
    heap = [(0.0, -1)]
    heappop(heap)
    for j in range(ny):
        for i in range(nx):
            if not valid[j, i]:
                continue
            edge = j == 0 or i == 0 or j == ny - 1 or i == nx - 1
            if not edge:
                for y in range(j - 1, j + 2):
                    for x in range(i - 1, i + 2):
                        if not valid[y, x]:
                            edge = True
            if edge:
                k = j * nx + i
                border[k] = 1
                seen[j, i] = 1
                heappush(heap, (z[j, i], k))
    count = 0
    while heap:
        level, k = heappop(heap)
        order[count] = k
        count += 1
        j, i = k // nx, k % nx
        for y in range(max(0, j - 1), min(ny, j + 2)):
            for x in range(max(0, i - 1), min(nx, i + 2)):
                if not valid[y, x] or seen[y, x] != 0:
                    continue
                seen[y, x] = 1
                surface[y, x] = max(level, z[y, x])
                parent[y * nx + x] = k
                heappush(heap, (surface[y, x], y * nx + x))
    receiver = parent.copy()
    accum = cell_area.flatten()
    outlet = np.full(size, -1, np.int64)
    for n in range(count):
        k = order[n]
        j, i = k // nx, k % nx
        best = 0.0
        if border[k] == 0:
            for y in range(max(0, j - 1), min(ny, j + 2)):
                for x in range(max(0, i - 1), min(nx, i + 2)):
                    if not valid[y, x] or (y == j and x == i):
                        continue
                    slope = (surface[j, i] - surface[y, x]) / (
                        spacing * math.hypot(y - j, x - i)
                    )
                    if slope > best:
                        best = slope
                        receiver[k] = y * nx + x
        to = receiver[k]
        outlet[k] = k if to < 0 else outlet[to]
    for n in range(count - 1, -1, -1):
        k = order[n]
        to = receiver[k]
        if to >= 0:
            accum[to] += accum[k]
    return (
        surface,
        accum.reshape((ny, nx)),
        outlet.reshape((ny, nx)),
        receiver.reshape((ny, nx)),
    )


def weights(*, n, source, target):
    count = math.ceil(n * source / target - 1e-12)
    rows = []
    cols = []
    data = []
    for i in range(count):
        lo, hi = i * target, min(n * source, (i + 1) * target)
        for j in range(math.floor(lo / source), min(n, math.ceil(hi / source))):
            overlap = min(hi, (j + 1) * source) - max(lo, j * source)
            if overlap > 0:
                rows.append(i)
                cols.append(j)
                data.append(overlap)
    return csr_matrix((data, (rows, cols)), shape=(count, n))


def aggregate(*, z, source, target):
    y = weights(n=z.shape[0], source=source, target=target)
    x = weights(n=z.shape[1], source=source, target=target)
    valid = np.isfinite(z)
    area = (x @ (y @ valid).T).T
    sums = (x @ (y @ np.nan_to_num(z, nan=0.0)).T).T
    mean = np.divide(sums, area, out=np.full(area.shape, np.nan), where=area > 0)
    sq = (x @ (y @ np.nan_to_num(z**2, nan=0.0)).T).T
    variance = (
        np.divide(sq, area, out=np.zeros_like(sq), where=area > 0)
        - np.nan_to_num(mean) ** 2
    )
    return mean, area, np.sqrt(np.maximum(variance, 0))


def safe_gradient(array, spacing):
    g = []
    for axis in (0, 1):
        g.append(
            np.gradient(array, spacing, axis=axis)
            if array.shape[axis] > 1
            else np.zeros_like(array)
        )
    return g


def terrain(*, z, dx, areas, channel_area, window_m):
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("DEM has no valid terrain cells")
    nearest = nd.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    extended = z[tuple(nearest)]
    gy, gx = safe_gradient(extended, dx)
    gyy, gyx = safe_gradient(gy, dx)
    gxy, gxx = safe_gradient(gx, dx)
    cross = 0.5 * (gyx + gxy)
    g2 = gx * gx + gy * gy
    size = max(3, 2 * int(round(window_m / (2 * dx))) + 1)
    high = nd.maximum_filter(np.where(valid, z, -np.inf), size=size, mode="nearest")
    low = nd.minimum_filter(np.where(valid, z, np.inf), size=size, mode="nearest")
    relief = high - low
    position = np.divide(z - low, relief, out=np.zeros_like(z), where=relief > 0)
    filled, area, outlet, receiver = drainage(
        np.ascontiguousarray(z),
        np.ascontiguousarray(valid),
        np.ascontiguousarray(areas),
        float(dx),
    )
    channel = (area >= channel_area) & valid
    valley = (position <= 0.25) & valid
    width = 2 * nd.distance_transform_edt(np.pad(valley, 1))[1:-1, 1:-1] * dx
    slope = np.sqrt(g2)
    s = g2 > 1e-16
    plan = np.divide(
        gy**2 * gxx - 2 * gx * gy * cross + gx**2 * gyy,
        g2**1.5,
        out=np.zeros_like(z),
        where=s,
    )
    profile = np.divide(
        gx**2 * gxx + 2 * gx * gy * cross + gy**2 * gyy,
        g2 * (1 + g2) ** 1.5,
        out=np.zeros_like(z),
        where=s,
    )
    valid_area = areas.sum()
    links = np.flatnonzero(channel.ravel() & (receiver.ravel() >= 0))
    dest = receiver.ravel()[links]
    length = (
        dx
        * np.hypot(
            links // z.shape[1] - dest // z.shape[1],
            links % z.shape[1] - dest % z.shape[1],
        )
    ).sum()
    sj = slope[valid]
    outcells = valid & (receiver < 0)
    qs = np.percentile(sj, [5, 25, 50, 75, 95, 99])
    summary = dict(
        dx_m=dx,
        rows=z.shape[0],
        columns=z.shape[1],
        cells=int(valid.sum()),
        area_m2=float(valid_area),
        mean_slope=float(np.average(sj, weights=areas[valid])),
        slope_p05=qs[0],
        slope_p25=qs[1],
        median_slope=qs[2],
        slope_p75=qs[3],
        slope_p95=qs[4],
        slope_p99=qs[5],
        steep_fraction=float(areas[slope > 0.05].sum() / valid_area),
        gentle_fraction=float(areas[slope < 0.002].sum() / valid_area),
        relief_m=np.ptp(z[valid]),
        local_relief_p50_m=np.median(relief[valid]),
        valley_fraction=areas[valley].sum() / valid_area,
        mean_convergence_per_m=float(np.mean(-plan[valid])),
        channel_density_m_per_m2=length / valid_area,
        channel_threshold_m2=channel_area,
        channel_width_proxy_m=float(np.median(width[channel]))
        if channel.any()
        else None,
        width_to_grid=float(np.median(width[channel]) / dx) if channel.any() else None,
        depression_storage_m3=float(np.sum((filled[valid] - z[valid]) * areas[valid])),
        largest_contributing_area_m2=float(area[valid].max()),
        outlet_count=int(outcells.sum()),
        largest_outlet_fraction=float(area[outcells].max() / valid_area),
        elevation_p05_m=float(np.percentile(z[valid], 5)),
        elevation_p95_m=float(np.percentile(z[valid], 95)),
    )
    maps = dict(
        elevation=z,
        slope=slope,
        local_relief=relief,
        valley_position=position,
        plan_curvature=plan,
        profile_curvature=profile,
        contributing_area=area,
        channel=channel.astype(float),
        valley_width_proxy=width,
        fill_depth=filled - z,
        outlet_id=outlet.astype(float),
        saddle=((gxx * gyy - cross**2 < 0) & (slope <= qs[2])).astype(float),
    )
    return summary, {k: np.where(valid, v, np.nan) for k, v in maps.items()}


def analyse_array(
    *, elevation, native_resolution, resolution, window_m=30.0, channel_area_m2=None, represented_elevation=None
):
    z = np.asarray(elevation, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).any() or np.isinf(z).any():
        raise ValueError(
            "Provide a two-dimensional metric DEM with valid finite elevations or NaN"
        )
    if (
        not np.isfinite([native_resolution, resolution, window_m]).all()
        or min(native_resolution, window_m) <= 0
        or resolution < native_resolution
    ):
        raise ValueError("Resolution must be at least the native grid spacing")
    cell = np.where(np.isfinite(z), native_resolution**2, 0.0)
    threshold = (
        max(100.0, 0.02 * cell.sum()) if channel_area_m2 is None else channel_area_m2
    )
    native, nmap = terrain(
        z=z, dx=native_resolution, areas=cell, channel_area=threshold, window_m=window_m
    )
    coarse, areas, interior_std = aggregate(
        z=z, source=native_resolution, target=resolution
    )
    if represented_elevation is not None:
        actual = np.asarray(represented_elevation, dtype=float)
        if actual.shape != coarse.shape or not np.array_equal(np.isfinite(actual), areas > 0):
            raise ValueError("Represented bed must match the active coarse raster")
        coarse = actual
    model, mmap = terrain(
        z=coarse, dx=resolution, areas=areas, channel_area=threshold, window_m=window_m
    )
    y = np.minimum(
        ((np.arange(z.shape[0]) + 0.5) * native_resolution / resolution).astype(int),
        coarse.shape[0] - 1,
    )
    x = np.minimum(
        ((np.arange(z.shape[1]) + 0.5) * native_resolution / resolution).astype(int),
        coarse.shape[1] - 1,
    )
    select = np.ix_(y, x)
    change = coarse[select] - z
    reachable = nd.maximum_filter(np.nan_to_num(mmap["channel"]), size=3)[select] > 0
    channels = nmap["channel"] == 1
    saddle = nmap["saddle"] == 1
    # Distance between the native terminal drainage pixel and its coarse counterpart.
    valid = np.isfinite(z)
    end = np.nan_to_num(nmap["outlet_id"]).astype(int)
    modeled = np.nan_to_num(mmap["outlet_id"][select]).astype(int)
    end_y, end_x = np.divmod(end, z.shape[1])
    out_y, out_x = np.divmod(modeled, coarse.shape[1])
    distance = np.hypot(
        (end_y + 0.5) * native_resolution - (out_y + 0.5) * resolution,
        (end_x + 0.5) * native_resolution - (out_x + 0.5) * resolution,
    )
    loss = np.where(channels & ~reachable, 1.0, 0.0)
    result = dict(
        native=native,
        model=model,
        comparison=dict(
            elevation_rmse_m=float(np.nanmean(change**2) ** 0.5),
            within_cell_std_mean_m=float(np.average(interior_std, weights=areas)),
            channel_loss_fraction=float(loss[channels].mean())
            if channels.any()
            else None,
            storage_change_m3=model["depression_storage_m3"]
            - native["depression_storage_m3"],
            outlet_displacement_p95_m=float(np.percentile(distance[valid], 95)),
            changed_outlet_fraction=float(np.mean(distance[valid] > 2 * resolution)),
            saddle_elevation_rmse_m=float(np.mean(change[saddle] ** 2) ** 0.5)
            if saddle.any()
            else None,
            barrier_lowering_p95_m=float(
                np.percentile(np.maximum(-change[valid], 0), 95)
            ),
            channel_density_change=model["channel_density_m_per_m2"]
            - native["channel_density_m_per_m2"],
        ),
    )
    maps = dict(
        nmap,
        elevation_change=change,
        lost_channel=np.where(valid, loss, np.nan),
        outlet_displacement=np.where(valid, distance, np.nan),
        within_cell_std=np.where(valid, interior_std[select], np.nan),
        saddle_change=np.where(saddle, change, np.nan),
        barrier_lowering=np.maximum(-change, 0),
    )
    return result, maps, mmap
