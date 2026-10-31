"""Exact-zero uniform-grid scheduling; no alternative hydraulic operator.

The compact list, count and tile flags stay on the device. Bounded batched
launches walk the list with a grid stride, including inside CUDA graphs.
"""

import warp as wp

from floodwarp.full_swe_kernels import faces_at, draining_at, limit_faces_at, update_at


@wp.kernel
def mark_tiles(q: wp.array(dtype=wp.vec3d), active: wp.array(dtype=wp.int32),
               bed: wp.array(dtype=wp.float64), bx: wp.array(dtype=wp.int32),
               by: wp.array(dtype=wp.int32), sx: wp.array(dtype=wp.float64),
               sy: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32),
               nx: int, tile_size: int, tiles_x: int):
    k = wp.tid()
    if active[k] == 0:
        return
    j = k / nx
    i = k % nx
    seed = q[k][0] != wp.float64(0.0) or q[k][1] != wp.float64(0.0) or q[k][2] != wp.float64(0.0)
    for offset in range(2):
        x = j * (nx + 1) + i + offset
        y = k + offset * nx
        if bx[x] == 3 and sx[x] > bed[k]:
            seed = True
        if by[y] == 3 and sy[y] > bed[k]:
            seed = True
    if seed:
        wp.atomic_max(flags, (j / tile_size) * tiles_x + i / tile_size, 1)


@wp.kernel
def compact_tiles(flags: wp.array(dtype=wp.int32), work: wp.array(dtype=wp.int32),
                  count: wp.array(dtype=wp.int32), tiles_x: int, tiles_y: int):
    tile = wp.tid()
    row = tile / tiles_x
    column = tile % tiles_x
    include = False
    for dj in range(-1, 2):
        for di in range(-1, 2):
            j = row + dj
            i = column + di
            if j >= 0 and j < tiles_y and i >= 0 and i < tiles_x:
                if flags[j * tiles_x + i] != 0:
                    include = True
    if include:
        slot = wp.atomic_add(count, 0, 1)
        work[slot] = tile


@wp.func
def work_size(count: wp.array(dtype=wp.int32), nx: int, ny: int, tile_size: int,
              dense_cutoff: int):
    result = count[0] * tile_size * tile_size
    if count[0] >= dense_cutoff:
        result = nx * ny
    return result


@wp.func
def work_cell(index: int, work: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
              nx: int, ny: int, tile_size: int, tiles_x: int, dense_cutoff: int):
    k = index
    if count[0] < dense_cutoff:
        area = tile_size * tile_size
        tile = work[index / area]
        local = index % area
        j = (tile / tiles_x) * tile_size + local / tile_size
        i = (tile % tiles_x) * tile_size + local % tile_size
        k = -1
        if j < ny and i < nx:
            k = j * nx + i
    return k


@wp.kernel
def clear_previous(fx: wp.array(dtype=wp.vec3d), fy: wp.array(dtype=wp.vec3d),
                   px: wp.array(dtype=wp.vec2d), py: wp.array(dtype=wp.vec2d),
                   theta: wp.array(dtype=wp.float64), work: wp.array(dtype=wp.int32),
                   count: wp.array(dtype=wp.int32), nx: int, ny: int, tile_size: int,
                   tiles_x: int, dense_cutoff: int, stride: int):
    for index in range(wp.tid(), work_size(count, nx, ny, tile_size, dense_cutoff), stride):
        k = work_cell(index, work, count, nx, ny, tile_size, tiles_x, dense_cutoff)
        if k >= 0:
            j = k / nx
            i = k % nx
            x = j * (nx + 1) + i
            fx[x] = wp.vec3d(wp.float64(0.0))
            fy[k] = wp.vec3d(wp.float64(0.0))
            px[x] = wp.vec2d(wp.float64(0.0))
            py[k] = wp.vec2d(wp.float64(0.0))
            theta[k] = wp.float64(1.0)
            if i == nx - 1:
                fx[x + 1] = wp.vec3d(wp.float64(0.0))
                px[x + 1] = wp.vec2d(wp.float64(0.0))
            if j == ny - 1:
                fy[k + nx] = wp.vec3d(wp.float64(0.0))
                py[k + nx] = wp.vec2d(wp.float64(0.0))


@wp.kernel
def sparse_faces(q: wp.array(dtype=wp.vec3d), bed: wp.array(dtype=wp.float64),
                 active: wp.array(dtype=wp.int32), types: wp.array(dtype=wp.int32),
                 stages: wp.array(dtype=wp.float64), flux: wp.array(dtype=wp.vec3d),
                 pressure: wp.array(dtype=wp.vec2d), nx: int, ny: int, axis: int,
                 g: wp.float64, dry: wp.float64, stage_momentum_mode: int,
                 ctrl: wp.array(dtype=wp.float64), flux_mode: int,
                 work: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                 tile_size: int, tiles_x: int, dense_cutoff: int, stride: int):
    for index in range(wp.tid(), work_size(count, nx, ny, tile_size, dense_cutoff), stride):
        k = work_cell(index, work, count, nx, ny, tile_size, tiles_x, dense_cutoff)
        if k >= 0:
            j = k / nx
            i = k % nx
            face = k
            if axis == 0:
                face = j * (nx + 1) + i
            faces_at(face, q, bed, active, types, stages, flux, pressure, nx, ny, axis,
                     g, dry, stage_momentum_mode, ctrl, flux_mode)
            if axis == 0 and i == nx - 1:
                faces_at(face + 1, q, bed, active, types, stages, flux, pressure, nx, ny,
                         axis, g, dry, stage_momentum_mode, ctrl, flux_mode)
            if axis == 1 and j == ny - 1:
                faces_at(face + nx, q, bed, active, types, stages, flux, pressure, nx, ny,
                         axis, g, dry, stage_momentum_mode, ctrl, flux_mode)


@wp.kernel
def sparse_draining(q: wp.array(dtype=wp.vec3d), fx: wp.array(dtype=wp.vec3d),
                    fy: wp.array(dtype=wp.vec3d), theta: wp.array(dtype=wp.float64),
                    ctrl: wp.array(dtype=wp.float64), nx: int, dx: wp.float64, dy: wp.float64,
                    ny: int, work: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                    tile_size: int, tiles_x: int, dense_cutoff: int, stride: int):
    for index in range(wp.tid(), work_size(count, nx, ny, tile_size, dense_cutoff), stride):
        k = work_cell(index, work, count, nx, ny, tile_size, tiles_x, dense_cutoff)
        if k >= 0:
            draining_at(k, q, fx, fy, theta, ctrl, nx, dx, dy)


@wp.kernel
def sparse_limit_faces(flux: wp.array(dtype=wp.vec3d), theta: wp.array(dtype=wp.float64),
                       active: wp.array(dtype=wp.int32), volume: wp.array(dtype=wp.float64),
                       ctrl: wp.array(dtype=wp.float64), nx: int, ny: int, axis: int,
                       width: wp.float64, work: wp.array(dtype=wp.int32),
                       count: wp.array(dtype=wp.int32), tile_size: int, tiles_x: int,
                       dense_cutoff: int, stride: int):
    for index in range(wp.tid(), work_size(count, nx, ny, tile_size, dense_cutoff), stride):
        k = work_cell(index, work, count, nx, ny, tile_size, tiles_x, dense_cutoff)
        if k >= 0:
            j = k / nx
            i = k % nx
            face = k
            if axis == 0:
                face = j * (nx + 1) + i
            limit_faces_at(face, flux, theta, active, volume, ctrl, nx, ny, axis, width)
            if axis == 0 and i == nx - 1:
                limit_faces_at(face + 1, flux, theta, active, volume, ctrl, nx, ny, axis, width)
            if axis == 1 and j == ny - 1:
                limit_faces_at(face + nx, flux, theta, active, volume, ctrl, nx, ny, axis, width)


@wp.kernel
def sparse_update(q: wp.array(dtype=wp.vec3d), fx: wp.array(dtype=wp.vec3d), fy: wp.array(dtype=wp.vec3d),
                  px: wp.array(dtype=wp.vec2d), py: wp.array(dtype=wp.vec2d),
                  active: wp.array(dtype=wp.int32), manning: wp.array(dtype=wp.float64),
                  maximum: wp.array(dtype=wp.float64), inflow: wp.array(dtype=wp.float64),
                  outflow: wp.array(dtype=wp.float64), ctrl: wp.array(dtype=wp.float64),
                  nx: int, ny: int, dx: wp.float64, dy: wp.float64, g: wp.float64, dry: wp.float64,
                  work: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                  tile_size: int, tiles_x: int, dense_cutoff: int, stride: int):
    for index in range(wp.tid(), work_size(count, nx, ny, tile_size, dense_cutoff), stride):
        k = work_cell(index, work, count, nx, ny, tile_size, tiles_x, dense_cutoff)
        if k >= 0:
            update_at(k, q, fx, fy, px, py, active, manning, maximum, inflow, outflow,
                      ctrl, nx, ny, dx, dy, g, dry)
