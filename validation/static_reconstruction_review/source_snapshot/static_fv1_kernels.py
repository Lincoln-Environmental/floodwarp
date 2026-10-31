"""Conservative shared-subface FV1 evolution on synchronised static leaves."""

import warp as wp

from floodwarp.full_swe_kernels import faces_at


@wp.kernel
def restrict_sources(q: wp.array(dtype=wp.vec3d), owner: wp.array(dtype=wp.int32),
                     counts: wp.array(dtype=wp.int32), leaves: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    leaf = owner[k]
    if leaf >= 0:
        wp.atomic_add(leaves, leaf, q[k] / wp.float64(counts[leaf]))


@wp.kernel
def project_leaves(leaves: wp.array(dtype=wp.vec3d), owner: wp.array(dtype=wp.int32),
                   counts: wp.array(dtype=wp.int32), bed: wp.array(dtype=wp.float64),
                   mean_bed: wp.array(dtype=wp.float64), q: wp.array(dtype=wp.vec3d),
                   theta: wp.array(dtype=wp.float64), leaf_theta: wp.array(dtype=wp.float64),
                   maximum: wp.array(dtype=wp.float64), ctrl: wp.array(dtype=wp.float64),
                   dry: wp.float64, record_history: int):
    k = wp.tid()
    leaf = owner[k]
    if leaf < 0:
        return
    state = leaves[leaf]
    if counts[leaf] > 1 and state[0] > wp.float64(0.0):
        depth = state[0] + (mean_bed[leaf] - bed[k])
        if depth <= wp.float64(0.0):
            # This fixed representation cannot encode partly dry coarse cells.
            # Fail the run; never publish this as an accepted approximation.
            wp.atomic_add(ctrl, 7, wp.float64(1.0))
            return
        scale = depth / state[0]
        state = wp.vec3d(depth, state[1] * scale, state[2] * scale)
        if depth <= dry and (state[1] != wp.float64(0.0) or state[2] != wp.float64(0.0)):
            wp.atomic_add(ctrl, 7, wp.float64(1.0))
            return
    q[k] = state
    if record_history != 0:
        theta[k] = leaf_theta[leaf]
        maximum[k] = wp.max(maximum[k], state[0])


@wp.func
def face_cells(face: int, nx: int, ny: int, axis: int):
    left = int(-1)
    right = int(-1)
    if axis == 0:
        j = face / (nx + 1)
        i = face % (nx + 1)
        if i > 0:
            left = j * nx + i - 1
        if i < nx:
            right = j * nx + i
    else:
        j = face / nx
        i = face % nx
        if j > 0:
            left = (j - 1) * nx + i
        if j < ny:
            right = j * nx + i
    return wp.vec2i(left, right)


@wp.kernel
def subfaces(ids: wp.array(dtype=wp.int32), q: wp.array(dtype=wp.vec3d),
             bed: wp.array(dtype=wp.float64), active: wp.array(dtype=wp.int32),
             types: wp.array(dtype=wp.int32), stages: wp.array(dtype=wp.float64),
             flux: wp.array(dtype=wp.vec3d), pressure: wp.array(dtype=wp.vec2d),
             nx: int, ny: int, axis: int, g: wp.float64, dry: wp.float64,
             momentum_mode: int, ctrl: wp.array(dtype=wp.float64), flux_mode: int):
    face = ids[wp.tid()]
    faces_at(face, q, bed, active, types, stages, flux, pressure, nx, ny, axis,
             g, dry, momentum_mode, ctrl, flux_mode)


@wp.kernel
def collect_outflows(ids: wp.array(dtype=wp.int32), flux: wp.array(dtype=wp.vec3d),
                     owner: wp.array(dtype=wp.int32), outgoing: wp.array(dtype=wp.float64),
                     nx: int, ny: int, axis: int, width: wp.float64):
    face = ids[wp.tid()]
    cells = face_cells(face, nx, ny, axis)
    donor = cells[0]
    if flux[face][0] < wp.float64(0.0):
        donor = cells[1]
    if donor >= 0:
        leaf = owner[donor]
        if leaf >= 0:
            wp.atomic_add(outgoing, leaf, wp.abs(flux[face][0]) * width)


@wp.kernel
def donor_budgets(leaves: wp.array(dtype=wp.vec3d), counts: wp.array(dtype=wp.int32),
                  outgoing: wp.array(dtype=wp.float64), theta: wp.array(dtype=wp.float64),
                  ctrl: wp.array(dtype=wp.float64), cell_area: wp.float64):
    leaf = wp.tid()
    value = wp.float64(1.0)
    demand = outgoing[leaf] * ctrl[2]
    if demand > wp.float64(0.0):
        value = wp.min(value, leaves[leaf][0] * wp.float64(counts[leaf]) * cell_area / demand)
    theta[leaf] = value


@wp.kernel
def accept_subfaces(ids: wp.array(dtype=wp.int32), flux: wp.array(dtype=wp.vec3d),
                    pressure: wp.array(dtype=wp.vec2d), owner: wp.array(dtype=wp.int32),
                    theta: wp.array(dtype=wp.float64), delta: wp.array(dtype=wp.vec3d),
                    volume: wp.array(dtype=wp.float64), inflow: wp.array(dtype=wp.float64),
                    outflow: wp.array(dtype=wp.float64), ctrl: wp.array(dtype=wp.float64),
                    nx: int, ny: int, axis: int, width: wp.float64):
    face = ids[wp.tid()]
    cells = face_cells(face, nx, ny, axis)
    left = int(-1)
    right = int(-1)
    if cells[0] >= 0:
        left = owner[cells[0]]
    if cells[1] >= 0:
        right = owner[cells[1]]
    donor = left
    if flux[face][0] < wp.float64(0.0):
        donor = right
    f = flux[face]
    if donor >= 0:
        f = f * theta[donor]
    flux[face] = f
    volume[face] += f[0] * ctrl[2] * width
    pl = wp.vec3d(wp.float64(0.0), pressure[face][0], wp.float64(0.0))
    pr = wp.vec3d(wp.float64(0.0), pressure[face][1], wp.float64(0.0))
    if axis == 1:
        pl = wp.vec3d(wp.float64(0.0), wp.float64(0.0), pressure[face][0])
        pr = wp.vec3d(wp.float64(0.0), wp.float64(0.0), pressure[face][1])
    if left >= 0:
        wp.atomic_add(delta, left, -width * (f - pl))
    if right >= 0:
        wp.atomic_add(delta, right, width * (f - pr))
    exterior_cell = int(-1)
    amount = wp.float64(0.0)
    if left >= 0 and right < 0:
        exterior_cell = cells[0]
        amount = f[0] * ctrl[2] * width
    elif right >= 0 and left < 0:
        exterior_cell = cells[1]
        amount = -f[0] * ctrl[2] * width
    if exterior_cell >= 0:
        wp.atomic_add(outflow, exterior_cell, wp.max(amount, wp.float64(0.0)))
        wp.atomic_add(inflow, exterior_cell, -wp.min(amount, wp.float64(0.0)))


@wp.kernel
def update_leaves(leaves: wp.array(dtype=wp.vec3d), delta: wp.array(dtype=wp.vec3d),
                  counts: wp.array(dtype=wp.int32), manning: wp.array(dtype=wp.float64),
                  ctrl: wp.array(dtype=wp.float64), cell_area: wp.float64,
                  g: wp.float64, dry: wp.float64):
    leaf = wp.tid()
    value = leaves[leaf] + ctrl[2] / (cell_area * wp.float64(counts[leaf])) * delta[leaf]
    if value[0] < wp.float64(-1.e-12) or not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
        wp.atomic_add(ctrl, 7, wp.float64(1.0))
    depth = wp.max(value[0], wp.float64(0.0))
    hu = wp.float64(0.0)
    hv = wp.float64(0.0)
    if depth > dry:
        drag = ctrl[2] * g * manning[leaf] * manning[leaf] * wp.sqrt(value[1] * value[1] + value[2] * value[2]) / wp.pow(depth, wp.float64(7.0) / wp.float64(3.0))
        hu = value[1] / (wp.float64(1.0) + drag)
        hv = value[2] / (wp.float64(1.0) + drag)
    leaves[leaf] = wp.vec3d(depth, hu, hv)
