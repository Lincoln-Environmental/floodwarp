"""FP64 MUSCL, hydrostatic balance and transactional SSP-RK2 stage kernels."""

import warp as wp

from floodwarp.full_swe_kernels import hll, hll_speeds, stage_boundary_state

# FV2-only HLL flux reusing the wave bounds already needed for the CFL.
# Preserve the exact arithmetic of full_swe_kernels.hll; test parity on wet and dry faces.
@wp.func
def hll_fv2_precomputed(left: wp.vec3d, right: wp.vec3d, g: wp.float64, dry: wp.float64, flux_mode: int, speeds: wp.vec2d):
    ul = wp.vec2d(wp.float64(0.0))
    ur = wp.vec2d(wp.float64(0.0))
    if left[0] > dry:
        ul = wp.vec2d(left[1],left[2])/left[0]
    if right[0] > dry:
        ur = wp.vec2d(right[1],right[2])/right[0]
    sl = speeds[0]
    sr = speeds[1]
    fl = wp.vec3d(left[1], left[1] * ul[0] + wp.float64(0.5) * g * left[0] * left[0], left[1] * ul[1])
    fr = wp.vec3d(right[1], right[1] * ur[0] + wp.float64(0.5) * g * right[0] * right[0], right[1] * ur[1])
    result = wp.vec3d(wp.float64(0.0))
    if sl >= wp.float64(0.0):
        result = fl
    elif sr <= wp.float64(0.0):
        result = fr
    elif sr > sl:
        result = (sr * fl - sl * fr + sl * sr * (right - left)) / (sr - sl)
        if flux_mode == 1 or flux_mode == 3:
            # Algebraically equivalent to the contact speed estimate; its
            # denominator is positive for a nondegenerate wet HLL star state.
            dl = left[0]*(ul[0]-sl)
            dr = right[0]*(sr-ur[0])
            denominator = dl+dr
            if denominator > wp.float64(0.0):
                contact = (sr*dl+sl*dr)/denominator
                tangent = ur[1]
                if contact >= wp.float64(0.0):
                    tangent = ul[1]
                result[2] = result[0]*tangent
            # Degenerate star states retain the finite conservative HLL value.
    return result



@wp.func
def scheduled_index(thread: int, work: wp.array(dtype=wp.int32)):
    if work.shape[0] > 0:
        return work[thread]
    return thread


@wp.func
def limited(a: wp.float64, b: wp.float64, mc: int):
    zero = wp.float64(0.0)
    if a * b <= zero:
        return zero
    mag = wp.min(wp.abs(a), wp.abs(b))
    if mc != 0:
        mag = wp.min(wp.float64(2.0) * mag, wp.abs(a + b) * wp.float64(0.5))
    return wp.sign(a) * mag


@wp.func
def surface_bed_slopes(
    dz_back: wp.float64,
    dz_ahead: wp.float64,
    dh_back: wp.float64,
    dh_ahead: wp.float64,
    mc: int,
):
    bed = limited(dz_back, dz_ahead, mc)
    surface = limited(dh_back + dz_back, dh_ahead + dz_ahead, mc)
    # Neither neighbour's bed trace may cross the midpoint between their
    # centres. Otherwise a monotone descent can acquire an uphill face step.
    bound = wp.min(wp.abs(dz_back), wp.abs(dz_ahead))
    if wp.abs(bed) > bound:
        surface = surface * (bound / wp.abs(bed))
        bed = wp.sign(bed) * bound
    return wp.vec2d(surface, bed)


@wp.func
def neighbor(k: int, axis: int, side: int, nx: int, ny: int):
    if axis == 0:
        if k % nx + side >= 0 and k % nx + side < nx:
            return k + side
    elif k / nx + side >= 0 and k / nx + side < ny:
        return k + side * nx
    return int(-1)


@wp.func
def indices(face: int, axis: int, nx: int, ny: int):
    if axis == 0:
        i = face % (nx + 1)
        k = face - face / (nx + 1)
        return wp.vec2i(wp.where(i > 0, k - 1, -1), wp.where(i < nx, k, -1))
    return wp.vec2i(wp.where(face >= nx, face - nx, -1), wp.where(face < nx * ny, face, -1))


@wp.func
def drag(q: wp.vec3d, n: wp.float64, dt: wp.float64, g: wp.float64, dry: wp.float64):
    if q[0] <= dry:
        return wp.vec3d(q[0], wp.float64(0.0), wp.float64(0.0))
    if n == wp.float64(0.0) or (q[1] == wp.float64(0.0) and q[2] == wp.float64(0.0)):
        return q
    beta = (
        dt * g * n * n * wp.length(wp.vec2d(q[1], q[2])) / wp.pow(q[0], (wp.float64(7.0) / wp.float64(3.0)))
    )
    return wp.vec3d(q[0], q[1] / (wp.float64(1.0) + beta), q[2] / (wp.float64(1.0) + beta))


@wp.func
def decay(x: wp.float64):
    # Taylor remainder <= (1/256)^6 / 6! < 5e-18 on the polynomial branch.
    if x >= wp.float64(0.0) and x < wp.float64(1.0 / 256.0):
        return wp.float64(1.0) - x * (wp.float64(1.0) - x * (wp.float64(0.5) -
            x * (wp.float64(1.0 / 6.0) - x * (wp.float64(1.0 / 24.0) - x * wp.float64(1.0 / 120.0)))))
    return wp.exp(-x)


@wp.func
def water_exchange(
    q: wp.vec3d,
    soil: wp.float64,
    rain: wp.float64,
    dt: wp.float64,
    infil: wp.float64,
    f0: wp.float64,
    fd: wp.float64,
    cond: wp.float64,
):
    # Explicit midpoint for cumulative infiltration, with a finite supply.
    added = rain * dt
    if f0 == cond and infil == wp.float64(0.0) and cond == wp.float64(0.0):
        factor = wp.where(q[0] > wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        return wp.vec4d(q[0] + added, factor, added, wp.float64(0.0))
    initial = (f0 - cond) * wp.exp(-soil / fd)
    base = infil + cond
    mid = wp.float64(0.5) * wp.min(q[0] + added, dt * (base + initial))
    removed = wp.min(q[0] + added, dt * (base + initial * decay(mid / fd)))
    h = (q[0] + added) - removed
    factor = wp.float64(0.0)
    if h > wp.float64(0.0) and q[0] > wp.float64(0.0):
        # Integrate dm/dt=-I*m/h for constant interval-mean rain and loss.
        delta = added - removed
        inverse = wp.float64(2.0) / (q[0] + h)
        if wp.abs(delta) > wp.float64(1.0e-5) * q[0]:
            r = delta / (q[0] + h)
            if wp.abs(r) < wp.float64(0.01):
                # log(h/h0)/delta = 2*atanh(r)/(r*(h+h0)).
                # Relative tail beyond r^6 is below 1.2e-17.
                rr = r * r
                inverse *= wp.float64(1.0) + rr * (wp.float64(1.0/3.0) + rr *
                           (wp.float64(1.0/5.0) + rr * wp.float64(1.0/7.0)))
            else:
                inverse = wp.log(h / q[0]) / delta
        factor = decay(removed * inverse)
    return wp.vec4d(h, factor, added, removed)


@wp.func
def wet_inverse(h: wp.float64, dry: wp.float64):
    if h > dry:
        return wp.float64(1.0) / h
    return wp.float64(0.0)


@wp.kernel
def split_sources(
    q: wp.array(dtype=wp.vec3d),
    start: wp.array(dtype=wp.vec3d),
    end: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    n: wp.array(dtype=wp.float64),
    soil: wp.array(dtype=wp.float64),
    rain: wp.array(dtype=wp.float64),
    mult: wp.array(dtype=wp.float64),
    infil: wp.array(dtype=wp.float64),
    f0: wp.array(dtype=wp.float64),
    fd: wp.array(dtype=wp.float64),
    cond: wp.array(dtype=wp.float64),
    amounts: wp.array(dtype=wp.vec2d),
    rain_sum: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    g: wp.float64,
    dry: wp.float64,
    stage: int,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    if active[k] == 0:
        return
    dt = ctrl[2] * wp.float64(0.5)
    state = q[k]
    rate = rain[k] + mult[k] * (ctrl[9] + (wp.float64(0.5) + wp.float64(stage)) * dt * ctrl[11])
    history = soil[k]
    if stage == 1:
        if ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
            return
        state = drag(wp.float64(0.5) * (start[k] + end[k]), n[k], dt, g, dry)
        history += amounts[k][1]
    change = water_exchange(state, history, rate, dt, infil[k], f0[k], fd[k], cond[k])
    state = wp.vec3d(change[0], state[1] * change[1], state[2] * change[1])
    if stage == 0:
        start[k] = drag(state, n[k], dt, g, dry)
        inverse[k] = wet_inverse(state[0], dry)
        amounts[k] = wp.vec2d(change[2], change[3])
    else:
        if state[0] <= dry:
            state[1] = wp.float64(0.0)
            state[2] = wp.float64(0.0)
        q[k] = state
        soil[k] += amounts[k][1] + change[3]
        rain_sum[k] += amounts[k][0] + change[2]
        maximum[k] = wp.max(maximum[k], state[0])


@wp.func
def compression_rule(left: wp.vec3d, right: wp.vec3d, axis: int, gravity: wp.float64):
    # A compressing acoustic family straddles zero at a stationary jump.
    ul = left[axis + 1] / left[0]
    ur = right[axis + 1] / right[0]
    if ul > ur and wp.abs(left[0] - right[0]) > wp.float64(0.04) * wp.min(left[0], right[0]):
        cl = wp.sqrt(gravity * left[0])
        cr = wp.sqrt(gravity * right[0])
        if (ul > wp.float64(0.8)*cl and ur < wp.float64(1.25)*cr and right[0] > left[0]) or (ul > -wp.float64(1.25)*cl and ur < -wp.float64(0.8)*cr and left[0] > right[0]):
            return 0
    return 1


@wp.kernel
def inverse_depth(q: wp.array(dtype=wp.vec3d), inverse: wp.array(dtype=wp.float64), dry: wp.float64,
                  work: wp.array(dtype=wp.int32)):
    k = scheduled_index(wp.tid(), work)
    value = wp.float64(0.0)
    if q[k][0] > dry:
        value = wp.float64(1.0) / q[k][0]
    inverse[k] = value


@wp.func
def slope(
    k: int,
    axis: int,
    q: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    bed: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    dry: wp.float64,
    mc: int,
    gravity: wp.float64,
):
    a = neighbor(k, axis, -1, nx, ny)
    b = neighbor(k, axis, 1, nx, ny)
    zero = wp.float64(0.0)
    if active[k] == 0 or a < 0 or b < 0:
        return wp.vec4d(zero)
    if active[a] == 0 or active[b] == 0 or wp.min(q[k][0], wp.min(q[a][0], q[b][0])) <= dry:
        return wp.vec4d(zero)
    h = q[k][0]
    if q[a] == q[k] and q[b] == q[k] and bed[a] == bed[k] and bed[b] == bed[k]:
        return wp.vec4d(zero)
    da = bed[k] - bed[a]
    db = bed[b] - bed[k]
    rule = mc
    if mc == 2:
        rule = compression_rule(q[a], q[b], axis, gravity)
    geometry = surface_bed_slopes(da, db, h - q[a][0], q[b][0] - h, rule)
    se = geometry[0]
    sz = geometry[1]
    velocities = wp.vec2d(q[k][1], q[k][2]) * inverse[k]
    va = wp.vec2d(q[a][1], q[a][2]) * inverse[a]
    vb = wp.vec2d(q[b][1], q[b][2]) * inverse[b]
    s = wp.vec4d(
        se,
        sz,
        limited(velocities[0] - va[0], vb[0] - velocities[0], rule),
        limited(velocities[1] - va[1], vb[1] - velocities[1], rule),
    )
    if wp.abs(se - sz) >= wp.float64(2.0) * h:
        # A zero order trace is safer than an almost empty moving face.
        return wp.vec4d(zero)
    return s


@wp.kernel
def slopes(
    q: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    bed: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    nx: int,
    ny: int,
    dry: wp.float64,
    mc: int,
    gravity: wp.float64,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    sx[k] = slope(k, 0, q, inverse, bed, active, nx, ny, dry, mc, gravity)
    sy[k] = slope(k, 1, q, inverse, bed, active, nx, ny, dry, mc, gravity)


@wp.func
def trace(q: wp.vec3d, s: wp.vec4d, side: wp.float64, dry: wp.float64, axis: int):
    half = wp.float64(0.5) * side
    delta = half * (s[0] - s[1])
    h = q[0] + delta
    v = wp.vec2d(wp.float64(0.0))
    if q[0] > dry and (q[1] != wp.float64(0.0) or q[2] != wp.float64(0.0) or s[2] != wp.float64(0.0) or s[3] != wp.float64(0.0)):
        v = (wp.vec2d(q[1], q[2]) + half * (q[0] - delta) * wp.vec2d(s[2], s[3])) / q[0]
    if axis == 1:
        v = wp.vec2d(v[1], v[0])
    return wp.vec3d(h, h * v[0], h * v[1])


@wp.func
def cut(q: wp.vec3d, rise: wp.float64, dry: wp.float64):
    h = wp.max(wp.float64(0.0), q[0] - wp.max(wp.float64(0.0), rise))
    factor = wp.float64(0.0)
    if q[0] > dry:
        if rise <= wp.float64(0.0):
            return q
        factor = h / q[0]
    return wp.vec3d(h, q[1] * factor, q[2] * factor)


@wp.func
def cut_trace(q: wp.vec3d, inverse: wp.float64, s: wp.vec4d, side: wp.float64, rise: wp.float64, axis: int, dry: wp.float64):
    # Return the hydrostatic depth and the two velocities that multiply it.
    # A trace already below the dry threshold carries no post-cut momentum.
    half = wp.float64(0.5) * side
    delta = half * (s[0] - s[1])
    original = q[0] + delta
    depth = wp.max(wp.float64(0.0), original - wp.max(rise, wp.float64(0.0)))
    velocity = wp.vec2d(wp.float64(0.0))
    if original > dry and q[0] > dry:
        velocity = (wp.vec2d(q[1], q[2]) + half * (q[0] - delta) * wp.vec2d(s[2], s[3])) * inverse
    if axis == 1:
        velocity = wp.vec2d(velocity[1], velocity[0])
    return wp.vec3d(depth, velocity[0], velocity[1])


@wp.func
def primitive_contact(left: wp.vec3d, right: wp.vec3d, g: wp.float64, dry: wp.float64):
    zero = wp.float64(0.0)
    l = wp.vec3d(left[0], left[0]*left[1], left[0]*left[2])
    r = wp.vec3d(right[0], right[0]*right[1], right[0]*right[2])
    ul = left[1]
    ur = right[1]
    vl = left[2]
    vr = right[2]
    if l[0] <= dry:
        ul = zero
        vl = zero
    if r[0] <= dry:
        ur = zero
        vr = zero
    cl = wp.sqrt(g*l[0])
    cr = wp.sqrt(g*r[0])
    sl = wp.min(ul-cl,ur-cr)
    sr = wp.max(ul+cl,ur+cr)
    if r[0] <= dry:
        sl = ul-cl
        sr = ul+wp.float64(2.0)*cl
    if l[0] <= dry:
        sl = ur-wp.float64(2.0)*cr
        sr = ur+cr
    fl = wp.vec3d(l[1], l[1]*ul + wp.float64(0.5)*g*l[0]*l[0], l[1]*vl)
    fr = wp.vec3d(r[1], r[1]*ur + wp.float64(0.5)*g*r[0]*r[0], r[1]*vr)
    result = fl
    if sl < zero:
        result = fr
        if sr > zero:
            result = (sr*fl-sl*fr+sl*sr*(r-l))/(sr-sl)
            dl = l[0]*(ul-sl)
            dr = r[0]*(sr-ur)
            if dl+dr > zero:
                result[2] = result[0]*wp.where(sr*dl+sl*dr >= zero, vl, vr)
    return wp.vec4d(result[0], result[1], result[2], wp.max(wp.abs(sl),wp.abs(sr)))



# Compact contiguous runs of face indices. The list is rebuilt from each
# post-source RK stage. Every excluded face has exactly zero incident depth.
TILE_FACES = 32


@wp.kernel
def reset_tiles(x: wp.array(dtype=wp.int32), y: wp.array(dtype=wp.int32)):
    x[0] = 0
    y[0] = 0


@wp.kernel
def schedule_faces(
    q: wp.array(dtype=wp.vec3d),
    active: wp.array(dtype=wp.int32),
    types: wp.array(dtype=wp.int32),
    flux: wp.array(dtype=wp.vec3d),
    pressure: wp.array(dtype=wp.vec2d),
    tiles: wp.array(dtype=wp.int32),
    nx: int, ny: int, axis: int,
):
    tile = wp.tid()
    first = tile * TILE_FACES
    end = wp.min(first + TILE_FACES, flux.shape[0])
    live = bool(False)
    for k in range(first, end):
        ab = indices(k, axis, nx, ny)
        if types[k] == 3:
            live = True
            break
        for side in range(2):
            cell = ab[side]
            if cell >= 0 and active[cell] != 0 and q[cell][0] != wp.float64(0.0):
                live = True
        if live:
            break
    if live:
        slot = wp.atomic_add(tiles, 0, 1)
        tiles[1 + slot] = tile
    else:
        for k in range(first, end):
            flux[k] = wp.vec3d(wp.float64(0.0))
            pressure[k] = wp.vec2d(wp.float64(0.0))


@wp.kernel
def faces(
    q: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    bed: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    s: wp.array(dtype=wp.vec4d),
    types: wp.array(dtype=wp.int32),
    stages: wp.array(dtype=wp.float64),
    stage_series: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    flux: wp.array(dtype=wp.vec3d),
    pressure: wp.array(dtype=wp.vec2d),
    nx: int,
    ny: int,
    axis: int,
    spacing: wp.float64,
    g: wp.float64,
    dry: wp.float64,
    cfl: wp.float64,
    mode: int,
    sparse: int,
    tiles: wp.array(dtype=wp.int32),
    work: wp.array(dtype=wp.int32),
):
    thread = wp.tid()
    k = thread
    if tiles.shape[0] > 0:
        slot = thread / TILE_FACES
        if slot >= tiles[0]:
            return
        k = tiles[slot + 1] * TILE_FACES + thread % TILE_FACES
        if k >= flux.shape[0]:
            return
    else:
        k = scheduled_index(thread, work)
    pair = indices(k, axis, nx, ny)
    a = pair[0]
    b = pair[1]
    al = a >= 0 and active[wp.max(a, 0)] != 0
    ar = b >= 0 and active[wp.max(b, 0)] != 0
    zero = wp.float64(0.0)
    flux[k] = wp.vec3d(zero)
    pressure[k] = wp.vec2d(zero)
    if not al and not ar:
        return
    if sparse != 0 and types[k] != 3 and q[wp.max(a, 0)][0] == zero and q[wp.max(b, 0)][0] == zero:
        return
    l = wp.vec3d(zero)
    r = wp.vec3d(zero)
    speed = zero
    f = wp.vec3d(zero)
    if al and ar:
        rise = (bed[b] - bed[a]) - wp.float64(0.5) * (s[a][1] + s[b][1])
        l = cut_trace(q[a], inverse[a], s[a], wp.float64(1.0), rise, axis, dry)
        r = cut_trace(q[b], inverse[b], s[b], wp.float64(-1.0), -rise, axis, dry)
        result = primitive_contact(l, r, g, dry)
        f = wp.vec3d(result[0], result[1], result[2])
        speed = result[3]
    else:
        if al:
            l = trace(q[a], s[a], wp.float64(1.0), dry, axis)
        if ar:
            r = trace(q[b], s[b], wp.float64(-1.0), dry, axis)
        inside = r
        cell = b
        outward = wp.float64(-1.0)
        if al:
            inside = l
            cell = a
            outward = wp.float64(1.0)
        ghost = inside
        if types[k] <= 1:
            ghost[1] = -inside[1]
        elif types[k] == 3:
            level = stages[k]
            if stage_series[0] > zero:
                level = stage_series[1]
            hg = wp.max(zero, (level - bed[cell]) - outward * wp.float64(0.5) * s[cell][1])
            ghost = stage_boundary_state(inside, hg, dry, mode, g, outward)
            if ghost[0] < zero:
                wp.atomic_min(ctrl, 7, wp.float64(-1.0))
                return
        if al:
            r = ghost
        else:
            l = ghost
        speeds = hll_speeds(l, r, g, dry, 1)
        f = hll_fv2_precomputed(l, r, g, dry, 1, speeds)
        speed = wp.max(wp.abs(speeds[0]), wp.abs(speeds[1]))
    if speed > zero:
        wp.atomic_min(ctrl, 17, wp.float64(0.5) * spacing * cfl / speed)
    if al != ar and types[k] <= 1:
        f = wp.vec3d(zero, f[1], zero)
    if al != ar and types[k] == 2 and ((al and f[0] < zero) or (ar and f[0] > zero)):
        inside_h = wp.where(al, l[0], r[0])
        f = wp.vec3d(zero, wp.float64(0.5) * g * inside_h * inside_h, zero)
    if axis == 1:
        f = wp.vec3d(f[0], f[2], f[1])
    flux[k] = f
    pressure[k] = wp.float64(0.5) * g * wp.vec2d(l[0] * l[0], r[0] * r[0])


@wp.kernel
def donor_budget(
    q: wp.array(dtype=wp.vec3d),
    theta: wp.array(dtype=wp.float64),
    fx: wp.array(dtype=wp.vec3d),
    fy: wp.array(dtype=wp.vec3d),
    ctrl: wp.array(dtype=wp.float64),
    nx: int,
    dx: wp.float64,
    dy: wp.float64,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    zero = wp.float64(0.0)
    dt = ctrl[2]
    w = k + k / nx
    out = (wp.max(zero, fx[w + 1][0]) - wp.min(zero, fx[w][0])) / dx
    out += (wp.max(zero, fy[k + nx][0]) - wp.min(zero, fy[k][0])) / dy
    factor = wp.float64(1.0)
    if out * dt > zero:
        budget = q[k][0]
        if out * dt >= budget:
            factor = (wp.float64(1.0) - wp.float64(2.0e-15)) * budget / (out * dt)
    theta[k] = factor


@wp.kernel
def limit_faces(
    f: wp.array(dtype=wp.vec3d),
    theta: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    axis: int,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    pair = indices(k, axis, nx, ny)
    donor = wp.where(f[k][0] >= wp.float64(0.0), pair[0], pair[1])
    if donor >= 0 and active[donor] != 0:
        f[k] *= theta[donor]


@wp.kernel
def euler(
    q: wp.array(dtype=wp.vec3d),
    dest: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    stage: int,
    active: wp.array(dtype=wp.int32),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    fx: wp.array(dtype=wp.vec3d),
    fy: wp.array(dtype=wp.vec3d),
    px: wp.array(dtype=wp.vec2d),
    py: wp.array(dtype=wp.vec2d),
    ctrl: wp.array(dtype=wp.float64),
    nx: int,
    dx: wp.float64,
    dy: wp.float64,
    g: wp.float64,
    dry: wp.float64,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    if active[k] == 0 or ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
        dest[k] = q[k]
        if stage == 0:
            inverse[k] = wet_inverse(q[k][0], dry)
        return
    w = k + k / nx
    h = q[k][0]
    dt = ctrl[2]
    v = q[k]
    west = fx[w]
    east = fx[w + 1]
    south = fy[k]
    north = fy[k + nx]
    west[1] -= px[w][1]
    east[1] -= px[w + 1][0]
    south[2] -= py[k][1]
    north[2] -= py[k + nx][0]
    v += (dt / dx) * (west - east) + (dt / dy) * (south - north)
    v[1] -= dt / dx * g * h * sx[k][0]
    v[2] -= dt / dy * g * h * sy[k][0]
    if v[0] < wp.float64(0.0) or not wp.isfinite(v):
        wp.atomic_add(ctrl, 7, wp.float64(1.0))
    if v[0] <= dry:
        v[1] = wp.float64(0.0)
        v[2] = wp.float64(0.0)
    dest[k] = v
    if stage == 0:
        inverse[k] = wet_inverse(v[0], dry)


@wp.kernel
def accept_faces(
    first: wp.array(dtype=wp.vec3d),
    second: wp.array(dtype=wp.vec3d),
    flux: wp.array(dtype=wp.vec3d),
    volume: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    inflow: wp.array(dtype=wp.float64),
    outflow: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    axis: int,
    width: wp.float64,
    work: wp.array(dtype=wp.int32),
):
    if ctrl[2] <= ctrl[17] and ctrl[7] == wp.float64(0.0):
        k = scheduled_index(wp.tid(), work)
        mean = wp.float64(0.5) * (first[k] + second[k])
        flux[k] = mean
        volume[k] += ctrl[2] * width * mean[0]
        pair = indices(k, axis, nx, ny)
        a = pair[0] >= 0 and active[wp.max(pair[0], 0)] != 0
        b = pair[1] >= 0 and active[wp.max(pair[1], 0)] != 0
        if a != b:
            sign = wp.where(a, wp.float64(1.0), wp.float64(-1.0))
            cell = wp.where(a, pair[0], pair[1])
            scale = wp.float64(0.5) * ctrl[2] * width
            p = sign * first[k][0]
            r = sign * second[k][0]
            wp.atomic_add(outflow, cell, scale * (wp.max(p, wp.float64(0.0)) + wp.max(r, wp.float64(0.0))))
            wp.atomic_add(inflow, cell, -scale * (wp.min(p, wp.float64(0.0)) + wp.min(r, wp.float64(0.0))))


@wp.kernel
def predictor(ctrl: wp.array(dtype=wp.float64), previous: wp.array(dtype=wp.float64)):
    if previous[0] > wp.float64(0.0):
        ctrl[2] = wp.min(ctrl[2], previous[0])
    ctrl[17] = wp.float64(1.0e100)


@wp.kernel
def forcing_time(
    times: wp.array(dtype=wp.float64),
    levels: wp.array(dtype=wp.float64),
    sample: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    stage: int,
):
    if times.shape[0] == 1:
        sample[0] = wp.float64(0.0)
        return
    t = ctrl[0]
    i = int(0)
    while i < times.shape[0] - 2 and t >= times[i + 1]:
        i += 1
    if t < times[0] or t >= times[times.shape[0] - 1]:
        wp.atomic_add(ctrl, 7, wp.float64(1.0))
        return
    if stage == 0:
        ctrl[2] = wp.min(ctrl[2], times[i + 1] - t)
    sample[0] = wp.float64(1.0)
    sample[1] = levels[i] + ((t - times[i]) + wp.float64(stage) * ctrl[2]) / (times[i + 1] - times[i]) * (
        levels[i + 1] - levels[i]
    )


@wp.kernel
def finish(
    ctrl: wp.array(dtype=wp.float64),
    previous: wp.array(dtype=wp.float64),
    condition: wp.array(dtype=wp.int32),
    histogram: wp.array(dtype=wp.int64),
):
    dt = ctrl[2]
    bound = ctrl[17]
    previous[0] = wp.float64(0.95) * bound
    if ctrl[7] != wp.float64(0.0):
        condition[0] = 0
        return
    if dt > bound:
        ctrl[16] += wp.float64(1.0)
        condition[0] = 1
        return
    ctrl[0] += dt
    ctrl[3] += wp.float64(1.0)
    ctrl[4] = wp.min(ctrl[4], dt)
    ctrl[5] = wp.max(ctrl[5], dt)
    ctrl[6] += dt * dt
    ctrl[14] = wp.min(ctrl[14], bound)
    bucket = wp.clamp(int(wp.floor(wp.log(dt) / wp.log(wp.float64(2.0)))) + 32, 0, 63)
    histogram[bucket] += wp.int64(1)
    condition[0] = int(ctrl[0] < ctrl[1] and ctrl[3] < ctrl[13])


@wp.kernel
def validate_trial(ctrl: wp.array(dtype=wp.float64), minimum: wp.float64):
    if ctrl[7] == wp.float64(0.0) and (
        ctrl[17] < minimum or ctrl[2] <= wp.float64(0.0) or ctrl[0] + ctrl[2] == ctrl[0]
    ):
        ctrl[14] = wp.min(ctrl[14], ctrl[17])
        ctrl[7] += wp.float64(1.0)
