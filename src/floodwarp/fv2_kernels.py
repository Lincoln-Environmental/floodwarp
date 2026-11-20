"""FP64 MUSCL, hydrostatic balance and transactional SSP-RK2 stage kernels."""

import warp as wp

from floodwarp.full_swe_kernels import hll, hll_speeds, stage_boundary_state


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
    if n == wp.float64(0.0):
        return q
    beta = (
        dt * g * n * n * wp.length(wp.vec2d(q[1], q[2])) / wp.pow(q[0], (wp.float64(7.0) / wp.float64(3.0)))
    )
    return wp.vec3d(q[0], q[1] / (wp.float64(1.0) + beta), q[2] / (wp.float64(1.0) + beta))


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
    initial = (f0 - cond) * wp.exp(-soil / fd)
    base = infil + cond
    mid = wp.float64(0.5) * wp.min(q[0] + added, dt * (base + initial))
    removed = wp.min(q[0] + added, dt * (base + initial * wp.exp(-mid / fd)))
    h = (q[0] + added) - removed
    factor = wp.float64(0.0)
    if h > wp.float64(0.0) and q[0] > wp.float64(0.0):
        # Integrate dm/dt=-I*m/h for constant interval-mean rain and loss.
        delta = added - removed
        inverse = wp.float64(2.0) / (q[0] + h)
        if wp.abs(delta) > wp.float64(1.0e-5) * q[0]:
            inverse = wp.log(h / q[0]) / delta
        factor = wp.exp(-removed * inverse)
    return wp.vec4d(h, factor, added, removed)


@wp.kernel
def split_sources(
    q: wp.array(dtype=wp.vec3d),
    start: wp.array(dtype=wp.vec3d),
    end: wp.array(dtype=wp.vec3d),
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
def slope(
    k: int,
    axis: int,
    q: wp.array(dtype=wp.vec3d),
    bed: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    dry: wp.float64,
    mc: int,
):
    a = neighbor(k, axis, -1, nx, ny)
    b = neighbor(k, axis, 1, nx, ny)
    zero = wp.float64(0.0)
    if active[k] == 0 or a < 0 or b < 0:
        return wp.vec4d(zero)
    if active[a] == 0 or active[b] == 0 or wp.min(q[k][0], wp.min(q[a][0], q[b][0])) <= dry:
        return wp.vec4d(zero)
    h = q[k][0]
    da = bed[k] - bed[a]
    db = bed[b] - bed[k]
    sz = limited(da, db, mc)
    se = limited((h - q[a][0]) + da, (q[b][0] - h) + db, mc)
    velocities = wp.vec2d(q[k][1], q[k][2]) / h
    va = wp.vec2d(q[a][1], q[a][2]) / q[a][0]
    vb = wp.vec2d(q[b][1], q[b][2]) / q[b][0]
    s = wp.vec4d(
        se,
        sz,
        limited(velocities[0] - va[0], vb[0] - velocities[0], mc),
        limited(velocities[1] - va[1], vb[1] - velocities[1], mc),
    )
    if wp.abs(se - sz) >= wp.float64(2.0) * h:
        # A zero order trace is safer than an almost empty moving face.
        return wp.vec4d(zero)
    return s


@wp.kernel
def slopes(
    q: wp.array(dtype=wp.vec3d),
    bed: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    nx: int,
    ny: int,
    dry: wp.float64,
    mc: int,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    sx[k] = slope(k, 0, q, bed, active, nx, ny, dry, mc)
    sy[k] = slope(k, 1, q, bed, active, nx, ny, dry, mc)


@wp.func
def trace(q: wp.vec3d, s: wp.vec4d, side: wp.float64, dry: wp.float64, axis: int):
    half = wp.float64(0.5) * side
    delta = half * (s[0] - s[1])
    h = q[0] + delta
    v = wp.vec2d(wp.float64(0.0))
    if q[0] > dry:
        v = (wp.vec2d(q[1], q[2]) + half * (q[0] - delta) * wp.vec2d(s[2], s[3])) / q[0]
    if axis == 1:
        v = wp.vec2d(v[1], v[0])
    return wp.vec3d(h, h * v[0], h * v[1])


@wp.func
def cut(q: wp.vec3d, rise: wp.float64, dry: wp.float64):
    h = wp.max(wp.float64(0.0), q[0] - wp.max(wp.float64(0.0), rise))
    factor = wp.float64(0.0)
    if q[0] > dry:
        factor = h / q[0]
    return wp.vec3d(h, q[1] * factor, q[2] * factor)


@wp.kernel
def faces(
    q: wp.array(dtype=wp.vec3d),
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
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
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
    if al:
        l = trace(q[a], s[a], wp.float64(1.0), dry, axis)
    if ar:
        r = trace(q[b], s[b], wp.float64(-1.0), dry, axis)
    if al and ar:
        rise = (bed[b] - bed[a]) - wp.float64(0.5) * (s[a][1] + s[b][1])
        l = cut(l, rise, dry)
        r = cut(r, -rise, dry)
    else:
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
    f = hll(l, r, g, dry, 1)
    speeds = hll_speeds(l, r, g, dry, 1)
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
