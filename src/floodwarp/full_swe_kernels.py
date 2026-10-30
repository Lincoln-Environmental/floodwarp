"""Independent FP64 finite-volume SWE kernels. No local-inertial operators."""

import warp as wp


@wp.func
def stage_ghost(inside: wp.vec3d, depth: wp.float64, dry: wp.float64, momentum_mode: int):
    """Prescribed depth with explicit discharge/velocity extrapolation semantics."""
    ghost=wp.vec3d(depth,wp.float64(0.0),wp.float64(0.0))
    if depth>dry:
        if momentum_mode==1:
            if inside[0]>dry:
                ghost[1]=(inside[1]/inside[0])*depth
                ghost[2]=(inside[2]/inside[0])*depth
        else:
            ghost[1]=inside[1]
            ghost[2]=inside[2]
    return ghost


@wp.func
def stage_boundary_state(inside: wp.vec3d, depth: wp.float64, dry: wp.float64,
                         momentum_mode: int, g: wp.float64, outward: wp.float64):
    """Normal/tangent state at the local hydraulic sill; negative h rejects.

    Incoming transverse velocity is zero (external flow normal to the face).
    A stage below critical depth gives a critical free discharge, not an
    imposed supercritical downstream state. Dry-land inundation and incoming
    supercritical states require additional external momentum data.
    """
    if momentum_mode != 2:
        return stage_ghost(inside, depth, dry, momentum_mode)
    zero = wp.float64(0.0)
    invalid = wp.vec3d(wp.float64(-1.0), zero, zero)
    if inside[0] <= dry:
        if depth > dry:
            return invalid
        return wp.vec3d(zero)
    un = outward * inside[1] / inside[0]
    ci = wp.sqrt(g * inside[0])
    if un >= ci:
        return inside
    if un <= -ci:
        return invalid
    invariant = un + wp.float64(2.0) * ci
    cb = wp.sqrt(g * depth)
    # The rarefaction becomes sonic at the fixed outlet face.
    cb = wp.max(cb, invariant / wp.float64(3.0))
    hb = cb * cb / g
    ub = invariant - wp.float64(2.0) * cb
    if ub <= -cb:
        return invalid
    if hb <= dry:
        return wp.vec3d(hb, zero, zero)
    tangent = zero
    if ub >= zero:
        tangent = inside[2] / inside[0]
    return wp.vec3d(hb, outward * ub * hb, tangent * hb)


@wp.func
def source_state(inside: wp.vec3d, added: wp.float64, removed: wp.float64):
    available = inside[0] + added
    remaining = wp.max(wp.float64(0.0), available - removed)
    scale = wp.float64(0.0)
    if available > wp.float64(0.0):
        scale = remaining / available
    return wp.vec3d(remaining, inside[1] * scale, inside[2] * scale)


@wp.func
def hll_speeds(left: wp.vec3d, right: wp.vec3d, g: wp.float64, dry: wp.float64, flux_mode: int):
    ul = wp.vec2d(wp.float64(0.0))
    ur = wp.vec2d(wp.float64(0.0))
    if left[0] > dry:
        ul = wp.vec2d(left[1], left[2]) / left[0]
    if right[0] > dry:
        ur = wp.vec2d(right[1], right[2]) / right[0]
    cl = wp.sqrt(g * left[0])
    cr = wp.sqrt(g * right[0])
    sl = wp.min(ul[0] - cl, ur[0] - cr)
    sr = wp.max(ul[0] + cl, ur[0] + cr)
    if flux_mode >= 2 and left[0] > dry and right[0] > dry:
        predictor = wp.float64(0.5)*(cl+cr)+wp.float64(0.25)*(ul[0]-ur[0])
        # A negative two-rarefaction predictor denotes vacuum. Squaring it
        # would create an unphysical positive star depth; use endpoint bounds.
        if predictor > wp.float64(0.0):
            us = wp.float64(0.5)*(ul[0]+ur[0])+cl-cr
            sl = wp.min(ul[0]-cl,us-predictor)
            sr = wp.max(ur[0]+cr,us+predictor)
    if right[0] <= dry:
        sl = ul[0] - cl
        sr = ul[0] + wp.float64(2.0) * cl
    if left[0] <= dry:
        sl = ur[0] - wp.float64(2.0) * cr
        sr = ur[0] + cr
    return wp.vec2d(sl,sr)


@wp.func
def hll(left: wp.vec3d, right: wp.vec3d, g: wp.float64, dry: wp.float64, flux_mode: int = 0):
    ul = wp.vec2d(wp.float64(0.0))
    ur = wp.vec2d(wp.float64(0.0))
    if left[0] > dry:
        ul = wp.vec2d(left[1],left[2])/left[0]
    if right[0] > dry:
        ur = wp.vec2d(right[1],right[2])/right[0]
    speeds = hll_speeds(left,right,g,dry,flux_mode)
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


@wp.kernel
def begin_step(ctrl: wp.array(dtype=wp.float64), times: wp.array(dtype=wp.float64),
               rates: wp.array(dtype=wp.float64), count: int):
    index = int(ctrl[10])
    while index < count - 2 and ctrl[0] >= times[index + 1]:
        index += 1
    ctrl[10] = wp.float64(index)
    slope = (rates[index + 1] - rates[index]) / (times[index + 1] - times[index])
    rate = rates[index] + slope * (ctrl[0] - times[index])
    ctrl[9] = rate
    ctrl[11] = slope
    ctrl[2] = wp.max(wp.float64(0.0), wp.min(ctrl[8], wp.min(ctrl[1] - ctrl[0], times[index + 1] - ctrl[0])))
    ctrl[12] = wp.max(rate, rate + slope * ctrl[2])


@wp.kernel
def cfl_reduce(q: wp.array(dtype=wp.vec3d), active: wp.array(dtype=wp.int32),
               rain: wp.array(dtype=wp.float64), multiplier: wp.array(dtype=wp.float64),
               bed: wp.array(dtype=wp.float64), bx: wp.array(dtype=wp.int32),
               by: wp.array(dtype=wp.int32), sx: wp.array(dtype=wp.float64),
               sy: wp.array(dtype=wp.float64), nx: int,
               ctrl: wp.array(dtype=wp.float64), dx: wp.float64, dy: wp.float64,
               g: wp.float64, dry: wp.float64, cfl: wp.float64, stage_momentum_mode: int):
    k = wp.tid()
    if active[k] != 0:
        u = wp.float64(0.0)
        v = wp.float64(0.0)
        if q[k][0] > dry:
            u = q[k][1] / q[k][0]
            v = q[k][2] / q[k][0]
        # Use the immutable cap, not the concurrently reduced dt.
        wave = wp.sqrt(g * (q[k][0] + ctrl[8] * (rain[k] + multiplier[k] * ctrl[12])))
        ax = wp.abs(u) + wave
        ay = wp.abs(v) + wave
        j = k / nx
        i = k % nx
        west = j * (nx + 1) + i
        # A wet stage ghost can be deeper than initially dry land, or have a
        # greater velocity when extrapolated momentum meets a shallower stage.
        # Its incoming characteristics must participate in the stability bound.
        for offset in range(2):
            if bx[west + offset] == 3:
                hghost = wp.max(wp.float64(0.0), sx[west + offset] - bed[k])
                outward = wp.float64(2 * offset - 1)
                ghost = stage_boundary_state(q[k],hghost,dry,stage_momentum_mode,g,outward)
                velocity = wp.float64(0.0)
                if ghost[0] > dry:
                    velocity = wp.abs(ghost[1]) / ghost[0]
                ax = wp.max(ax, velocity + wp.sqrt(g * wp.max(ghost[0],wp.float64(0.0))))
            face_y = k + offset * nx
            if by[face_y] == 3:
                hghost = wp.max(wp.float64(0.0), sy[face_y] - bed[k])
                outward = wp.float64(2 * offset - 1)
                normal = wp.vec3d(q[k][0],q[k][2],q[k][1])
                ghost = stage_boundary_state(normal,hghost,dry,stage_momentum_mode,g,outward)
                velocity = wp.float64(0.0)
                if ghost[0] > dry:
                    velocity = wp.abs(ghost[1]) / ghost[0]
                ay = wp.max(ay, velocity + wp.sqrt(g * wp.max(ghost[0],wp.float64(0.0))))
        rate = ax / dx + ay / dy
        if rate > wp.float64(0.0):
            wp.atomic_min(ctrl, 2, cfl / rate)
            wp.atomic_min(ctrl, 14, cfl / rate)


@wp.func
def cell_wave_interval(k: int, axis: int, top: wp.float64,
        q: wp.array(dtype=wp.vec3d), bed: wp.array(dtype=wp.float64),
        rain: wp.array(dtype=wp.float64), multiplier: wp.array(dtype=wp.float64),
        infil: wp.array(dtype=wp.float64), f0: wp.array(dtype=wp.float64), fd: wp.array(dtype=wp.float64),
        conductivity: wp.array(dtype=wp.float64), soil: wp.array(dtype=wp.float64),
        ctrl: wp.array(dtype=wp.float64), g: wp.float64, dry: wp.float64):
    """Bounds on normal velocity and celerity for every dt <= immutable cap."""
    zero = wp.float64(0.0)
    h = q[k][0]
    added = ctrl[8]*(rain[k]+multiplier[k]*ctrl[12])
    capacity = infil[k]+conductivity[k]+(f0[k]-conductivity[k])*wp.exp(-soil[k]/fd[k])
    low = wp.max(zero,h-ctrl[8]*capacity)
    high = h+added
    u0 = zero
    u1 = zero
    if h > dry:
        u0 = q[k][axis+1]/h
        u1 = q[k][axis+1]/high
    umin = wp.min(u0,u1)
    umax = wp.max(u0,u1)
    hl = wp.max(zero,low+bed[k]-top)
    hh = wp.max(zero,high+bed[k]-top)
    if hl <= dry:
        umin = wp.min(umin,zero)
        umax = wp.max(umax,zero)
    return wp.vec4d(umin,umax,wp.sqrt(g*hl),wp.sqrt(g*hh))


@wp.func
def wave_interval_bound(left: wp.vec4d, right: wp.vec4d, g: wp.float64, dry: wp.float64):
    """Envelope of endpoints, valid star predictors and dry rarefactions.

    With positive predictor, star speeds are affine in u_L,u_R,c_L,c_R.
    Bounding those affine forms over source intervals is conservative without
    estimating a timestep from an inconsistent pre-source Riemann state.
    """
    sl = wp.min(left[0]-left[3],right[0]-right[3])
    sr = wp.max(left[1]+left[3],right[1]+right[3])
    sl = wp.min(sl,wp.float64(.75)*left[0]+wp.float64(.25)*right[0]+wp.float64(.5)*left[2]-wp.float64(1.5)*right[3])
    sr = wp.max(sr,wp.float64(.25)*left[1]+wp.float64(.75)*right[1]+wp.float64(1.5)*left[3]-wp.float64(.5)*right[2])
    if left[2]*left[2] <= g*dry:
        sl = wp.min(sl,right[0]-wp.float64(2.0)*right[3])
    if right[2]*right[2] <= g*dry:
        sr = wp.max(sr,left[1]+wp.float64(2.0)*left[3])
    return wp.max(wp.abs(sl),wp.abs(sr))


@wp.kernel
def star_cfl_reduce(q: wp.array(dtype=wp.vec3d), active: wp.array(dtype=wp.int32),
        bed: wp.array(dtype=wp.float64), rain: wp.array(dtype=wp.float64), multiplier: wp.array(dtype=wp.float64),
        infil: wp.array(dtype=wp.float64), f0: wp.array(dtype=wp.float64), fd: wp.array(dtype=wp.float64),
        conductivity: wp.array(dtype=wp.float64), soil: wp.array(dtype=wp.float64),
        bx: wp.array(dtype=wp.int32), by: wp.array(dtype=wp.int32),
        sx: wp.array(dtype=wp.float64), sy: wp.array(dtype=wp.float64),
        nx: int, ny: int, ctrl: wp.array(dtype=wp.float64), dx: wp.float64, dy: wp.float64,
        g: wp.float64, dry: wp.float64, cfl: wp.float64, stage_mode: int):
    k = wp.tid()
    if active[k] == 0:
        return
    j = k/nx
    i = k%nx
    rates = wp.vec2d(wp.float64(0.0))
    for axis in range(2):
        for side in range(2):
            neighbor = int(-1)
            kind = int(0)
            stage = wp.float64(0.0)
            if axis == 0:
                face = j*(nx+1)+i+side
                kind = bx[face]
                stage = sx[face]
                if i+side > 0 and i+side < nx:
                    neighbor = k+2*side-1
            else:
                face = k+side*nx
                kind = by[face]
                stage = sy[face]
                if j+side > 0 and j+side < ny:
                    neighbor = k+(2*side-1)*nx
            interior = bool(False)
            if neighbor >= 0:
                interior = active[neighbor] != 0
            top = bed[k]
            if interior:
                top = wp.max(top,bed[neighbor])
            inside = cell_wave_interval(k,axis,top,q,bed,rain,multiplier,infil,f0,fd,conductivity,soil,ctrl,g,dry)
            outside = inside
            if interior:
                outside = cell_wave_interval(neighbor,axis,top,q,bed,rain,multiplier,infil,f0,fd,conductivity,soil,ctrl,g,dry)
            elif kind <= 1:
                outside = wp.vec4d(-inside[1],-inside[0],inside[2],inside[3])
            elif kind == 3:
                hghost = wp.max(wp.float64(0.0),stage-bed[k])
                cb = wp.sqrt(g*hghost)
                outside = wp.vec4d(wp.float64(0.0),wp.float64(0.0),cb,cb)
                sources = rain[k]+multiplier[k]*ctrl[12]+infil[k]+f0[k] > wp.float64(0.0)
                if stage_mode == 2:
                    if sources:
                        maximum = wp.max(wp.abs(inside[0]),wp.abs(inside[1]))+wp.float64(2.0)*(inside[3]+cb)
                        outside = wp.vec4d(-maximum,maximum,wp.float64(0.0),wp.max(inside[3],cb))
                    else:
                        normal = q[k]
                        if axis == 1:
                            normal = wp.vec3d(q[k][0],q[k][2],q[k][1])
                        ghost = stage_boundary_state(normal,hghost,dry,stage_mode,g,wp.float64(2*side-1))
                        if ghost[0] > dry:
                            ug = ghost[1]/ghost[0]
                            cg = wp.sqrt(g*ghost[0])
                            outside = wp.vec4d(ug,ug,cg,cg)
                elif hghost > dry:
                    if stage_mode == 1:
                        outside = wp.vec4d(inside[0],inside[1],cb,cb)
                    else:
                        ug = q[k][axis+1]/hghost
                        outside = wp.vec4d(ug,ug,cb,cb)
                        if sources:
                            outside = wp.vec4d(wp.min(ug,wp.float64(0.0)),wp.max(ug,wp.float64(0.0)),cb,cb)
            speed = wave_interval_bound(inside,outside,g,dry)
            if side == 0:
                speed = wave_interval_bound(outside,inside,g,dry)
            rates[axis] = wp.max(rates[axis],speed)
    rate = rates[0]/dx+rates[1]/dy
    if rate > wp.float64(0.0):
        wp.atomic_min(ctrl,2,cfl/rate)
        wp.atomic_min(ctrl,14,cfl/rate)


@wp.kernel
def characteristic_source_bound(q: wp.array(dtype=wp.vec3d), active: wp.array(dtype=wp.int32),
               rain: wp.array(dtype=wp.float64), multiplier: wp.array(dtype=wp.float64),
               infil: wp.array(dtype=wp.float64), f0: wp.array(dtype=wp.float64),
               bx: wp.array(dtype=wp.int32), by: wp.array(dtype=wp.int32),
               sx: wp.array(dtype=wp.float64), sy: wp.array(dtype=wp.float64), bed: wp.array(dtype=wp.float64),
               nx: int, ctrl: wp.array(dtype=wp.float64), dx: wp.float64, dy: wp.float64,
               g: wp.float64, dry: wp.float64, cfl: wp.float64):
    """Conservative envelope for every source-updated characteristic ghost.

    Rain dilutes velocity and withdrawal preserves it until the dry threshold.
    Thus |u_post| <= |u_old| and c_post <= sqrt(g*(h+cap*rain_max)).
    The characteristic state has |ub| <= |u_old|+2*c_max+2*c_external;
    critical and supercritical-outflow states satisfy the same envelope.
    Only cells with actual sources need this additional (loose) bound.
    """
    k = wp.tid()
    if active[k] != 0 and rain[k] + multiplier[k]*ctrl[12] + infil[k] + f0[k] > wp.float64(0.0):
        u = wp.float64(0.0)
        v = wp.float64(0.0)
        if q[k][0] > dry:
            u = wp.abs(q[k][1]/q[k][0])
            v = wp.abs(q[k][2]/q[k][0])
        ci = wp.sqrt(g*(q[k][0]+ctrl[8]*(rain[k]+multiplier[k]*ctrl[12])))
        ax = u+ci
        ay = v+ci
        j = k/nx
        i = k%nx
        for offset in range(2):
            face_x = j*(nx+1)+i+offset
            if bx[face_x] == 3:
                cb = wp.sqrt(g*wp.max(wp.float64(0.0),sx[face_x]-bed[k]))
                ax = wp.max(ax,u+wp.float64(2.0)*ci+wp.float64(2.0)*cb+wp.max(ci,cb))
            face_y = k+offset*nx
            if by[face_y] == 3:
                cb = wp.sqrt(g*wp.max(wp.float64(0.0),sy[face_y]-bed[k]))
                ay = wp.max(ay,v+wp.float64(2.0)*ci+wp.float64(2.0)*cb+wp.max(ci,cb))
        rate = ax/dx+ay/dy
        if rate > wp.float64(0.0):
            wp.atomic_min(ctrl,2,cfl/rate)
            wp.atomic_min(ctrl,14,cfl/rate)


@wp.kernel
def validate_characteristic(q: wp.array(dtype=wp.vec3d), active: wp.array(dtype=wp.int32),
               rain: wp.array(dtype=wp.float64), multiplier: wp.array(dtype=wp.float64),
               infil: wp.array(dtype=wp.float64), f0: wp.array(dtype=wp.float64),
               fd: wp.array(dtype=wp.float64), conductivity: wp.array(dtype=wp.float64), soil: wp.array(dtype=wp.float64),
               bed: wp.array(dtype=wp.float64), bx: wp.array(dtype=wp.int32), by: wp.array(dtype=wp.int32),
               sx: wp.array(dtype=wp.float64), sy: wp.array(dtype=wp.float64), nx: int,
               ctrl: wp.array(dtype=wp.float64), g: wp.float64, dry: wp.float64):
    """Reject unsupported *post-source* states before any state/budget mutation."""
    k = wp.tid()
    if active[k] != 0:
        dt = ctrl[2]
        added = dt*(rain[k]+multiplier[k]*(ctrl[9]+wp.float64(0.5)*ctrl[11]*dt))
        capacity = infil[k]+conductivity[k]+(f0[k]-conductivity[k])*wp.exp(-soil[k]/fd[k])
        inside = source_state(q[k],added,wp.min(q[k][0]+added,dt*capacity))
        j = k/nx
        i = k%nx
        for offset in range(2):
            outward = wp.float64(2*offset-1)
            face_x = j*(nx+1)+i+offset
            if bx[face_x] == 3:
                ghost = stage_boundary_state(inside,wp.max(wp.float64(0.0),sx[face_x]-bed[k]),dry,2,g,outward)
                if ghost[0] < wp.float64(0.0):
                    wp.atomic_min(ctrl,7,wp.float64(-1.0))
            face_y = k+offset*nx
            if by[face_y] == 3:
                normal = wp.vec3d(inside[0],inside[2],inside[1])
                ghost = stage_boundary_state(normal,wp.max(wp.float64(0.0),sy[face_y]-bed[k]),dry,2,g,outward)
                if ghost[0] < wp.float64(0.0):
                    wp.atomic_min(ctrl,7,wp.float64(-1.0))


@wp.kernel
def apply_sources(q: wp.array(dtype=wp.vec3d), active: wp.array(dtype=wp.int32),
                  rain: wp.array(dtype=wp.float64), multiplier: wp.array(dtype=wp.float64),
                  infil: wp.array(dtype=wp.float64), f0: wp.array(dtype=wp.float64),
                  fd: wp.array(dtype=wp.float64), conductivity: wp.array(dtype=wp.float64),
                  soil: wp.array(dtype=wp.float64), rain_sum: wp.array(dtype=wp.float64),
                  ctrl: wp.array(dtype=wp.float64)):
    k = wp.tid()
    if active[k] != 0 and ctrl[7] >= wp.float64(0.0):
        dt = ctrl[2]
        added = dt * (rain[k] + multiplier[k] * (ctrl[9] + wp.float64(0.5) * ctrl[11] * dt))
        available = q[k][0] + added
        capacity = infil[k] + conductivity[k] + (f0[k] - conductivity[k]) * wp.exp(-soil[k] / fd[k])
        removed = wp.min(available, dt * capacity)
        scale = wp.float64(0.0)
        if available > wp.float64(0.0):
            scale = (available - removed) / available
        q[k] = wp.vec3d(available - removed, q[k][1] * scale, q[k][2] * scale)
        soil[k] += removed
        rain_sum[k] += added


@wp.func
def reconstruct(q: wp.vec3d, z: wp.float64, top: wp.float64, dry: wp.float64):
    depth = wp.max(wp.float64(0.0), q[0] + z - top)
    scale = wp.float64(0.0)
    if q[0] > dry:
        scale = depth / q[0]
    return wp.vec3d(depth, q[1] * scale, q[2] * scale)


@wp.kernel
def faces(q: wp.array(dtype=wp.vec3d), bed: wp.array(dtype=wp.float64),
          active: wp.array(dtype=wp.int32), types: wp.array(dtype=wp.int32),
          stages: wp.array(dtype=wp.float64), flux: wp.array(dtype=wp.vec3d),
          pressure: wp.array(dtype=wp.vec2d), nx: int, ny: int, axis: int,
          g: wp.float64, dry: wp.float64, stage_momentum_mode: int,
          ctrl: wp.array(dtype=wp.float64), flux_mode: int):
    k = wp.tid()
    if ctrl[7] < wp.float64(0.0):
        return
    left = int(-1)
    right = int(-1)
    if axis == 0:
        j = k / (nx + 1)
        i = k % (nx + 1)
        if i > 0:
            left = j * nx + i - 1
        if i < nx:
            right = j * nx + i
    else:
        j = k / nx
        i = k % nx
        if j > 0:
            left = (j - 1) * nx + i
        if j < ny:
            right = j * nx + i
    al = bool(False)
    ar = bool(False)
    if left >= 0:
        al = active[left] != 0
    if right >= 0:
        ar = active[right] != 0
    ql = wp.vec3d(wp.float64(0.0))
    qr = wp.vec3d(wp.float64(0.0))
    zl = wp.float64(0.0)
    zr = wp.float64(0.0)
    if al:
        ql = q[left]
        zl = bed[left]
    if ar:
        qr = q[right]
        zr = bed[right]
    if axis == 1:
        ql = wp.vec3d(ql[0], ql[2], ql[1])
        qr = wp.vec3d(qr[0], qr[2], qr[1])
    wall = bool(False)
    if al != ar:
        inside = qr
        zin = zr
        if al:
            inside = ql
            zin = zl
        ghost = inside
        wall = types[k] <= 1
        if wall:
            ghost = wp.vec3d(inside[0], -inside[1], inside[2])
        elif types[k] == 3:
            outward = wp.float64(-1.0)
            if al:
                outward = wp.float64(1.0)
            ghost = stage_boundary_state(inside,wp.max(wp.float64(0.0),stages[k]-zin),dry,stage_momentum_mode,g,outward)
        if al:
            qr = ghost
        else:
            ql = ghost
        zl = zin
        zr = zin
    top = wp.max(zl, zr)
    ls = reconstruct(ql, zl, top, dry)
    rs = reconstruct(qr, zr, top, dry)
    f = hll(ls, rs, g, dry, flux_mode)
    if wall:
        f = wp.vec3d(wp.float64(0.0), f[1], wp.float64(0.0))
    if al != ar and types[k] == 2 and ((al and f[0] < wp.float64(0.0)) or (ar and f[0] > wp.float64(0.0))):
        # Block advective exchange while retaining the inside hydrostatic
        # traction. Removing pressure creates an O(g h^2) acceleration even
        # as the incoming velocity tends to zero.
        f = wp.vec3d(wp.float64(0.0), wp.float64(0.5) * g * inside[0] * inside[0], wp.float64(0.0))
    if not al and not ar:
        f = wp.vec3d(wp.float64(0.0))
    if axis == 1:
        f = wp.vec3d(f[0], f[2], f[1])
    flux[k] = f
    pressure[k] = wp.vec2d(wp.float64(0.5) * g * ls[0] * ls[0], wp.float64(0.5) * g * rs[0] * rs[0])


@wp.kernel
def draining(q: wp.array(dtype=wp.vec3d), fx: wp.array(dtype=wp.vec3d),
             fy: wp.array(dtype=wp.vec3d), theta: wp.array(dtype=wp.float64),
             ctrl: wp.array(dtype=wp.float64), nx: int, dx: wp.float64, dy: wp.float64):
    k = wp.tid()
    if ctrl[7] < wp.float64(0.0):
        return
    j = k / nx
    i = k % nx
    rate = (wp.max(fx[j * (nx + 1) + i + 1][0], wp.float64(0.0)) - wp.min(fx[j * (nx + 1) + i][0], wp.float64(0.0))) / dx
    rate += (wp.max(fy[(j + 1) * nx + i][0], wp.float64(0.0)) - wp.min(fy[j * nx + i][0], wp.float64(0.0))) / dy
    value = wp.float64(1.0)
    if rate * ctrl[2] > wp.float64(0.0):
        value = wp.min(value, q[k][0] / (rate * ctrl[2]))
    theta[k] = value


@wp.kernel
def limit_faces(flux: wp.array(dtype=wp.vec3d), theta: wp.array(dtype=wp.float64),
                active: wp.array(dtype=wp.int32), volume: wp.array(dtype=wp.float64),
                ctrl: wp.array(dtype=wp.float64), nx: int, ny: int, axis: int, width: wp.float64):
    k = wp.tid()
    if ctrl[7] < wp.float64(0.0):
        return
    donor = int(-1)
    if axis == 0:
        j = k / (nx + 1)
        i = k % (nx + 1)
        if flux[k][0] >= wp.float64(0.0) and i > 0:
            donor = j * nx + i - 1
        elif flux[k][0] < wp.float64(0.0) and i < nx:
            donor = j * nx + i
    else:
        j = k / nx
        i = k % nx
        if flux[k][0] >= wp.float64(0.0) and j > 0:
            donor = (j - 1) * nx + i
        elif flux[k][0] < wp.float64(0.0) and j < ny:
            donor = j * nx + i
    if donor >= 0:
        if active[donor] != 0:
            flux[k] = flux[k] * theta[donor]
    volume[k] += flux[k][0] * ctrl[2] * width


@wp.kernel
def update(q: wp.array(dtype=wp.vec3d), fx: wp.array(dtype=wp.vec3d), fy: wp.array(dtype=wp.vec3d),
           px: wp.array(dtype=wp.vec2d), py: wp.array(dtype=wp.vec2d),
           active: wp.array(dtype=wp.int32), manning: wp.array(dtype=wp.float64),
           maximum: wp.array(dtype=wp.float64), inflow: wp.array(dtype=wp.float64),
           outflow: wp.array(dtype=wp.float64), ctrl: wp.array(dtype=wp.float64),
           nx: int, ny: int, dx: wp.float64, dy: wp.float64, g: wp.float64, dry: wp.float64):
    k = wp.tid()
    if active[k] != 0 and ctrl[7] >= wp.float64(0.0):
        j = k / nx
        i = k % nx
        w = j * (nx + 1) + i
        e = w + 1
        s = k
        n = k + nx
        dt = ctrl[2]
        # Remove each reconstructed pressure before differencing. The omitted
        # cell-local 0.5*g*h^2 cancels, preserving a lake even at large datums.
        fw = fx[w] - wp.vec3d(wp.float64(0.0), px[w][1], wp.float64(0.0))
        fe = fx[e] - wp.vec3d(wp.float64(0.0), px[e][0], wp.float64(0.0))
        fs = fy[s] - wp.vec3d(wp.float64(0.0), wp.float64(0.0), py[s][1])
        fn = fy[n] - wp.vec3d(wp.float64(0.0), wp.float64(0.0), py[n][0])
        value = q[k] + dt / dx * (fw - fe) + dt / dy * (fs - fn)
        if value[0] < wp.float64(-1.e-12) or not wp.isfinite(value[0]) or not wp.isfinite(value[1]) or not wp.isfinite(value[2]):
            wp.atomic_add(ctrl, 7, wp.float64(1.0))
        depth = wp.max(value[0], wp.float64(0.0))
        hu = wp.float64(0.0)
        hv = wp.float64(0.0)
        if depth > dry:
            drag = dt * g * manning[k] * manning[k] * wp.sqrt(value[1] * value[1] + value[2] * value[2]) / wp.pow(depth, wp.float64(7.0) / wp.float64(3.0))
            hu = value[1] / (wp.float64(1.0) + drag)
            hv = value[2] / (wp.float64(1.0) + drag)
        q[k] = wp.vec3d(depth, hu, hv)
        maximum[k] = wp.max(maximum[k], depth)
        bw = bool(i == 0)
        be = bool(i == nx - 1)
        bs = bool(j == 0)
        bn = bool(j == ny - 1)
        if i > 0:
            bw = active[k - 1] == 0
        if i < nx - 1:
            be = active[k + 1] == 0
        if j > 0:
            bs = active[k - nx] == 0
        if j < ny - 1:
            bn = active[k + nx] == 0
        a = wp.float64(0.0)
        b = wp.float64(0.0)
        c = wp.float64(0.0)
        d = wp.float64(0.0)
        if bw:
            a = -fx[w][0] * dy * dt
        if be:
            b = fx[e][0] * dy * dt
        if bs:
            c = -fy[s][0] * dx * dt
        if bn:
            d = fy[n][0] * dx * dt
        outflow[k] += wp.max(a, wp.float64(0.0)) + wp.max(b, wp.float64(0.0)) + wp.max(c, wp.float64(0.0)) + wp.max(d, wp.float64(0.0))
        inflow[k] -= wp.min(a, wp.float64(0.0)) + wp.min(b, wp.float64(0.0)) + wp.min(c, wp.float64(0.0)) + wp.min(d, wp.float64(0.0))


@wp.kernel
def finish_step(ctrl: wp.array(dtype=wp.float64), condition: wp.array(dtype=wp.int32),
                histogram: wp.array(dtype=wp.int64), minimum_dt: wp.float64):
    if ctrl[7] < wp.float64(0.0):
        condition[0] = 0
        return
    dt = ctrl[2]
    if dt <= wp.float64(0.0) or ctrl[0] + dt == ctrl[0]:
        ctrl[7] += wp.float64(1.0)
    if ctrl[14] < minimum_dt:
        ctrl[7] += wp.float64(1.0)
    ctrl[0] += dt
    ctrl[3] += wp.float64(1.0)
    ctrl[4] = wp.min(ctrl[4], dt)
    ctrl[5] = wp.max(ctrl[5], dt)
    ctrl[6] += dt * dt
    if dt > wp.float64(0.0):
        bucket = wp.clamp(int(wp.floor(wp.log(dt) / wp.log(wp.float64(2.0)))) + 32, 0, 63)
        histogram[bucket] += wp.int64(1)
    condition[0] = int(ctrl[0] < ctrl[1] and ctrl[7] == wp.float64(0.0) and ctrl[3] < ctrl[13])
