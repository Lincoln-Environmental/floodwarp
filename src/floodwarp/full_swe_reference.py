"""Independent, vectorised FP64 NumPy oracle for first-order full SWE.

Cell order is south to north; conservative state is (h, hu, hv). Hydrostatic
reconstruction follows Audusse et al. (2004), with two-wave HLL fluxes. This
module deliberately has no Warp or existing hydraulic-solver dependencies.
"""

import numpy as np


def exponential_infiltration_capacity(*, cumulative_infiltration_m, initial_capacity_m_s,
                                      decay_depth_m, conductivity_m_s):
    """Independent explicit capacity law, accepting broadcastable FP64 fields."""
    cumulative, initial, decay, conductivity = np.broadcast_arrays(
        np.asarray(cumulative_infiltration_m, dtype=np.float64),
        np.asarray(initial_capacity_m_s, dtype=np.float64),
        np.asarray(decay_depth_m, dtype=np.float64),
        np.asarray(conductivity_m_s, dtype=np.float64))
    return conductivity + (initial - conductivity) * np.exp(-cumulative / decay)


def exact_ponded_infiltration(*, time_s, initial_capacity_m_s, decay_depth_m, conductivity_m_s):
    """Analytical cumulative infiltration from F(0)=0 with unlimited water.

    Substituting E=exp(F/fd) in F'=k+(f0-k)exp(-F/fd) gives a linear ODE.
    The k=0 logarithmic limit is evaluated directly, without division by zero.
    This is a continuous-time oracle, independent of the explicit stepper.
    """
    initial, decay, conductivity = np.broadcast_arrays(
        np.asarray(initial_capacity_m_s, dtype=np.float64),
        np.asarray(decay_depth_m, dtype=np.float64),
        np.asarray(conductivity_m_s, dtype=np.float64))
    if time_s < 0 or np.any(decay <= 0) or np.any(conductivity < 0) or np.any(initial < conductivity):
        raise ValueError("Nonnegative time/capacities, positive decay depth and f0 >= k required")
    growth = np.divide(initial * np.expm1(conductivity * time_s / decay), conductivity,
        out=np.zeros_like(initial), where=conductivity > 0)
    growth = np.where(conductivity > 0, growth, initial * time_s / decay)
    return decay * np.log1p(growth)


def hll_wave_speeds(*, left, right, gravity, dry_depth, flux_scheme='legacy_hll'):
    """Normal/tangential states; include the exact dry-side rarefaction bounds."""
    hl, hr = left[..., 0], right[..., 0]
    ul = np.divide(left[..., 1:], hl[..., None], out=np.zeros_like(left[..., 1:]), where=hl[..., None] > dry_depth)
    ur = np.divide(right[..., 1:], hr[..., None], out=np.zeros_like(right[..., 1:]), where=hr[..., None] > dry_depth)
    cl, cr = np.sqrt(gravity * hl), np.sqrt(gravity * hr)
    sl = np.minimum(ul[..., 0] - cl, ur[..., 0] - cr)
    sr = np.maximum(ul[..., 0] + cl, ur[..., 0] + cr)
    if flux_scheme not in ('legacy_hll','contact_hll','star_hll','star_contact_hll'):
        raise ValueError('Unknown FV1 flux_scheme')
    if flux_scheme in ('star_hll','star_contact_hll'):
        predictor = (cl+cr)/2+(ul[..., 0]-ur[..., 0])/4
        valid = (hl > dry_depth) & (hr > dry_depth) & (predictor > 0.)
        us = (ul[..., 0]+ur[..., 0])/2+cl-cr
        sl = np.where(valid,np.minimum(ul[..., 0]-cl,us-predictor),sl)
        sr = np.where(valid,np.maximum(ur[..., 0]+cr,us+predictor),sr)
    sl = np.where(hr <= dry_depth, ul[..., 0] - cl, sl)
    sr = np.where(hr <= dry_depth, ul[..., 0] + 2 * cl, sr)
    sl = np.where(hl <= dry_depth, ur[..., 0] - 2 * cr, sl)
    sr = np.where(hl <= dry_depth, ur[..., 0] + cr, sr)
    return sl,sr,ul,ur


def hll_flux(*, left, right, gravity, dry_depth, flux_scheme='legacy_hll'):
    """Conservative HLL with independently selectable wave/contact operators."""
    hl,hr = left[..., 0],right[..., 0]
    sl,sr,ul,ur = hll_wave_speeds(left=left,right=right,gravity=gravity,dry_depth=dry_depth,flux_scheme=flux_scheme)
    fl, fr = np.zeros_like(left), np.zeros_like(right)
    fl[..., 0], fr[..., 0] = left[..., 1], right[..., 1]
    fl[..., 1] = left[..., 1] * ul[..., 0] + .5 * gravity * hl**2
    fr[..., 1] = right[..., 1] * ur[..., 0] + .5 * gravity * hr**2
    fl[..., 2], fr[..., 2] = left[..., 1] * ul[..., 1], right[..., 1] * ur[..., 1]
    middle = np.divide(sr[..., None] * fl - sl[..., None] * fr + (sl * sr)[..., None] * (right - left),
                       (sr - sl)[..., None], out=np.zeros_like(left), where=(sr > sl)[..., None])
    if flux_scheme in ('contact_hll','star_contact_hll'):
        dl = hl*(ul[..., 0]-sl); dr = hr*(sr-ur[..., 0])
        denominator = dl+dr
        contact = np.divide(sr*dl+sl*dr,denominator,out=np.zeros_like(hl),where=denominator > 0.)
        tangent = np.where(contact >= 0.,ul[..., 1],ur[..., 1])
        middle[..., 2] = np.where(denominator > 0.,middle[..., 0]*tangent,middle[..., 2])
    return np.where((sl >= 0)[..., None], fl, np.where((sr <= 0)[..., None], fr, middle))


def characteristic_stage_state(*, inside, depth, outward, gravity, dry_depth, selected):
    """Independent vectorised characteristic closure on a local flat sill.

    No outgoing information exists for supercritical inflow or dry land with
    a wet exterior. Those cases need external momentum and are rejected.
    Incoming transverse velocity is prescribed as zero. Critical free drainage
    uses u=c=J+/3 when the downstream stage is below the sonic depth.
    """
    h = inside[..., 0]
    velocity = np.divide(inside[..., 1:], h[..., None], out=np.zeros_like(inside[..., 1:]), where=h[..., None] > dry_depth)
    un = outward*velocity[..., 0]
    ci = np.sqrt(gravity*h)
    super_out = (h > dry_depth) & (un >= ci)
    cb = np.maximum(np.sqrt(gravity*depth), (un+2*ci)/3)
    ub = un+2*ci-2*cb
    invalid = ((h <= dry_depth) & (depth > dry_depth)) | ((h > dry_depth) & ((un <= -ci) | ((~super_out) & (ub <= -cb))))
    if np.any(selected & invalid):
        raise ValueError('Characteristic stage boundary requires incoming momentum data')
    hb = cb**2/gravity
    ghost = np.stack([hb, outward*ub*hb, np.where(ub >= 0, velocity[..., 1]*hb, 0.)], axis=-1)
    ghost[..., 1:] = np.where((hb > dry_depth)[..., None], ghost[..., 1:], 0.)
    ghost = np.where(super_out[..., None], inside, ghost)
    return np.where(((h <= dry_depth) & (depth <= dry_depth))[..., None], 0., ghost)


def face_fluxes(*, state, bed, active, boundary_type, boundary_stage, axis, gravity, dry_depth,
                stage_boundary_momentum='velocity_extrapolation',flux_scheme='contact_hll'):
    """Return shared oriented flux and the left/right reconstructed pressures.

    Boundary codes: 0 internal (wall at mask), 1 reflecting wall, 2 one-way
    transmissive outflow, 3 stage with wet extrapolated/dry zero momentum.
    """
    padding = [(0, 0), (0, 0), (0, 0)]
    padding[axis] = (1, 1)
    padded = np.pad(state, padding)
    z = np.pad(bed, padding[:2])
    a = np.pad(active, padding[:2])
    l, r = [slice(None)] * 2, [slice(None)] * 2
    l[axis], r[axis] = slice(None, -1), slice(1, None)
    l, r = tuple(l), tuple(r)
    ql, qr, zl, zr, al, ar = padded[l].copy(), padded[r].copy(), z[l].copy(), z[r].copy(), a[l], a[r]
    order = [0, 1, 2] if axis == 1 else [0, 2, 1]
    ql, qr = ql[..., order], qr[..., order]
    edge = al != ar
    inside = np.where(al[..., None], ql, qr)
    zin = np.where(al, zl, zr)
    ghost = inside.copy()
    wall = edge & (boundary_type <= 1)
    ghost[..., 1] = np.where(wall, -inside[..., 1], ghost[..., 1])
    stage = edge & (boundary_type == 3)
    ghost[..., 0] = np.where(stage, np.maximum(0., boundary_stage - zin), ghost[..., 0])
    if stage_boundary_momentum=='velocity_extrapolation':
        velocity=np.divide(inside[...,1:],inside[...,0,None],out=np.zeros_like(inside[...,1:]),
            where=inside[...,0,None]>dry_depth)
        ghost[...,1:]=np.where(stage[...,None],ghost[...,0,None]*velocity,ghost[...,1:])
    elif stage_boundary_momentum=='characteristic':
        prescribed = characteristic_stage_state(inside=inside, depth=ghost[..., 0], outward=np.where(al, 1., -1.),
            gravity=gravity, dry_depth=dry_depth, selected=stage)
        ghost = np.where(stage[..., None], prescribed, ghost)
    elif stage_boundary_momentum!='discharge_extrapolation':
        raise ValueError('Unknown stage momentum boundary treatment')
    ghost[..., 1:] = np.where((stage & (ghost[..., 0] <= dry_depth))[..., None], 0., ghost[..., 1:])
    ql = np.where((edge & ~al)[..., None], ghost, ql)
    qr = np.where((edge & ~ar)[..., None], ghost, qr)
    zl, zr = np.where(edge, zin, zl), np.where(edge, zin, zr)
    top = np.maximum(zl, zr)
    hl, hr = np.maximum(0., ql[..., 0] + zl - top), np.maximum(0., qr[..., 0] + zr - top)
    qls, qrs = ql.copy(), qr.copy()
    qls[..., 0], qrs[..., 0] = hl, hr
    qls[..., 1:] *= np.divide(hl, ql[..., 0], out=np.zeros_like(hl), where=ql[..., 0] > dry_depth)[..., None]
    qrs[..., 1:] *= np.divide(hr, qr[..., 0], out=np.zeros_like(hr), where=qr[..., 0] > dry_depth)[..., None]
    if flux_scheme == 'legacy_hll':
        flux = hll_flux(left=qls,right=qrs,gravity=gravity,dry_depth=dry_depth)
    else:
        flux = hll_flux(left=qls,right=qrs,gravity=gravity,dry_depth=dry_depth,flux_scheme=flux_scheme)
    # Enforce impermeability exactly, including for a shallow wet wall.
    flux[..., 0] = np.where(wall, 0., flux[..., 0])
    flux[..., 2] = np.where(wall, 0., flux[..., 2])
    blocked = edge & (boundary_type == 2) & np.where(al, flux[..., 0] < 0, flux[..., 0] > 0)
    blocked_flux = np.zeros_like(flux)
    blocked_flux[..., 1] = .5 * gravity * inside[..., 0]**2
    flux = np.where(blocked[..., None], blocked_flux, flux)
    flux = np.where((~(al | ar))[..., None], 0., flux)
    return flux[..., order], .5 * gravity * hl**2, .5 * gravity * hr**2


def stable_timestep(*, state, active, dx, dy, gravity, dry_depth, cfl, maximum_dt, rain_rate):
    h = state[..., 0]
    velocity = np.divide(state[..., 1:], h[..., None], out=np.zeros_like(state[..., 1:]), where=h[..., None] > dry_depth)
    wave = np.sqrt(gravity * (h + maximum_dt * rain_rate))
    rate = (np.abs(velocity[..., 0]) + wave) / dx + (np.abs(velocity[..., 1]) + wave) / dy
    maximum_rate = np.max(np.where(active, rate, 0.))
    return min(maximum_dt, cfl / maximum_rate) if maximum_rate > 0 else maximum_dt


def numpy_step(*, state, bed, active, manning, dx, dy, gravity, dry_depth, dt,
               boundary_type_x, boundary_type_y, boundary_stage_x, boundary_stage_y,
               rain_rate, infiltration_rate,stage_boundary_momentum='velocity_extrapolation',flux_scheme='contact_hll'):
    """One unsplit Euler step with finite-availability sources then transport.

    The donor draining limiter shares one flux between adjacent cells. A
    semi-implicit vector Manning drag follows the conservative hydraulic step.
    Returns state and explicitly integrated source/boundary volumes.
    """
    q = state.copy()
    rain = np.where(active, rain_rate * dt, 0.)
    available = q[..., 0] + rain
    infil = np.where(active, np.minimum(available, infiltration_rate * dt), 0.)
    q[..., 1:] *= np.divide(available - infil, available, out=np.zeros_like(available), where=available > 0)[..., None]
    q[..., 0] = available - infil
    fx, pxl, pxr = face_fluxes(state=q, bed=bed, active=active, boundary_type=boundary_type_x,
        boundary_stage=boundary_stage_x, axis=1, gravity=gravity, dry_depth=dry_depth,stage_boundary_momentum=stage_boundary_momentum,flux_scheme=flux_scheme)
    fy, pyl, pyr = face_fluxes(state=q, bed=bed, active=active, boundary_type=boundary_type_y,
        boundary_stage=boundary_stage_y, axis=0, gravity=gravity, dry_depth=dry_depth,stage_boundary_momentum=stage_boundary_momentum,flux_scheme=flux_scheme)
    outgoing = (np.maximum(fx[:, 1:, 0], 0.) - np.minimum(fx[:, :-1, 0], 0.)) / dx
    outgoing += (np.maximum(fy[1:, :, 0], 0.) - np.minimum(fy[:-1, :, 0], 0.)) / dy
    theta = np.minimum(1., np.divide(q[..., 0], dt * outgoing, out=np.ones_like(outgoing), where=outgoing > 0))
    tx, ty = np.pad(theta, ((0, 0), (1, 1)), constant_values=1.), np.pad(theta, ((1, 1), (0, 0)), constant_values=1.)
    fx *= np.where(fx[..., 0] >= 0, tx[:, :-1], tx[:, 1:])[..., None]
    fy *= np.where(fy[..., 0] >= 0, ty[:-1], ty[1:])[..., None]
    q -= dt / dx * (fx[:, 1:] - fx[:, :-1]) + dt / dy * (fy[1:] - fy[:-1])
    q[..., 1] += dt / dx * (pxl[:, 1:] - pxr[:, :-1])
    q[..., 2] += dt / dy * (pyl[1:] - pyr[:-1])
    q[..., 0] = np.maximum(q[..., 0], 0.)
    speed = np.sqrt(np.sum(q[..., 1:]**2, axis=-1))
    drag = np.divide(dt * gravity * manning**2 * speed, q[..., 0]**(7. / 3.), out=np.zeros_like(speed), where=q[..., 0] > dry_depth)
    q[..., 1:] /= (1. + drag)[..., None]
    q[..., 1:] = np.where((q[..., 0] > dry_depth)[..., None], q[..., 1:], 0.)
    q = np.where(active[..., None], q, 0.)
    ax, ay = np.pad(active, ((0, 0), (1, 1))), np.pad(active, ((1, 1), (0, 0)))
    bx = fx[..., 0] * (ax[:, :-1].astype(int) - ax[:, 1:].astype(int)) * dy * dt
    by = fy[..., 0] * (ay[:-1].astype(int) - ay[1:].astype(int)) * dx * dt
    return q, dict(rain=float(rain.sum() * dx * dy), infiltration=float(infil.sum() * dx * dy),
                   boundary_outflow=float(np.maximum(bx, 0.).sum() + np.maximum(by, 0.).sum()),
                   boundary_inflow=float(-np.minimum(bx, 0.).sum() - np.minimum(by, 0.).sum()))
