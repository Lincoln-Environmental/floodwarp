"""Vectorised independent FP64 MUSCL / Heun reference, with reflecting boundaries."""

import numpy as np

from floodwarp.full_swe_reference import hll_flux


def limited(*, left, right, limiter):
    if limiter == "mc":
        mag = np.minimum(np.minimum(2 * np.abs(left), 2 * np.abs(right)), np.abs(left + right) / 2)
    elif limiter == "minmod":
        mag = np.minimum(np.abs(left), np.abs(right))
    else:
        raise ValueError("Unknown limiter")
    return np.where(left * right > 0, np.sign(left) * mag, 0.0)


def traces(*, q, z, dry, axis, limiter, gravity=9.80665):
    h = q[..., 0]
    vel = np.divide(q[..., 1:], h[..., None], out=np.zeros_like(q[..., 1:]), where=h[..., None] > dry)
    variables = np.concatenate((h[..., None], z[..., None], vel), axis=-1)
    back = variables - np.roll(variables, 1, axis=axis)
    ahead = np.roll(variables, -1, axis=axis) - variables
    back[..., 0] += back[..., 1]
    ahead[..., 0] += ahead[..., 1]
    if limiter == "adaptive":
        ha, hb = np.roll(h, 1, axis=axis), np.roll(h, -1, axis=axis)
        normal = vel[..., 1-axis]
        ua, ub = np.roll(normal, 1, axis=axis), np.roll(normal, -1, axis=axis)
        ca, cb = np.sqrt(gravity * ha), np.sqrt(gravity * hb)
        crossing = ((ua > .8*ca) & (ub < 1.25*cb) & (hb > ha)) | ((ua > -1.25*ca) & (ub < -.8*cb) & (ha > hb))
        shock = (ua > ub) & (np.abs(ha-hb) > 0.04*np.minimum(ha,hb)) & crossing
        shock &= (ha > dry) & (hb > dry) & (h > dry)
        boundary = [slice(None)] * 2
        boundary[axis] = [0, -1]
        shock[tuple(boundary)] = False
        shock = shock | np.roll(shock, 1, axis=axis) | np.roll(shock, -1, axis=axis)
        slope = np.where(shock[..., None], limited(left=back, right=ahead, limiter="minmod"),
                         limited(left=back, right=ahead, limiter="mc"))
    else:
        slope = limited(left=back, right=ahead, limiter=limiter)
    # Limit each bed extrapolation to half the centre-to-centre bed change.
    original_bed = slope[..., 1].copy()
    bound = np.minimum(np.abs(back[..., 1]), np.abs(ahead[..., 1]))
    constrained = np.abs(original_bed) > bound
    slope[..., 0][constrained] *= bound[constrained] / np.abs(original_bed[constrained])
    slope[..., 1] = np.clip(original_bed, -bound, bound)
    wet = (h > dry) & (np.roll(h, 1, axis=axis) > dry) & (np.roll(h, -1, axis=axis) > dry)
    ends = [slice(None)] * 2
    ends[axis] = [0, -1]
    wet[tuple(ends)] = False
    slope *= wet[..., None]
    slope[np.abs(slope[..., 0] - slope[..., 1]) >= 2 * h] = 0.0
    dh = slope[..., 0] - slope[..., 1]
    hm, hp = h - 0.5 * dh, h + 0.5 * dh
    weight = np.divide(hm * hp, 2 * h, out=np.zeros_like(h), where=h > dry)
    minus = np.concatenate((hm[..., None], hm[..., None] * vel - weight[..., None] * slope[..., 2:]), axis=-1)
    plus = np.concatenate((hp[..., None], hp[..., None] * vel + weight[..., None] * slope[..., 2:]), axis=-1)
    return minus, plus, slope[..., 0], slope[..., 1]


def directional(*, q, z, g, dry, axis, limiter):
    minus, plus, eta, dz = traces(q=q, z=z, dry=dry, axis=axis, limiter=limiter, gravity=g)
    order = [0, 1, 2] if axis == 1 else [0, 2, 1]
    lower = np.take(minus, [0], axis=axis)[..., order].copy()
    upper = np.take(plus, [-1], axis=axis)[..., order].copy()
    lower[..., 1] *= -1
    upper[..., 1] *= -1
    l = np.concatenate((lower, plus[..., order]), axis=axis)
    r = np.concatenate((minus[..., order], upper), axis=axis)
    inner = np.diff(z, axis=axis) - 0.5 * (
        np.take(dz, range(1, z.shape[axis]), axis=axis) + np.take(dz, range(z.shape[axis] - 1), axis=axis)
    )
    pad = [(0, 0)] * 2
    pad[axis] = (1, 1)
    step = np.pad(inner, pad)
    for side, rise in ((l, step), (r, -step)):
        depth = np.maximum(0.0, side[..., 0] - np.maximum(0.0, rise))
        side[..., 1:] *= np.divide(depth, side[..., 0], out=np.zeros_like(depth), where=side[..., 0] > dry)[
            ..., None
        ]
        side[..., 0] = depth
    f = hll_flux(left=l, right=r, gravity=g, dry_depth=dry, flux_scheme="contact_hll")
    for end in (0, -1):
        sl = [slice(None)] * 2
        sl[axis] = end
        f[tuple(sl)][..., ::2] = 0.0
    return f[..., order], 0.5 * g * l[..., 0] ** 2, 0.5 * g * r[..., 0] ** 2, eta


def friction(*, q, n, g, dry, dt):
    h = q[..., 0]
    result = q.copy()
    drag = np.divide(
        dt * g * n * n * np.linalg.norm(q[..., 1:], axis=-1),
        h ** (7 / 3),
        out=np.zeros_like(h),
        where=h > dry,
    )
    result[..., 1:] = np.where((h > dry)[..., None], q[..., 1:] / (1 + drag[..., None]), 0.0)
    return result


def euler(*, q, z, dt, spacing, g, dry, limiter):
    h = q[..., 0]
    result = q.copy()
    faces = [directional(q=q, z=z, g=g, dry=dry, axis=axis, limiter=limiter) for axis in (0, 1)]
    outgoing = np.zeros_like(h)
    for axis, ((flux, _, _, _), delta) in enumerate(zip(faces, spacing, strict=True)):
        outgoing += (
            np.take(np.maximum(flux[..., 0], 0.0), range(1, h.shape[axis] + 1), axis=axis)
            - np.take(np.minimum(flux[..., 0], 0.0), range(h.shape[axis]), axis=axis)
        ) / delta
    theta = np.minimum(1.0, np.divide(h, dt * outgoing, out=np.ones_like(h), where=outgoing > 0))
    for axis, ((flux, pl, pr, eta), delta) in enumerate(zip(faces, spacing, strict=True)):
        pad = [(0, 0)] * 2
        pad[axis] = (1, 1)
        padded = np.pad(theta, pad, constant_values=1)
        flux *= np.where(
            flux[..., 0] >= 0,
            np.take(padded, range(h.shape[axis] + 1), axis=axis),
            np.take(padded, range(1, h.shape[axis] + 2), axis=axis),
        )[..., None]
        result -= dt / delta * np.diff(flux, axis=axis)
        result[..., 2 - axis] += (
            dt
            / delta
            * (
                np.take(pl, range(1, h.shape[axis] + 1), axis=axis)
                - np.take(pr, range(h.shape[axis]), axis=axis)
                - g * h * eta
            )
        )
    result[..., 1:] = np.where(result[..., :1] > dry, result[..., 1:], 0.0)
    return result


def reference_step(
    *,
    q,
    z,
    soil,
    rain_start,
    rain_end,
    infil,
    f0,
    fd,
    conductivity,
    manning,
    dt,
    dx,
    dy,
    gravity,
    dry,
    limiter,
):
    half = dt / 2
    rain = 0.75 * rain_start + 0.25 * rain_end
    initial, history = exchange(
        q=q, soil=soil, rate=rain, dt=half, infil=infil, f0=f0, fd=fd, conductivity=conductivity
    )
    start = friction(q=initial, n=manning, g=gravity, dry=dry, dt=half)
    spatial = dict(z=z, dt=dt, spacing=(dy, dx), g=gravity, dry=dry, limiter=limiter)
    mid = euler(q=start, **spatial)
    final = friction(q=0.5 * (start + euler(q=mid, **spatial)), n=manning, g=gravity, dry=dry, dt=half)
    result, history = exchange(
        q=final,
        soil=history,
        rate=0.25 * rain_start + 0.75 * rain_end,
        dt=half,
        infil=infil,
        f0=f0,
        fd=fd,
        conductivity=conductivity,
    )
    result[..., 1:] *= result[..., :1] > dry
    return result, history


def exchange(*, q, soil, rate, dt, infil, f0, fd, conductivity):
    h = q[..., 0]
    rise = rate * dt
    capacity = (f0 - conductivity) * np.exp(-soil / fd)
    base = infil + conductivity
    midpoint = 0.5 * np.minimum(h + rise, dt * (base + capacity))
    loss = np.minimum(h + rise, dt * (base + capacity * np.exp(-midpoint / fd)))
    end = h + rise - loss
    avg = np.divide(2.0, h + end, out=np.zeros_like(h), where=h + end > 0.0)
    wet = (h > 0.0) & (end > 0.0)
    varying = wet & (np.abs(rise - loss) > 1.0e-5 * h)
    avg[varying] = np.log(end[varying] / h[varying]) / (rise - loss)[varying]
    momentum = q[..., 1:] * np.where(wet, np.exp(-loss * avg), 0.0)[..., None]
    return np.concatenate((end[..., None], momentum), axis=-1), soil + loss
