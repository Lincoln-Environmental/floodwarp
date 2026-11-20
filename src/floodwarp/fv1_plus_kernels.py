"""Local two-dimensional Taylor predictor for the experimental MUSCL–Hancock scheme."""

import warp as wp
from floodwarp.fv2_kernels import scheduled_index, wet_inverse, slope


@wp.func
def predict_cell(
    k: int,
    x: wp.vec4d,
    y: wp.vec4d,
    q: wp.array(dtype=wp.vec3d),
    out: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    inverse_mid: wp.array(dtype=wp.float64),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    active: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float64),
    dx: wp.float64,
    dy: wp.float64,
    g: wp.float64,
    dry: wp.float64,
    cfl: wp.float64,
):
    base = q[k]
    h = base[0]
    zero = wp.float64(0.0)
    out[k] = base
    inverse_mid[k] = inverse[k]
    sx[k] = x
    sy[k] = y
    fallback[k] = 0
    if h <= dry or active[k] == 0:
        return
    u = base[1] * inverse[k]
    v = base[2] * inverse[k]

    dt = wp.float64(0.5) * ctrl[2]
    dhx = (x[0] - x[1]) / dx
    dhy = (y[0] - y[1]) / dy
    dh = -dt * (u * dhx + v * dhy + h * (x[2] / dx + y[3] / dy))
    du = -dt * (u * x[2] / dx + v * y[2] / dy + g * x[0] / dx)
    dv = -dt * (u * x[3] / dx + v * y[3] / dy + g * y[0] / dy)
    candidate = base + wp.vec3d(dh, u * dh + h * du, v * dh + h * dv)
    # Nonconservative midpoint is a trace predictor only. All accepted
    # transport is the single, shared conservative face flux.
    excursion = wp.float64(0.5) * wp.max(wp.abs(x[0] - x[1]), wp.abs(y[0] - y[1]))
    if not wp.isfinite(candidate):
        wp.atomic_min(ctrl, 7, wp.float64(-1.0))
    elif candidate[0] <= dry + excursion:
        fallback[k] = 1
        sx[k] = wp.vec4d(zero)
        sy[k] = wp.vec4d(zero)
    else:
        out[k] = candidate
        inverse_mid[k] = wet_inverse(candidate[0], dry)
    c = wp.sqrt(g * h)
    rate = wp.max((wp.abs(u) + c) / dx, (wp.abs(v) + c) / dy)
    wp.atomic_min(ctrl, 17, wp.float64(0.5) * cfl / rate)


@wp.kernel
def predict(
    q: wp.array(dtype=wp.vec3d),
    out: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    inverse_mid: wp.array(dtype=wp.float64),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    active: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float64),
    dx: wp.float64,
    dy: wp.float64,
    g: wp.float64,
    dry: wp.float64,
    cfl: wp.float64,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    predict_cell(
        k, sx[k], sy[k], q, out, inverse, inverse_mid, sx, sy, active, fallback, ctrl, dx, dy, g, dry, cfl
    )


@wp.kernel
def slopes_predict(
    q: wp.array(dtype=wp.vec3d),
    out: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    inverse_mid: wp.array(dtype=wp.float64),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    active: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float64),
    dx: wp.float64,
    dy: wp.float64,
    g: wp.float64,
    dry: wp.float64,
    cfl: wp.float64,
    flags: wp.array(dtype=wp.int32),
    bed: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    mc: int,
    work: wp.array(dtype=wp.int32),
):
    k = scheduled_index(wp.tid(), work)
    x = slope(k, 0, q, inverse, flags, bed, active, nx, ny, dry, mc)
    y = slope(k, 1, q, inverse, flags, bed, active, nx, ny, dry, mc)
    predict_cell(k, x, y, q, out, inverse, inverse_mid, sx, sy, active, fallback, ctrl, dx, dy, g, dry, cfl)
