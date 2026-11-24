"""Local two-dimensional Taylor predictor for the experimental MUSCL–Hancock scheme."""

from typing import Any

import warp as wp
from floodwarp.hancock_tiles import scheduled_index, scheduled_kernel
from floodwarp.fv2_kernels import DIRECTIONAL_CONTROL_SIZE, X_SIGNAL_RATE, Y_SIGNAL_RATE, wet_inverse, slope


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
    fallback: wp.array(dtype=wp.uint8),
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
    fallback[k] = wp.uint8(0)
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
        fallback[k] = wp.uint8(1)
        sx[k] = wp.vec4d(zero)
        sy[k] = wp.vec4d(zero)
    else:
        out[k] = candidate
        inverse_mid[k] = wet_inverse(candidate[0], dry)
    c = wp.sqrt(g * h)
    ax = (wp.abs(u) + c) / dx
    ay = (wp.abs(v) + c) / dy
    if ctrl.shape[0] == DIRECTIONAL_CONTROL_SIZE:
        wp.atomic_max(ctrl, X_SIGNAL_RATE, ax)
        wp.atomic_max(ctrl, Y_SIGNAL_RATE, ay)
    else:
        wp.atomic_min(ctrl, 17, wp.float64(0.5) * cfl / wp.max(ax, ay))


@scheduled_kernel
def predict(
    q: wp.array(dtype=wp.vec3d),
    out: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    inverse_mid: wp.array(dtype=wp.float64),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    active: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.uint8),
    ctrl: wp.array(dtype=wp.float64),
    dx: wp.float64,
    dy: wp.float64,
    g: wp.float64,
    dry: wp.float64,
    cfl: wp.float64,
    work: Any,
):
    k = scheduled_index(wp.tid(), work)
    if k < 0:
        return
    predict_cell(
        k, sx[k], sy[k], q, out, inverse, inverse_mid, sx, sy, active, fallback, ctrl, dx, dy, g, dry, cfl
    )


@scheduled_kernel
def slopes_predict(
    q: wp.array(dtype=wp.vec3d),
    out: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    inverse_mid: wp.array(dtype=wp.float64),
    sx: wp.array(dtype=wp.vec4d),
    sy: wp.array(dtype=wp.vec4d),
    active: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.uint8),
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
    types_x: wp.array(dtype=wp.int32),
    types_y: wp.array(dtype=wp.int32),
    free_slope: int,
    work: Any,
):
    k = scheduled_index(wp.tid(), work)
    if k < 0:
        return
    x = slope(k, 0, q, inverse, flags, bed, active, types_x, nx, ny, dry, mc, free_slope)
    y = slope(k, 1, q, inverse, flags, bed, active, types_y, nx, ny, dry, mc, free_slope)
    predict_cell(k, x, y, q, out, inverse, inverse_mid, sx, sy, active, fallback, ctrl, dx, dy, g, dry, cfl)
