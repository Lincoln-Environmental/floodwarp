"""NumPy unsplit Taylor half-step oracle, formed from intra-cell face traces."""

import numpy as np
from floodwarp.fv2_reference import directional, euler, traces


def velocity(*, state, dry):
    return np.divide(
        state[..., 1:],
        state[..., 0, None],
        out=np.zeros_like(state[..., 1:]),
        where=state[..., 0, None] > dry,
    )


def hancock(*, q, z, dt, spacing, g, dry, limiter, flux_scheme="contact_hll"):
    geometries = []
    u = velocity(state=q, dry=dry)
    h = q[..., 0]
    dhdt = np.zeros_like(h)
    dudt = np.zeros_like(u)
    excursion = np.zeros_like(h)
    for axis, delta in enumerate(spacing):
        low, high, eta, bed = traces(q=q, z=z, dry=dry, axis=axis, limiter=limiter, gravity=g)
        du = velocity(state=high, dry=0.0) - velocity(state=low, dry=0.0)
        dh = high[..., 0] - low[..., 0]
        normal = 1 - axis
        dhdt -= (u[..., normal] * dh + h * du[..., normal]) / delta
        dudt -= u[..., normal, None] * du / delta
        dudt[..., normal] -= g * eta / delta
        excursion = np.maximum(excursion, abs(dh) / 2)
        geometries.append((du, eta, bed))
    pred = q.copy()
    pred[..., 0] += dt / 2 * dhdt
    pred[..., 1:] += dt / 2 * (h[..., None] * dudt + u * dhdt[..., None])
    fallback = (h <= dry) | (pred[..., 0] <= dry + excursion)
    pred[fallback] = q[fallback]
    hmid = pred[..., 0]
    v = velocity(state=pred, dry=dry)
    faces = []
    for axis, (du, eta, bed) in enumerate(geometries):
        du = np.where(fallback[..., None], 0.0, du)
        eta = np.where(fallback, 0.0, eta)
        bed = np.where(fallback, 0.0, bed)
        heights = (hmid - (eta - bed) / 2, hmid + (eta - bed) / 2)
        scale = np.divide(heights[0] * heights[1], 2 * hmid, out=np.zeros_like(hmid), where=hmid > dry)
        states = []
        for sign, height in zip((-1, 1), heights, strict=True):
            m = height[..., None] * v + sign * scale[..., None] * du
            states.append(np.concatenate((height[..., None], m), axis=-1))
        faces.append(
            directional(
                q=pred,
                z=z,
                g=g,
                dry=dry,
                axis=axis,
                limiter=limiter,
                flux_scheme=flux_scheme,
                face_traces=(*states, eta, bed),
            )
        )
    return euler(
        q=q, z=z, dt=dt, spacing=spacing, g=g, dry=dry, limiter=limiter, faces=faces, source_depth=hmid
    )
