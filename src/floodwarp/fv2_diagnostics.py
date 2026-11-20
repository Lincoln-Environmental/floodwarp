"""Optional accepted-step MUSCL fallback counts, volume and location."""

import numpy as np
import warp as wp

from floodwarp.fv2_kernels import indices, neighbor, limited


@wp.func
def reason(
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
    if k < 0 or active[wp.max(0, k)] == 0:
        return 0
    a = neighbor(k, axis, -1, nx, ny)
    b = neighbor(k, axis, 1, nx, ny)
    if q[k][0] <= dry:
        return 3
    if a < 0 or b < 0 or active[wp.max(a, 0)] == 0 or active[wp.max(b, 0)] == 0:
        return 1
    if wp.min(q[a][0], q[b][0]) <= dry:
        return 3
    la = bed[k] - bed[a]
    rb = bed[b] - bed[k]
    dh = limited(q[k][0] - q[a][0] + la, q[b][0] - q[k][0] + rb, mc) - limited(la, rb, mc)
    if wp.abs(dh) >= wp.float64(2.0) * q[k][0]:
        return 2
    return 0


@wp.kernel
def measure(
    q: wp.array(dtype=wp.vec3d),
    f: wp.array(dtype=wp.vec3d),
    bed: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float64),
    edges: wp.array(dtype=wp.float64),
    counts: wp.array3d(dtype=wp.float64),
    locations: wp.array(dtype=wp.float64),
    nx: int,
    ny: int,
    axis: int,
    width: wp.float64,
    dry: wp.float64,
    mc: int,
):
    if ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
        return
    k = wp.tid()
    pair = indices(k, axis, nx, ny)
    depth = wp.float64(0.0)
    why = int(0)
    for side in range(2):
        cell = pair[side]
        if cell >= 0 and active[wp.max(0, cell)] != 0:
            depth = wp.max(depth, q[cell][0])
            why = wp.max(why, reason(cell, axis, q, bed, active, nx, ny, dry, mc))
    if depth > wp.float64(0.0) or f[k][0] != wp.float64(0.0):
        band = int(0)
        for i in range(edges.shape[0]):
            if depth >= edges[i]:
                band += 1
        weight = wp.float64(0.5) * ctrl[2] * width * wp.abs(f[k][0])
        wp.atomic_add(counts, band, why, 0, wp.float64(1.0))
        wp.atomic_add(counts, band, why, 1, weight)
        if why > 0:
            locations[k] += weight


class ReconstructionDiagnostics:
    def __init__(self, *, solver, depth_edges):
        edges = np.asarray(depth_edges, dtype=float)
        if not (
            edges.ndim == 1 and np.isfinite(edges).all() and (edges > 0).all() and (np.diff(edges) > 0).all()
        ):
            raise ValueError("Depth thresholds must be finite, positive and increasing")
        self.edges = wp.array(edges, dtype=wp.float64, device=solver.device)
        self.counts = wp.zeros((len(edges) + 1, 4, 2), dtype=wp.float64, device=solver.device)
        self.maps = [wp.zeros(f.size, dtype=wp.float64, device=solver.device) for f in solver._flux[0]]

    def enqueue(self, *, solver):
        s = solver
        for stage, q in enumerate((s._start, s._stage)):
            for axis, f in enumerate(s._flux[stage]):
                width = s.dy if axis == 0 else s.dx
                s._launch(
                    measure,
                    f.size,
                    [
                        q,
                        f,
                        s._fields["bed"],
                        s._fields["active"],
                        s.ctrl,
                        self.edges,
                        self.counts,
                        self.maps[axis],
                        s.nx,
                        s.ny,
                        axis,
                        width,
                        s.dry_depth,
                        int(s.limiter == "mc"),
                    ],
                )

    def summary(self):
        return dict(
            edges_m=self.edges.numpy().tolist(),
            reasons=["unrestricted_stencil", "boundary", "positivity", "dry_stencil"],
            metrics=["stage_faces", "absolute_volume_m3"],
            bands=self.counts.numpy().tolist(),
        )
