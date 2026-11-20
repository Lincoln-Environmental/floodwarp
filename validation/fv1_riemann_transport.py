"""Opt-in, flux-weighted diagnostics for interior faces of a dense or sparse FV1 replay."""

import numpy as np
import warp as wp

from floodwarp.full_swe_kernels import reconstruct, research_flux


@wp.kernel
def classify_faces(q: wp.array(dtype=wp.vec3d), bed: wp.array(dtype=wp.float64),
                   active: wp.array(dtype=wp.int32), channel: wp.array(dtype=wp.int32),
                   edges: wp.array(dtype=wp.float64), wet: wp.float64, g: wp.float64,
                   dry: wp.float64, mode: int, axis: int, nx: int, ny: int,
                   labels: wp.array(dtype=wp.int32)):
    face = wp.tid()
    labels[face] = -1
    stride = 1
    if axis == 0:
        i = face % (nx + 1)
        if i == 0 or i == nx:
            return
        right = face - face / (nx + 1)
    else:
        if face < nx or face >= nx*ny:
            return
        right = face
        stride = nx
    left = right - stride
    if active[left] == 0 or active[right] == 0:
        return
    top = wp.max(bed[left], bed[right])
    a = reconstruct(q[left], bed[left], top, dry)
    b = reconstruct(q[right], bed[right], top, dry)
    depth = wp.max(a[0], b[0])
    if depth == wp.float64(0.0):
        return
    if axis == 1:
        a = wp.vec3d(a[0], a[2], a[1])
        b = wp.vec3d(b[0], b[2], b[1])
    result = research_flux(left=a, right=b, g=g, dry=dry, mode=mode)
    band = int(0)
    for i in range(edges.shape[0]):
        if depth >= edges[i]:
            band += 1
    flags = int(0)
    if channel[left] != 0 or channel[right] != 0:
        flags += 1
    if wp.min(a[0], b[0]) >= wet:
        flags += 2
        fa = a[1] / (a[0]*wp.sqrt(g*a[0]))
        fb = b[1] / (b[0]*wp.sqrt(g*b[0]))
        if ((fa > 1.0 and fb > 0.0 and fb < 1.0 and b[0] > a[0]) or
                (fb < -1.0 and fa < 0.0 and fa > -1.0 and a[0] > b[0])):
            flags += 4
    labels[face] = (8*band + flags)*4 + int(result[3])


@wp.kernel
def sum_transport(labels: wp.array(dtype=wp.int32), flux: wp.array(dtype=wp.vec3d),
                  ctrl: wp.array(dtype=wp.float64), width: wp.float64,
                  total: wp.array3d(dtype=wp.float64)):
    face = wp.tid()
    label = labels[face]
    if label >= 0:
        f = flux[face]
        scale = width*ctrl[2]
        shard = face % total.shape[0]
        wp.atomic_add(total, shard, label, 0, wp.float64(1.0))
        wp.atomic_add(total, shard, label, 1, scale*wp.abs(f[0]))
        wp.atomic_add(total, shard, label, 2, scale*wp.sqrt(f[1]*f[1]+f[2]*f[2]))


class InteriorTransport:
    """Wrap only an instrumented replay. Counters use the donor-limited face flux."""

    def __init__(self, *, solver, edges, wet, channel_depth, channel_discharge):
        self.solver = solver
        self.advance = solver._enqueue_hydraulics
        state = solver.get_conservative_state()
        mask = (state[..., 0] >= channel_depth) & (np.hypot(state[..., 1], state[..., 2]) >= channel_discharge)
        self.channel = wp.array(mask.ravel().astype(np.int32), device=solver.device)
        self.edges = wp.array(edges, dtype=wp.float64, device=solver.device)
        self.wet = wet
        self.labels = [wp.empty(shape=solver._fields['flux_'+axis].size, dtype=wp.int32,
                                device=solver.device) for axis in ('x', 'y')]
        self.total = wp.zeros(shape=(128, (len(edges)+1)*32, 3), dtype=wp.float64, device=solver.device)
        self.settings = dict(depth_edges_m=list(edges), jointly_wet_m=wet, initial_channel_depth_m=channel_depth,
                             initial_channel_discharge_m2_s=channel_discharge, fixed_channel_cells=int(mask.sum()))
        solver._enqueue_hydraulics = self.enqueue
        solver._graph = None

    def enqueue(self):
        s = self.solver
        for axis, labels in enumerate(self.labels):
            wp.launch(kernel=classify_faces, dim=labels.size, inputs=[s.q, s._fields['bed'], s._fields['active'],
                self.channel, self.edges, self.wet, s.gravity, s.dry_depth, s._flux_mode, axis, s.nx, s.ny, labels],
                device=s.device)
        self.advance()
        for axis, labels in enumerate(self.labels):
            direction, width = ('x', s.dy) if axis == 0 else ('y', s.dx)
            wp.launch(kernel=sum_transport, dim=labels.size, inputs=[labels, s._fields['flux_'+direction],
                      s.ctrl, width, self.total], device=s.device)

    def results(self):
        values = self.total.numpy().sum(axis=0).reshape(-1, 8, 4, 3)
        groups = {}
        for name, flag in (('all', 0), ('channel_proxy', 1), ('jointly_wet', 2), ('jump_proxy', 4)):
            selected = [i for i in range(8) if i & flag == flag]
            groups[name] = values[:, selected].sum(axis=1).tolist()
        # Release bound method references before the next full-domain run.
        self.solver._enqueue_hydraulics = self.advance
        self.solver._graph = None
        return dict(settings=self.settings, axes=['depth_band', 'outcome', 'metric'],
                    outcomes=['usual', 'shallow_hll', 'inadmissible_hll', 'entropy_corrected'],
                    metrics=['face_steps', 'absolute_volume_m3', 'momentum_flux_impulse_per_density_m4_s'],
                    groups=groups)
