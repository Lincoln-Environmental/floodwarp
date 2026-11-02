"""Opt-in FP64 culvert exchange for full SWE, using the established opening law.

A culvert is an internal, quasi-steady link between distinct surface cells.
This version requires disjoint endpoint cells. It conserves water, withdraws
proportional donor momentum and mixes the inflow with zero resolved momentum.
It does not resolve barrel inertia, pipe-network pressure or a submerged jet.
"""
import hashlib
import json

import numpy as np
import warp as wp

from floodwarp.config import PRECISION_MODE


@wp.func
def circular_discharge(eta_a: wp.float64, eta_b: wp.float64, diameter: wp.float64,
                       invert_a: wp.float64, invert_b: wp.float64, length: wp.float64,
                       manning: wp.float64, entrance: wp.float64, exit_loss: wp.float64,
                       one_way: wp.int32, max_q: wp.float64, gravity: wp.float64) -> wp.float64:
    # Same Manning + entrance/exit energy-loss model as structure_kernels,
    # with explicit FP64 gravity and rational exponents throughout.
    delta = eta_a - eta_b
    if delta == wp.float64(0.0) or (one_way == 1 and delta < wp.float64(0.0)):
        return wp.float64(0.0)
    upstream = eta_a if delta > wp.float64(0.0) else eta_b
    invert = invert_a if delta > wp.float64(0.0) else invert_b
    depth = wp.min(diameter, wp.max(wp.float64(0.0), upstream - invert))
    if depth <= wp.float64(0.0):
        return wp.float64(0.0)
    radius = diameter * wp.float64(0.5)
    cosine = wp.clamp((radius - depth) / radius, wp.float64(-1.0), wp.float64(1.0))
    angle = wp.acos(cosine)
    area = radius * radius * angle - (radius - depth) * wp.sqrt(wp.max(wp.float64(0.0), diameter * depth - depth * depth))
    perimeter = diameter * angle
    if area <= wp.float64(1.e-12) or perimeter <= wp.float64(1.e-12):
        return wp.float64(0.0)
    hydraulic_radius = area / perimeter
    loss = wp.float64(1.0) + entrance + exit_loss
    loss += wp.float64(2.0) * gravity * manning * manning * length / wp.pow(hydraulic_radius, wp.float64(4.0) / wp.float64(3.0))
    rate = wp.min(max_q, area * wp.sqrt(wp.float64(2.0) * gravity * wp.abs(delta) / wp.max(wp.float64(1.0), loss)))
    return rate if delta > wp.float64(0.0) else -rate


@wp.kernel
def snapshot_maximum(cells: wp.array2d(dtype=wp.int32), maximum: wp.array(dtype=wp.float64),
                     previous: wp.array2d(dtype=wp.float64)):
    s = wp.tid()
    previous[s, 0] = maximum[cells[s, 0]]
    previous[s, 1] = maximum[cells[s, 1]]


@wp.kernel
def exchange(q: wp.array(dtype=wp.vec3d), bed: wp.array(dtype=wp.float64),
             maximum: wp.array(dtype=wp.float64), cells: wp.array2d(dtype=wp.int32),
             parameters: wp.array2d(dtype=wp.float64), previous: wp.array2d(dtype=wp.float64),
             records: wp.array2d(dtype=wp.float64), ctrl: wp.array(dtype=wp.float64),
             area: wp.float64, dry: wp.float64, gravity: wp.float64):
    s = wp.tid()
    if ctrl[7] != wp.float64(0.0):
        return
    a = cells[s, 0]
    b = cells[s, 1]
    va = q[a]
    vb = q[b]
    # Water below either opening sill cannot drive flow through that opening.
    eta_a = wp.max(bed[a] + va[0], parameters[s, 1])
    eta_b = wp.max(bed[b] + vb[0], parameters[s, 2])
    rate = circular_discharge(eta_a, eta_b, parameters[s, 0],
        parameters[s, 1], parameters[s, 2], parameters[s, 3], parameters[s, 4],
        parameters[s, 5], parameters[s, 6], int(parameters[s, 7]), parameters[s, 8], gravity)
    request = wp.abs(rate) * ctrl[2]
    donor = va
    receiver = vb
    invert = parameters[s, 1]
    donor_bed = bed[a]
    if rate < wp.float64(0.0):
        donor = vb
        receiver = va
        invert = parameters[s, 2]
        donor_bed = bed[b]
    available = wp.min(donor[0], wp.max(wp.float64(0.0), donor_bed + donor[0] - invert))
    # Equal-area, disjoint endpoints: do not reverse their head difference
    # within one split step, even if a large barrel empties the donor quickly.
    limit = wp.min(available * area, wp.abs(eta_a - eta_b) * area * wp.float64(0.5))
    volume = wp.min(request, limit)
    dh = wp.min(donor[0], volume / area)
    retained = wp.float64(0.0)
    if donor[0] > wp.float64(0.0):
        retained = wp.max(wp.float64(0.0), wp.float64(1.0) - dh / donor[0])
    donor = wp.vec3d(wp.max(wp.float64(0.0), donor[0] - dh), donor[1] * retained, donor[2] * retained)
    receiver = wp.vec3d(receiver[0] + dh, receiver[1], receiver[2])
    if donor[0] <= dry:
        donor = wp.vec3d(donor[0], wp.float64(0.0), wp.float64(0.0))
    if receiver[0] <= dry:
        receiver = wp.vec3d(receiver[0], wp.float64(0.0), wp.float64(0.0))
    signed = dh * area
    if rate < wp.float64(0.0):
        q[b] = donor
        q[a] = receiver
        records[s, 2] += signed
        signed = -signed
    else:
        q[a] = donor
        q[b] = receiver
        records[s, 1] += signed
    records[s, 0] = signed / ctrl[2]
    records[s, 3] = wp.max(records[s, 3], wp.abs(records[s, 0]))
    records[s, 4] = wp.max(records[s, 4], wp.abs(eta_a - eta_b))
    if request > limit:
        records[s, 5] += wp.float64(1.0)
    # The composed surface + culvert state is the accepted step. Discard the
    # surface-only intermediate maximum at these endpoints.
    maximum[a] = wp.max(previous[s, 0], q[a][0])
    maximum[b] = wp.max(previous[s, 1], q[b][0])


class FV1Culverts:
    """Disjoint circular links; all arrays participate in native checkpoints."""

    def __init__(self, *, solver, structures):
        if PRECISION_MODE != 'fp64':
            raise ValueError('FV1 culverts require FLOODWARP_PRECISION=fp64')
        records = []
        cells = []
        parameters = []
        used = set()
        ids = set()
        for item in structures:
            if item.get('type') != 'culvert' or item.get('shape') != 'circular':
                raise ValueError('FV1 structures currently support circular culverts only')
            identifier = item['id']
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError('Culvert identifiers must be nonempty and unique')
            ids.add(identifier)
            ends = []
            for prefix in ('inlet', 'outlet'):
                row, col = item[prefix+'_row'], item[prefix+'_col']
                if int(row) != row or int(col) != col or not (0 <= row < solver.ny and 0 <= col < solver.nx):
                    raise ValueError('Culvert endpoint is outside the grid')
                if not solver._active[int(row), int(col)]:
                    raise ValueError('Culvert endpoint must be active')
                cell = int(row)*solver.nx+int(col)
                if cell in used:
                    raise ValueError('FV1 culvert endpoint cells must be distinct across all links')
                used.add(cell)
                ends.append(cell)
            names = ('diameter_m', 'inlet_invert_m', 'outlet_invert_m', 'length_m', 'manning_n',
                     'entrance_loss_coefficient', 'exit_loss_coefficient', 'one_way', 'max_discharge_m3_s')
            values = [float(item[name]) for name in names]
            if not np.isfinite(values).all() or min(values[0], values[3], values[8]) <= 0. or min(values[4:7]) < 0. or values[7] not in (0., 1.):
                raise ValueError('Invalid culvert dimensions, losses or direction constraint')
            records.append(dict(id=identifier, type='culvert', shape='circular',
                inlet_row=int(item['inlet_row']), inlet_col=int(item['inlet_col']),
                outlet_row=int(item['outlet_row']), outlet_col=int(item['outlet_col']),
                **dict(zip(names, values))))
            cells.append(ends)
            parameters.append(values)
        self.definitions = records
        self.count = len(records)
        self.identity = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.cells = wp.array(np.asarray(cells, dtype=np.int32), dtype=wp.int32, device=solver.device)
        self.parameters = wp.array(np.asarray(parameters, dtype=np.float64), dtype=wp.float64, device=solver.device)
        self.previous = wp.zeros((self.count, 2), dtype=wp.float64, device=solver.device)
        self.records = wp.zeros((self.count, 6), dtype=wp.float64, device=solver.device)
        for name in ('cells', 'parameters', 'previous', 'records'):
            solver._fields['culvert_'+name] = getattr(self, name)

    def snapshot(self, *, solver):
        wp.launch(kernel=snapshot_maximum, dim=self.count,
            inputs=[self.cells, solver._fields['maximum'], self.previous], device=solver.device)

    def enqueue(self, *, solver):
        wp.launch(kernel=exchange, dim=self.count,
            inputs=[solver.q, solver._fields['bed'], solver._fields['maximum'], self.cells,
                    self.parameters, self.previous, self.records, solver.ctrl,
                    solver.dx*solver.dy, solver.dry_depth, solver.gravity], device=solver.device)

    def diagnostics(self):
        records = self.records.numpy()
        return [dict(id=item['id'], discharge_m3_s=float(values[0]),
            forward_volume_m3=float(values[1]), reverse_volume_m3=float(values[2]),
            net_volume_m3=float(values[1]-values[2]), maximum_abs_discharge_m3_s=float(values[3]),
            maximum_head_difference_m=float(values[4]), limited_steps=int(values[5]))
            for item, values in zip(self.definitions, records)]
