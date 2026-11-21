"""Conservative, accepted-step surface links for Hancock on CPU and CUDA."""

from collections.abc import Mapping
import hashlib
import json

import numpy as np
import warp as wp

from floodwarp.config import PRECISION_MODE
from floodwarp.structure_kernels import structure_discharge


@wp.struct
class Opening:
    kind: int
    width: wp.float64
    height: wp.float64
    invert_a: wp.float64
    invert_b: wp.float64
    length: wp.float64
    manning: wp.float64
    entrance: wp.float64
    exit_loss: wp.float64
    crest: wp.float64
    crest_length: wp.float64
    weir_factor: wp.float64
    one_way: int
    capacity: wp.float64


@wp.kernel
def save_maximum(
    cells: wp.array(dtype=int),
    maximum: wp.array(dtype=wp.float64),
    old: wp.array(dtype=wp.float64),
):
    slot = wp.tid()
    old[slot] = maximum[cells[slot]]


@wp.kernel
def request_links(
    q: wp.array(dtype=wp.vec3d),
    bed: wp.array(dtype=wp.float64),
    nodes: wp.array(dtype=int),
    ends: wp.array(dtype=wp.vec2i),
    openings: wp.array(dtype=Opening),
    ctrl: wp.array(dtype=wp.float64),
    area: wp.float64,
    gravity: wp.float64,
    request: wp.array(dtype=wp.float64),
    head_room: wp.array(dtype=wp.float64),
    records: wp.array2d(dtype=wp.float64),
):
    if ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
        return
    link = wp.tid()
    a = nodes[ends[link][0]]
    b = nodes[ends[link][1]]
    p = openings[link]
    ea = wp.max(bed[a] + q[a][0], p.invert_a)
    eb = wp.max(bed[b] + q[b][0], p.invert_b)
    flow = structure_discharge(
        p.kind,
        ea,
        eb,
        p.width,
        p.height,
        p.invert_a,
        p.invert_b,
        p.length,
        p.manning,
        p.entrance,
        p.exit_loss,
        p.crest,
        p.crest_length,
        p.weir_factor,
        p.one_way,
        p.capacity,
        gravity,
    )
    volume = flow * ctrl[2]
    limit = wp.abs(ea - eb) * wp.float64(0.5) * area
    head_room[link] = limit
    request[link] = wp.sign(volume) * wp.min(wp.abs(volume), limit)
    records[link, 4] = wp.max(records[link, 4], wp.abs(ea - eb))
    records[link, 6] = volume


@wp.kernel
def budget_endpoints(
    q: wp.array(dtype=wp.vec3d),
    bed: wp.array(dtype=wp.float64),
    cells: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    incident: wp.array(dtype=int),
    sill: wp.array(dtype=wp.float64),
    request: wp.array(dtype=wp.float64),
    head_room: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.vec2d),
    area: wp.float64,
):
    slot = wp.tid()
    cell = cells[slot]
    total = wp.float64(0.0)
    incoming = wp.float64(0.0)
    receive_limit = wp.float64(1.0e100)
    donate_limit = wp.float64(1.0e100)
    factor = wp.float64(1.0)
    # Incident links are sorted by descending sill. Each prefix can only
    # withdraw water above its lowest sill, even for links sharing a donor.
    for i in range(offsets[slot], offsets[slot + 1]):
        side = incident[i]
        link = wp.abs(side) - 1
        signed = wp.float64(wp.sign(side)) * request[link]
        if signed > wp.float64(0.0):
            donate_limit = wp.min(donate_limit, head_room[link])
            total += signed
            available = (
                wp.min(
                    q[cell][0],
                    wp.max(wp.float64(0.0), bed[cell] + q[cell][0] - sill[i]),
                )
                * area
            )
            if total >= available:
                factor = wp.min(
                    factor, (wp.float64(1.0) - wp.float64(2.0e-15)) * available / total
                )
        elif signed < wp.float64(0.0):
            receive_limit = wp.min(receive_limit, head_room[link])
            incoming -= signed
    # Aggregate links cannot jointly overfill a receiver or reverse an active
    # pair's effective head gradient in this explicit structure substep.
    receive = wp.float64(1.0)
    if incoming > receive_limit:
        receive = receive_limit / incoming
    if total > donate_limit:
        factor = wp.min(factor, donate_limit / total)
    scale[slot] = wp.vec2d(factor, receive)


@wp.kernel
def accept_links(
    ends: wp.array(dtype=wp.vec2i),
    request: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.vec2d),
    actual: wp.array(dtype=wp.float64),
    records: wp.array2d(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
):
    if ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
        return
    link = wp.tid()
    volume = request[link]
    donor = ends[link][0] if volume >= wp.float64(0.0) else ends[link][1]
    receiver = ends[link][1] if volume >= wp.float64(0.0) else ends[link][0]
    volume *= wp.min(scale[donor][0], scale[receiver][1])
    actual[link] = volume
    records[link, 0] = volume / ctrl[2]
    records[link, 1] += wp.max(wp.float64(0.0), volume)
    records[link, 2] -= wp.min(wp.float64(0.0), volume)
    records[link, 3] = wp.max(records[link, 3], wp.abs(volume / ctrl[2]))
    if wp.abs(records[link, 6]) > wp.abs(volume):
        records[link, 5] += wp.float64(1.0)


@wp.kernel
def update_endpoints(
    q: wp.array(dtype=wp.vec3d),
    cells: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    incident: wp.array(dtype=int),
    actual: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float64),
    old: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    area: wp.float64,
    dry: wp.float64,
):
    if ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
        return
    slot = wp.tid()
    cell = cells[slot]
    incoming = wp.float64(0.0)
    outgoing = wp.float64(0.0)
    for i in range(offsets[slot], offsets[slot + 1]):
        side = incident[i]
        volume = wp.float64(wp.sign(side)) * actual[wp.abs(side) - 1]
        outgoing += wp.max(wp.float64(0.0), volume)
        incoming -= wp.min(wp.float64(0.0), volume)
    state = q[cell]
    remaining = wp.max(wp.float64(0.0), state[0] - outgoing / area)
    if outgoing > wp.float64(0.0) and state[0] > wp.float64(0.0):
        fraction = remaining / state[0]
        state[1] *= fraction
        state[2] *= fraction
    state[0] = remaining + incoming / area
    if state[0] <= dry:
        state[1] = wp.float64(0.0)
        state[2] = wp.float64(0.0)
    maximum[cell] = wp.max(old[slot], state[0])
    q[cell] = state


def opening(*, record, gravity):
    p = Opening()
    kind = record.get("type")
    if kind == "weir":
        kinds = {"sharp_crested": 2, "broad_crested": 3}
        if record.get("weir_type") not in kinds:
            raise ValueError("Unsupported weir type")
        p.kind = kinds[record["weir_type"]]
        p.crest = p.invert_a = p.invert_b = float(record["crest_elevation_m"])
        p.crest_length = float(record["crest_length_m"])
        coefficient = float(record["discharge_coefficient"])
        p.weir_factor = coefficient * (
            np.sqrt(8 * gravity / 27) if p.kind == 3 else 2 / 3 * np.sqrt(2 * gravity)
        )
        if min(coefficient, p.crest_length) <= 0.0:
            raise ValueError("Positive weir coefficient and crest length required")
    elif kind in ("bridge", "culvert"):
        if kind == "bridge":
            p.kind = 4
            p.width, p.height = (
                float(record["opening_width_m"]),
                float(record["opening_height_m"]),
            )
            p.invert_a = p.invert_b = float(record["opening_invert_m"])
        else:
            if record.get("shape") not in ("circular", "box", "rectangular"):
                raise ValueError("Unsupported culvert shape")
            p.kind = 0 if record["shape"] == "circular" else 1
            p.width = float(record["diameter_m"] if p.kind == 0 else record["width_m"])
            p.height = p.width if p.kind == 0 else float(record["height_m"])
            p.invert_a, p.invert_b = (
                float(record["inlet_invert_m"]),
                float(record["outlet_invert_m"]),
            )
        p.length, p.manning = (
            float(record.get("length_m", 0.0)),
            float(record.get("manning_n", 0.0)),
        )
        p.entrance = float(record.get("entrance_loss_coefficient", 0.0))
        p.exit_loss = float(record.get("exit_loss_coefficient", 0.0))
        if (
            min(p.width, p.height) <= 0.0
            or min(p.length, p.manning, p.entrance, p.exit_loss) < 0.0
        ):
            raise ValueError("Invalid structure dimensions or losses")
    else:
        raise ValueError("Unsupported Hancock structure type")
    cap = record.get("maximum_discharge_m3s")
    legacy_cap = record.get("max_discharge_m3_s")
    if cap is not None and legacy_cap is not None and cap != legacy_cap:
        raise ValueError("Conflicting structure discharge caps")
    p.capacity = float(
        cap if cap is not None else legacy_cap if legacy_cap is not None else 1.0e30
    )
    if p.capacity <= 0.0 or not np.isfinite(list(p.numpy_value())).all():
        raise ValueError(
            "Structure parameters must be finite with a positive discharge cap"
        )
    one_way = record.get("one_way", False)
    if one_way not in (True, False):
        raise ValueError("Invalid one_way flag")
    p.one_way = int(one_way)
    return p


def endpoint(*, record, prefix, solver):
    row, col = record[prefix + "_row"], record[prefix + "_col"]
    if (
        int(row) != row
        or int(col) != col
        or not (0 <= row < solver.ny and 0 <= col < solver.nx)
    ):
        raise ValueError("Invalid structure endpoint")
    if not solver._active[int(row), int(col)]:
        raise ValueError("Structure endpoints must be active")
    return int(row) * solver.nx + int(col)


class HancockStructures:
    def __init__(self, *, solver, structures):
        if PRECISION_MODE != "fp64":
            raise ValueError("Hancock structures require FP64")
        self.ids = []
        pairs, openings = [], []
        for record in structures:
            if not isinstance(record, Mapping):
                raise ValueError("Supply hydraulic structure records as mappings")
            name = record.get("id")
            if not isinstance(name, str) or not name or name in self.ids:
                raise ValueError("Structure IDs must be unique nonempty strings")
            self.ids.append(name)
            pair = [
                endpoint(record=record, prefix=side, solver=solver)
                for side in ("inlet", "outlet")
            ]
            if pair[0] == pair[1]:
                raise ValueError("Structure endpoints must be distinct")
            pairs.append(pair)
            openings.append(opening(record=record, gravity=solver.gravity))
        cells, indices = np.unique(pairs, return_inverse=True)
        indices = indices.reshape(-1, 2)
        adjacency = [[] for _ in cells]
        for n, (a, b) in enumerate(indices):
            adjacency[a].append((openings[n].invert_a, n + 1))
            adjacency[b].append((openings[n].invert_b, -(n + 1)))
        incident, sill, offsets = [], [], [0]
        for edges in adjacency:
            for elevation, side in sorted(edges, reverse=True):
                incident.append(side)
                sill.append(elevation)
            offsets.append(len(incident))
        self.identity = hashlib.sha256(
            json.dumps(
                [self.ids, pairs, [p.numpy_value() for p in openings]],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.cells = wp.array(cells, dtype=int, device=solver.device)
        self.ends = wp.array(indices, dtype=wp.vec2i, device=solver.device)
        self.openings = wp.array(openings, dtype=Opening, device=solver.device)
        self.offsets = wp.array(offsets, dtype=int, device=solver.device)
        self.incident = wp.array(incident, dtype=int, device=solver.device)
        self.sill = wp.array(sill, dtype=wp.float64, device=solver.device)
        self.request = wp.zeros(len(pairs), dtype=wp.float64, device=solver.device)
        self.actual = wp.zeros_like(self.request)
        self.head_room = wp.zeros_like(self.request)
        self.old = wp.zeros(len(cells), dtype=wp.float64, device=solver.device)
        self.scale = wp.zeros(len(cells), dtype=wp.vec2d, device=solver.device)
        self.records = wp.zeros((len(pairs), 7), dtype=wp.float64, device=solver.device)
        solver._fields["structure_records"] = self.records

    def snapshot(self, *, solver):
        solver._launch(
            kernel=save_maximum,
            count=self.cells.size,
            inputs=[self.cells, solver._fields["maximum"], self.old],
        )

    def enqueue(self, *, solver):
        area = solver.dx * solver.dy
        f = solver._fields
        solver._launch(
            kernel=request_links,
            count=self.ends.size,
            inputs=[
                solver.q,
                f["bed"],
                self.cells,
                self.ends,
                self.openings,
                solver.ctrl,
                area,
                solver.gravity,
                self.request,
                self.head_room,
                self.records,
            ],
        )
        solver._launch(
            kernel=budget_endpoints,
            count=self.cells.size,
            inputs=[
                solver.q,
                f["bed"],
                self.cells,
                self.offsets,
                self.incident,
                self.sill,
                self.request,
                self.head_room,
                self.scale,
                area,
            ],
        )
        solver._launch(
            kernel=accept_links,
            count=self.ends.size,
            inputs=[
                self.ends,
                self.request,
                self.scale,
                self.actual,
                self.records,
                solver.ctrl,
            ],
        )
        solver._launch(
            kernel=update_endpoints,
            count=self.cells.size,
            inputs=[
                solver.q,
                self.cells,
                self.offsets,
                self.incident,
                self.actual,
                f["maximum"],
                self.old,
                solver.ctrl,
                area,
                solver.dry_depth,
            ],
        )

    def diagnostics(self):
        return [
            dict(
                id=name,
                discharge_m3_s=v[0],
                forward_volume_m3=v[1],
                reverse_volume_m3=v[2],
                net_volume_m3=v[1] - v[2],
                maximum_abs_discharge_m3_s=v[3],
                maximum_head_difference_m=v[4],
                limited_steps=int(v[5]),
            )
            for name, v in zip(self.ids, self.records.numpy(), strict=True)
        ]
