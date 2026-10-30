"""Sparse internal hydraulic-structure transfer kernels.

One Warp thread operates each structure; endpoint scratch arrays are compact
and sized only to the unique structure endpoint cells. The donor limiter first
shares available endpoint storage proportionally among outgoing requests.
"""
from __future__ import annotations

import warp as wp
from floodwarp.config import BED_FLOAT, WP_FLOAT

G = 9.80665


@wp.kernel
def snapshot_structure_endpoints(
    depth: wp.array2d(dtype=WP_FLOAT),
    bed: wp.array2d(dtype=BED_FLOAT),
    active: wp.array2d(dtype=wp.int32),
    subgrid_stage: wp.array2d(dtype=WP_FLOAT),
    endpoint_cells: wp.array(dtype=wp.int32),
    endpoint_stage: wp.array(dtype=WP_FLOAT),
    endpoint_storage: wp.array(dtype=WP_FLOAT),
    requested_outflow: wp.array(dtype=WP_FLOAT),
    nx: wp.int32,
    cell_area: WP_FLOAT,
    use_subgrid: wp.int32,
):
    k = wp.tid()
    flat = endpoint_cells[k]
    j = flat // nx
    i = flat - j * nx
    if active[j, i] == 1:
        if use_subgrid == 1:
            endpoint_stage[k] = subgrid_stage[j, i]
        else:
            endpoint_stage[k] = WP_FLOAT(bed[j, i]) + depth[j, i]
        endpoint_storage[k] = wp.max(WP_FLOAT(0.0), depth[j, i]) * cell_area
    else:
        endpoint_stage[k] = WP_FLOAT(0.0)
        endpoint_storage[k] = WP_FLOAT(0.0)
    requested_outflow[k] = WP_FLOAT(0.0)


@wp.func
def _circular_area(diameter: WP_FLOAT, depth: WP_FLOAT) -> WP_FLOAT:
    radius = diameter * WP_FLOAT(0.5)
    h = wp.min(diameter, wp.max(WP_FLOAT(0.0), depth))
    if h <= WP_FLOAT(0.0):
        return WP_FLOAT(0.0)
    if h >= diameter:
        return WP_FLOAT(0.25) * WP_FLOAT(3.141592653589793) * diameter * diameter
    x = wp.max(WP_FLOAT(-1.0), wp.min(WP_FLOAT(1.0), (radius - h) / radius))
    return radius * radius * wp.acos(x) - (radius - h) * wp.sqrt(wp.max(WP_FLOAT(0.0), WP_FLOAT(2.0) * radius * h - h * h))


@wp.func
def _structure_q(
    kind: wp.int32,
    eta_a: WP_FLOAT,
    eta_b: WP_FLOAT,
    width: WP_FLOAT,
    height: WP_FLOAT,
    invert_a: WP_FLOAT,
    invert_b: WP_FLOAT,
    length: WP_FLOAT,
    manning: WP_FLOAT,
    entrance_loss: WP_FLOAT,
    exit_loss: WP_FLOAT,
    crest: WP_FLOAT,
    weir_length: WP_FLOAT,
    weir_factor: WP_FLOAT,
    one_way: wp.int32,
    max_q: WP_FLOAT,
) -> WP_FLOAT:
    delta = eta_a - eta_b
    if one_way == 1 and delta <= WP_FLOAT(0.0):
        return WP_FLOAT(0.0)
    if delta == WP_FLOAT(0.0):
        return WP_FLOAT(0.0)
    forward = delta > WP_FLOAT(0.0)
    upstream = eta_a if forward else eta_b
    downstream = eta_b if forward else eta_a
    if kind == 2 or kind == 3:  # sharp/broad-crested weir
        head = wp.max(WP_FLOAT(0.0), upstream - crest)
        if head <= WP_FLOAT(0.0):
            return WP_FLOAT(0.0)
        submergence = wp.max(WP_FLOAT(0.0), downstream - crest) / head
        correction = wp.pow(wp.max(WP_FLOAT(0.0), WP_FLOAT(1.0) - wp.pow(wp.min(WP_FLOAT(1.0), submergence), WP_FLOAT(1.5))), WP_FLOAT(0.385))
        q = weir_factor * weir_length * wp.pow(head, WP_FLOAT(1.5)) * correction
    else:  # culvert or bridge opening; Manning + entrance/exit energy losses
        invert = invert_a if forward else invert_b
        opening_h = wp.min(height, wp.max(WP_FLOAT(0.0), upstream - invert))
        if opening_h <= WP_FLOAT(0.0):
            return WP_FLOAT(0.0)
        if kind == 0:  # circular culvert, partially full circular segment
            area = _circular_area(width, opening_h)
            radius = width * WP_FLOAT(0.5)
            x = wp.max(WP_FLOAT(-1.0), wp.min(WP_FLOAT(1.0), (radius - opening_h) / radius))
            perimeter = WP_FLOAT(2.0) * radius * wp.acos(x)
            full_area = WP_FLOAT(0.25) * WP_FLOAT(3.141592653589793) * width * width
            if opening_h >= height:
                area = full_area
                perimeter = WP_FLOAT(3.141592653589793) * width
        else:
            area = width * opening_h
            perimeter = width + WP_FLOAT(2.0) * opening_h
        if area <= WP_FLOAT(1.0e-12) or perimeter <= WP_FLOAT(1.0e-12):
            return WP_FLOAT(0.0)
        hydraulic_radius = area / perimeter
        head = wp.abs(delta)
        # Manning friction expressed as an equivalent quadratic energy loss.
        loss = WP_FLOAT(1.0) + entrance_loss + exit_loss
        loss += WP_FLOAT(2.0) * WP_FLOAT(G) * manning * manning * length / wp.pow(hydraulic_radius, WP_FLOAT(4.0 / 3.0))
        q = area * wp.sqrt(WP_FLOAT(2.0) * WP_FLOAT(G) * head / wp.max(WP_FLOAT(1.0), loss))
    q = wp.min(q, max_q)
    return q if forward else -q


@wp.kernel
def calculate_structure_requests(
    endpoint_stage: wp.array(dtype=WP_FLOAT),
    endpoint_storage: wp.array(dtype=WP_FLOAT),
    endpoint_cells: wp.array(dtype=wp.int32),
    endpoint_slot_a: wp.array(dtype=wp.int32),
    endpoint_slot_b: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    one_way: wp.array(dtype=wp.int32),
    width: wp.array(dtype=WP_FLOAT),
    height: wp.array(dtype=WP_FLOAT),
    invert_a: wp.array(dtype=WP_FLOAT),
    invert_b: wp.array(dtype=WP_FLOAT),
    length: wp.array(dtype=WP_FLOAT),
    manning: wp.array(dtype=WP_FLOAT),
    entrance_loss: wp.array(dtype=WP_FLOAT),
    exit_loss: wp.array(dtype=WP_FLOAT),
    crest: wp.array(dtype=WP_FLOAT),
    weir_length: wp.array(dtype=WP_FLOAT),
    weir_factor: wp.array(dtype=WP_FLOAT),
    max_q: wp.array(dtype=WP_FLOAT),
    dt: WP_FLOAT,
    donor_requested: wp.array(dtype=WP_FLOAT),
    requested_volume: wp.array(dtype=WP_FLOAT),
    instantaneous_q: wp.array(dtype=WP_FLOAT),
    max_head: wp.array(dtype=WP_FLOAT),
):
    s = wp.tid()
    a = endpoint_slot_a[s]
    b = endpoint_slot_b[s]
    eta_a = endpoint_stage[a]
    eta_b = endpoint_stage[b]
    q = _structure_q(kind[s], eta_a, eta_b, width[s], height[s], invert_a[s], invert_b[s], length[s], manning[s], entrance_loss[s], exit_loss[s], crest[s], weir_length[s], weir_factor[s], one_way[s], max_q[s])
    signed_volume = q * dt
    requested_volume[s] = signed_volume
    instantaneous_q[s] = q
    max_head[s] = wp.max(max_head[s], wp.abs(eta_a - eta_b))
    if signed_volume > WP_FLOAT(0.0):
        wp.atomic_add(donor_requested, a, signed_volume)
    elif signed_volume < WP_FLOAT(0.0):
        wp.atomic_add(donor_requested, b, -signed_volume)


@wp.kernel
def apply_structure_requests(
    depth: wp.array2d(dtype=WP_FLOAT),
    endpoint_cells: wp.array(dtype=wp.int32),
    endpoint_slot_a: wp.array(dtype=wp.int32),
    endpoint_slot_b: wp.array(dtype=wp.int32),
    endpoint_storage: wp.array(dtype=WP_FLOAT),
    donor_requested: wp.array(dtype=WP_FLOAT),
    requested_volume: wp.array(dtype=WP_FLOAT),
    cumulative_positive: wp.array(dtype=WP_FLOAT),
    cumulative_negative: wp.array(dtype=WP_FLOAT),
    net_volume: wp.array(dtype=WP_FLOAT),
    maximum_abs_q: wp.array(dtype=WP_FLOAT),
    instantaneous_q: wp.array(dtype=WP_FLOAT),
    limiter_activations: wp.array(dtype=wp.int32),
    actual_volume: wp.array(dtype=WP_FLOAT),
    nx: wp.int32,
    cell_area: WP_FLOAT,
):
    s = wp.tid()
    request = requested_volume[s]
    if request == WP_FLOAT(0.0):
        actual_volume[s] = WP_FLOAT(0.0)
        return
    donor_slot = endpoint_slot_a[s] if request > WP_FLOAT(0.0) else endpoint_slot_b[s]
    receiver_slot = endpoint_slot_b[s] if request > WP_FLOAT(0.0) else endpoint_slot_a[s]
    total = donor_requested[donor_slot]
    available = endpoint_storage[donor_slot]
    magnitude = wp.min(wp.abs(request), available * wp.abs(request) / wp.max(total, WP_FLOAT(1.0e-30)))
    signed = magnitude if request > WP_FLOAT(0.0) else -magnitude
    donor_flat = endpoint_cells[donor_slot]
    receiver_flat = endpoint_cells[receiver_slot]
    donor_j = donor_flat // nx
    donor_i = donor_flat - donor_j * nx
    receiver_j = receiver_flat // nx
    receiver_i = receiver_flat - receiver_j * nx
    wp.atomic_add(depth, donor_j, donor_i, -magnitude / cell_area)
    wp.atomic_add(depth, receiver_j, receiver_i, magnitude / cell_area)
    actual_volume[s] = signed
    if request > WP_FLOAT(0.0):
        wp.atomic_add(cumulative_positive, s, magnitude)
    else:
        wp.atomic_add(cumulative_negative, s, -magnitude)
    wp.atomic_add(net_volume, s, signed)
    wp.atomic_max(maximum_abs_q, s, wp.abs(instantaneous_q[s]))
    if magnitude + WP_FLOAT(1.0e-12) < wp.abs(request):
        wp.atomic_add(limiter_activations, s, 1)
