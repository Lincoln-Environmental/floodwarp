"""Small-patch native flux lift, with the same grounded SPD equations.

No fastmath, geometry approximation, changed source weights or coefficient
closure. Ambiguous FP32 rounding and unsafe factorizations use the retained
NumPy/SciPy path. All matrices are local to a native hydraulic patch.
"""

import numpy as np
from numba import njit

from floodwarp.enhanced_preprocessing import wet_labels


def prepare_component(*, patch, owner_mask, entries, contacts, port_sills,
                      manning_n, contact_roughness, resolution, factor, formulation, inertia_weighting="uniform"):
    counts = np.array([len(entry[2]) for entry in entries], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    nodes = np.concatenate([entry[2] for entry in entries]).astype(np.int64)
    sills = np.concatenate([contacts[entry[0]] for entry in entries]).astype(np.float64)
    spatial = np.ndim(manning_n) != 0
    roughness = np.broadcast_to(np.asarray(manning_n, dtype=np.float64), patch.shape).copy()
    contact_n = (np.concatenate([contact_roughness[entry[0]] for entry in entries]).astype(np.float64)
        if spatial else np.full(len(nodes), float(manning_n)))
    # Enforce ownership here too: direct callers need not manufacture the
    # finite builder's inactive-bed sentinel before preparing a local system.
    native_patch = np.ascontiguousarray(patch).copy()
    native_patch[~owner_mask] = np.inf
    native = (native_patch, np.ascontiguousarray(owner_mask), roughness,
        offsets, nodes, sills, contact_n, np.asarray(port_sills, dtype=np.float64),
        float(resolution), bool(spatial), formulation == "transport_star", inertia_weighting == "energy_normalized")
    # Backend choice follows the owned numerical system, rather than the
    # enclosing hydraulic patch. Port-rich systems use bounded edge batches.
    from floodwarp.enhanced_native_sparse import DENSE_SYSTEM_NODES, RHS_BATCH_PORTS
    if patch.size <= 64 or (np.count_nonzero(owner_mask) <= DENSE_SYSTEM_NODES and len(entries) <= RHS_BATCH_PORTS):
        return native
    from floodwarp.enhanced_native_sparse import prepare_edges
    return prepare_edges(native=native)


@njit(cache=True, nogil=True)
def native_values(patch, owner_mask, roughness, offsets, nodes, sills, contact_n,
                  port_sills, resolution, spatial, star, energy_normalized, stage, reciprocal, reference, tail):
    count_ports = len(port_sills)
    values = np.zeros((count_ports, 2))
    statistics = np.zeros(5, np.int64)
    bed = patch.copy()
    if tail:
        for j in range(bed.shape[0]):
            for i in range(bed.shape[1]):
                bed[j, i] = (patch[j, i] - reference) * reciprocal if owner_mask[j, i] else 2.
    labels = wet_labels(bed, stage).ravel()
    wet_ids = np.flatnonzero(labels >= 0)
    count = len(wet_ids)
    mapping = np.full(bed.size, -1, np.int64)
    for i in range(count):
        mapping[wet_ids[i]] = i
    z = bed.ravel()
    n = roughness.ravel()
    active = np.zeros(count_ports, np.bool_)
    port_component = np.full(count_ports, -1, np.int32)
    capacities = np.zeros(count_ports)
    normalized_depth = np.zeros(count_ports)
    max_edges = 2 * count + len(nodes)
    u, v = np.empty(max_edges, np.int64), np.empty(max_edges, np.int64)
    c, inertia, resistance = np.empty(max_edges), np.empty(max_edges), np.empty(max_edges)
    basis = np.zeros((max_edges, count_ports))
    edge_count = 0
    ny, nx = bed.shape
    for direction in range(2):
        for j in range(ny if direction == 0 else ny - 1):
            for i in range(nx - 1 if direction == 0 else nx):
                a = j * nx + i
                b = a + (1 if direction == 0 else nx)
                if mapping[a] >= 0 and mapping[b] >= 0:
                    depth = stage - max(z[a], z[b])
                    edge_n = max(n[a], n[b]) if spatial else n[0]
                    u[edge_count], v[edge_count] = mapping[a], mapping[b]
                    c[edge_count] = resolution * depth**(5. / 3.) / (edge_n * resolution)
                    inertia[edge_count] = resolution / (resolution * depth)
                    resistance[edge_count] = edge_n**2 * resolution / (resolution**2 * depth**(10. / 3.))
                    edge_count += 1
    internal = edge_count
    active_count = 0
    for port in range(count_ports):
        sill = (port_sills[port] - reference) * reciprocal if tail else port_sills[port]
        if stage <= sill:
            continue
        active[port] = True
        active_count += 1
        normalized_depth[port] = stage - sill
        total = 0.
        begin = edge_count
        for index in range(offsets[port], offsets[port + 1]):
            sill = (sills[index] - reference) * reciprocal if tail else sills[index]
            if sill >= stage:
                continue
            node = nodes[index]
            component = labels[node]
            if component < 0 or (port_component[port] >= 0 and component != port_component[port]):
                return values, 2, statistics
            port_component[port] = component
            depth = stage - sill
            edge_n = contact_n[index]
            profile = depth**(5. / 3.)
            if spatial:
                profile /= edge_n
            total += profile
            u[edge_count], v[edge_count] = mapping[node], count + port
            c[edge_count] = resolution * depth**(5. / 3.) / (edge_n * (resolution / 2.))
            inertia[edge_count] = (resolution / 2.) / (resolution * depth)
            resistance[edge_count] = edge_n**2 * (resolution / 2.) / (resolution**2 * depth**(10. / 3.))
            basis[edge_count, port] = profile
            edge_count += 1
        if total <= 0:
            return values, 2, statistics
        capacities[port] = resolution * total if spatial else resolution * total / n[0]
        for edge in range(begin, edge_count):
            basis[edge, port] /= total
    laplacian = np.zeros((count, count))
    for edge in range(internal):
        a, b, weight = u[edge], v[edge], c[edge]
        laplacian[a, a] += weight
        laplacian[b, b] += weight
        laplacian[a, b] -= weight
        laplacian[b, a] -= weight
    source = np.empty(count)
    for i in range(count):
        source[i] = (stage - z[wet_ids[i]])**(5. / 3.)
    for component in range(int(labels.max()) + 1):
        native = np.flatnonzero(labels[wet_ids] == component)
        port_ids = np.flatnonzero(active & (port_component == component))
        if not len(port_ids):
            continue
        total_source = 0.
        ground = native[0]
        for node in native:
            total_source += source[node]
            if source[node] > source[ground]:
                ground = node
        free = native[native != ground]
        dimension = len(free)
        statistics[0] += 1
        statistics[1] = max(statistics[1], len(native))
        if dimension:
            statistics[2] += 1
            statistics[3] += len(port_ids)
            statistics[4] += len(port_ids)
        lower = np.zeros((dimension, dimension))
        for j in range(dimension):
            pivot = laplacian[free[j], free[j]]
            for k in range(j):
                pivot -= lower[j, k] * lower[j, k]
            if pivot <= 0. or not np.isfinite(pivot):
                return values, 2, statistics
            lower[j, j] = np.sqrt(pivot)
            for i in range(j + 1, dimension):
                entry = laplacian[free[i], free[j]]
                for k in range(j):
                    entry -= lower[i, k] * lower[j, k]
                lower[i, j] = entry / lower[j, j]
        for port in port_ids:
            rhs = source / total_source
            for edge in range(internal, edge_count):
                rhs[u[edge]] -= basis[edge, port]
            forward, potential = np.zeros(dimension), np.zeros(count)
            for j in range(dimension):
                entry = rhs[free[j]]
                for k in range(j):
                    entry -= lower[j, k] * forward[k]
                forward[j] = entry / lower[j, j]
            for j in range(dimension - 1, -1, -1):
                entry = forward[j]
                for k in range(j + 1, dimension):
                    entry -= lower[k, j] * potential[free[k]]
                potential[free[j]] = entry / lower[j, j]
            for edge in range(internal):
                basis[edge, port] = c[edge] * (potential[u[edge]] - potential[v[edge]])
    mass = np.zeros((count_ports, count_ports))
    contact_lower = np.zeros(count_ports)
    for edge in range(edge_count):
        for a in range(count_ports):
            if edge >= internal:
                contact_lower[a] += basis[edge, a]**2 * inertia[edge]
            for b in range(count_ports):
                mass[a, b] += basis[edge, a] * (inertia[edge] * basis[edge, b])
    diagonal = np.zeros(count_ports)
    for port in range(count_ports):
        for other in range(count_ports):
            diagonal[port] += abs(mass[port, other])
    if star and (active_count == 2 or active_count == 3) and np.any(labels < 0):
        ids = np.flatnonzero(active)
        if active_count == 2:
            for port in ids:
                diagonal[port] = max(mass[port, port] - mass[ids[0], ids[1]], contact_lower[port])
        else:
            pair = np.empty((3, 3))
            for a in range(3):
                for b in range(3):
                    pair[a, b] = mass[ids[a], ids[a]] + mass[ids[b], ids[b]] - 2. * mass[ids[a], ids[b]]
            diagonal[ids[0]] = max((pair[0, 1] + pair[0, 2] - pair[1, 2]) / 2., contact_lower[ids[0]])
            diagonal[ids[1]] = max((pair[0, 1] + pair[1, 2] - pair[0, 2]) / 2., contact_lower[ids[1]])
            diagonal[ids[2]] = max((pair[0, 2] + pair[1, 2] - pair[0, 1]) / 2., contact_lower[ids[2]])
    star_applied = star and (active_count == 2 or active_count == 3) and np.any(labels < 0)
    if energy_normalized and not star_applied:
        for port in range(count_ports):
            if active[port]:
                diagonal[port] = 0.
                for other in range(count_ports):
                    if active[other]:
                        diagonal[port] += abs(mass[port, other]) / np.sqrt(mass[other, other])
                diagonal[port] *= np.sqrt(mass[port, port])
    damping = np.zeros(count_ports)
    for edge in range(edge_count):
        weighted = 0.
        for port in range(count_ports):
            weighted += abs(basis[edge, port]) * capacities[port]
        for port in range(count_ports):
            damping[port] += abs(basis[edge, port]) * (resistance[edge] * weighted**2)
    ambiguous = False
    for port in range(count_ports):
        if active[port]:
            values[port, 0] = diagonal[port] * normalized_depth[port]
            values[port, 1] = damping[port] / capacities[port]**2 * normalized_depth[port]**(10. / 3.)
            for column in range(2):
                value = values[port, column]
                if value <= 0. or not np.isfinite(value):
                    return values, 2, statistics
                if (count > 1 or column == 1) and abs(float(np.float32(value)) - value) <= 2.e-12 * value:
                    ambiguous = True
    return values, 1 if ambiguous else 0, statistics


def evaluate_component(*, prepared, stage, reciprocal=1., reference=0., tail=False, diagnostics=None):
    from floodwarp.enhanced_native_sparse import evaluate_edges, record_backend
    diagnostics = {} if diagnostics is None else diagnostics
    record_backend(diagnostics=diagnostics, name='stage_evaluations')
    if isinstance(prepared, dict):
        try:
            return evaluate_edges(prepared=prepared, stage=float(stage), reciprocal=reciprocal,
                reference=reference, tail=tail, diagnostics=diagnostics)
        except (np.linalg.LinAlgError, RuntimeError) as error:
            record_backend(diagnostics=diagnostics, name='fallbacks')
            record_backend(diagnostics=diagnostics.setdefault('fallback_reasons', {}),
                name=type(error).__name__ + ': ' + str(error))
            return None
    values, status, statistics = native_values(*prepared, float(stage), float(reciprocal), float(reference), tail)
    for name, count in zip(('systems', 'maximum_system_nodes', 'dense_cholesky', 'solve_batches', 'port_rhs'), statistics):
        if name.startswith('maximum_'):
            diagnostics[name] = max(diagnostics.get(name, 0), int(count))
        else:
            record_backend(diagnostics=diagnostics, name=name, count=int(count))
    record_backend(diagnostics=diagnostics, name='factorizations', count=int(statistics[2]))
    if status:
        record_backend(diagnostics=diagnostics, name='fallbacks')
        record_backend(diagnostics=diagnostics.setdefault('fallback_reasons', {}),
            name='FP32_rounding_boundary' if status == 1 else 'unsafe_dense_reduction')
        return None
    return {index: values[index] for index in range(len(values)) if values[index, 0] > 0.}


def warm_native_kernel():
    """Compile one signature before spawning workers; no hydraulic state used."""
    patch = np.zeros((1, 1))
    native_values(patch, np.ones((1, 1), dtype=np.bool_), np.full((1, 1), .04),
        np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64), np.zeros(1),
        np.full(1, .04), np.zeros(1), 1., False, True, False, 1., 1., 0., False)
