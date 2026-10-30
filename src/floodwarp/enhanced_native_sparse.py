"""Edge-based native momentum reduction, independent of the dense oracle.

Only preprocessing uses this module. Systems are individual D4 wet components;
contacts impose fluxes and never become connections between native nodes.
"""

from contextlib import contextmanager
from tempfile import TemporaryFile

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from floodwarp.enhanced_preprocessing import wet_labels
from floodwarp.enhanced_reduction import raster_edges


# Internal numerical policy, included in the content-bound geometry cache key.
DENSE_SYSTEM_NODES = 128
RHS_BATCH_PORTS = 16
WORKSPACE_BYTES = 8 * 1024 * 1024


def record_backend(*, diagnostics, name, count=1):
    diagnostics[name] = diagnostics.get(name, 0) + count


def merge_diagnostics(*, destination, source):
    for name, value in source.items():
        if name.startswith("maximum_"):
            destination[name] = max(destination.get(name, 0), value)
        elif isinstance(value, dict):
            merge_diagnostics(destination=destination.setdefault(name, {}), source=value)
        else:
            destination[name] = destination.get(name, 0) + value


def prepare_edges(*, native):
    patch, owner, roughness, offsets, nodes, sills, contact_n, port_sills, resolution, spatial, star, energy_normalized = native
    u, v = raster_edges(shape=patch.shape)
    valid = owner.ravel()[u] & owner.ravel()[v]
    u, v = u[valid], v[valid]
    return dict(native=native, u=u, v=v,
        edge_sill=np.maximum(patch.ravel()[u], patch.ravel()[v]),
        edge_n=np.maximum(roughness.ravel()[u], roughness.ravel()[v]),
        activation=np.unique(patch[owner]), topology_key=None, systems=None)


def wet_systems(*, prepared, stage, reciprocal, reference, tail, diagnostics):
    patch, owner = prepared['native'][:2]
    # Within each open activation interval the indexing is identical. Keep only
    # the current interval, bounding memory even for a continuously sloping DEM.
    key = len(prepared['activation']) if tail else int(np.searchsorted(prepared['activation'], stage, side='left'))
    if prepared['topology_key'] == key:
        record_backend(diagnostics=diagnostics, name='topology_reuses')
        return prepared['systems']
    bed = np.where(owner, (patch - reference) * reciprocal if tail else patch, np.inf)
    labels = wet_labels(bed, float(stage)).ravel()
    systems = []
    mapping = np.full(patch.size, -1, dtype=np.int64)
    prepared["node_system"] = labels
    u, v = prepared['u'], prepared['v']
    for component in np.unique(labels[labels >= 0]):
        ids = np.flatnonzero(labels == component)
        edge_ids = np.flatnonzero((labels[u] == component) & (labels[v] == component))
        mapping[ids] = np.arange(len(ids))
        systems.append((ids, edge_ids, mapping[u[edge_ids]], mapping[v[edge_ids]], mapping))
    prepared['topology_key'], prepared['systems'] = key, systems
    record_backend(diagnostics=diagnostics, name='topology_builds')
    return systems


@njit(cache=True, nogil=True)
def dense_factor(count, u, v, conductance, free):
    """Direct Laplacian assembly and one grounded Cholesky, strict FP64."""
    laplacian = np.zeros((count, count))
    for edge in range(len(u)):
        a, b, weight = u[edge], v[edge], conductance[edge]
        laplacian[a, a] += weight
        laplacian[b, b] += weight
        laplacian[a, b] -= weight
        laplacian[b, a] -= weight
    dimension = len(free)
    lower = np.zeros((dimension, dimension))
    for j in range(dimension):
        pivot = laplacian[free[j], free[j]]
        for k in range(j):
            pivot -= lower[j, k] * lower[j, k]
        if pivot <= 0. or not np.isfinite(pivot):
            raise np.linalg.LinAlgError('Nonpositive grounded Laplacian pivot')
        lower[j, j] = np.sqrt(pivot)
        for i in range(j + 1, dimension):
            entry = laplacian[free[i], free[j]]
            for k in range(j):
                entry -= lower[i, k] * lower[j, k]
            lower[i, j] = entry / lower[j, j]
    return lower


@njit(cache=True, nogil=True)
def dense_solve(lower, rhs):
    """Reuse the factor for a bounded block of port right-hand sides."""
    result = rhs.copy()
    for port in range(rhs.shape[1]):
        for j in range(len(lower)):
            for k in range(j):
                result[j, port] -= lower[j, k] * result[k, port]
            result[j, port] /= lower[j, j]
        for j in range(len(lower) - 1, -1, -1):
            for k in range(j + 1, len(lower)):
                result[j, port] -= lower[k, j] * result[k, port]
            result[j, port] /= lower[j, j]
    return result


def warm_edge_kernels():
    """Compile the two numerical signatures outside timed construction."""
    factor = dense_factor(count=2, u=np.array([0], dtype=np.int64),
        v=np.array([1], dtype=np.int64), conductance=np.ones(1), free=np.array([1], dtype=np.int64))
    dense_solve(lower=factor, rhs=np.ones((1, 1)))


def factor_system(*, count, u, v, conductance, free, diagnostics):
    diagnostics['maximum_system_nodes'] = max(diagnostics.get('maximum_system_nodes', 0), count)
    record_backend(diagnostics=diagnostics, name='systems')
    if not len(free):
        record_backend(diagnostics=diagnostics, name='singleton_systems')
        return None
    record_backend(diagnostics=diagnostics, name='factorizations')
    if count <= DENSE_SYSTEM_NODES:
        record_backend(diagnostics=diagnostics, name='dense_cholesky')
        return dense_factor(count=count, u=u, v=v, conductance=conductance, free=free)
    record_backend(diagnostics=diagnostics, name='sparse_lu')
    # O(nodes + edges) assembly; no incidence or dense node-by-node matrix.
    diagonal = np.bincount(np.r_[u, v], weights=np.r_[conductance, conductance], minlength=count)
    indices = np.arange(count)
    matrix = coo_matrix((np.r_[diagonal, -conductance, -conductance],
        (np.r_[indices, u, v], np.r_[indices, v, u])), shape=(count, count)).tocsc()
    grounded = matrix[free][:, free].tocsc()
    return splu(grounded, permc_spec='COLAMD', options={'SymmetricMode': True})


@contextmanager
def basis_workspace(*, edges, ports, diagnostics):
    shape = (edges, ports)
    if edges * ports * np.dtype(np.float64).itemsize <= WORKSPACE_BYTES:
        yield np.zeros(shape, dtype=np.float64)
    else:
        record_backend(diagnostics=diagnostics, name='disk_backed_basis')
        with TemporaryFile() as stream:
            stream.truncate(edges * ports * np.dtype(np.float64).itemsize)
            values = np.memmap(stream, mode='r+', dtype=np.float64, shape=shape)
            try:
                yield values
            finally:
                values._mmap.close()


def system_coefficients(*, ids, u, v, edge_ids, mapping, prepared, stage,
                        reciprocal, reference, tail, port_ids, contacts,
                        capacities, active_count, partially_wet, diagnostics):
    native = prepared['native']
    patch, owner, roughness, offsets, nodes, sills, contact_n, port_sills, resolution, spatial, star, energy_normalized = native
    z = (patch.ravel()[ids] - reference) * reciprocal if tail else patch.ravel()[ids]
    edge_sill = prepared['edge_sill'][edge_ids]
    if tail:
        edge_sill = (edge_sill - reference) * reciprocal
    depth = stage - edge_sill
    edge_n = prepared['edge_n'][edge_ids]
    conductance = resolution * depth**(5./3.) / (edge_n * resolution)
    inertia = resolution / (resolution * depth)
    resistance = edge_n**2 * resolution / (resolution**2 * depth**(10./3.))
    internal = len(u)
    contact_ids = np.concatenate([contacts[p] for p in port_ids])
    contact_ports = np.repeat(np.arange(len(port_ids)), [len(contacts[p]) for p in port_ids])
    contact_nodes = mapping[nodes[contact_ids]]
    contact_sill = sills[contact_ids]
    if tail:
        contact_sill = (contact_sill - reference) * reciprocal
    contact_depth = stage - contact_sill
    profile = contact_depth**(5./3.)
    if spatial:
        profile /= contact_n[contact_ids]
    totals = np.bincount(contact_ports, weights=profile, minlength=len(port_ids))
    profile /= totals[contact_ports]
    inertia = np.r_[inertia, (resolution/2.)/(resolution * contact_depth)]
    resistance = np.r_[resistance, contact_n[contact_ids]**2*(resolution/2.) /
        (resolution**2 * contact_depth**(10./3.))]
    source = (stage - z)**(5./3.)
    ground = int(np.argmax(source))
    free = np.flatnonzero(np.arange(len(ids)) != ground)
    source /= source.sum()
    factor = factor_system(count=len(ids), u=u, v=v, conductance=conductance, free=free, diagnostics=diagnostics)
    count_ports, count_edges = len(port_ids), len(inertia)
    # Bounded port blocks; extremely port-rich systems spill their basis to a
    # temporary mapping rather than allocating an enormous edge x port array.
    batch = max(1, min(RHS_BATCH_PORTS, WORKSPACE_BYTES // (8 * max(count_edges, len(ids), 1))))
    diagnostics['maximum_rhs_ports'] = max(diagnostics.get('maximum_rhs_ports', 0), count_ports)
    diagnostics['maximum_system_edges'] = max(diagnostics.get('maximum_system_edges', 0), count_edges)
    diagonal, damping = np.zeros(count_ports), np.zeros(count_ports)
    contact_lower = np.bincount(contact_ports, weights=profile**2 * inertia[internal:], minlength=count_ports)
    with basis_workspace(edges=count_edges, ports=count_ports, diagnostics=diagnostics) as basis:
        for start in range(0, count_ports, batch):
            stop = min(start + batch, count_ports)
            rhs = np.repeat(source[:, None], stop - start, axis=1)
            selected = (contact_ports >= start) & (contact_ports < stop)
            np.add.at(rhs, (contact_nodes[selected], contact_ports[selected] - start), -profile[selected])
            potential = np.zeros((len(ids), stop - start))
            if len(free):
                potential[free] = (dense_solve(lower=factor, rhs=rhs[free]) if isinstance(factor, np.ndarray)
                    else factor.solve(rhs[free]))
                record_backend(diagnostics=diagnostics, name='solve_batches')
                record_backend(diagnostics=diagnostics, name='port_rhs', count=stop-start)
            basis[:internal, start:stop] = conductance[:, None] * (potential[u] - potential[v])
            basis[internal + np.flatnonzero(selected), contact_ports[selected]] = profile[selected]
        weighted = np.zeros(count_edges)
        scales = np.ones(count_ports)
        if energy_normalized and not (star and active_count in (2, 3) and partially_wet):
            for start in range(0, count_ports, batch):
                stop = min(start + batch, count_ports)
                scales[start:stop] = np.sqrt(np.sum(inertia[:,None] * basis[:,start:stop]**2, axis=0))
        small_mass = np.zeros((count_ports, count_ports)) if active_count in (2, 3) else None
        for start in range(0, count_ports, batch):
            stop = min(start + batch, count_ports)
            left = np.asarray(basis[:, start:stop])
            weighted += np.abs(left) @ capacities[port_ids[start:stop]]
            for other in range(0, count_ports, batch):
                end = min(other + batch, count_ports)
                mass = left.T @ (inertia[:, None] * basis[:, other:end])
                diagonal[start:stop] += scales[start:stop] * (np.abs(mass)/scales[other:end]).sum(axis=1)
                if small_mass is not None:
                    small_mass[start:stop, other:end] = mass
        if star and active_count in (2, 3) and partially_wet:
            # Cross-component entries are zero. The global star completion is
            # applied by the caller after collecting these tiny mass blocks.
            star_mass = small_mass
        else:
            star_mass = None
        for start in range(0, count_ports, batch):
            stop = min(start + batch, count_ports)
            damping[start:stop] = np.abs(basis[:, start:stop]).T @ (resistance * weighted**2)
    return diagonal, damping, contact_lower, star_mass


def evaluate_edges(*, prepared, stage, reciprocal, reference, tail, diagnostics):
    native = prepared['native']
    patch, owner, roughness, offsets, nodes, sills, contact_n, port_sills, resolution, spatial, star, energy_normalized = native
    systems = wet_systems(prepared=prepared, stage=stage, reciprocal=reciprocal,
        reference=reference, tail=tail, diagnostics=diagnostics)
    scaled_sills = (sills-reference)*reciprocal if tail else sills
    scaled_ports = (port_sills-reference)*reciprocal if tail else port_sills
    active = np.flatnonzero(stage > scaled_ports)
    contacts = {p: np.arange(offsets[p], offsets[p+1])[scaled_sills[offsets[p]:offsets[p+1]] < stage] for p in active}
    capacities = np.zeros(len(port_sills))
    system_ports = [[] for _ in systems]
    for port in active:
        selected = contacts[port]
        if not len(selected):
            raise ValueError('Active port has no wet contact')
        found = []
        for index, system in enumerate(systems):
            belongs = prepared['node_system'][nodes[selected]] == prepared['node_system'][system[0][0]]
            if np.any(belongs):
                if not np.all(belongs):
                    raise ValueError('Port spans disconnected wet systems')
                found.append(index)
        if len(found) != 1:
            raise ValueError('Port is not owned by one wet system')
        system_ports[found[0]].append(port)
        h = stage-scaled_sills[selected]
        capacities[port] = (resolution*np.sum(h**(5./3.)/contact_n[selected]) if spatial else
            resolution*np.sum(h**(5./3.))/roughness.flat[0])
    diagonal, damping, contact_lower = (np.zeros(len(port_sills)) for _ in range(3))
    mass = np.zeros((len(active), len(active))) if len(active) in (2, 3) else None
    partially_wet = sum(len(s[0]) for s in systems) != patch.size
    for system, ports in zip(systems, system_ports, strict=True):
        if not ports:
            continue
        ids, edge_ids, u, v, mapping = system
        port_ids = np.asarray(ports, dtype=np.int64)
        d, r, lower, block = system_coefficients(ids=ids, u=u, v=v, edge_ids=edge_ids,
            mapping=mapping, prepared=prepared, stage=stage, reciprocal=reciprocal,
            reference=reference, tail=tail, port_ids=port_ids, contacts=contacts,
            capacities=capacities, active_count=len(active), partially_wet=partially_wet, diagnostics=diagnostics)
        diagonal[port_ids], damping[port_ids], contact_lower[port_ids] = d, r, lower
        if block is not None:
            local = np.searchsorted(active, port_ids)
            mass[np.ix_(local, local)] = block
    if star and len(active) in (2, 3) and partially_wet:
        if len(active) == 2:
            arms = np.diag(mass) - mass[0, 1]
        else:
            pair = np.diag(mass)[:, None] + np.diag(mass)[None, :] - 2.*mass
            arms = np.array([(pair[0,1]+pair[0,2]-pair[1,2])/2.,
                (pair[0,1]+pair[1,2]-pair[0,2])/2., (pair[0,2]+pair[1,2]-pair[0,1])/2.])
        diagonal[active] = np.maximum(arms, contact_lower[active])
    result = {}
    for port in active:
        depth = stage-scaled_ports[port]
        value = np.array([diagonal[port]*depth, damping[port]/capacities[port]**2*depth**(10./3.)])
        if np.any(value <= 0.) or not np.all(np.isfinite(value)):
            raise np.linalg.LinAlgError('Nonpositive or nonfinite reduced coefficients')
        result[int(port)] = value
    return result
