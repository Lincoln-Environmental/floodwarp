"""Native D4 connectivity and passive four-port reductions used in preprocessing.

Migrated without mathematical changes from the Phase 8/9 reference. No native
network solve is used during a hydraulic timestep.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import solve

def raster_edges(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Undirected cardinal adjacency, using the existing D4 topology only."""
    ids = np.arange(np.prod(shape)).reshape(shape)
    return (
        np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel())),
        np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel())),
    )



def wet_components(bed: np.ndarray, stage_m: float) -> np.ndarray:
    from floodwarp.enhanced_preprocessing import wet_labels
    return wet_labels(np.asarray(bed, dtype=np.float64), float(stage_m))


def _wet_components_reference(bed: np.ndarray, stage_m: float) -> np.ndarray:
    """Label *all* components on raw hydraulic elevations, without filling."""
    z = np.asarray(bed, dtype=np.float64)
    wet = z.ravel() < stage_m
    parent = np.arange(z.size)
    u, v = raster_edges(shape=z.shape)
    for a, b in zip(u[wet[u] & wet[v]], v[wet[u] & wet[v]], strict=True):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        parent[b] = a
    for a in np.flatnonzero(wet):
        b = a
        while parent[b] != b:
            b = parent[b]
        parent[a] = b
    result = np.full(z.size, -1, dtype=np.int32)
    result[wet] = np.unique(parent[wet], return_inverse=True)[1]
    return result.reshape(z.shape)



def boundary_nodes(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """West, east, south, north native boundary segment node IDs."""
    ids = np.arange(np.prod(shape)).reshape(shape)
    return ids[:, 0], ids[:, -1], ids[0, :], ids[-1, :]


@lru_cache(maxsize=128)
def _raster_edges_cached(shape):
    values = raster_edges(shape=shape)
    for value in values:
        value.setflags(write=False)
    return values


@lru_cache(maxsize=128)
def _boundary_nodes_cached(shape):
    values = boundary_nodes(shape=shape)
    for value in values:
        value.setflags(write=False)
    return values



@dataclass(frozen=True)
class PortGroup:
    side: int
    component: int
    segments: np.ndarray



@dataclass(frozen=True)
class PortReduction:
    groups: tuple[PortGroup, ...]
    component_labels: np.ndarray
    stage_m: float
    conductance: np.ndarray | None  # Optional auxiliary DtN; momentum q is outward.
    flux_basis: np.ndarray
    inertance: np.ndarray
    edge_inertance: np.ndarray
    edge_resistance: np.ndarray
    edge_sill_m: np.ndarray
    edge_width_m: np.ndarray
    edge_length_m: np.ndarray
    edge_depth_m: np.ndarray
    contact_flux_basis: np.ndarray
    native_divergence: np.ndarray
    storage_weights: np.ndarray

    def damping(self, discharge_m3s: np.ndarray) -> np.ndarray:
        """PSD secant of cubic Manning dissipation, in units s/m2."""
        native_flow = self.flux_basis @ discharge_m3s
        return self.flux_basis.T @ (
            (self.edge_resistance * np.abs(native_flow))[:, None] * self.flux_basis
        )

    def dissipation(self, discharge_m3s: np.ndarray) -> float:
        """Sum R|Q|^3 [m4/s]; multiply by rho*g for power [W]."""
        return float(np.sum(self.edge_resistance * np.abs(self.flux_basis @ discharge_m3s) ** 3))



def boundary_groups(labels: np.ndarray) -> tuple[PortGroup, ...]:
    """Keep disjoint runs separate even on the same side and component."""
    groups = []
    for side, nodes in enumerate(boundary_nodes(shape=labels.shape)):
        values = labels.ravel()[nodes]
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[stop] == values[start]:
                stop += 1
            if values[start] >= 0:
                groups.append(PortGroup(side=side, component=int(values[start]), segments=np.arange(start, stop)))
            start = stop
    return tuple(groups)



def reduce_cell(
    bed_m: np.ndarray, stage_m: float, resolution_m: float, manning_n: float,
    port_groups: tuple[PortGroup, ...] | None = None,
    contact_profiles: tuple[np.ndarray, ...] | None = None,
    contact_sills_m: tuple[np.ndarray, ...] | None = None,
    momentum_source_weights: np.ndarray | None = None,
    contact_nodes: tuple[np.ndarray, ...] | None = None,
    contact_manning_n: tuple[np.ndarray, ...] | None = None,
    compute_conductance: bool = True,
    _wet_labels: np.ndarray | None = None,
) -> PortReduction:
    """Full-cell DtN operator and component-wise source-compatible flux lift.

    Production coefficient tables use only the transient flux lift. Set
    compute_conductance=False to omit the independent steady DtN solve; the
    returned conductance is then None. All transient arrays are unchanged.

    Contact edges extend native centres to the coarse boundary by delta/2.
    C=w*h^(5/3)/(n*l) defines only the auxiliary linear operator. Its numerical
    units are m^2/s; Q=C*phi is an auxiliary potential relation, NOT Manning
    flow under a physical head. Physical inertia and resistance use I=l/(w*h)
    [1/m], R=n^2*l/(w^2*h^(10/3)) [s^2/m5]. No Manhattan length multiplier is
    applied: the discrete network's energy determines the resistance/inertia.
    """
    bed = np.asarray(bed_m, dtype=np.float64)
    if bed.ndim != 2 or not np.all(np.isfinite(bed)) or not np.isfinite(stage_m):
        raise ValueError('finite 2D bed and finite stage required')
    roughness = np.asarray(manning_n, dtype=np.float64)
    spatial_roughness = roughness.ndim != 0
    if spatial_roughness and roughness.shape != bed.shape:
        raise ValueError('native Manning shape must match the reduction patch')
    if not np.isfinite(resolution_m) or resolution_m <= 0 or not np.all(np.isfinite(roughness)) or np.any(roughness <= 0):
        raise ValueError('positive finite resolution and Manning required')
    labels = wet_components(bed=bed, stage_m=stage_m) if _wet_labels is None else _wet_labels
    if labels.shape != bed.shape or not np.array_equal(labels >= 0, bed < stage_m):
        raise ValueError('Precomputed wet labels must match this patch and stage')
    groups = boundary_groups(labels=labels) if port_groups is None else port_groups
    if contact_profiles is not None and len(contact_profiles) != len(groups):
        raise ValueError("one contact profile required per port group")
    if contact_sills_m is not None and len(contact_sills_m) != len(groups):
        raise ValueError("one contact sill array required per port group")
    if contact_nodes is not None and len(contact_nodes) != len(groups):
        raise ValueError("one native node array required per port")
    if contact_manning_n is not None and len(contact_manning_n) != len(groups):
        raise ValueError('one native roughness array required per contact group')
    used_segments = set()
    for group_index, group in enumerate(groups):
        if group.side not in range(4) or np.asarray(group.segments).ndim != 1 or not len(group.segments):
            raise ValueError('nonempty valid boundary group required')
        boundary = (_boundary_nodes_cached(shape=bed.shape)[group.side] if contact_nodes is None
                    else np.asarray(contact_nodes[group_index]))
        if (boundary.ndim != 1 or not np.issubdtype(boundary.dtype, np.integer)
                or np.any(boundary < 0) or np.any(boundary >= bed.size)):
            raise ValueError("valid native contact node indices required")
        segments = np.asarray(group.segments)
        if not np.issubdtype(segments.dtype,np.integer) or np.any(segments < 0) or np.any(segments >= boundary.size):
            raise ValueError('integer boundary segment indices required')
        if np.any(labels.ravel()[boundary[segments]] != group.component) or group.component < 0:
            raise ValueError('boundary group must belong to one native wet component')
        for segment in segments:
            key = (group.side,int(boundary[segment]))
            if key in used_segments:
                raise ValueError('boundary segments must not be duplicated')
            used_segments.add(key)
    wet_ids = np.flatnonzero(labels.ravel() >= 0)
    count = wet_ids.size
    port_count = len(groups)
    mapping = np.full(bed.size, -1, dtype=int)
    mapping[wet_ids] = np.arange(count)
    u, v = _raster_edges_cached(shape=bed.shape)
    keep = (mapping[u] >= 0) & (mapping[v] >= 0)
    edge_n = np.maximum(roughness.ravel()[u[keep]],roughness.ravel()[v[keep]]) if spatial_roughness else manning_n
    sill = np.maximum(bed.ravel()[u[keep]], bed.ravel()[v[keep]])
    u, v = mapping[u[keep]], mapping[v[keep]]
    lengths = np.full(u.size, resolution_m)
    internal_count = u.size
    contact_basis_parts = []
    ports = _boundary_nodes_cached(shape=bed.shape)
    for p, group in enumerate(groups):
        ids = (ports[group.side][group.segments] if contact_nodes is None
               else np.asarray(contact_nodes[p])[group.segments])
        u = np.concatenate((u, mapping[ids]))
        v = np.concatenate((v, np.full(ids.size, count + p)))
        contact_sill = bed.ravel()[ids] if contact_sills_m is None else np.asarray(contact_sills_m[p])
        if (contact_sill.shape != ids.shape or not np.all(np.isfinite(contact_sill))
                or np.any(contact_sill < bed.ravel()[ids]) or np.any(contact_sill >= stage_m)):
            raise ValueError('wet finite contact sills at or above native bed required')
        sill = np.concatenate((sill, contact_sill))
        if spatial_roughness:
            contact_n=roughness.ravel()[ids] if contact_manning_n is None else np.asarray(contact_manning_n[p])
            if contact_n.shape!=ids.shape or np.any(contact_n<=0) or not np.all(np.isfinite(contact_n)):
                raise ValueError('positive finite native roughness required at every wet contact')
            edge_n=np.concatenate((edge_n,contact_n))
        lengths = np.concatenate((lengths, np.full(ids.size, resolution_m / 2)))
        profile = (stage_m-contact_sill)**(5/3) if contact_profiles is None else np.asarray(contact_profiles[p])
        if spatial_roughness and contact_profiles is None:
            profile=profile/contact_n
        if profile.shape != ids.shape or not np.all(np.isfinite(profile)) or np.any(profile <= 0):
            raise ValueError('positive finite contact weights matching each port required')
        block = np.zeros((ids.size,port_count))
        block[:,p] = profile/profile.sum()
        contact_basis_parts.append(block)
    widths = np.full(u.size, resolution_m)
    depths = stage_m - sill
    incidence = np.zeros((count + port_count, u.size))
    incidence[u, np.arange(u.size)] = 1
    incidence[v, np.arange(u.size)] = -1
    c = widths * depths ** (5 / 3) / (edge_n * lengths)
    edge_i = lengths / (widths * depths)
    edge_r = edge_n ** 2 * lengths / (widths ** 2 * depths ** (10 / 3))
    laplacian = (incidence * c) @ incidence.T if compute_conductance else None
    conductance = np.zeros((port_count, port_count)) if compute_conductance else None
    basis = np.zeros((u.size, port_count))
    weights = np.zeros((count, port_count))
    contact_basis = np.concatenate(contact_basis_parts,axis=0) if contact_basis_parts else np.zeros((0,port_count))
    basis[internal_count:] = contact_basis
    # DtN uses Dirichlet electrodes. The TRANSIENT lift instead fixes contact
    # flux profiles: a zero-net port must not permit a hidden boundary shortcut.
    # Never invert across components. Closed pools have no port operator.
    for component in np.unique(labels[labels >= 0]):
        native = np.flatnonzero(labels.ravel()[wet_ids] == component)
        port_ids = np.array([p for p, group in enumerate(groups) if group.component == component], dtype=int)
        if not port_ids.size:
            continue
        if compute_conductance:
            terminals = count + port_ids
            lii = laplacian[np.ix_(native, native)]
            lib = laplacian[np.ix_(native, terminals)]
            conductance[np.ix_(port_ids, port_ids)] = (
                laplacian[np.ix_(terminals, terminals)] - lib.T @ solve(lii, lib, assume_a='pos')
            )
        if momentum_source_weights is None:
            weights[np.ix_(native, port_ids)] = 1 / native.size
        else:
            source = np.asarray(momentum_source_weights, dtype=float)
            if source.shape != bed.shape or not np.all(np.isfinite(source)) or np.any(source < 0):
                raise ValueError('finite nonnegative momentum weights matching the patch required')
            local = source.ravel()[wet_ids[native]]
            if local.sum() <= 0:
                raise ValueError('positive total momentum weight per active component required')
            weights[np.ix_(native, port_ids)] = (local/local.sum())[:, None]
        native_incidence = incidence[:count,:internal_count]
        contact_incidence = incidence[:count,internal_count:]
        rhs = weights[np.ix_(native,port_ids)] - contact_incidence[native]@contact_basis[:,port_ids]
        native_laplacian = (native_incidence*c[:internal_count])@native_incidence.T
        potential = np.zeros((count,port_ids.size))
        if native.size > 1:
            ground = native[-1] if momentum_source_weights is None else native[np.argmax(local)]
            free = native[native != ground]
            relative_free = np.flatnonzero(native != ground)
            potential[free] = solve(native_laplacian[np.ix_(free,free)],rhs[relative_free],assume_a='pos')
        basis[:internal_count,port_ids] = c[:internal_count,None]*(native_incidence.T@potential)
    inertia = basis.T @ (edge_i[:, None] * basis)
    return PortReduction(
        groups=groups, component_labels=labels, stage_m=stage_m,
        conductance=conductance, flux_basis=basis, inertance=inertia,
        edge_inertance=edge_i, edge_resistance=edge_r, edge_sill_m=sill,
        edge_width_m=widths, edge_length_m=lengths, edge_depth_m=depths,
        contact_flux_basis=contact_basis, native_divergence=incidence[:count] @ basis, storage_weights=weights,
    )



def port_coefficients(reduction, formulation: str) -> tuple[np.ndarray, np.ndarray]:
    """Return port inertia [1/m] and Manning resistance [s²/m⁵].

    Edge quadrature majorizes both the kinetic quadratic and cubic dissipation.
    Matrix lumping uses a tighter quadratic majorant, with the same positive
    Manning quadrature. Neither is row-sum lumping of outward port momenta.
    """
    magnitude = np.abs(reduction.flux_basis)
    total = magnitude.sum(axis=1)
    if formulation == 'edge_lumped':
        inertia = magnitude.T @ (reduction.edge_inertance * total)
    elif formulation == 'matrix_lumped':
        inertia = np.abs(reduction.inertance).sum(axis=1)
    else:
        raise ValueError('unknown local momentum formulation')
    resistance = magnitude.T @ (reduction.edge_resistance * total**2)
    return inertia, resistance



def transport_star(*, reduction) -> tuple[np.ndarray, dict]:
    """Return completed arm inertia [1/m] and explicit approximation diagnostics."""
    inertia,_=port_coefficients(reduction=reduction,formulation='matrix_lumped')
    count=len(inertia)
    reason='not_two_or_three_ports'
    if count not in (2,3) or np.all(reduction.component_labels>=0):
        if np.all(reduction.component_labels>=0):
            reason='fully_wet_cell'
        return inertia,dict(applied=False,reason=reason)
    mass=reduction.inertance
    if count==2:
        arms=np.diag(mass)-mass[0,1]
    else:
        pair=np.diag(mass)[:,None]+np.diag(mass)[None,:]-2*mass
        arms=np.array([(pair[0,1]+pair[0,2]-pair[1,2])/2,
                       (pair[0,1]+pair[1,2]-pair[0,2])/2,
                       (pair[0,2]+pair[1,2]-pair[0,1])/2])
    contacts=reduction.contact_flux_basis
    contact_count=contacts.shape[0]
    contact_inertia=(contacts**2).T@reduction.edge_inertance[-contact_count:]
    completed=np.maximum(arms,contact_inertia)
    if np.any(completed<=0) or not np.all(np.isfinite(completed)):
        raise ValueError('transport-star requires positive finite boundary contact inertia')
    return completed,dict(applied=True,exact_arms=arms.tolist(),contact_lower_bound=contact_inertia.tolist(),
        completed_arms=completed.tolist(),negative_arms=int(np.count_nonzero(arms<0)))
