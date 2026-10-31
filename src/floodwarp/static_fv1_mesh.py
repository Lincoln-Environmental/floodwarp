"""Independent two-level forest of regular FV1 blocks, fixed during a run.

Only topology construction and geometry/source vetoes live here. Hydraulic
leaves and shared fine subfaces evolve in static_fv1_kernels.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticBlockMesh:
    owner: np.ndarray
    counts: np.ndarray
    mean_bed: np.ndarray
    manning: np.ndarray
    level: np.ndarray
    blocks: np.ndarray
    faces_x: np.ndarray
    faces_y: np.ndarray
    coarse_roots: np.ndarray
    veto_counts: dict


def affine_residual(*, bed):
    row, column = np.indices(dimensions=bed.shape)
    design = np.stack((np.ones_like(row), row, column), axis=-1).reshape(-1, 3)
    coefficients, _, _, _ = np.linalg.lstsq(a=design, b=bed.ravel(), rcond=None)
    return float(np.max(np.abs(design @ coefficients - bed.ravel())))


def root_eligibility(*, state, block_size, protected, surface_tolerance, momentum_tolerance,
                     geometry_tolerance, source_tolerance):
    f = state['arrays']
    config = state['configuration']
    shape = (config['ny'], config['nx'])
    width = 2 * block_size
    roots = np.zeros(shape=((shape[0] + width - 1) // width, (shape[1] + width - 1) // width), dtype=bool)
    q = f['q'].reshape(*shape, 3)
    active = f['active'].reshape(shape).astype(bool)
    bed = f['bed'].reshape(shape)
    if protected.shape != shape:
        raise ValueError('Protected geometry mask must match finest grid')
    vetoes = dict(boundary_or_mask=0, protected=0, geometry=0, mixed_wet_dry=0, source=0,
                  hydraulic=0, unrepresentable_transfer=0)
    # Every actual opening and every physical-domain interface remains fine,
    # including interfaces inside a masked root.
    bx = f['types_x'].reshape(shape[0], shape[1] + 1)
    by = f['types_y'].reshape(shape[0] + 1, shape[1])
    openings = (bx[:, :-1] != 0) | (bx[:, 1:] != 0) | (by[:-1] != 0) | (by[1:] != 0)
    padded = np.pad(array=active, pad_width=1)
    openings |= active & (~padded[1:-1, :-2] | ~padded[1:-1, 2:] |
                          ~padded[:-2, 1:-1] | ~padded[2:, 1:-1])
    fields = ('manning', 'soil', 'rain', 'infil', 'f0', 'fd', 'conductivity', 'multiplier')
    for r, c in np.ndindex(roots.shape):
        j, i = r * width, c * width
        selection = (slice(j, j + width), slice(i, i + width))
        if j == 0 or i == 0 or j + width >= shape[0] or i + width >= shape[1] or not active[selection].all() or openings[selection].any():
            vetoes['boundary_or_mask'] += 1
            continue
        if protected[selection].any():
            vetoes['protected'] += 1
            continue
        z = bed[selection]
        h = q[selection][..., 0]
        all_dry = np.all(h == 0.)
        all_wet = np.all(h > config['dry_depth'])
        if not all_dry and not all_wet:
            vetoes['mixed_wet_dry'] += 1
            continue
        # An affine patch has no interior sill, barrier, channel or saddle.
        # Dry sloping patches cannot use one wet/dry state when ponding starts.
        if affine_residual(bed=z) > geometry_tolerance or (all_dry and np.ptp(z) > geometry_tolerance):
            vetoes['geometry'] += 1
            continue
        if any(np.ptp(f[name].reshape(shape)[selection]) > source_tolerance for name in fields):
            vetoes['source'] += 1
            continue
        eta_error = float(np.ptp(h + z)) if all_wet else 0.
        momentum_error = float(np.max(np.ptp(q[selection][..., 1:], axis=(0, 1))))
        if eta_error > surface_tolerance or momentum_error > momentum_tolerance:
            vetoes['hydraulic'] += 1
            continue
        if all_wet:
            parent_h = h.reshape(block_size, 2, block_size, 2).mean(axis=(1, 3))
            parent_z = z.reshape(block_size, 2, block_size, 2).mean(axis=(1, 3))
            projected = parent_h.repeat(2, axis=0).repeat(2, axis=1) + (
                parent_z.repeat(2, axis=0).repeat(2, axis=1) - z)
            parent_momentum = q[selection][..., 1:].reshape(block_size, 2, block_size, 2, 2).mean(axis=(1, 3))
            bearing = np.any(parent_momentum != 0., axis=2).repeat(2, axis=0).repeat(2, axis=1)
            if np.any(projected <= 0.) or np.any((projected <= config['dry_depth']) & bearing):
                vetoes['unrepresentable_transfer'] += 1
                continue
        roots[r, c] = True
    return roots, vetoes


def interface_faces(*, owner):
    padded_x = np.pad(array=owner, pad_width=((0, 0), (1, 1)), constant_values=-1)
    padded_y = np.pad(array=owner, pad_width=((1, 1), (0, 0)), constant_values=-1)
    left, right = padded_x[:, :-1], padded_x[:, 1:]
    south, north = padded_y[:-1], padded_y[1:]
    x = np.flatnonzero(((left != right) & ((left >= 0) | (right >= 0))).ravel()).astype(np.int32)
    y = np.flatnonzero(((south != north) & ((south >= 0) | (north >= 0))).ravel()).astype(np.int32)
    return x, y


def build_mesh(*, state, block_size, protected, surface_tolerance, momentum_tolerance,
               geometry_tolerance, source_tolerance, requested_roots=None):
    if block_size < 2 or int(block_size) != block_size:
        raise ValueError('Static block_size must be an integer >= 2')
    tolerances = (surface_tolerance, momentum_tolerance, geometry_tolerance, source_tolerance)
    if not np.isfinite(tolerances).all() or min(tolerances) < 0.:
        raise ValueError('Finite nonnegative static-mesh tolerances required')
    coarse, vetoes = root_eligibility(state=state, block_size=block_size, protected=protected,
        surface_tolerance=surface_tolerance, momentum_tolerance=momentum_tolerance,
        geometry_tolerance=geometry_tolerance, source_tolerance=source_tolerance)
    if requested_roots is not None:
        if requested_roots.shape != coarse.shape:
            raise ValueError('Requested root mask shape differs')
        if np.any(requested_roots & ~coarse):
            raise ValueError('Requested coarsening violates hydraulic, geometry or source veto')
        coarse &= requested_roots
    f = state['arrays']
    config = state['configuration']
    shape = (config['ny'], config['nx'])
    active = f['active'].reshape(shape).astype(bool)
    owner = np.full(shape=shape, fill_value=-1, dtype=np.int32)
    counts, levels, blocks = [], [], []
    root_width = 2 * block_size
    for r, c in np.ndindex(coarse.shape):
        root_row, root_column = r * root_width, c * root_width
        step = 2 if coarse[r, c] else 1
        block_width = block_size * step
        for dj in range(0, root_width, block_width):
            for di in range(0, root_width, block_width):
                row, column = root_row + dj, root_column + di
                if row >= shape[0] or column >= shape[1]:
                    continue
                blocks.append((row, column, step // 2, len(counts)))
                for j in range(row, min(row + block_width, shape[0]), step):
                    for i in range(column, min(column + block_width, shape[1]), step):
                        if not active[j, i]:
                            continue
                        owner[j:j + step, i:i + step] = len(counts)
                        counts.append(step * step)
                        levels.append(step // 2)
    valid = owner.ravel() >= 0
    ids = owner.ravel()[valid]
    counts = np.asarray(counts, dtype=np.int32)
    mean_bed = np.bincount(ids, weights=f['bed'][valid], minlength=counts.size) / counts
    manning = np.bincount(ids, weights=f['manning'][valid], minlength=counts.size) / counts
    faces_x, faces_y = interface_faces(owner=owner)
    return StaticBlockMesh(owner=owner, counts=counts, mean_bed=mean_bed, manning=manning,
        level=np.asarray(levels, dtype=np.int32), blocks=np.asarray(blocks, dtype=np.int32),
        faces_x=faces_x, faces_y=faces_y, coarse_roots=coarse, veto_counts=vetoes)


def restrict_state(*, q, mesh):
    valid = mesh.owner.ravel() >= 0
    ids = mesh.owner.ravel()[valid]
    return np.stack([np.bincount(ids, weights=q[valid, component], minlength=mesh.counts.size) /
                     mesh.counts for component in range(3)], axis=-1)


def prolong_state(*, leaves, mesh, bed, dry_depth):
    """Conservative common-surface/common-velocity transfer where representable."""
    result = np.zeros(shape=(mesh.owner.size, 3))
    valid = mesh.owner.ravel() >= 0
    ids = mesh.owner.ravel()[valid]
    state = leaves[ids].copy()
    coarse = mesh.counts[ids] > 1
    wet = coarse & (state[:, 0] > 0.)
    depth = state[:, 0] + (mesh.mean_bed[ids] - bed[valid])
    if np.any(wet & (depth <= 0.)):
        raise ValueError('Coarse state has an unrepresentable mixed wet/dry configuration')
    scale = np.divide(depth, state[:, 0], out=np.ones_like(depth), where=wet)
    state[wet, 1:] *= scale[wet, None]
    state[wet, 0] = depth[wet]
    if np.any((state[:, 0] <= dry_depth) & np.any(state[:, 1:] != 0., axis=1)):
        raise ValueError('Coarse momentum cannot be represented across the fine dry threshold')
    result[valid] = state
    return result
