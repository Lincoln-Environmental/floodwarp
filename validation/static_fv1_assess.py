"""Geometry/source-gated coarsenability and allocation bounds from saved states.

Completed-event information is diagnostic only. These masks never enter an
execution or refinement decision in production.
"""

from pathlib import Path
import json

import numpy as np

from floodwarp.static_fv1_mesh import root_eligibility
from validation.sparse_fv1_benchmark import select_checkpoints, digest


def child_values(*, values):
    ny, nx = values.shape
    return values[:ny // 2 * 2, :nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2).transpose(0, 2, 1, 3).reshape(ny // 2, nx // 2, 4)


def active_cell_candidates(*, arrays, shape, surface_tolerance, momentum_tolerance, geometry_tolerance, source_tolerance):
    q = arrays['q'].reshape(*shape, 3)
    bed = child_values(values=arrays['bed'].reshape(shape))
    h = child_values(values=q[..., 0])
    physical = child_values(values=arrays['active'].reshape(shape)).all(axis=2)
    wet = physical & np.all(h > 0., axis=2)
    eta = h + bed
    hydraulic = wet & (np.ptp(eta, axis=2) <= surface_tolerance)
    for component in (1, 2):
        hydraulic &= np.ptp(child_values(values=q[..., component]), axis=2) <= momentum_tolerance
    affine = np.abs(bed[..., 0] + bed[..., 3] - bed[..., 1] - bed[..., 2]) <= geometry_tolerance
    homogeneous = np.ones_like(wet)
    for name in ('manning', 'soil', 'rain', 'infil', 'f0', 'fd', 'conductivity', 'multiplier'):
        homogeneous &= np.ptp(child_values(values=arrays[name].reshape(shape)), axis=2) <= source_tolerance
    return dict(fully_wet_physical_parents=int(wet.sum()),
        hydraulic_indicator_parents=int(hydraulic.sum()), geometry_admissible_parents=int(np.count_nonzero(hydraulic & affine)),
        geometry_and_source_admissible_parents=int(np.count_nonzero(hydraulic & affine & homogeneous)))


def assess(*, paths, block_sizes, surface_tolerance, momentum_tolerance, geometry_tolerance,
           source_tolerance, occupancy_path, benchmark_path, output):
    rows = []
    for path in paths:
        with np.load(file=path, allow_pickle=False) as saved:
            state = json.loads(str(saved['metadata']))
            state['arrays'] = {name: saved[name] for name in
                ('q', 'bed', 'active', 'manning', 'soil', 'rain', 'infil', 'f0', 'fd', 'conductivity', 'multiplier', 'types_x', 'types_y')}
        shape = (state['configuration']['ny'], state['configuration']['nx'])
        protected = np.zeros(shape=shape, dtype=bool)
        blocks = []
        for block_size in block_sizes:
            roots, vetoes = root_eligibility(state=state, block_size=block_size, protected=protected,
                surface_tolerance=surface_tolerance, momentum_tolerance=momentum_tolerance,
                geometry_tolerance=geometry_tolerance, source_tolerance=source_tolerance)
            blocks.append(dict(block_size=block_size, eligible_roots=int(roots.sum()),
                               total_roots=int(roots.size), veto_counts=vetoes))
        rows.append(dict(time_s=state['time'], checkpoint_sha256=digest(path=path), blocks=blocks,
            active_region_2x2=active_cell_candidates(arrays=state['arrays'], shape=shape,
                surface_tolerance=surface_tolerance, momentum_tolerance=momentum_tolerance,
                geometry_tolerance=geometry_tolerance, source_tolerance=source_tolerance)))
    occupancy = json.loads(occupancy_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    memory = benchmark['phases'][0]['configurations'][0]['memory_bytes']
    pool_bounds = []
    for size in (8, 16, 32):
        peak_cells = max(row['tiles'][str(size)]['evaluated_cells'] for row in occupancy['states'])
        peak_faces = max(row['tiles'][str(size)]['evaluated_faces_x'] + row['tiles'][str(size)]['evaluated_faces_y'] for row in occupancy['states'])
        pool_q = peak_cells * 24
        pool_scratch = peak_faces * 40 + peak_cells * 8
        savings = memory['conservative'] + memory['face_scratch'] - pool_q - pool_scratch
        pool_bounds.append(dict(tile_size=size, sampled_peak_live_cells=peak_cells,
            ideal_state_pool_bytes=pool_q, ideal_face_and_limiter_pool_bytes=pool_scratch,
            ideal_total_savings_bytes=savings, ideal_total_savings_fraction=savings / memory['total'],
            scope='Ideal sampled high-water capacity with no growth slack, maps, padding or allocation overhead; '
                  'a diagnostic upper bound, not a predictive pool size or measured allocation.'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(driver_sha256=digest(path=Path(__file__)),
        surface_tolerance_m=surface_tolerance, momentum_tolerance_m2_s=momentum_tolerance,
        affine_geometry_tolerance_m=geometry_tolerance, source_history_tolerance=source_tolerance,
        memory_bytes=memory, ideal_persistent_pool_bounds=pool_bounds, phases=rows), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 13, 25, 48, 60, 72)
    block_sizes = (4, 8, 16)
    surface_tolerance = .001
    momentum_tolerance = .001
    geometry_tolerance = 1.e-10
    source_tolerance = 0.
    occupancy_path = Path('validation/sparse_review/occupancy.json')
    benchmark_path = Path('validation/sparse_review/benchmark.json')
    output = Path('validation/static_review/coarsenability.json')
    paths = select_checkpoints(directory=directory, hours=hours)
    assess(paths=paths, block_sizes=block_sizes, surface_tolerance=surface_tolerance,
           momentum_tolerance=momentum_tolerance, geometry_tolerance=geometry_tolerance,
           source_tolerance=source_tolerance, occupancy_path=occupancy_path, benchmark_path=benchmark_path, output=output)
