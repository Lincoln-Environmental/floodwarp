"""Matched saved-state FV1 execution timings; profiling is a separate run.

All adjustable run settings are in the execution guard. Checkpoints and old
event results are read-only. No external solver or prepared model is imported.
"""

from pathlib import Path
from time import perf_counter
import gc
import hashlib
import json

import numpy as np
import warp as wp

from floodwarp.full_swe_solver import FullSWESolver
from validation.sparse_fv1_audit import checkpoint_time, dependency_tiles, tile_mask
from validation.gpu_benchmark import reset_device_allocation_peak


def digest(*, path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_solver(*, path, device, mode, tile_size, dense_fraction):
    with np.load(file=path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        config = metadata['configuration']
        shape = (config['ny'], config['nx'])
        solver = FullSWESolver(**config, bed_elevation=saved['bed'].reshape(shape),
            manning_n=saved['manning'].reshape(shape), active_mask=saved['active'].reshape(shape), device=device)
    solver.configure_execution(mode=mode, tile_size=tile_size, dense_fraction=dense_fraction)
    solver.load_checkpoint(path=path)
    return solver


def memory_layers(*, solver):
    f = solver._fields
    groups = dict(conservative=('q',), geometry=('bed', 'active', 'manning', 'types_x', 'types_y', 'stages_x', 'stages_y'),
        hydrology=('soil', 'rain', 'infil', 'f0', 'fd', 'conductivity', 'multiplier'),
        face_scratch=('flux_x', 'flux_y', 'pressure_x', 'pressure_y', 'theta'),
        diagnostics=('maximum', 'rain_sum', 'inflow', 'outflow', 'volume_x', 'volume_y', 'dt_histogram'))
    result = {name: sum(f[field].capacity for field in fields) for name, fields in groups.items()}
    result['control_forcing'] = sum(array.capacity for array in
        (solver.ctrl, solver.condition, solver._rain_times, solver._rain_rates))
    result['scheduling'] = solver.execution_diagnostics()['scheduling_bytes']
    result['total'] = sum(result.values())
    return result


def cfl_diagnostics(*, solver, before, post_source, ctrl, work_cells):
    """Vectorised diagnostic of the retained endpoint source/stage CFL policy."""
    f = {name: array.numpy() for name, array in solver._fields.items()}
    active = f['active'].astype(bool)
    h = before[:, 0]
    u = np.divide(before[:, 1], h, out=np.zeros_like(h), where=h > solver.dry_depth)
    v = np.divide(before[:, 2], h, out=np.zeros_like(h), where=h > solver.dry_depth)
    wave = np.sqrt(solver.gravity * (h + ctrl[8] * (f['rain'] + f['multiplier'] * ctrl[12])))
    ax, ay = np.abs(u) + wave, np.abs(v) + wave
    indices = np.arange(h.size)
    j, i = indices // solver.nx, indices % solver.nx
    for axis, speeds, base, stride in (('x', ax, j * (solver.nx + 1) + i, 1), ('y', ay, indices, solver.nx)):
        for offset in range(2):
            faces = base + offset * stride
            stage = active & (f['types_' + axis][faces] == 3)
            ghost_h = np.maximum(0., f['stages_' + axis][faces] - f['bed'])
            normal = before[:, 1 if axis == 'x' else 2]
            denominator = h if solver._stage_momentum_mode == 1 else ghost_h
            ghost_u = np.divide(np.abs(normal), denominator, out=np.zeros_like(h),
                where=(ghost_h > solver.dry_depth) & (denominator > solver.dry_depth))
            speeds[stage] = np.maximum(speeds[stage], (ghost_u + np.sqrt(solver.gravity * ghost_h))[stage])
    rate = ax / solver.dx + ay / solver.dy
    candidates = np.divide(solver.cfl, rate, out=np.full_like(h, np.inf), where=active & (rate > 0.))
    minimum = float(candidates.min())
    controlling = active & (candidates == minimum)
    before_wet = active & (before[:, 0] != 0.)
    after_wet = active & (post_source[:, 0] != 0.)
    return dict(accepted_dt_s=float(ctrl[2]), cfl_candidate_s=minimum if np.isfinite(minimum) else None,
        maximum_dt_controls=bool(minimum >= ctrl[8]), candidate_minimum_cells=int(controlling.sum()),
        candidate_minimum_outside_work=int(np.count_nonzero(controlling & ~work_cells)),
        source_activated_cells=int(np.count_nonzero(after_wet & ~before_wet)),
        source_deactivated_cells=int(np.count_nonzero(before_wet & ~after_wet)))


def inspect_step(*, solver, tile_size):
    initial = solver.capture_state()
    before = initial['arrays']['q']
    ctrl = np.zeros_like(initial['control'])
    ctrl[0:3] = [solver.time, solver.time + solver.maximum_dt, 0.]
    ctrl[3:9] = [solver.steps, np.inf, 0., 0., 0., solver.maximum_dt]
    ctrl[13:15] = [solver.steps + 1, solver.maximum_dt]
    solver.ctrl.assign(ctrl)
    solver._enqueue_sources()
    post_source = solver.q.numpy().copy()
    ctrl = solver.ctrl.numpy()
    active = initial['arrays']['active'].astype(bool)
    shape = (solver.ny, solver.nx)
    seeds = active & np.any(post_source != 0., axis=1)
    f = initial['arrays']
    k = np.arange(seeds.size)
    j, i = k // solver.nx, k % solver.nx
    for offset in range(2):
        x = j * (solver.nx + 1) + i + offset
        y = k + offset * solver.nx
        seeds |= active & (((f['types_x'][x] == 3) & (f['stages_x'][x] > f['bed'])) |
                           ((f['types_y'][y] == 3) & (f['stages_y'][y] > f['bed'])))
    before_tiles = tile_mask(cells=(active & np.any(before != 0., axis=1)).reshape(shape), tile_size=tile_size)
    after_tiles = tile_mask(cells=seeds.reshape(shape), tile_size=tile_size)
    closure = dependency_tiles(seeds=after_tiles)
    work_cells = closure.repeat(tile_size, axis=0).repeat(tile_size, axis=1)[:solver.ny, :solver.nx].ravel()
    cfl = cfl_diagnostics(solver=solver, before=before, post_source=post_source, ctrl=ctrl, work_cells=work_cells)
    solver._enqueue_hydraulics()
    execution = solver.execution_diagnostics()
    after = solver.q.numpy().copy()
    final_tiles = tile_mask(cells=(active & np.any(after != 0., axis=1)).reshape(shape), tile_size=tile_size)
    cfl.update(source_changed_seed_tiles=int(np.count_nonzero(before_tiles != after_tiles)),
               hydraulic_changed_seed_tiles=int(np.count_nonzero(after_tiles != final_tiles)),
               front_new_nonzero_cells=int(np.count_nonzero((post_source[:, 0] == 0.) & (after[:, 0] != 0.))))
    solver.restore_state(state=initial)
    return dict(cfl=cfl, execution=execution)


def timed_run(*, solver, duration, graph):
    began = perf_counter()
    result = solver.run_device_loop(duration=duration, use_cuda_graph=graph)
    wp.synchronize_device(device=solver.device)
    result['end_to_end_seconds'] = perf_counter() - began
    result['seconds_per_step'] = result['loop_seconds'] / result['steps']
    return result


def compare_fields(*, solver, reference):
    differences = {name: float(np.max(np.abs(array.numpy() - reference['arrays'][name])))
                   for name, array in solver._fields.items()}
    return dict(maximum_absolute_by_field=differences,
                identical_steps=solver.steps == reference['steps'], identical_time=solver.time == reference['time'])


def benchmark(*, paths, configurations, device, duration, repeats, graph, output):
    wp.init()
    report = dict(device=str(wp.get_device(device)), duration_s=duration, repeats=repeats, graph=graph,
        python_scope='Pinned Python 3.14 / FP64',
        sources={path.name: digest(path=path) for path in Path('src/floodwarp').glob('*fv1*.py')}, phases=[])
    report['sources'].update({path.name: digest(path=path) for path in
        (Path('src/floodwarp/full_swe_solver.py'), Path('src/floodwarp/full_swe_kernels.py'), Path(__file__))})
    for path in paths:
        phase = dict(checkpoint=str(path), checkpoint_sha256=digest(path=path), configurations=[])
        reference = None
        solver = None
        for mode, tile_size, dense_fraction in configurations:
            solver = None
            gc.collect()
            baseline = reset_device_allocation_peak(device=device)
            solver = load_solver(path=path, device=device, mode=mode, tile_size=tile_size, dense_fraction=dense_fraction)
            phase['time_s'] = solver.time
            inspection = inspect_step(solver=solver, tile_size=tile_size)
            initial = solver.capture_state()
            timed_run(solver=solver, duration=duration, graph=graph)
            runs = []
            for repeat in range(repeats):
                solver.restore_state(state=initial)
                runs.append(timed_run(solver=solver, duration=duration, graph=graph))
            if reference is None:
                reference = solver.capture_state()
            row = dict(mode=mode, tile_size=tile_size, dense_fraction=dense_fraction,
                inspection=inspection, runs=runs, memory_bytes=memory_layers(solver=solver),
                peak_live_device_bytes=int(wp.get_mempool_used_mem_high(device=device) - baseline),
                differences=compare_fields(solver=solver, reference=reference), accounting=solver.cumulative_accounting(),
                final_execution=solver.execution_diagnostics(),
                median_hydraulic_seconds=float(np.median([run['loop_seconds'] for run in runs])),
                median_end_to_end_seconds=float(np.median([run['end_to_end_seconds'] for run in runs])))
            phase['configurations'].append(row)
            print(path.name, mode, tile_size, row['median_hydraulic_seconds'], flush=True)
        report['phases'].append(phase)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def select_checkpoints(*, directory, hours):
    paths = {checkpoint_time(path=path): path for path in directory.glob('checkpoint-*.npz')}
    return [paths[hour * 3600.] for hour in hours]


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 13, 25, 48, 60, 72)
    configurations = (('dense', 16, .65), ('sparse', 8, .65), ('sparse', 16, .65), ('sparse', 32, .65))
    device = 'cuda:0'
    duration = 30.
    repeats = 3
    graph = True
    output = Path('validation/sparse_review/benchmark.json')
    paths = select_checkpoints(directory=directory, hours=hours)
    benchmark(paths=paths, configurations=configurations, device=device, duration=duration,
              repeats=repeats, graph=graph, output=output)
