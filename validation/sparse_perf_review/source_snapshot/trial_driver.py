"""Interleaved development controls against frozen FloodWarp v0.1.1 scheduling.

No numerical field or accepted timestep is approximated. Profiling is separate
from these graph timings. Original checkpoints and reports are read-only.
"""

from pathlib import Path
import gc
import json

import numpy as np
import warp as wp

from floodwarp.sparse_fv1 import SparseFV1Execution
from validation.sparse_fv1_benchmark import (
    compare_fields, digest, load_solver, memory_layers, select_checkpoints, timed_run,
)
from validation.sparse_perf_review.source_snapshot.reference_execution import SparseFV1Execution as ReleasedExecution
from validation.sparse_perf_review.source_snapshot import released_sparse_fv1_kernels


def prepare_solver(*, path, variant, device, tile_size, dense_fraction):
    solver = load_solver(path=path, device=device, mode='dense',
        tile_size=tile_size, dense_fraction=dense_fraction)
    if variant == 'dense':
        return solver
    if variant == 'released':
        solver._sparse_execution = ReleasedExecution(solver=solver,
            tile_size=tile_size, dense_fraction=dense_fraction)
    else:
        fraction = .01 if variant == 'forced_dense' else dense_fraction
        engine = SparseFV1Execution(solver=solver, tile_size=tile_size, dense_fraction=fraction)
        engine._clear_released_only = variant in ('clear_only', 'combined', 'forced_dense')
        engine._fuse_activity = variant in ('fused_only', 'combined', 'forced_dense')
        solver._sparse_execution = engine
    return solver


def benchmark(*, paths, variants, device, tile_size, dense_fraction, duration, repeats, output):
    wp.load_module(module=released_sparse_fv1_kernels, device=device)
    report = dict(device=str(wp.get_device(device)), duration_s=duration,
        repeats=repeats, warmup='One full continuation per variant, then exact restore',
        order='Rotate variant order between measured repetitions',
        scope='Saved-state continuations, not a new complete-event speedup claim',
        sources={name: digest(path=Path('src/floodwarp') / name) for name in
            ('full_swe_solver.py', 'full_swe_kernels.py', 'sparse_fv1.py', 'sparse_fv1_kernels.py')},
        reference_loader_sha256=digest(path=Path('validation/sparse_perf_review/source_snapshot/reference_execution.py')),
        driver_sha256=digest(path=Path(__file__)), phases=[])
    for path in paths:
        solvers = {variant: prepare_solver(path=path, variant=variant, device=device,
            tile_size=tile_size, dense_fraction=dense_fraction) for variant in variants}
        initial = solvers['dense'].capture_state()
        rows = {variant: dict(runs=[]) for variant in variants}
        for solver in solvers.values():
            solver.restore_state(state=initial)
            timed_run(solver=solver, duration=duration, graph=True)
        reference = None
        for repeat in range(repeats):
            order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
            for variant in order:
                solver = solvers[variant]
                solver.restore_state(state=initial)
                stats = timed_run(solver=solver, duration=duration, graph=True)
                if reference is None:
                    if variant != 'dense':
                        raise ValueError('First measurement must establish the dense control')
                    reference = solver.capture_state()
                differences = compare_fields(solver=solver, reference=reference)
                if not differences['identical_steps'] or not differences['identical_time']:
                    raise ArithmeticError('Accepted time/steps differ: ' + variant)
                if max(differences['maximum_absolute_by_field'].values()) != 0.:
                    raise ArithmeticError('Hydraulic fields differ: ' + variant)
                rows[variant]['runs'].append(dict(stats=stats, differences=differences))
        for variant, solver in solvers.items():
            row = rows[variant]
            row['median_hydraulic_seconds'] = float(np.median([run['stats']['loop_seconds'] for run in row['runs']]))
            row['median_end_to_end_seconds'] = float(np.median([run['stats']['end_to_end_seconds'] for run in row['runs']]))
            row['memory_bytes'] = memory_layers(solver=solver)
            row['execution'] = solver.execution_diagnostics()
            row['accounting'] = solver.cumulative_accounting()
        phase = dict(time_s=initial['time'], checkpoint_sha256=digest(path=path), configurations=rows,
            concurrent_control_pool_peak_bytes=int(wp.get_mempool_used_mem_high(device=device)),
            memory_scope='Persistent arrays are per solver; pool high-water includes concurrent controls, not an isolated candidate peak')
        report['phases'].append(phase)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(initial['time'] / 3600., {name: round(row['median_hydraulic_seconds'], 6) for name, row in rows.items()}, flush=True)
        solvers.clear()
        solver = None
        initial = None
        reference = None
        gc.collect()


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 13, 25, 48, 60, 72)
    variants = ('dense', 'released', 'clear_only', 'fused_only', 'combined', 'forced_dense')
    device = 'cuda:0'
    tile_size = 8
    dense_fraction = .65
    duration = 30.
    repeats = 3
    output = Path('validation/sparse_perf_review/benchmark.json')
    paths = select_checkpoints(directory=directory, hours=hours)
    benchmark(paths=paths, variants=variants, device=device, tile_size=tile_size,
        dense_fraction=dense_fraction, duration=duration, repeats=repeats, output=output)
