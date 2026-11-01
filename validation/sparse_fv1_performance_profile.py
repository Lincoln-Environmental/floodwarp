"""Separate CUDA-event attribution for released and candidate sparse execution."""

from pathlib import Path
import json

import warp as wp

from validation.sparse_fv1_benchmark import digest, select_checkpoints
from validation.sparse_fv1_performance import prepare_solver


def profile(*, paths, variants, device, tile_size, dense_fraction,
            forced_dense_fraction, duration, output):
    rows = []
    for path in paths:
        for variant in variants:
            solver = prepare_solver(path=path, variant=variant, device=device,
                tile_size=tile_size, dense_fraction=dense_fraction,
                forced_dense_fraction=forced_dense_fraction)
            state = solver.capture_state()
            solver.run_device_loop(duration=duration, use_cuda_graph=False)
            solver.restore_state(state=state)
            with wp.ScopedTimer(name='Development sparse kernel attribution',
                    cuda_filter=wp.TIMING_ALL, print=False) as timer:
                stats = solver.run_device_loop(duration=duration, use_cuda_graph=False)
            kernels = {}
            for event in timer.timing_results:
                row = kernels.setdefault(event.name, dict(calls=0, milliseconds=0.))
                row['calls'] += 1
                row['milliseconds'] += event.elapsed
            for row in kernels.values():
                row['microseconds_per_step'] = row['milliseconds'] * 1000. / stats['steps']
            rows.append(dict(time_s=state['time'], variant=variant, steps=stats['steps'], kernels=kernels))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Instrumented host launches, separate from graph throughput timing. '
        'Dependency halo classification is fused into compaction; source/activity costs are combined in the fused kernel.',
        driver_sha256=digest(path=Path(__file__)), rows=rows), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 25, 60)
    variants = ('released', 'combined', 'forced_dense')
    device = 'cuda:0'
    tile_size = 8
    dense_fraction = .65
    forced_dense_fraction = .01
    duration = 2.
    output = Path('validation/sparse_perf_review/profile.json')
    paths = select_checkpoints(directory=directory, hours=hours)
    profile(paths=paths, variants=variants, device=device, tile_size=tile_size,
        dense_fraction=dense_fraction, forced_dense_fraction=forced_dense_fraction,
        duration=duration, output=output)
