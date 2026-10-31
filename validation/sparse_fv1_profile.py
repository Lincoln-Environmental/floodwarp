"""Separate, instrumented GPU kernel attribution; not a throughput benchmark."""

from pathlib import Path
import json

import warp as wp

from validation.sparse_fv1_benchmark import load_solver, select_checkpoints, digest


def profile(*, paths, device, configurations, duration, output):
    wp.init()
    rows = []
    for path in paths:
        for mode, tile_size, dense_fraction in configurations:
            solver = load_solver(path=path, device=device, mode=mode, tile_size=tile_size, dense_fraction=dense_fraction)
            initial = solver.capture_state()
            solver.run_device_loop(duration=duration, use_cuda_graph=False)
            solver.restore_state(state=initial)
            with wp.ScopedTimer(name='FV1 attributed kernels', cuda_filter=wp.TIMING_ALL, print=False) as timer:
                stats = solver.run_device_loop(duration=duration, use_cuda_graph=False)
            totals = {}
            for result in timer.timing_results:
                entry = totals.setdefault(result.name, dict(calls=0, milliseconds=0.))
                entry['calls'] += 1
                entry['milliseconds'] += result.elapsed
            for entry in totals.values():
                entry['microseconds_per_step'] = entry['milliseconds'] * 1000. / stats['steps']
            rows.append(dict(checkpoint=str(path), time_s=initial['time'], mode=mode, tile_size=tile_size,
                             steps=stats['steps'], activities=totals))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Instrumented CUDA events with host launches; '
        'separate from graph throughput measurements. Tile halo handling is fused into compact_tiles.',
        driver_sha256=digest(path=Path(__file__)), rows=rows), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 13, 25, 48, 60)
    device = 'cuda:0'
    configurations = (('dense', 8, .65), ('sparse', 8, .65))
    duration = 2.
    output = Path('validation/sparse_review/profile.json')
    paths = select_checkpoints(directory=directory, hours=hours)
    profile(paths=paths, device=device, configurations=configurations, duration=duration, output=output)
