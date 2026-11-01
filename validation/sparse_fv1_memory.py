"""Separate live-allocation probe for independently measured sparse variants."""

from pathlib import Path
import gc
import json

import warp as wp

from validation.gpu_benchmark import reset_device_allocation_peak
from validation.sparse_fv1_benchmark import digest, memory_layers, select_checkpoints
from validation.sparse_fv1_performance import prepare_solver


def measure(*, path, variants, device, tile_size, dense_fraction,
            forced_dense_fraction, duration, output):
    rows = []
    for variant in variants:
        baseline = reset_device_allocation_peak(device=device)
        solver = prepare_solver(path=path, variant=variant, device=device,
            tile_size=tile_size, dense_fraction=dense_fraction,
            forced_dense_fraction=forced_dense_fraction)
        solver.run_device_loop(duration=duration)
        wp.synchronize_device(device=device)
        rows.append(dict(variant=variant, persistent_bytes=memory_layers(solver=solver),
            baseline_pool_bytes=baseline,
            peak_additional_live_pool_bytes=int(wp.get_mempool_used_mem_high(device=device) - baseline)))
        solver = None
        gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Separate memory probe with one solver at a time. '
        'Live allocator high-water excludes CUDA context, code and reserved pool memory. No runtime claim.',
        checkpoint_sha256=digest(path=path), rows=rows), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (25,)
    variants = ('dense', 'released', 'clear_only', 'fused_only', 'combined', 'forced_dense')
    device = 'cuda:0'
    tile_size = 8
    dense_fraction = .65
    forced_dense_fraction = .01
    duration = 1.
    output = Path('validation/sparse_perf_review/memory.json')
    path = select_checkpoints(directory=directory, hours=hours)[0]
    measure(path=path, variants=variants, device=device, tile_size=tile_size,
        dense_fraction=dense_fraction, forced_dense_fraction=forced_dense_fraction,
        duration=duration, output=output)
