"""Persistent and live CUDA pool allocations for fixed block configurations."""

from pathlib import Path
import gc
import json

import warp as wp

from floodwarp.static_fv1 import StaticBlockFV1Solver
from validation.gpu_benchmark import reset_device_allocation_peak
from validation.static_fv1_benchmark import build_control


def array_bytes(*, solver):
    arrays = list(solver._fields.values())
    arrays.extend((solver.ctrl, solver.condition, solver._rain_times, solver._rain_rates))
    engine = getattr(solver, '_static_execution', None)
    if engine is not None:
        arrays.extend(engine.fields.values())
    unique = {array.ptr: array for array in arrays}
    return sum(array.capacity for array in unique.values())


def measure(*, names, shape, device, options, mesh_options, block_sizes, duration, output):
    rows = []
    for name in names:
        baseline = reset_device_allocation_peak(device=device)
        reference, protected = build_control(name=name, shape=shape, device=device, options=options)
        initial = reference.capture_state()
        reference.run_device_loop(duration=duration)
        reference.restore_state(state=initial)
        rows.append(dict(case=name, mode='dense', persistent_bytes=array_bytes(solver=reference),
            peak_additional_live_pool_bytes=int(wp.get_mempool_used_mem_high(device=device) - baseline)))
        for block_size in block_sizes:
            baseline = reset_device_allocation_peak(device=device)
            solver = StaticBlockFV1Solver.from_solver(solver=reference, block_size=block_size,
                protected=protected, **mesh_options)
            solver.run_device_loop(duration=duration)
            wp.synchronize_device(device=device)
            rows.append(dict(case=name, mode='static', block_size=block_size,
                persistent_bytes=array_bytes(solver=solver),
                peak_additional_live_pool_bytes=int(wp.get_mempool_used_mem_high(device=device) - baseline),
                resident_reference_bytes=array_bytes(solver=reference),
                whole_process_peak_pool_bytes=int(wp.get_mempool_used_mem_high(device=device))))
            solver = None
            gc.collect()
        reference = None
        gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Separate memory probe, not a runtime benchmark. '
        'Peak additional live bytes exclude the resident uniform source model required by the constructor; '
        'whole-process peaks include it. CUDA context/code/reserved pool memory excluded.', rows=rows), indent=2) + '\n')


if __name__ == "__main__":
    names = ('lake', 'front', 'ponding', 'infiltrating', 'heterogeneous', 'channel')
    shape = (64, 64)
    device = 'cuda:0'
    options = dict(depth=.4, spacing=1., manning=.03, maximum_dt=.02, bed_slope=.002,
        bed_step=.1, lake_stage=.8, barrier_height=2., front_protection_cells=4,
        velocity_x=.2, velocity_y=.1, rain=.001, high_infiltration=.002, low_infiltration=.0002,
        soil_period_cells=3, decay_depth=.01)
    mesh_options = dict(surface_tolerance=.001, momentum_tolerance=.001,
                        geometry_tolerance=1.e-12, source_tolerance=0.)
    block_sizes = (4, 8)
    duration = .2
    output = Path('validation/static_review/memory.json')
    measure(names=names, shape=shape, device=device, options=options, mesh_options=mesh_options,
            block_sizes=block_sizes, duration=duration, output=output)
