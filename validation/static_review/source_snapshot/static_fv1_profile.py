"""Separate CUDA-event attribution of static transfers and subface evolution."""

from pathlib import Path
import json

import warp as wp

from floodwarp.static_fv1 import StaticBlockFV1Solver
from validation.static_fv1_benchmark import build_control


def profile(*, names, shape, device, options, mesh_options, block_size, duration, output):
    rows = []
    for name in names:
        reference, protected = build_control(name=name, shape=shape, device=device, options=options)
        static = StaticBlockFV1Solver.from_solver(solver=reference, block_size=block_size,
                                                  protected=protected, **mesh_options)
        for method, solver in (('dense', reference), ('static', static)):
            initial = solver.capture_state()
            solver.run_device_loop(duration=duration, use_cuda_graph=False)
            solver.restore_state(state=initial)
            with wp.ScopedTimer(name='static FV1 kernel attribution', cuda_filter=wp.TIMING_ALL, print=False) as timer:
                stats = solver.run_device_loop(duration=duration, use_cuda_graph=False)
            activities = {}
            for result in timer.timing_results:
                entry = activities.setdefault(result.name, dict(calls=0, milliseconds=0.))
                entry['calls'] += 1
                entry['milliseconds'] += result.elapsed
            for entry in activities.values():
                entry['microseconds_per_step'] = entry['milliseconds'] * 1000. / stats['steps']
            rows.append(dict(case=name, method=method, steps=stats['steps'], activities=activities))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Instrumented host-launch GPU kernels, separate from graph timing; '
        'restriction/projection are spatial transfer overhead, not removed source work.', rows=rows), indent=2) + '\n')


if __name__ == "__main__":
    names = ('lake', 'front', 'infiltrating', 'channel')
    shape = (64, 64)
    device = 'cuda:0'
    options = dict(depth=.4, spacing=1., manning=.03, maximum_dt=.02, bed_slope=.002,
        bed_step=.1, lake_stage=.8, barrier_height=2., front_protection_cells=4,
        velocity_x=.2, velocity_y=.1, rain=.001, high_infiltration=.002, low_infiltration=.0002,
        soil_period_cells=3, decay_depth=.01)
    mesh_options = dict(surface_tolerance=.001, momentum_tolerance=.001,
                        geometry_tolerance=1.e-12, source_tolerance=0.)
    block_size = 4
    duration = .4
    output = Path('validation/static_review/profile.json')
    profile(names=names, shape=shape, device=device, options=options, mesh_options=mesh_options,
            block_size=block_size, duration=duration, output=output)
