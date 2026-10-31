"""Restartable matched 72-hour sparse FV1 event, using retained native inputs.

The ordinary retained event used 60 s output boundaries. Replaying those same
boundaries keeps its accepted timestep sequence directly comparable. Original
scenarios are read-only; all outputs go to a new directory.
"""

from pathlib import Path
from time import perf_counter
import json

import numpy as np

from validation.sparse_fv1_benchmark import load_solver, select_checkpoints, digest, memory_layers


def publish(*, path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(target=path)


def initialise_event(*, solver):
    state = solver.capture_state()
    for name in ('q', 'maximum', 'soil', 'rain_sum', 'inflow', 'outflow', 'volume_x', 'volume_y',
                 'flux_x', 'flux_y', 'pressure_x', 'pressure_y', 'dt_histogram'):
        state['arrays'][name].fill(0.)
    state['arrays']['theta'].fill(1.)
    state['time'] = 0.
    state['steps'] = 0
    state['initial_volume'] = 0.
    state['control'].fill(0.)
    solver.restore_state(state=state)


def hourly_differences(*, solver, path):
    with np.load(file=path, allow_pickle=False) as saved:
        differences = {name: float(np.max(np.abs(field.numpy() - saved[name])))
                       for name, field in solver._fields.items()}
        metadata = json.loads(str(saved['metadata']))
    return dict(reference_checkpoint=str(path), reference_sha256=digest(path=path),
                maximum_absolute_by_field=differences,
                identical_steps=solver.steps == metadata['steps'])


def verify_recovery(*, solver, checkpoint, device, tile_size, dense_fraction):
    restored = load_solver(path=checkpoint, device=device, mode='sparse',
                           tile_size=tile_size, dense_fraction=dense_fraction)
    assert restored.time == solver.time and restored.steps == solver.steps
    before = solver.capture_state()
    solver.step(remaining_time=1.)
    restored.step(remaining_time=1.)
    maximum = max(float(np.max(np.abs(restored._fields[name].numpy() - field.numpy())))
                  for name, field in solver._fields.items())
    assert maximum == 0.
    solver.restore_state(state=before)
    return dict(checkpoint_sha256=digest(path=checkpoint), continuation_maximum_absolute=maximum,
                restored_time_s=before['time'])


def run_event(*, checkpoints, device, tile_size, dense_fraction, duration, sample_interval,
              checkpoint_interval, output):
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path('src/floodwarp') / name for name in
                    ('full_swe_solver.py', 'full_swe_kernels.py', 'sparse_fv1.py', 'sparse_fv1_kernels.py')]
    signature = dict(sources={path.name: digest(path=path) for path in source_paths},
        driver_sha256=digest(path=Path(__file__)), initial_input_sha256=digest(path=checkpoints[0]),
        tile_size=tile_size, dense_fraction=dense_fraction, duration_s=duration,
        sample_interval_s=sample_interval, checkpoint_interval_s=checkpoint_interval)
    progress_path = output / 'progress.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['signature'] != signature:
            raise ValueError('Event resume signature differs')
        checkpoint = output / progress['checkpoint']
        if digest(path=checkpoint) != progress['checkpoint_sha256']:
            raise ValueError('Event resume checkpoint checksum differs')
        solver = load_solver(path=checkpoint, device=device, mode='sparse', tile_size=tile_size, dense_fraction=dense_fraction)
    else:
        solver = load_solver(path=checkpoints[0], device=device, mode='sparse', tile_size=tile_size, dense_fraction=dense_fraction)
        initialise_event(solver=solver)
        progress = dict(signature=signature, checkpoint=None, rows=[], hourly=[], recovery=[], process_seconds=0.)
    # Compile/capture outside hydraulic timing and restore exactly afterwards.
    initial = solver.capture_state()
    solver.run_device_loop(duration=min(sample_interval, duration - solver.time))
    solver.restore_state(state=initial)
    checkpoint_by_time = {float((index + 1) * 3600): path for index, path in enumerate(checkpoints)}
    process_started = perf_counter()
    while solver.time < duration:
        target = min(solver.time + sample_interval, duration)
        stats = solver.run_device_loop(duration=target - solver.time)
        ledger = solver.cumulative_accounting()
        progress['rows'].append(dict(stats=stats, accounting=ledger, execution=solver.execution_diagnostics()))
        if target % checkpoint_interval == 0. or target == duration:
            differences = hourly_differences(solver=solver, path=checkpoint_by_time[target])
            if not differences['identical_steps'] or max(differences['maximum_absolute_by_field'].values()) != 0.:
                raise ArithmeticError('Sparse event differs from its retained hourly dense reference')
            progress['hourly'].append(dict(time_s=target, differences=differences))
            checkpoint = output / ('checkpoint-' + str(int(target)) + '.npz')
            solver.save_checkpoint(path=checkpoint)
            if not progress['recovery']:
                progress['recovery'].append(verify_recovery(solver=solver, checkpoint=checkpoint,
                    device=device, tile_size=tile_size, dense_fraction=dense_fraction))
            previous = progress['checkpoint']
            progress['checkpoint'] = checkpoint.name
            progress['checkpoint_sha256'] = digest(path=checkpoint)
            progress['time_s'] = target
            progress['steps'] = solver.steps
            progress['memory_bytes'] = memory_layers(solver=solver)
            progress['process_seconds'] += perf_counter() - process_started
            publish(path=progress_path, value=progress)
            if previous is not None and previous != checkpoint.name:
                (output / previous).unlink()
            process_started = perf_counter()
            print('hour', target / 3600., 'steps', solver.steps,
                  'hydraulic_seconds', sum(row['stats']['loop_seconds'] for row in progress['rows']), flush=True)
    progress['hydraulic_seconds'] = sum(row['stats']['loop_seconds'] for row in progress['rows'])
    progress['final_accounting'] = solver.cumulative_accounting()
    progress['complete'] = True
    publish(path=output / 'complete.json', value=progress)


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    device = 'cuda:0'
    tile_size = 8
    dense_fraction = .65
    duration = 259200.
    sample_interval = 60.
    checkpoint_interval = 3600.
    output = Path('outputs/sparse_fv1_event_8')
    checkpoints = select_checkpoints(directory=directory, hours=tuple(range(1, 73)))
    run_event(checkpoints=checkpoints, device=device, tile_size=tile_size, dense_fraction=dense_fraction,
              duration=duration, sample_interval=sample_interval, checkpoint_interval=checkpoint_interval, output=output)
