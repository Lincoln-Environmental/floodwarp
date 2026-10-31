"""Full matched event for the phase-qualified modified sparse FV1 on dev.

Retained dense and released-sparse results are read-only. The existing durable
event driver verifies hourly fields and checkpoint continuation. This wrapper
records the qualified source identity, hardware and complete process timing.
"""

from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json
import subprocess

import numpy as np
import warp as wp

from validation.gpu_benchmark import reset_device_allocation_peak
from validation.sparse_fv1_benchmark import digest, select_checkpoints
from validation.sparse_fv1_event import publish, run_event


def run_modified_event(*, directory, output, qualification_path, prior_summary_path,
                       device, tile_size, dense_fraction, duration, sample_interval,
                       checkpoint_interval, expected_shape, expected_active_cells,
                       gpu_command):
    began = perf_counter()
    qualification = json.loads(qualification_path.read_text())
    for name, expected in qualification['sources'].items():
        if digest(path=Path('src/floodwarp') / name) != expected:
            raise ValueError('Runtime differs from phase-qualified candidate: ' + name)
    hours = tuple(range(1, int(duration // checkpoint_interval) + 1))
    checkpoints = select_checkpoints(directory=directory, hours=hours)
    with np.load(file=checkpoints[0], allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        shape = (metadata['configuration']['ny'], metadata['configuration']['nx'])
        if shape != expected_shape or np.count_nonzero(saved['active']) != expected_active_cells:
            raise ValueError('Unexpected full-domain reference grid')
    context = dict(schema=1, qualified_sources=qualification['sources'],
        qualification_sha256=digest(path=qualification_path),
        prior_sparse_summary_sha256=digest(path=prior_summary_path),
        wrapper_sha256=digest(path=Path(__file__)),
        event_driver_sha256=digest(path=Path('validation/sparse_fv1_event.py')),
        source_commit=subprocess.run(args=['git', 'rev-parse', 'HEAD'], capture_output=True,
            text=True, check=True).stdout.strip(),
        configuration=metadata['configuration'], tile_size=tile_size,
        dense_fraction=dense_fraction, duration_s=duration, sample_interval_s=sample_interval,
        checkpoint_interval_s=checkpoint_interval, active_cells=expected_active_cells)
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / 'run_context.json'
    if context_path.exists():
        if json.loads(context_path.read_text())['identity'] != context:
            raise ValueError('Modified-event resume identity differs')
    else:
        if (output / 'progress.json').exists() or (output / 'complete.json').exists():
            raise ValueError('Existing event lacks this wrapper identity')
        publish(path=context_path, value=dict(identity=context,
            started_utc=datetime.now(tz=timezone.utc).isoformat(),
            gpu_before=subprocess.run(args=gpu_command, capture_output=True, text=True, check=True).stdout.strip()))
    if (output / 'complete.json').exists():
        raise ValueError('Completed event is preserved; assess it or choose a fresh output directory')
    baseline = reset_device_allocation_peak(device=device)
    run_event(checkpoints=checkpoints, device=device, tile_size=tile_size,
        dense_fraction=dense_fraction, duration=duration, sample_interval=sample_interval,
        checkpoint_interval=checkpoint_interval, output=output)
    elapsed = perf_counter() - began
    record = json.loads((output / 'complete.json').read_text())
    publish(path=output / 'wrapper_completion.json', value=dict(
        completed_utc=datetime.now(tz=timezone.utc).isoformat(),
        current_process_seconds_including_preflight=elapsed,
        complete_record_sha256=digest(path=output / 'complete.json'),
        baseline_live_pool_bytes=baseline,
        peak_additional_live_pool_bytes=int(wp.get_mempool_used_mem_high(device=device) - baseline),
        peak_scope='CUDA live allocator high-water includes warm-up and the temporary recovery solver; '
            'excludes context/code/reserved memory. Persistent allocation is in complete.json.',
        hydraulic_seconds=record['hydraulic_seconds'], driver_active_process_seconds=record['process_seconds'],
        gpu_after=subprocess.run(args=gpu_command, capture_output=True, text=True, check=True).stdout.strip()))


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    output = Path('outputs/sparse_fv1_event_modified_8')
    qualification_path = Path('validation/sparse_perf_review/benchmark.json')
    prior_summary_path = Path('validation/sparse_review/full_event_summary.json')
    device = 'cuda:0'
    tile_size = 8
    dense_fraction = .65
    duration = 259200.
    sample_interval = 60.
    checkpoint_interval = 3600.
    expected_shape = (1123, 843)
    expected_active_cells = 559985
    gpu_command = ['nvidia-smi', '--query-gpu=name,driver_version,pstate,temperature.gpu,power.limit,clocks.sm,clocks.mem', '--format=csv,noheader']
    run_modified_event(directory=directory, output=output, qualification_path=qualification_path,
        prior_summary_path=prior_summary_path, device=device, tile_size=tile_size,
        dense_fraction=dense_fraction, duration=duration, sample_interval=sample_interval,
        checkpoint_interval=checkpoint_interval, expected_shape=expected_shape,
        expected_active_cells=expected_active_cells, gpu_command=gpu_command)
