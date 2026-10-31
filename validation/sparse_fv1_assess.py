"""Assess the completed exact-sparse event without rerunning hydraulics."""

from pathlib import Path
from datetime import datetime
import gzip
import hashlib
import json
import shutil
import subprocess

import numpy as np


def digest(*, path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assess(*, directory, dense_record_path, output, phase_edges_h):
    record_path = directory / 'complete.json'
    record = json.loads(record_path.read_text())
    dense = json.loads(dense_record_path.read_text())
    if not record['complete'] or record['steps'] != dense['steps'] or record['time_s'] != dense['time_s']:
        raise ValueError('Incomplete or differently stepped event')
    if len(record['hourly']) != 72 or len(record['rows']) != 4320:
        raise ValueError('Missing hourly verification or minute diagnostics')
    maximum = max(max(hour['differences']['maximum_absolute_by_field'].values()) for hour in record['hourly'])
    if maximum != 0. or not all(hour['differences']['identical_steps'] for hour in record['hourly']):
        raise ValueError('Uniform-grid sparse execution differs from dense')
    dense_stats = {row['time_s']: row for row in dense['statistics']}
    nx, ny = dense['signature']['shape'][1], dense['signature']['shape'][0]
    cells = nx * ny
    phases = []
    for start, end in zip(phase_edges_h[:-1], phase_edges_h[1:]):
        rows = [row for row in record['rows'] if start * 3600. < row['stats']['time_s'] <= end * 3600.]
        hydraulic = sum(row['stats']['loop_seconds'] for row in rows)
        reference_cost = sum(dense_stats[row['stats']['time_s']]['loop_seconds'] for row in rows)
        weighted_cells = sum(dense_stats[row['stats']['time_s']]['loop_seconds'] * row['execution']['evaluated_cells'] for row in rows)
        phases.append(dict(start_h=start, end_h=end, hydraulic_seconds=hydraulic,
            retained_dense_hydraulic_seconds=reference_cost, measured_hydraulic_ratio=reference_cost / hydraulic,
            accepted_steps=sum(row['stats']['steps'] for row in rows),
            dense_cost_weighted_endpoint_work_cell_fraction=weighted_cells / (reference_cost * cells)))
    raw_birth = subprocess.run(args=['stat', '--printf=%w', str(directory)], capture_output=True, text=True, check=True).stdout.strip()
    date, time, offset = raw_birth.split()
    began = datetime.fromisoformat(date + 'T' + time + offset).timestamp()
    observed_wall = record_path.stat().st_mtime - began
    dense_ledger = {row['seconds']: row for row in dense['ledger']}
    sparse_volume = np.asarray([row['accounting']['boundary_outflow'] - row['accounting']['boundary_inflow'] for row in record['rows']])
    dense_volume = np.asarray([dense_ledger[row['stats']['time_s']]['outflow_m3'] - dense_ledger[row['stats']['time_s']]['inflow_m3'] for row in record['rows']])
    dq = np.diff(np.r_[0., sparse_volume - dense_volume]) / 60.
    source_snapshot = output / 'source_snapshot'
    source_snapshot.mkdir(parents=True, exist_ok=True)
    for name, expected in record['signature']['sources'].items():
        path = Path('src/floodwarp') / name
        if digest(path=path) != expected:
            raise ValueError('Runtime source changed before preservation: ' + name)
        shutil.copy2(src=path, dst=source_snapshot / name)
    driver = Path('validation/sparse_fv1_event.py')
    if digest(path=driver) != record['signature']['driver_sha256']:
        raise ValueError('Event driver changed before preservation')
    shutil.copy2(src=driver, dst=source_snapshot / driver.name)
    summary = dict(complete_record_sha256=digest(path=record_path),
        dense_record_sha256=digest(path=dense_record_path), source_signature=record['signature'],
        duration_s=record['time_s'], accepted_steps=record['steps'],
        hourly_comparisons=len(record['hourly']), maximum_absolute_field_difference=maximum,
        hydraulic_seconds=record['hydraulic_seconds'], retained_dense_hydraulic_seconds=dense['hydraulic_seconds'],
        measured_hydraulic_speedup=dense['hydraulic_seconds'] / record['hydraulic_seconds'],
        active_process_seconds_excluding_pause=record['process_seconds'],
        retained_dense_end_to_end_component_seconds=dense['end_to_end_seconds'],
        observed_wall_seconds_including_release_pause=observed_wall,
        timing_scope='One candidate run across two processes with verified restart; active process time includes '
            'additional hourly dense-field checks/hashes and IO. Wall time includes the authorised release-work pause. '
            'Hydraulic ratios use the retained same-device dense run, not a fresh interleaved full-event repetition.',
        phases=phases, occupancy_scope='Minute-end work counts weighted by matched dense-window costs; '
            'not exact per-step integrated cell/face counters.',
        hydrograph_rmse_vs_dense_m3_s=float(np.sqrt(np.mean(dq**2))),
        cumulative_net_export_difference_vs_dense_m3=float(sparse_volume[-1] - dense_volume[-1]),
        memory_bytes=record['memory_bytes'], checkpoint_recovery=record['recovery'], final_accounting=record['final_accounting'],
        final_checkpoint_sha256=record['checkpoint_sha256'], final_checkpoint=str(directory / record['checkpoint']))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'full_event_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with gzip.open(filename=output / 'full_event_record.json.gz', mode='wb') as stream:
        stream.write(record_path.read_bytes())
    lines = ['# Matched 72-hour exact-sparse FV1 event', '',
        f'Completed {record["time_s"] / 3600.:g} h with **{record["steps"]:,} accepted timesteps**. '
        'All 72 hourly comparisons have zero numerical field difference against retained dense FP64 FV1, '
        'including conservative state, sources/soil, boundaries, maxima, accepted face volumes and timestep histograms.', '',
        f'Hydraulic time: **{record["hydraulic_seconds"]:.3f} s**, retained dense **{dense["hydraulic_seconds"]:.3f} s** '
        f'(**{summary["measured_hydraulic_speedup"]:.3f}×**). Active-process time including validation/IO: '
        f'{record["process_seconds"]:.3f} s. Observed publication wall time: {observed_wall:.1f} s, '
        'including the deliberate pause for the v0.1.0 release. Do not describe that pause as solver compute time.', '',
        '| Event interval [h] | Sparse hydraulic [s] | Retained dense [s] | Dense / sparse | Dense-cost-weighted endpoint cell fraction |',
        '|---|---:|---:|---:|---:|']
    for row in phases:
        lines.append(f'| {row["start_h"]:g}–{row["end_h"]:g} | {row["hydraulic_seconds"]:.3f} | '
            f'{row["retained_dense_hydraulic_seconds"]:.3f} | {row["measured_hydraulic_ratio"]:.3f} | '
            f'{row["dense_cost_weighted_endpoint_work_cell_fraction"]:.4f} |')
    lines.extend(['', 'Weights use actual retained dense-window cost. Occupancy is sampled at minute ends and '
        'must not be claimed as an exact per-step work integral. The full-event comparison is one candidate '
        'run against the retained matched control; the saved-state short windows separately have warm-up '
        'and three repetitions. Profilers ran outside timed replay.', '',
        f'Steady allocation remains dense: {record["memory_bytes"]["total"]:,} bytes, including '
        f'{record["memory_bytes"]["scheduling"]:,} scheduling bytes. There is no memory-saving claim. '
        'Saved-state peak allocations are recorded in benchmark.json; whole-run allocation peaks were '
        'not instrumented, and checkpoint recovery temporarily constructs a second solver.', '',
        'Sources and full source/boundary-aware CFL remain active over the physical domain; no shallow '
        'water is deleted. Dry and recession workloads benefit, while widespread runoff retains almost '
        'all physical hydraulic work and can be slower. Dense execution remains the default/fallback. '
        'No sparse-storage or AMR speedup is inferred or multiplied into these results.', '',
        'The uninterrupted physical trajectory survives the authorised release-work interruption. The '
        'native checkpoint hash, exact recovery continuation, complete minute records and 72 hourly '
        'field checks are retained. The source snapshot binds the actual timed implementation.', ''])
    (output / 'FULL_EVENT.md').write_text('\n'.join(lines))


if __name__ == "__main__":
    directory = Path('outputs/sparse_fv1_event_8')
    dense_record_path = Path('outputs/qualification_optimized/purau_event_72h/fv1_0/record.json')
    output = Path('validation/sparse_review')
    phase_edges_h = (0., 11., 20., 31., 48., 72.)
    assess(directory=directory, dense_record_path=dense_record_path, output=output, phase_edges_h=phase_edges_h)
