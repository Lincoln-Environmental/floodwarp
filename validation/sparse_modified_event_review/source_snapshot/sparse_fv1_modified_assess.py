"""Assess a completed modified sparse event against preserved implementations."""

from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil

import numpy as np

from examples.orton_bradley.reproduce import compare
from validation.benchmark_restart import payload_digest
from validation.sparse_fv1_benchmark import digest


def read_event(*, path):
    content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def canonical_event(*, record, label):
    ledger = []
    for row in record['rows']:
        a = row['accounting']
        ledger.append(dict(seconds=row['stats']['time_s'], rain_m3=a['rain'],
            infiltration_m3=a['infiltration'], inflow_m3=a['boundary_inflow'],
            outflow_m3=a['boundary_outflow'], storage_m3=a['storage'], residual_m3=a['residual']))
    return dict(solver=label, ledger=ledger, steps=record['steps'],
        time_s=record['time_s'], hydraulic_seconds=record['hydraulic_seconds'],
        end_to_end_seconds=record['process_seconds'],
        dt_min_s=min(row['stats']['dt_min_s'] for row in record['rows']),
        dt_mean_s=record['time_s'] / record['steps'],
        dt_max_s=max(row['stats']['dt_max_s'] for row in record['rows']))


def outlet_series(*, record, times, interval):
    ledger = {row['seconds']: row for row in record['ledger']}
    if any(time not in ledger for time in times):
        raise ValueError('Missing exact common cumulative-outlet time')
    volume = np.asarray([ledger[time]['outflow_m3'] - ledger[time]['inflow_m3'] for time in times])
    return volume, np.diff(np.r_[0., volume]) / interval


def read_checkpoint(*, path):
    with np.load(file=path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        arrays = {}
        for name, expected in metadata['sha256'].items():
            values = saved[name]
            if hashlib.sha256(values.tobytes()).hexdigest() != expected:
                raise ValueError('Checkpoint checksum differs: ' + name)
            arrays[name] = values.copy()
    return metadata, arrays


def checkpoint_differences(*, current, reference_path):
    metadata, reference = read_checkpoint(path=reference_path)
    if current.keys() != reference.keys():
        raise ValueError('Checkpoint field sets differ')
    differences = {name: float(np.max(np.abs(values - reference[name]))) for name, values in current.items()}
    return metadata, differences


def export_hydrographs(*, records, times, interval, path):
    series = {name: outlet_series(record=record, times=times, interval=interval) for name, record in records.items()}
    labels = dict(modified='floodwarp_modified_sparse_fv1', released='floodwarp_released_sparse_fv1',
        dense='floodwarp_dense_fv1', reference='lisflood_fv1')
    names = ['interval_start_s', 'interval_end_s', 'interval_mid_h']
    for name in records:
        names.extend([labels[name] + '_mean_net_outlet_m3_s', labels[name] + '_cumulative_net_export_m3'])
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index, time in enumerate(times):
            row = [time - interval, time, (time - interval / 2.) / 3600.]
            for name in records:
                volume, discharge = series[name]
                row.extend([discharge[index], volume[index]])
            writer.writerow(row)


def phase_comparison(*, record, prior, dense, phase_edges_h):
    dense_rows = {row['time_s']: row for row in dense['statistics']}
    prior_rows = {row['stats']['time_s']: row for row in prior['rows']}
    result = []
    for start, end in zip(phase_edges_h[:-1], phase_edges_h[1:]):
        rows = [row for row in record['rows'] if start * 3600. < row['stats']['time_s'] <= end * 3600.]
        modified_seconds = sum(row['stats']['loop_seconds'] for row in rows)
        prior_seconds = sum(prior_rows[row['stats']['time_s']]['stats']['loop_seconds'] for row in rows)
        dense_seconds = sum(dense_rows[row['stats']['time_s']]['loop_seconds'] for row in rows)
        result.append(dict(start_h=start, end_h=end, modified_hydraulic_seconds=modified_seconds,
            released_sparse_hydraulic_seconds=prior_seconds, dense_hydraulic_seconds=dense_seconds,
            reduction_vs_released_sparse=1. - modified_seconds / prior_seconds,
            reduction_vs_dense=1. - modified_seconds / dense_seconds,
            accepted_steps=sum(row['stats']['steps'] for row in rows)))
    return result


def export_runtime(*, record, prior, dense, path):
    sources = dict(modified=[row['stats'] for row in record['rows']],
        released=[row['stats'] for row in prior['rows']], dense=dense['statistics'])
    cumulative = {name: np.cumsum([row['loop_seconds'] for row in rows]) for name, rows in sources.items()}
    names = ['interval_end_s', 'time_h']
    for name in sources:
        names.extend([name + '_hydraulic_seconds_interval', name + '_hydraulic_seconds_cumulative', name + '_accepted_steps_interval'])
    names.extend(['modified_evaluated_cells_endpoint', 'modified_evaluated_faces_x_endpoint', 'modified_evaluated_faces_y_endpoint'])
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index, current in enumerate(record['rows']):
            time = current['stats']['time_s']
            row = [time, time / 3600.]
            for name, rows in sources.items():
                if rows[index]['time_s'] != time:
                    raise ValueError('Runtime sampling times differ')
                row.extend([rows[index]['loop_seconds'], cumulative[name][index], rows[index]['steps']])
            row.extend(current['execution'][name] for name in ('evaluated_cells', 'evaluated_faces_x', 'evaluated_faces_y'))
            writer.writerow(row)


def assess(*, directory, dense_directory, external_directory, prior_record_path,
           prior_summary_path, output, duration, interval, thresholds,
           wet_union_threshold, phase_edges_h):
    record, record_hash = read_event(path=directory / 'complete.json')
    prior, prior_hash = read_event(path=prior_record_path)
    prior_summary = json.loads(prior_summary_path.read_text())
    dense = json.loads((dense_directory / 'record.json').read_text())
    external = json.loads((external_directory / 'record.json').read_text())
    wrapper = json.loads((directory / 'wrapper_completion.json').read_text())
    context = json.loads((directory / 'run_context.json').read_text())
    if prior_hash != prior_summary['complete_record_sha256'] or wrapper['complete_record_sha256'] != record_hash:
        raise ValueError('Complete-record identity differs')
    for saved in (dense, external):
        if payload_digest(value=saved) != saved['payload_sha256']:
            raise ValueError('Retained benchmark record payload checksum differs')
    if not record['complete'] or record['time_s'] != duration or record['steps'] != dense['steps']:
        raise ValueError('Incomplete event or changed accepted timestep sequence')
    times = np.arange(interval, duration + interval / 2., interval)
    np.testing.assert_array_equal(actual=[row['stats']['time_s'] for row in record['rows']], desired=times)
    np.testing.assert_array_equal(actual=[row['stats']['steps'] for row in record['rows']],
        desired=[row['stats']['steps'] for row in prior['rows']])
    np.testing.assert_array_equal(actual=[row['stats']['steps'] for row in record['rows']],
        desired=[row['steps'] for row in dense['statistics']])
    if len(record['hourly']) != len(prior['hourly']) or not all(row['differences']['identical_steps'] for row in record['hourly']):
        raise ValueError('Missing or changed hourly state comparisons')
    hourly_maximum = max(max(row['differences']['maximum_absolute_by_field'].values()) for row in record['hourly'])
    if hourly_maximum != 0.:
        raise ValueError('Hourly accepted hydraulic fields differ')
    final_checkpoint = directory / record['checkpoint']
    if digest(path=final_checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Final checkpoint file checksum differs')
    metadata, arrays = read_checkpoint(path=final_checkpoint)
    old_checkpoint = Path(prior_summary['final_checkpoint'])
    if digest(path=old_checkpoint) != prior_summary['final_checkpoint_sha256']:
        raise ValueError('Retained released-sparse checkpoint checksum differs')
    old_metadata, old_differences = checkpoint_differences(current=arrays, reference_path=old_checkpoint)
    if metadata['configuration'] != old_metadata['configuration'] or metadata['time'] != old_metadata['time']:
        raise ValueError('Retained sparse configuration/time differs')
    config = metadata['configuration']
    shape = (config['ny'], config['nx'])
    active = arrays['active'].reshape(shape).astype(bool)
    current_fields = dict(depth=arrays['q'][:, 0].reshape(shape), maximum=arrays['maximum'].reshape(shape))
    if np.any(current_fields['depth'][active] < 0.):
        raise ValueError('Negative accepted depth')
    fields = {'modified': current_fields}
    for name, source, saved_record in (('reference', external_directory, external), ('dense', dense_directory, dense)):
        if digest(path=source / 'fields.npz') != saved_record['fields_sha256']:
            raise ValueError('Retained final/maximum fields checksum differs')
        with np.load(file=source / 'fields.npz', allow_pickle=False) as saved:
            fields[name] = {key: saved[key].copy() for key in ('depth', 'maximum')}
    with np.load(file=old_checkpoint, allow_pickle=False) as saved:
        fields['released'] = dict(depth=saved['q'][:, 0].reshape(shape), maximum=saved['maximum'].reshape(shape))
    records = dict(modified=canonical_event(record=record, label='Modified sparse FloodWarp FV1'),
        released=canonical_event(record=prior, label='Released sparse FloodWarp FV1'), dense=dense, reference=external)
    reference_metrics = compare(records=records, fields=fields, active=active,
        thresholds=thresholds, cell_area=config['dx'] * config['dy'], wet_union_threshold=wet_union_threshold)
    execution_metrics = compare(records=dict(modified=records['modified'], released=records['released'], reference=dense),
        fields=dict(modified=fields['modified'], released=fields['released'], reference=fields['dense']),
        active=active, thresholds=thresholds, cell_area=config['dx'] * config['dy'], wet_union_threshold=wet_union_threshold)
    output.mkdir(parents=True, exist_ok=True)
    export_hydrographs(records=records, times=times, interval=interval, path=output / 'hydrographs_72h.csv')
    export_runtime(record=record, prior=prior, dense=dense, path=output / 'runtime_72h.csv')
    phases = phase_comparison(record=record, prior=prior, dense=dense, phase_edges_h=phase_edges_h)
    work_fields = ('evaluated_cells', 'evaluated_faces_x', 'evaluated_faces_y', 'work_tiles')
    work_equal = all(all(a['execution'][name] == b['execution'][name] for name in work_fields)
        for a, b in zip(record['rows'], prior['rows']))
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name, expected in record['signature']['sources'].items():
        source = Path('src/floodwarp') / name
        if digest(path=source) != expected:
            raise ValueError('Runtime changed before source preservation: ' + name)
        shutil.copy2(src=source, dst=snapshot / name)
    assessment_sources = {}
    for source in (Path('validation/sparse_fv1_event.py'), Path('validation/sparse_fv1_modified_event.py'), Path(__file__),
                   Path('validation/sparse_fv1_modified_figures.py'), Path('validation/sparse_fv1_benchmark.py'),
                   Path('validation/benchmark_restart.py'), Path('examples/orton_bradley/reproduce.py')):
        if source.name == 'sparse_fv1_event.py' and digest(path=source) != record['signature']['driver_sha256']:
            raise ValueError('Event driver changed before preservation')
        if source.name == 'sparse_fv1_modified_event.py' and digest(path=source) != context['identity']['wrapper_sha256']:
            raise ValueError('Event wrapper changed before preservation')
        assessment_sources[source.as_posix()] = digest(path=source)
        shutil.copy2(src=source, dst=snapshot / source.name)
    summary = dict(complete_record_sha256=record_hash, prior_complete_record_sha256=prior_hash,
        source_signature=record['signature'], assessment_sources=assessment_sources,
        run_context=context, wrapper_completion=wrapper,
        duration_s=record['time_s'], accepted_steps=record['steps'], hourly_comparisons=len(record['hourly']),
        maximum_hourly_field_difference=hourly_maximum, final_differences_vs_released_sparse=old_differences,
        minute_end_hydraulic_work_counts_equal=work_equal, phases=phases,
        metrics_vs_dense=execution_metrics, metrics_vs_external_fv1=reference_metrics,
        hydraulic_seconds=record['hydraulic_seconds'], released_sparse_hydraulic_seconds=prior['hydraulic_seconds'],
        dense_hydraulic_seconds=dense['hydraulic_seconds'],
        runtime_reduction_vs_released_sparse=1. - record['hydraulic_seconds'] / prior['hydraulic_seconds'],
        runtime_reduction_vs_dense=1. - record['hydraulic_seconds'] / dense['hydraulic_seconds'],
        memory_bytes=record['memory_bytes'], checkpoint_recovery=record['recovery'],
        final_accounting=record['final_accounting'],
        final_checkpoint=str(final_checkpoint), final_checkpoint_sha256=record['checkpoint_sha256'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with gzip.open(filename=output / 'complete_record.json.gz', mode='wb') as stream:
        stream.write((directory / 'complete.json').read_bytes())
    write_report(summary=summary, records=records, output=output / 'REPORT.md')


def write_report(*, summary, records, output):
    lines = ['# Modified exact-sparse FV1 — matched 72-hour Orton Bradley event', '',
        f'Completed **72 hours** on the full **1123×843, 5 m** domain with **{summary["accepted_steps"]:,} accepted steps**. '
        f'All **{summary["hourly_comparisons"]} hourly comparisons** against retained dense FV1 have '
        f'maximum field difference **{summary["maximum_hourly_field_difference"]:g}**.', '',
        '| Implementation | Hydraulic [s] | Recorded end-to-end component [s] | Steps |',
        '|---|---:|---:|---:|']
    labels = dict(dense='Dense FloodWarp FV1 (retained)', released='Released sparse FV1 (retained)',
        modified='Modified sparse FV1 (new)', reference='External LISFLOOD-FP FV1 (retained)')
    for name in ('dense', 'released', 'modified', 'reference'):
        row = records[name]
        lines.append(f'| {labels[name]} | {row["hydraulic_seconds"]:.3f} | {row["end_to_end_seconds"]:.3f} | {row["steps"]:,} |')
    lines.extend(['', f'Modified sparse hydraulic runtime is **{100. * summary["runtime_reduction_vs_released_sparse"]:.2f}% lower** '
        f'than released sparse and **{100. * summary["runtime_reduction_vs_dense"]:.2f}% lower** than retained dense. '
        'This is one new candidate event against retained same-device implementations, not interleaved full-event repetitions.', '',
        'End-to-end columns have different diagnostic workloads: sparse runs include hourly full-state validation and checkpoint hashing; '
        'dense/external timers have their original component scopes. The old sparse release-work pause is excluded. '
        'Wrapper completion separately records current-process wall time including preflight and live allocator peak; '
        'that peak includes the temporary checkpoint-recovery solver and excludes CUDA context/code/reserved memory. '
        f'Current-process time including preflight: **{summary["wrapper_completion"]["current_process_seconds_including_preflight"]:.3f} s**; '
        f'peak additional live pool allocation: **{summary["wrapper_completion"]["peak_additional_live_pool_bytes"]:,} B**. '
        'The independent process_time.txt records external wall time and CPU resident-memory peak.', '',
        '![Matched hydraulic runtime and phase comparison](runtime_comparison.png)', '',
        '| Event interval [h] | Dense [s] | Released sparse [s] | Modified sparse [s] | Reduction vs released |',
        '|---|---:|---:|---:|---:|'])
    for row in summary['phases']:
        lines.append(f'| {row["start_h"]:g}–{row["end_h"]:g} | {row["dense_hydraulic_seconds"]:.3f} | '
            f'{row["released_sparse_hydraulic_seconds"]:.3f} | {row["modified_hydraulic_seconds"]:.3f} | '
            f'{100. * row["reduction_vs_released_sparse"]:.2f}% |')
    lines.extend(['', '## Hydraulic verification', '',
        'Hourly comparisons include depth/momentum, soil/source partition, cumulative rainfall and accepted face volumes, '
        'boundary accounting arrays, maximum depth, limiter/pressure/flux scratch and timestep histograms. '
        'Final native checkpoint arrays are directly compared to released sparse, including checkpoint controls and forcing arrays. '
        'Minute-end evaluated-cell/face/work-tile counts are compared separately; endpoint samples are not per-step integrated work counters.', '',
        '| Metric | Modified versus dense | Modified versus external FV1 |', '|---|---:|---:|'])
    a, b = summary['metrics_vs_dense']['modified'], summary['metrics_vs_external_fv1']['modified']
    for label, key in (('Final-depth RMSE [m]', 'final_depth_rmse_m'), ('Final wet-union RMSE [m]', 'final_wet_union_rmse_m'),
        ('Maximum-envelope RMSE [m]', 'maximum_depth_envelope_rmse_m'), ('Maximum-envelope max absolute difference [m]', 'maximum_depth_envelope_maximum_absolute_error_m'),
        ('Hydrograph RMSE [m³/s]', 'hydrograph_rmse_m3_s'), ('Hydrograph NRMSE / reference peak', 'hydrograph_nrmse'),
        ('Peak discharge error [m³/s]', 'peak_discharge_error_m3_s'), ('Peak timing error [s]', 'peak_interval_end_timing_error_s'),
        ('Net export difference [m³]', 'cumulative_net_export_difference_m3'), ('Infiltration difference [m³]', 'infiltration_difference_m3')):
        lines.append(f'| {label} | {a[key]:.9g} | {b[key]:.9g} |')
    for threshold, metrics in a['maximum_event_extent'].items():
        lines.append(f'| Maximum-event IoU ≥{float(threshold):g} m | {metrics["iou"]:.9g} | {b["maximum_event_extent"][threshold]["iou"]:.9g} |')
    lines.extend(['', f'Final direct checkpoint maximum difference versus released sparse: '
        f'**{max(summary["final_differences_vs_released_sparse"].values()):.9g}**. '
        f'Minute-end hydraulic work counts unchanged: **{summary["minute_end_hydraulic_work_counts_equal"]}**. '
        f'Final water-balance residual: **{summary["final_accounting"]["residual"]:.9g} m³**. '
        f'Persistent solver allocation: **{summary["memory_bytes"]["total"]:,} B**.', '',
        'Depth-envelope/extent metrics use each cell’s event maximum, not final depth. Hydrographs are actual 60-second interval means '
        'from cumulative net export, with separate peak magnitude and timing criteria. Both run settings and original inputs are preserved: '
        'FP64, contact HLL, explicit discharge-extrapolated matched stage boundaries, full-domain source/boundary CFL and fine source/soil updates. '
        'No shallow water is deleted, and no source/precision/timestep/formulation change is used to obtain performance.', '',
        'New savings concern redundant scratch clearing and source/activity memory traffic/launches; hydraulic work counts and grid resolution '
        'remain unchanged relative to released sparse execution. No sparse-storage or AMR speedup is inferred. Original v0.1.0/v0.1.1 assets '
        'and prior benchmark records remain untouched. Source snapshots and lossless minute/hourly records bind this result.', '',
        '![Full-event, peak-detail and execution-difference hydrographs](hydrograph_comparison.png)', '',
        '## Reproduce the comparison', '',
        '`python -m validation.sparse_fv1_modified_assess` checks the completed candidate and retained references, '
        'then writes summary.json, the lossless complete record, hydrographs_72h.csv and runtime_72h.csv. '
        '`python -m validation.sparse_fv1_modified_figures` regenerates both figures from just the exported tables '
        'and summary, without hydraulic execution or access to the original private working directory. '
        'Editable paths and display settings are in the execution guards. A new hydraulic replay uses '
        '`python -m validation.sparse_fv1_modified_event` with a fresh output directory; completed results are never overwritten. '
        'The research driver preserves hourly progress with source/configuration identity checks and verifies native '
        'checkpoint continuation separately. This does not change the published standalone replay’s explicit-restart contract.', '',
        '**Research demonstration only — not an official flood-hazard assessment. '
        'Independent site-specific validation is required before decision-making.**', ''])
    output.write_text('\n'.join(lines))


if __name__ == "__main__":
    directory = Path('outputs/sparse_fv1_event_modified_8')
    dense_directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    external_directory = Path('outputs/qualification_optimized/purau_event_72h/reference_0')
    prior_record_path = Path('validation/sparse_review/full_event_record.json.gz')
    prior_summary_path = Path('validation/sparse_review/full_event_summary.json')
    output = Path('validation/sparse_modified_event_review')
    duration = 259200.
    interval = 60.
    thresholds = (.01, .1)
    wet_union_threshold = .001
    phase_edges_h = (0., 11., 20., 31., 48., 72.)
    assess(directory=directory, dense_directory=dense_directory, external_directory=external_directory,
        prior_record_path=prior_record_path, prior_summary_path=prior_summary_path,
        output=output, duration=duration, interval=interval, thresholds=thresholds,
        wet_union_threshold=wet_union_threshold, phase_edges_h=phase_edges_h)
