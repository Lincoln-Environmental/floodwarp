"""Lossless, resumable full-domain FV1 visualisation export; no solver changes."""
from pathlib import Path
import json
import sys
from importlib.metadata import version
from datetime import datetime, timezone
import os
import subprocess
import time

import numpy as np
import rasterio

from examples.orton_bradley.reproduce import file_hash, verify_sources
from examples.orton_bradley.run_model import load_forcing, sample_targets
from floodwarp.inputs import boundary_faces, read_terrain
from floodwarp.model import (advance, create_model, diagnostics, load_checkpoint,
                             model_time, save_checkpoint)


def atomic_json(*, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target=path)


def atomic_arrays(*, path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(file=stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target=path)


def device_status(*, device, maximum_idle_utilisation):
    """Refuse unknown/busy CUDA devices before allocating any solver state."""
    if device == 'cpu':
        return dict(free=True, reason='CPU selected explicitly')
    index = device.split(':')[-1]
    processes = subprocess.run(args=['nvidia-smi', '-i', index,
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
        check=True, capture_output=True, text=True).stdout.strip()
    utilisation = int(subprocess.run(args=['nvidia-smi', '-i', index,
        '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True).stdout.strip())
    entries = [line.strip() for line in processes.splitlines() if line.strip()]
    own_context = any(line.split(',', 1)[0].strip() == str(os.getpid()) for line in entries)
    foreign = [line for line in entries if line.split(',', 1)[0].strip() != str(os.getpid())]
    return dict(free=not foreign and (own_context or utilisation <= maximum_idle_utilisation),
                compute_processes='; '.join(foreign), utilisation_percent=utilisation)


def require_free_device(*, device, maximum_idle_utilisation):
    status = device_status(device=device, maximum_idle_utilisation=maximum_idle_utilisation)
    if not status['free']:
        raise RuntimeError('Device busy; no model started. Re-run when free: ' + json.dumps(status))
    return status


def prepare_dataset(*, source, example, output, duration, frame_interval,
                    diagnostic_interval, checkpoint_interval, maximum_dt):
    """Audit immutable local products and save geospatial inputs without running FV1."""
    for interval in (duration, frame_interval, diagnostic_interval, checkpoint_interval):
        if not np.isfinite(interval) or interval <= 0. or interval != int(interval):
            raise ValueError('Export times must be positive integer seconds')
    if frame_interval % diagnostic_interval or checkpoint_interval % frame_interval:
        raise ValueError('Frames must align with diagnostics and hourly checkpoints with frames')
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError('Use the pinned Python 3.14 environment')
    packages = {name: version(distribution_name=name) for name in ('numpy', 'scipy', 'numba', 'warp-lang')}
    if packages != dict(numpy='2.3.5', scipy='1.16.3', numba='0.67.0', **{'warp-lang': '1.17.0'}):
        raise RuntimeError('Hydraulic dependency versions differ from the frozen pins')
    provenance = json.loads((example / 'provenance.json').read_text())
    verify_sources(directory=source, provenance=provenance)
    grid = provenance['grid']
    if grid['shape'] != [1123, 843] or grid['active_cells'] != 559985:
        raise ValueError('Only the complete published 5 m model is supported')
    times, rates, fields = load_forcing(path=source / 'inputs/inputs_f64.bin')
    terrain = read_terrain(path=source / 'inputs/original_model_dem.tif', vertical_datum='NZVD2016')
    # Original DEM contains surrounding terrain; physical mask comes from its own raster.
    with rasterio.open(source / 'inputs/original_active_mask.tif') as mask_file:
        if list(mask_file.transform) != grid['transform'] or mask_file.crs.to_epsg() != grid['epsg']:
            raise ValueError('Mask geographic alignment differs')
        active = mask_file.read(1)[::-1].astype(bool)
    np.testing.assert_array_equal(actual=active, desired=fields[0].astype(bool))
    if np.count_nonzero(active) != grid['active_cells'] or list(terrain.metadata['transform']) != grid['transform']:
        raise ValueError('Full-domain terrain alignment differs')
    with np.load(file=source / 'inputs/comparison_inputs.npz', allow_pickle=False) as comparison:
        np.testing.assert_array_equal(actual=terrain.elevation[active], desired=comparison['bed'][::-1][active])
    ports = np.r_[fields[5, 1:], np.zeros_like(fields[5, :1])].astype(bool)
    if int(ports.sum()) != 212:
        raise ValueError('Expected 212 original stage contacts')
    root = Path(__file__).resolve().parents[2]
    numerical_files = [root / 'src/floodwarp' / name for name in (
        'full_swe_solver.py', 'full_swe_kernels.py', 'sparse_fv1.py', 'sparse_fv1_kernels.py',
        'model.py', 'solver_factory.py', 'inputs.py')]
    identity = dict(schema=1, source=str(source.resolve()),
        python='3.14', packages=packages,
        source_sha256=provenance['published_source_files'],
        numerical_sha256={path.name: file_hash(path=path) for path in numerical_files},
        driver_sha256=file_hash(path=Path(__file__)), duration_s=duration,
        frame_interval_s=frame_interval, diagnostic_interval_s=diagnostic_interval,
        checkpoint_interval_s=checkpoint_interval, maximum_dt_s=maximum_dt,
        grid=grid, array_row_order='south_to_north', depth_dtype='float64',
        water_surface='DEM + instantaneous depth; dry/inactive cells are not water',
        rainfall_end_behavior='finite_support', coastal_stage_contacts=212,
        interpretation='Uncalibrated 100-year design rainfall demonstration; not an observed flood',
        provenance=provenance['elevation_attribution'],
        software_attribution='Patrick Durney / Lincoln Agritech Ltd; AGPL-3.0-only',
        elevation_license='CC BY 4.0')
    manifest_path = output / 'metadata/dataset.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            if any((output / 'frames').glob('*.npz')) or any((output / 'checkpoints').glob('*')):
                raise ValueError('Existing dataset identity differs; preserve it and select a separate output root')
            atomic_json(path=manifest_path, value=identity)
        with np.load(file=output / 'metadata/terrain.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(actual=saved['dem'], desired=terrain.elevation)
            np.testing.assert_array_equal(actual=saved['active'], desired=active)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Unidentified nonempty output directory')
        atomic_arrays(path=output / 'metadata/terrain.npz', arrays=dict(
            dem=terrain.elevation, active=active, valid_dem=np.isfinite(terrain.elevation) &
            (terrain.elevation != terrain.metadata['nodata']), rain_times_s=times,
            rain_rates_m_s=rates, rainfall_multiplier=fields[1]))
        atomic_json(path=manifest_path, value=identity)
    return identity


def build_model(*, source, device, maximum_dt):
    """Configure the published ordinary FV1 adapter, preserving every physical field."""
    from floodwarp.inputs import with_active_mask
    times, rates, fields = load_forcing(path=source / 'inputs/inputs_f64.bin')
    terrain = read_terrain(path=source / 'inputs/original_model_dem.tif', vertical_datum='NZVD2016')
    terrain = with_active_mask(terrain=terrain, active=fields[0].astype(bool))
    roughness = np.loadtxt(fname=source / 'inputs/roughness.asc', skiprows=6)[::-1].copy()
    ports = np.r_[fields[5, 1:], np.zeros_like(fields[5, :1])].astype(bool)
    qx, qy = boundary_faces(terrain=terrain)
    rows, columns = np.nonzero(ports)
    qy[rows + 1, columns] = 3
    model = create_model(terrain=terrain, solver_method='fv1', device=device, manning_n=roughness,
        rainfall_series=(times, rates), rainfall_end_behavior='finite_support', initial_depth=0.,
        boundary_qx=qx, boundary_qy=qy, stage_qy=0., stage_datum='NZVD2016',
        explicit_minimum_dt=1.e-6, explicit_maximum_dt=maximum_dt, gravity=9.80665,
        dry_depth=.0001, cfl=.4, flux_scheme='contact_hll',
        stage_boundary_momentum='discharge_extrapolation')
    model.solver.set_exponential_infiltration(f0=fields[2], fd=fields[3], conductivity=fields[4])
    model.solver.set_rainfall_spatial_multiplier(multiplier=fields[1])
    return model


def frame_path(*, output, seconds):
    return output / 'frames' / f'depth-{int(seconds):06d}.npz'


def write_frame(*, output, seconds, depth, active):
    if depth.shape != active.shape or not np.isfinite(depth).all() or np.any(depth < 0.):
        raise ValueError('Invalid instantaneous depth')
    if np.any(depth[~active] != 0.):
        raise ValueError('Nonzero depth outside hydraulic mask')
    path = frame_path(output=output, seconds=seconds)
    if path.exists():
        with np.load(file=path, allow_pickle=False) as saved:
            if float(saved['time_s']) != seconds or not np.array_equal(saved['depth'], depth):
                raise ValueError('Retained frame differs on replay: ' + str(path))
    else:
        atomic_arrays(path=path, arrays=dict(time_s=np.float64(seconds), depth=depth))


def diagnostic_row(*, current, previous):
    interval = current['time_s'] - previous['time_s']
    if interval <= 0.:
        raise ValueError('Diagnostics require increasing times')
    return dict(time_s=current['time_s'], storage_m3=current['storage_m3'],
        rain_m3=current['rain'], infiltration_m3=current['infiltration'],
        outflow_m3=current['boundary_outflow'], inflow_m3=current['boundary_inflow'],
        residual_m3=current['residual_m3'],
        discharge_m3_s=(current['boundary_outflow'] - previous['boundary_outflow'] -
                        current['boundary_inflow'] + previous['boundary_inflow']) / interval,
        rainfall_m3_s=(current['rain'] - previous['rain']) / interval)


def commit_restart(*, model, output, rows, hydraulic_seconds, elapsed_seconds):
    seconds = int(model_time(model=model))
    checkpoint = output / 'checkpoints' / f'restart-{seconds:06d}.npz'
    save_checkpoint(model=model, path=checkpoint)
    # Commit marker is published last: a checkpoint without a marker is never restored.
    atomic_json(path=checkpoint.with_suffix('.json'), value=dict(time_s=seconds,
        checkpoint_sha256=file_hash(path=checkpoint), rows=rows,
        hydraulic_seconds=hydraulic_seconds, elapsed_seconds=elapsed_seconds))
    atomic_json(path=output / 'diagnostics/ledger.json', value=rows)


def restore_latest(*, model, output):
    markers = sorted((output / 'checkpoints').glob('restart-*.json'), reverse=True)
    for marker in markers:
        try:
            saved = json.loads(marker.read_text())
            checkpoint = marker.with_suffix('.npz')
            if not checkpoint.is_file() or file_hash(path=checkpoint) != saved['checkpoint_sha256']:
                continue
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        load_checkpoint(model=model, path=checkpoint)
        if model_time(model=model) != saved['time_s']:
            raise ValueError('Restart marker time differs')
        return saved
    return dict(rows=[], hydraulic_seconds=0., elapsed_seconds=0.)


def run_export(*, source, output, identity, device, maximum_idle_utilisation, stop_time):
    require_free_device(device=device, maximum_idle_utilisation=maximum_idle_utilisation)
    started = time.perf_counter()
    atomic_json(path=output / 'metadata/last_execution.json', value=dict(
        started_utc=datetime.now(tz=timezone.utc).isoformat(), device=device,
        source_commit=subprocess.run(args=['git', 'rev-parse', 'HEAD'], check=True,
            capture_output=True, text=True).stdout.strip()))
    model = build_model(source=source, device=device, maximum_dt=identity['maximum_dt_s'])
    progress = restore_latest(model=model, output=output)
    rows = progress['rows']
    hydraulic_seconds = progress['hydraulic_seconds']
    previous = diagnostics(model=model)
    start_time = model_time(model=model)
    end_time = identity['duration_s'] if stop_time is None else stop_time
    if end_time % identity['frame_interval_s'] != 0.:
        raise ValueError('Partial stop time must align with a frame')
    if end_time > identity['duration_s'] or end_time < start_time:
        raise ValueError('Stop time must lie between the saved restart and event end')
    write_frame(output=output, seconds=start_time, depth=model.solver.get_water_depth().copy(),
                active=model.terrain.active)
    targets = sample_targets(duration=end_time, interval=identity['diagnostic_interval_s'])
    for target in targets[targets > start_time]:
        tick = time.perf_counter()
        advance(model=model, end_time=float(target))
        hydraulic_seconds += time.perf_counter() - tick
        current = diagnostics(model=model)
        rows.append(diagnostic_row(current=current, previous=previous))
        previous = current
        if target % identity['frame_interval_s'] == 0. or target == identity['duration_s']:
            write_frame(output=output, seconds=target, depth=model.solver.get_water_depth().copy(),
                        active=model.terrain.active)
        if target % identity['checkpoint_interval_s'] == 0. or target == end_time:
            elapsed = progress['elapsed_seconds'] + time.perf_counter() - started
            commit_restart(model=model, output=output, rows=rows,
                           hydraulic_seconds=hydraulic_seconds, elapsed_seconds=elapsed)
            print(f't={target / 3600:.2f} h; hydraulic={hydraulic_seconds:.1f} s; '
                  f'residual={current["residual_m3"]:.3g} m3', flush=True)
    if end_time == identity['duration_s']:
        atomic_arrays(path=output / 'metadata/native_final.npz', arrays=dict(
            depth=model.solver.get_water_depth(), maximum=model.solver.get_max_water_depth()))
        return validate_dataset(source=source, output=output, identity=identity)
    return dict(status='partial', time_s=model_time(model=model), hydraulic_seconds=hydraulic_seconds)


def validate_dataset(*, source, output, identity):
    expected = np.r_[0., sample_targets(duration=identity['duration_s'], interval=identity['frame_interval_s'])]
    paths = sorted((output / 'frames').glob('depth-*.npz'))
    if len(paths) != len(expected):
        raise ValueError('Missing or unexpected frames')
    with np.load(file=output / 'metadata/terrain.npz', allow_pickle=False) as saved:
        active, dem = saved['active'], saved['dem']
    sampled = np.zeros(shape=active.shape, dtype=np.float64)
    times, frame_manifest = [], []
    for path in paths:
        with np.load(file=path, allow_pickle=False) as saved:
            depth = saved['depth']
            seconds = float(saved['time_s'])
            if path != frame_path(output=output, seconds=seconds):
                raise ValueError('Frame name/time differs')
            if depth.dtype != np.float64 or depth.shape != active.shape or not np.isfinite(depth).all() or np.any(depth < 0.) or np.any(depth[~active] != 0.):
                raise ValueError('Invalid exported field')
            times.append(seconds)
            frame_manifest.append(dict(time_s=seconds, path=str(path.relative_to(output)),
                sha256=file_hash(path=path), bytes=path.stat().st_size))
            np.maximum(sampled, depth, out=sampled)
    np.testing.assert_array_equal(actual=times, desired=expected)
    with np.load(file=output / 'metadata/native_final.npz', allow_pickle=False) as saved:
        final, maximum = saved['depth'], saved['maximum']
    with np.load(file=paths[-1], allow_pickle=False) as saved:
        np.testing.assert_array_equal(actual=saved['depth'], desired=final)
    # Lossless FP64 export; allowance only for roundoff in comparison arithmetic.
    roundoff = 64. * np.finfo(np.float64).eps * max(1., float(maximum.max()))
    if np.max(sampled - maximum) > roundoff:
        raise ArithmeticError('Sampled depths exceed native all-step maxima')
    rows = json.loads((output / 'diagnostics/ledger.json').read_text())
    np.testing.assert_array_equal(actual=[row['time_s'] for row in rows],
        desired=sample_targets(duration=identity['duration_s'], interval=identity['diagnostic_interval_s']))
    reference = json.loads((source / 'results/fv1_record.json').read_text())
    reference_rows = [row for row in reference['ledger'] if row['seconds'] % 60. == 0. and row['seconds'] > 0.]
    if identity['duration_s'] != 259200. or identity['diagnostic_interval_s'] != 60.:
        raise ValueError('Published validation requires the complete event and minute diagnostics')
    ref_out = np.array([row['outflow_m3'] - row['inflow_m3'] for row in reference_rows])
    ref_q = np.diff(np.r_[0., ref_out]) / 60.
    q = np.array([row['discharge_m3_s'] for row in rows])
    q_nrmse = float(np.sqrt(np.mean((q - ref_q)**2)) / max(float(np.max(np.abs(ref_q))), np.finfo(float).eps))
    with np.load(file=source / 'results/fv1_fields.npz', allow_pickle=False) as saved:
        rmse = float(np.sqrt(np.mean((final[active] - saved['depth'][active])**2)))
        max_rmse = float(np.sqrt(np.mean((maximum[active] - saved['maximum'][active])**2)))
    mass_limit = 1.e-8 * max(1., rows[-1]['rain_m3'] + rows[-1]['inflow_m3'])
    export_error = abs(rows[-1]['outflow_m3'] - rows[-1]['inflow_m3'] - ref_out[-1])
    storage_error = abs(rows[-1]['storage_m3'] - reference_rows[-1]['storage_m3'])
    residual = max(abs(row['residual_m3']) for row in rows)
    passed = residual <= mass_limit and q_nrmse <= .05 and rmse <= .001 and max_rmse <= .001 and export_error <= 5. + .02 * abs(ref_out[-1]) and storage_error <= 5. + .02 * reference_rows[-1]['storage_m3']
    report = dict(status='passed' if passed else 'failed', final_time_s=times[-1],
        frames=len(paths), diagnostics=len(rows), final_depth_rmse_m=rmse,
        native_maximum_rmse_m=max_rmse, hydrograph_nrmse=q_nrmse,
        cumulative_export_error_m3=export_error, final_storage_error_m3=storage_error,
        maximum_mass_residual_m3=residual, mass_limit_m3=mass_limit,
        sampled_maximum_deficit_m=float(np.max(maximum - sampled)),
        sampled_maximum_excess_m=float(np.max(sampled - maximum)),
        sampled_maximum_note='Five-minute samples may miss subinterval peaks; equality is not required.',
        finite_active_dem=bool(np.isfinite(dem[active]).all()), crs='EPSG:2193', vertical_datum='NZVD2016',
        dataset_bytes=sum(path.stat().st_size for directory in ('metadata', 'frames', 'checkpoints')
                          for path in (output / directory).rglob('*') if path.is_file()),
        tolerances='Frozen acceptance: mass 1e-8 supplied volume; depth RMSE 1 mm; Q NRMSE 5%; volume 5 m3 + 2%.')
    atomic_json(path=output / 'diagnostics/frame_manifest.json', value=frame_manifest)
    atomic_json(path=output / 'diagnostics/validation.json', value=report)
    if not passed:
        raise ArithmeticError('Validation gates failed; see diagnostics/validation.json')
    return report


def smoke_parity(*, source, example, output, identity, device, maximum_idle_utilisation, duration):
    """Compare full-domain smoke fields against the unchanged public replay."""
    from examples.orton_bradley.run_model import run
    require_free_device(device=device, maximum_idle_utilisation=maximum_idle_utilisation)
    inputs = output / 'smoke/reference_inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for name, target in (('inputs', source / 'inputs'), ('outputs', example / 'outputs')):
        link = inputs / name
        if not link.exists():
            link.symlink_to(target=target.resolve(), target_is_directory=True)
    reference_output = output / 'smoke/reference_replay'
    if not (reference_output / 'restart.npz').exists():
        reference = run(directory=inputs, output=reference_output, method='fv1', device=device,
            duration=duration, sample_interval=identity['diagnostic_interval_s'],
            maximum_dt=identity['maximum_dt_s'])
        del reference
    model = build_model(source=source, device=device, maximum_dt=identity['maximum_dt_s'])
    checkpoint = output / 'checkpoints' / f'restart-{int(duration):06d}.npz'
    load_checkpoint(model=model, path=checkpoint)
    with np.load(file=reference_output / 'fields.npz', allow_pickle=False) as saved:
        depth_error = float(np.max(np.abs(model.solver.get_water_depth() - saved['depth'])))
        maximum_error = float(np.max(np.abs(model.solver.get_max_water_depth() - saved['maximum'])))
    own_rows = json.loads(checkpoint.with_suffix('.json').read_text())['rows']
    reference_rows = json.loads((reference_output / 'ledger.json').read_text())
    np.testing.assert_array_equal(actual=[row['time_s'] for row in own_rows],
                                  desired=[row['time_s'] for row in reference_rows])
    ledger_error = max(abs(own[name] - ref[reference_name])
        for own, ref in zip(own_rows, reference_rows)
        for name, reference_name in (('rain_m3', 'rain'), ('infiltration_m3', 'infiltration'),
            ('outflow_m3', 'boundary_outflow'), ('inflow_m3', 'boundary_inflow'),
            ('storage_m3', 'storage_m3'), ('residual_m3', 'residual_m3')))
    tolerance = 64. * np.finfo(np.float64).eps
    if depth_error > tolerance or maximum_error > tolerance or ledger_error > tolerance * max(1., own_rows[-1]['rain_m3']):
        raise ArithmeticError('Smoke parity with public replay failed')
    report = dict(status='passed', time_s=duration, full_domain=True,
        depth_max_error_m=depth_error, maximum_depth_max_error_m=maximum_error,
        maximum_ledger_error_m3=ledger_error, tolerance='64 FP64 eps, volume-scaled for ledger',
        limitation='Initial rainfall is infiltration-limited; wet geometry is checked separately with retained recession data.')
    atomic_json(path=output / 'diagnostics/smoke_parity.json', value=report)
    return report


if __name__ == "__main__":
    source = Path('outputs/orton_bradley_release_staging')
    example = Path('examples/orton_bradley')
    output = Path('outputs/orton_bradley_visualisation')
    action = os.environ.get('FLOODWARP_VIS_ACTION', 'prepare')  # prepare, smoke, run, validate
    device = 'cuda:0'
    duration = 259200.
    frame_interval = 300.
    diagnostic_interval = 60.
    checkpoint_interval = 3600.
    maximum_dt = 1.
    maximum_idle_utilisation = 10
    smoke_duration = 600.
    identity = prepare_dataset(source=source, example=example, output=output, duration=duration,
        frame_interval=frame_interval, diagnostic_interval=diagnostic_interval,
        checkpoint_interval=checkpoint_interval, maximum_dt=maximum_dt)
    if action == 'run':
        smoke_report = json.loads((output / 'diagnostics/smoke_parity.json').read_text())
        render_report = json.loads((output / 'renders/preflight/render_report.json').read_text())
        if smoke_report['status'] != 'passed' or render_report['status'] != 'pipeline check':
            raise RuntimeError('Complete the hydraulic smoke and render preflight before the full export')
    if action in ('smoke', 'run'):
        result = run_export(source=source, output=output, identity=identity, device=device,
            maximum_idle_utilisation=maximum_idle_utilisation,
            stop_time=smoke_duration if action == 'smoke' else None)
        if action == 'smoke':
            result['parity'] = smoke_parity(source=source, example=example, output=output,
                identity=identity, device=device, maximum_idle_utilisation=maximum_idle_utilisation,
                duration=smoke_duration)
        print(json.dumps(result, indent=2))
    elif action == 'validate':
        print(json.dumps(validate_dataset(source=source, output=output, identity=identity), indent=2))
    elif action == 'prepare':
        print('Inputs verified; metadata prepared. No model started.', flush=True)
    else:
        raise ValueError('Unknown action')
