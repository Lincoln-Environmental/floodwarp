"""Matched native FV1 flux replays on the installation crop and a wet event window."""

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import warp as wp

from examples.run_purau import load_inputs
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.model import create_model
from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_transport import InteriorTransport
from validation.sparse_fv1_benchmark import digest, select_checkpoints


def crop_solver(*, scheme, device, directory, maximum_dt):
    terrain, bx, by, forcing = load_inputs(directory=directory)
    return create_model(terrain=terrain, solver_method='fv1', manning_n=.035, infiltration=0.,
        rainfall_series=forcing, device=device, boundary_qx=bx, boundary_qy=by,
        explicit_maximum_dt=maximum_dt, flux_scheme=scheme).solver


def checkpoint_solver(*, scheme, device, path):
    # Validate the archived original first. Changing the flux is an explicit
    # initial-condition experiment within a common event window.
    with np.load(file=path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        config = metadata['configuration']
        shape = config['ny'], config['nx']
        geometry = dict(bed_elevation=saved['bed'].reshape(shape),
            manning_n=saved['manning'].reshape(shape), active_mask=saved['active'].reshape(shape))
    original = FullSWESolver(device=device, **geometry, **config)
    original.load_checkpoint(path=path)
    initial = original.capture_state()
    del original
    config['flux_scheme'] = scheme
    solver = FullSWESolver(device=device, **geometry, **config)
    initial['configuration'] = solver._configuration()
    if scheme == 'hllc':
        # This is an explicit cross-flux initial condition from the validated archive.
        initial['hllc_revision'] = 2
    initial['control'] = np.zeros(solver.ctrl.size)
    initial['arrays']['maximum'] = initial['arrays']['q'][:, 0].copy()
    solver.restore_state(state=initial)
    return solver


def jump_candidates(*, state, active, threshold, gravity):
    h, qx, qy = np.moveaxis(state, -1, 0)
    valid = active & (h > threshold)
    froude = np.divide(np.hypot(qx, qy), h * np.sqrt(gravity * h), out=np.zeros_like(h), where=valid)
    oriented = 0
    for axis, momentum in ((0, qy), (1, qx)):
        left, right = [slice(None)] * 2, [slice(None)] * 2
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(left), tuple(right)
        positive = (momentum[a] > 0.) & (momentum[b] > 0.) & (froude[a] > 1.15) & (froude[b] < .85) & (h[b] > 1.1 * h[a])
        negative = (momentum[a] < 0.) & (momentum[b] < 0.) & (froude[b] > 1.15) & (froude[a] < .85) & (h[a] > 1.1 * h[b])
        oriented += int(np.count_nonzero(valid[a] & valid[b] & (positive | negative)))
    return oriented


def window(*, solver, device, duration, interval, threshold):
    start = solver.time
    initial = solver.cumulative_accounting()
    previous = initial['boundary_outflow']
    hydrograph, timings = [], []
    for offset in np.arange(interval, duration + interval, interval):
        wp.synchronize_device(device=device)
        began = perf_counter()
        stats = solver.run_device_loop(duration=float(start + offset - solver.time), use_cuda_graph=True)
        wp.synchronize_device(device=device)
        stats['wall_seconds'] = perf_counter() - began
        accounting = solver.cumulative_accounting()
        flux = (accounting['boundary_outflow'] - previous) / interval
        hydrograph.append((float(offset - interval / 2.), flux))
        timings.append(stats)
        previous = accounting['boundary_outflow']
    state = solver.get_conservative_state()
    active = solver._active
    maximum = solver.get_max_water_depth()
    wet = active & (state[..., 0] > threshold)
    speed = np.hypot(state[..., 1][wet], state[..., 2][wet]) / state[..., 0][wet]
    changes = {key: accounting[key] - initial[key] for key in accounting}
    hydro = np.asarray(hydrograph)
    peak = int(np.argmax(hydro[:, 1]))
    row = dict(start_s=start, duration_s=duration, shape=state.shape[:2], active_cells=int(active.sum()),
        timings=timings, ledger_delta_m3=changes, initial_storage_m3=initial['storage'],
        maximum_depth_m=float(maximum[active].max()), peak_wet_area_m2=float(np.count_nonzero((maximum > threshold) & active) * solver.dx * solver.dy),
        hydrograph_peak_m3_s=float(hydro[peak, 1]), peak_interval_centre_s=float(hydro[peak, 0]), hydrograph=hydrograph,
        speed_p95_m_s=float(np.percentile(speed, 95.)) if speed.size else None,
        jump_candidate_faces=jump_candidates(state=state, active=active, threshold=threshold, gravity=solver.gravity))
    return row, state, maximum


def compare(*, output, schemes, device, checkpoint_directory, checkpoint_hour, crop_directory, crop_duration,
            crop_interval, crop_cap, event_duration, event_interval, threshold, diagnostic_settings):
    has_event = checkpoint_directory.is_dir()
    checkpoint = select_checkpoints(directory=checkpoint_directory, hours=(checkpoint_hour,))[0] if has_event else None
    cases = [('installation_crop', crop_duration, crop_interval)]
    if has_event:
        cases.append(('orton_bradley_event_window', event_duration, event_interval))
    else:
        print(f'No retained Orton Bradley checkpoint directory at {checkpoint_directory}', flush=True)
    rows = []
    for case, duration, interval in cases:
        if duration % interval:
            raise ValueError('Equal sampling intervals must partition the catchment window')
        for scheme in schemes:
            if case == 'installation_crop':
                solver = crop_solver(scheme=scheme, device=device, directory=crop_directory, maximum_dt=crop_cap)
            else:
                solver = checkpoint_solver(scheme=scheme, device=device, path=checkpoint)
            try:
                row, final, maximum = window(solver=solver, device=device, duration=duration, interval=interval, threshold=threshold)
                row.update(case=case, scheme=scheme, status='completed')
                if scheme == 'contact_hll':
                    base, base_maximum = final, maximum
                    baseline_peak = row['hydrograph_peak_m3_s']
                    baseline_peak_time = row['peak_interval_centre_s']
                    baseline_volume = row['ledger_delta_m3']['boundary_outflow']
                active = solver._active
                row['relative_to_contact_hll'] = errors(result=final, expected=base, wet_depth=threshold, dx=solver.dx, window=active)
                delta = (maximum - base_maximum)[active]
                row['relative_to_contact_hll'].update(maximum_depth_rmse_m=float(np.sqrt(np.mean(delta**2))),
                    maximum_depth_difference_percentiles_m=np.percentile(delta, (0., 5., 50., 95., 100.)).tolist(),
                    hydrograph_peak_difference_m3_s=row['hydrograph_peak_m3_s'] - baseline_peak,
                    hydrograph_peak_interval_shift_s=row['peak_interval_centre_s'] - baseline_peak_time,
                    outlet_volume_difference_m3=row['ledger_delta_m3']['boundary_outflow'] - baseline_volume)
                if scheme in ('roe', 'hllc'):
                    if case == 'installation_crop':
                        tracked = crop_solver(scheme=scheme, device=device, directory=crop_directory, maximum_dt=crop_cap)
                    else:
                        tracked = checkpoint_solver(scheme=scheme, device=device, path=checkpoint)
                    probe = InteriorTransport(solver=tracked, **diagnostic_settings)
                    counts = dict(faces=0, shallow_hll=0, inadmissible_hll=0, entropy_corrected=0)
                    for _ in row['timings']:
                        stats = tracked.run_device_loop(duration=interval, use_cuda_graph=True, track_flux=True)
                        for key in counts:
                            counts[key] += stats['flux_outcomes'][key]
                    np.testing.assert_array_equal(actual=tracked.get_conservative_state(), desired=final)
                    row['interior_transport'] = probe.results()
                    del probe
                    row['flux_outcomes'] = counts
                    del tracked
            except (ArithmeticError, AssertionError) as failure:
                row = dict(case=case, scheme=scheme, status='failed', failure=str(failure))
                if scheme == 'contact_hll':
                    raise
            rows.append(row)
            del solver
            print(case, scheme, row['status'], flush=True)
            report_json(path=output, content=dict(wet_threshold_m=threshold, checkpoint=str(checkpoint) if has_event else None,
                                                rows=rows))
    report_json(path=output, content=dict(wet_threshold_m=threshold, checkpoint=str(checkpoint) if has_event else None,
        checkpoint_sha256=digest(path=checkpoint) if has_event else None,
        crop_files={name: digest(path=crop_directory / name) for name in ('terrain.npy', 'active.npy', 'boundaries.npz', 'rainfall.csv')},
        rows=rows))


if __name__ == "__main__":
    directory = Path('validation/fv1_riemann_review/corrected')
    device = 'cuda:0'
    schemes = ('contact_hll', 'legacy_hll', 'star_contact_hll', 'roe', 'hllc')
    checkpoint_directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    checkpoint_hour = 25
    crop_directory = Path('examples/purau')
    crop_duration, crop_interval, crop_cap = 120., 10., .5
    windows = ((600., 'catchment_600s.json'), (1200., 'catchment.json'))
    event_interval = 60.
    threshold = .01
    diagnostic_settings = dict(edges=(.001, .01, .1, 1.), wet=.01, channel_depth=.1, channel_discharge=.05)
    for event_duration, filename in windows:
        compare(output=directory / filename, schemes=schemes, device=device, checkpoint_directory=checkpoint_directory,
            checkpoint_hour=checkpoint_hour, crop_directory=crop_directory, crop_duration=crop_duration,
            crop_interval=crop_interval, crop_cap=crop_cap, event_duration=event_duration, event_interval=event_interval,
            threshold=threshold, diagnostic_settings=diagnostic_settings)
