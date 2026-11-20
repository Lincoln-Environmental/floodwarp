"""Run the controlled analytical and synthetic FV1 Riemann-flux comparisons."""

from pathlib import Path
from time import perf_counter
import json
import platform
import subprocess

import numpy as np
import warp as wp

from floodwarp.full_swe_reference import numpy_step
from floodwarp.solver_factory import create_overland_solver
from validation.fv1_riemann_cases import analytical_domain, cell_averages, coordinates, exact_strip, synthetic_domain


def configure(*, domain, scheme, device, cap, gravity, dry, cfl, mode='dense'):
    ny, nx = domain.bed.shape
    active = np.ones((ny, nx), dtype=bool)
    bx, by = np.zeros((ny, nx + 1), dtype=np.int32), np.zeros((ny + 1, nx), dtype=np.int32)
    sx, sy = np.zeros_like(bx, dtype=float), np.zeros_like(by, dtype=float)
    if domain.stage_x is not None:
        bx[:, [0, -1]], sx[:, [0, -1]] = 3, domain.stage_x
    if domain.stage_y is not None:
        by[[0, -1], :], sy[[0, -1], :] = 3, domain.stage_y
    solver = create_overland_solver(solver_method='fv1', ny=ny, nx=nx, dx=domain.dx, dy=domain.dx,
        bed_elevation=domain.bed, manning_n=domain.manning, active_mask=active, device=device,
        explicit_minimum_dt=0., explicit_maximum_dt=cap, gravity=gravity, dry_depth=dry, cfl=cfl,
        flux_scheme=scheme, execution_mode=mode)
    solver.set_water_depth(depth=domain.initial[..., 0])
    wet = domain.initial[..., 0] > dry
    solver.set_momentum(hu=np.where(wet, domain.initial[..., 1], 0.), hv=np.where(wet, domain.initial[..., 2], 0.))
    solver.set_source_rates(rain=domain.rain)
    solver.set_boundary_type_qx(boundary_type=bx)
    solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=sx)
    solver.set_boundary_stage_qy(stage=sy)
    return solver


def report_json(*, path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


def cap_at_grid(*, dx, scale):
    return float(2.**np.floor(np.log2(dx * scale)))


def rms(*, value):
    return float(np.sqrt(np.mean(np.square(value))))


def errors(*, result, expected, wet_depth, dx, window=None):
    selected = np.ones(result.shape[:2], dtype=bool) if window is None else window
    diff = (result - expected)[selected]
    h, h_exact = result[..., 0][selected], expected[..., 0][selected]
    wet, expected_wet = h > wet_depth, h_exact > wet_depth
    both = wet & expected_wet
    velocity = result[selected][both, 1:] / h[both, None]
    velocity_error = velocity - expected[selected][both, 1:] / h_exact[both, None]
    return dict(depth_rmse_m=rms(value=diff[:, 0]), depth_max_error_m=float(np.abs(diff[:, 0]).max()),
        wet_depth_rmse_m=rms(value=diff[expected_wet, 0]) if expected_wet.any() else None,
        momentum_rmse_m2_s=[rms(value=diff[:, i]) for i in (1, 2)],
        velocity_rmse_m_s=[rms(value=velocity_error[:, i]) for i in (0, 1)] if both.any() else None,
        wet_area_error_m2=float((np.count_nonzero(wet) - np.count_nonzero(expected_wet)) * dx**2),
        wet_symmetric_difference_m2=float(np.count_nonzero(wet != expected_wet) * dx**2),
        depth_min_m=float(h.min()), depth_max_m=float(h.max()),
        depth_range_overshoot_m=float(max(0., h.max() - h_exact.max(), h_exact.min() - h.min())))


def crossing(*, x, profile, level, target):
    left, right = profile[:-1] - level, profile[1:] - level
    found = (left * right <= 0.) & (left != right)
    locations = x[:-1][found] - np.diff(x)[found] * left[found] / (right[found] - left[found])
    return float(locations[np.argmin(np.abs(locations - target))]) if locations.size else None


def feature_errors(*, name, x, numerical, exact, time, gravity, wet_depth):
    _, _, positions = exact_strip(name=name, x=x, time=time, gravity=gravity)
    depth, momentum = numerical[:, 0], numerical[:, 2]
    levels = dict(front=wet_depth, left_front=wet_depth, right_front=wet_depth, rarefaction_head=1.9, shear=.55)
    for key, position in positions.items():
        if key == 'shock':
            sides = exact_strip(name=name, x=np.array([position - .001, position + .001]), time=time, gravity=gravity)[0]
            levels[key] = float(sides[:, 0].mean())
    # Use the same physical threshold for numerical and exact front/head
    # positions. Also preserve each true zero-depth front in the output.
    threshold_positions = positions.copy()
    if name in ('ritter', 'stoker'):
        threshold_positions['rarefaction_head'] = (2. * np.sqrt(2. * gravity) - 3. * np.sqrt(1.9 * gravity)) * time
        if name == 'ritter':
            threshold_positions['front'] = positions['front'] - 3. * np.sqrt(wet_depth * gravity) * time
    if name == 'thacker':
        inset = 120. - np.sqrt(120.**2 - wet_depth / .0001)
        threshold_positions['left_front'] += inset
        threshold_positions['right_front'] -= inset
    features = {}
    for key, position in threshold_positions.items():
        if key not in levels:
            continue
        profile = momentum if key == 'shear' else depth
        location = crossing(x=x, profile=profile, level=levels[key], target=position)
        features[key] = dict(exact_m=float(position), true_feature_m=float(positions[key]),
            simulated_m=location, error_m=None if location is None else float(location - position))
        if key in ('shock', 'shear'):
            sides = exact_strip(name=name, x=np.array([position - .001, position + .001]), time=time, gravity=gravity)[0]
            values = np.sort(sides[:, 2 if key == 'shear' else 0])
            thresholds = values[0] + np.array([.1, .9]) * (values[1] - values[0])
            a, b = [crossing(x=x, profile=profile, level=threshold, target=position) for threshold in thresholds]
            exact_profile = exact[:, 2 if key == 'shear' else 0]
            c, d = [crossing(x=x, profile=exact_profile, level=threshold, target=position) for threshold in thresholds]
            features[key]['numerical_10_to_90_width_m'] = abs(b - a) if a is not None and b is not None else None
            features[key]['exact_cell_average_10_to_90_width_m'] = abs(d - c) if c is not None and d is not None else None
    numerical_tv = np.abs(np.diff(numerical, axis=0)).sum(axis=0)
    exact_tv = np.abs(np.diff(exact, axis=0)).sum(axis=0)
    return features, (numerical_tv - exact_tv).tolist()


def analytical_metrics(*, domain, name, states, times, gravity, wet_depth):
    result, midpoint = states[-1], states[0]
    if name == 'lake':
        return errors(result=result, expected=domain.initial, wet_depth=wet_depth, dx=domain.dx)
    x, _ = coordinates(length=domain.bed.shape[1] * domain.dx, width=domain.bed.shape[0] * domain.dx, dx=domain.dx)
    exact = cell_averages(name=name, x=x, dx=domain.dx, time=times[-1], gravity=gravity)
    window = np.abs(x) <= 120. if name in ('jump', 'jump_offset', 'manufactured') else None
    metrics = errors(result=result, expected=exact, wet_depth=wet_depth, dx=domain.dx, window=window)
    middle = result.shape[0] // 2
    features, variation = feature_errors(name=name, x=x[middle], numerical=result[middle], exact=exact[middle],
        time=times[-1], gravity=gravity, wet_depth=wet_depth)
    first, _ = feature_errors(name=name, x=x[middle], numerical=midpoint[middle], exact=exact[middle],
        time=times[0], gravity=gravity, wet_depth=wet_depth)
    for key, feature in features.items():
        initial_error, final_error = first[key]['error_m'], feature['error_m']
        feature['midpoint'] = first[key]
        feature['propagation_error_m'] = None if initial_error is None or final_error is None else final_error - initial_error
    metrics.update(features=features, total_variation_excess=variation,
                   expected_volume_difference_m3=float((result[..., 0] - exact[..., 0]).sum() * domain.dx**2))
    if name == 'stoker':
        plateau = (x > 28.) & (x < 106.)
        difference = (result - exact)[plateau].mean(axis=0)
        metrics['intermediate_state_mean_error'] = difference.tolist()
    return metrics


def trial(*, solver, duration, device, track=False, snapshots=2, graph=True):
    stats, states = [], []
    times = np.linspace(duration / snapshots, duration, snapshots)
    for time in times:
        if device.startswith('cuda'):
            wp.synchronize_device(device=device)
        began = perf_counter()
        result = solver.run_device_loop(duration=float(time - solver.time), use_cuda_graph=graph, track_flux=track)
        if device.startswith('cuda'):
            wp.synchronize_device(device=device)
        result['wall_seconds'] = perf_counter() - began
        stats.append(result)
        states.append(solver.get_conservative_state())
    balance = solver.cumulative_accounting()
    balance['normalised_residual'] = abs(balance['residual']) / max(1., solver.initial_volume + balance['rain'] + balance['boundary_inflow'])
    return dict(stats=stats, states=states, times=times, balance=balance)


def checked_configuration(*, domain, scheme, device, cap, controls):
    return configure(domain=domain, scheme=scheme, device=device, cap=cap,
        gravity=controls['gravity'], dry=controls['dry'], cfl=controls['cfl'])


def face_outcomes(*, domain, scheme, device, cap, controls, baseline, snapshots=2):
    solver = checked_configuration(domain=domain, scheme=scheme, device=device, cap=cap, controls=controls)
    observed = trial(solver=solver, duration=domain.duration, device=device, track=True, snapshots=snapshots)
    np.testing.assert_array_equal(actual=observed['states'][-1], desired=baseline['states'][-1])
    return {key: sum(row['flux_outcomes'][key] for row in observed['stats']) for key in observed['stats'][0]['flux_outcomes']}


def analytical_runs(*, names, schemes, spacings, length, width, policies, device, controls, output):
    rows = []
    for name in names:
        for dx in spacings:
            domain = analytical_domain(name=name, dx=dx, length=length, width=width, gravity=controls['gravity'])
            for policy in policies:
                cap = cap_at_grid(dx=dx, scale=controls['dt_scale']) if policy == 'common' else controls['adaptive_cap']
                for scheme in schemes:
                    row = dict(case=name, dx_m=dx, scheme=scheme, policy=policy, cap_s=cap)
                    try:
                        solver = checked_configuration(domain=domain, scheme=scheme, cap=cap, device=device, controls=controls)
                        result = trial(solver=solver, duration=domain.duration, device=device)
                        stats = result['stats']
                        dt_equal = all(item['dt_min_s'] == item['dt_max_s'] == cap for item in stats)
                        if policy == 'common' and not dt_equal:
                            raise ArithmeticError(f'CFL reduced the common timestep: {stats}')
                        row.update(status='completed', shape=domain.bed.shape, stats=stats, balance=result['balance'],
                            **analytical_metrics(domain=domain, name=name, states=result['states'], times=result['times'],
                                gravity=controls['gravity'], wet_depth=controls['wet_depth']))
                        if scheme in ('roe', 'hllc'):
                            row['flux_outcomes'] = face_outcomes(domain=domain, scheme=scheme, device=device, cap=cap,
                                controls=controls, baseline=result)
                    except (ArithmeticError, AssertionError) as failure:
                        row.update(status='failed', failure=str(failure))
                    rows.append(row)
            print(f'Analytical {name}, {dx:g} m', flush=True)
            report_json(path=output / 'analytical.json', content=dict(controls=controls, rows=rows))


def restrict(*, state, coarse_shape):
    ny, nx = coarse_shape
    return state.reshape(ny, state.shape[0] // ny, nx, state.shape[1] // nx, 3).mean(axis=(1, 3))


def numpy_reference(*, domain, scheme, dt, gravity, dry):
    shape = domain.bed.shape
    state = domain.initial.copy()
    state[..., 1:] = np.where((state[..., 0] > dry)[..., None], state[..., 1:], 0.)
    bx = np.zeros((shape[0], shape[1] + 1), dtype=int)
    by = np.zeros((shape[0] + 1, shape[1]), dtype=int)
    steps = int(round(domain.duration / dt))
    if steps * dt != domain.duration:
        raise ValueError('NumPy reference requires an exact uniform partition')
    kwargs = dict(bed=domain.bed, active=np.ones(shape, dtype=bool), manning=np.full(shape, domain.manning),
        dx=domain.dx, dy=domain.dx, gravity=gravity, dry_depth=dry, dt=dt, boundary_type_x=bx,
        boundary_type_y=by, boundary_stage_x=np.zeros_like(bx), boundary_stage_y=np.zeros_like(by),
        rain_rate=domain.rain, infiltration_rate=0., flux_scheme=scheme)
    for _ in range(steps):
        state, _ = numpy_step(state=state, **kwargs)
    if not np.isfinite(state).all() or np.min(state[..., 0]) < 0.:
        raise ArithmeticError(f'Invalid NumPy {scheme} reference state')
    residual = (state[..., 0] - domain.initial[..., 0]).sum() - np.broadcast_to(domain.rain, shape).sum() * domain.duration
    return state, float(residual * domain.dx**2)


def grid_runs(*, names, schemes, spacings, reference_spacings, length, width, device, controls, output):
    rows, checks = [], []
    for name in names:
        candidates, references = {}, {}
        for dx in (*spacings, *reference_spacings):
            domain = synthetic_domain(name=name, dx=dx, length=length, width=width)
            cap = cap_at_grid(dx=dx, scale=controls['dt_scale'])
            selected = schemes if dx in spacings else ('contact_hll', 'roe')
            for scheme in selected:
                for refinement in (1, 2):
                    dt = cap / refinement
                    solver = checked_configuration(domain=domain, scheme=scheme, cap=dt, device=device, controls=controls)
                    complete = trial(solver=solver, duration=domain.duration, device=device, snapshots=1)
                    row = dict(case=name, dx_m=dx, scheme=scheme, dt_s=dt, refinement=refinement,
                               stats=complete['stats'][0], balance=complete['balance'])
                    if not row['stats']['dt_min_s'] == row['stats']['dt_max_s'] == dt:
                        raise ArithmeticError(f'Synthetic {name} CFL reduced the uniform timestep: {row}')
                    if scheme in ('roe', 'hllc') and dx in spacings and refinement == 1:
                        row['flux_outcomes'] = face_outcomes(domain=domain, scheme=scheme, device=device, cap=dt,
                            controls=controls, baseline=complete, snapshots=1)
                    candidates[dx, scheme, refinement] = (complete['states'][0], row)
            print(f'Synthetic {name}, {dx:g} m', flush=True)
        fine = reference_spacings[-1]
        mid = reference_spacings[0]
        for scheme in ('contact_hll', 'roe'):
            finest, row = candidates[fine, scheme, 2]
            smaller_dt_gap = rms(value=finest[..., 0] - candidates[fine, scheme, 1][0][..., 0])
            previous = candidates[mid, scheme, 2][0]
            spatial_gap = rms(value=restrict(state=finest, coarse_shape=previous.shape[:2])[..., 0] - previous[..., 0])
            same_dt_fine = candidates[fine, scheme, 1][0]
            same_dt_gap = rms(value=restrict(state=same_dt_fine, coarse_shape=previous.shape[:2])[..., 0] - previous[..., 0])
            coarse = candidates[spacings[-1], scheme, 2][0]
            previous_gap = rms(value=restrict(state=previous, coarse_shape=coarse.shape[:2])[..., 0] - coarse[..., 0])
            checks.append(dict(case=name, scheme=scheme, coarse_to_mid_rmse_m=previous_gap,
                mid_to_fine_rmse_m=spatial_gap, same_dt_mid_to_fine_rmse_m=same_dt_gap,
                finest_dt_halving_rmse_m=smaller_dt_gap,
                fine_monotone=spatial_gap < previous_gap))
            references[scheme] = finest
        independent_domain = synthetic_domain(name=name, dx=mid, length=length, width=width)
        independent_dt = cap_at_grid(dx=mid, scale=controls['dt_scale']) / 2.
        independent, residual = numpy_reference(domain=independent_domain, scheme='legacy_hll', dt=independent_dt,
                                                 gravity=controls['gravity'], dry=controls['dry'])
        independent_coarse_domain = synthetic_domain(name=name, dx=spacings[-1], length=length, width=width)
        independent_coarse, _ = numpy_reference(domain=independent_coarse_domain, scheme='legacy_hll', dt=independent_dt,
                                                 gravity=controls['gravity'], dry=controls['dry'])
        independent_shape = independent.shape[:2]
        disagreement = rms(value=references['contact_hll'][..., 0] - references['roe'][..., 0])
        checks.append(dict(case=name, cross_formulation_finest_rmse_m=disagreement,
            independent_dx_m=mid, independent_dt_s=independent_dt, independent_scheme='NumPy legacy_hll',
            independent_residual_m3=residual,
            independent_spatial_gap_rmse_m=rms(value=independent_coarse[..., 0] - restrict(state=independent, coarse_shape=independent_coarse.shape[:2])[..., 0]),
            cross_formulation_by_grid_rmse_m={str(dx): rms(value=candidates[dx, 'contact_hll', 2][0][..., 0]
                - candidates[dx, 'roe', 2][0][..., 0]) for dx in (*spacings, *reference_spacings)},
            independent_to_contact_rmse_m=rms(value=independent[..., 0] - restrict(state=references['contact_hll'], coarse_shape=independent_shape)[..., 0]),
            independent_to_roe_rmse_m=rms(value=independent[..., 0] - restrict(state=references['roe'], coarse_shape=independent_shape)[..., 0])))
        for (dx, scheme, refinement), (state, row) in candidates.items():
            if dx in spacings:
                row['deep_reference'] = {name: errors(result=state, expected=restrict(state=reference, coarse_shape=state.shape[:2]),
                    wet_depth=controls['wet_depth'], dx=dx) for name, reference in references.items()}
            if refinement == 2:
                row['dt_halving_depth_rmse_m'] = rms(value=state[..., 0] - candidates[dx, scheme, 1][0][..., 0])
            rows.append(row)
        report_json(path=output / 'grid.json', content=dict(controls=controls, spacings_m=spacings,
            reference_spacings_m=reference_spacings, rows=rows, reference_checks=checks))


def run(*, output, device, schemes, controls, analytical_names, analytical_spacings, analytical_length,
        analytical_width, policies, grid_names, grid_spacings, reference_spacings, grid_length, grid_width):
    report_json(path=output / 'environment.json', content=dict(python=platform.python_version(), numpy=np.__version__,
        warp=wp.__version__, system=platform.platform(), device=str(wp.get_device(device)),
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
        if device.startswith('cuda') else None))
    if analytical_names:
        analytical_runs(names=analytical_names, schemes=schemes, spacings=analytical_spacings, length=analytical_length,
                        width=analytical_width, policies=policies, device=device, controls=controls, output=output)
    if grid_names:
        grid_runs(names=grid_names, schemes=schemes, spacings=grid_spacings, reference_spacings=reference_spacings,
                  length=grid_length, width=grid_width, device=device, controls=controls, output=output)


if __name__ == "__main__":
    output = Path('validation/fv1_riemann_review')
    device = 'cuda:0'
    schemes = ('legacy_hll', 'contact_hll', 'star_contact_hll', 'roe', 'hllc')
    controls = dict(gravity=9.81, dry=.0001, wet_depth=.01, cfl=.4, dt_scale=.015625, adaptive_cap=1.)
    analytical_names = ('ritter', 'stoker', 'lake', 'thacker', 'manufactured', 'shear', 'stationary_shear', 'jump', 'jump_offset')
    analytical_spacings = (20., 10., 5., 2.5)
    analytical_length, analytical_width = 640., 40.
    policies = ('common', 'adaptive')
    grid_names = ('plane', 'irregular', 'steep', 'expansion', 'rainfall')
    grid_spacings, reference_spacings = (20., 10., 5., 2.5), (1.25, .625)
    grid_length, grid_width = 320., 160.
    run(output=output, device=device, schemes=schemes, controls=controls, analytical_names=analytical_names,
        analytical_spacings=analytical_spacings, analytical_length=analytical_length, analytical_width=analytical_width,
        policies=policies, grid_names=grid_names, grid_spacings=grid_spacings, reference_spacings=reference_spacings,
        grid_length=grid_length, grid_width=grid_width)
