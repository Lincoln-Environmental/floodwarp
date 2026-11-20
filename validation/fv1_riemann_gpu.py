"""Repeat warmed FV1 GPU timings and attribute CUDA kernel costs separately."""

from pathlib import Path
from time import perf_counter
import gc

import numpy as np
import warp as wp

from validation.fv1_riemann import (cap_at_grid, checked_configuration, errors,
                                    report_json, trial)
from validation.fv1_riemann_cases import analytical_domain, cell_averages, coordinates, synthetic_domain
from validation.gpu_benchmark import reset_device_allocation_peak
from validation.sparse_fv1_benchmark import memory_layers


def timed_graph(*, solver, device, duration):
    wp.synchronize_device(device=device)
    began = perf_counter()
    stats = solver.run_device_loop(duration=duration, use_cuda_graph=True)
    wp.synchronize_device(device=device)
    stats['wall_seconds'] = perf_counter() - began
    return stats


def make_solver(*, domain, scheme, policy, controls, device):
    cap = cap_at_grid(dx=domain.dx, scale=controls['dt_scale']) if policy == 'common' else controls['adaptive_cap']
    return checked_configuration(domain=domain, scheme=scheme, cap=cap, device=device, controls=controls)


def repetitions(*, domain, schemes, policy, controls, device, repeats, exact):
    rows = {scheme: dict(scheme=scheme, policy=policy, dx_m=domain.dx, shape=domain.bed.shape, repetitions=[]) for scheme in schemes}
    for scheme in schemes:
        solver = make_solver(domain=domain, scheme=scheme, policy=policy, controls=controls, device=device)
        timed_graph(solver=solver, device=device, duration=domain.duration)
    del solver
    for repeat in range(repeats):
        offset = repeat % len(schemes)
        for scheme in schemes[offset:] + schemes[:offset]:
            solver = make_solver(domain=domain, scheme=scheme, policy=policy, controls=controls, device=device)
            initial = solver.capture_state()
            first = timed_graph(solver=solver, device=device, duration=domain.duration)
            if repeat == 0:
                if exact is not None:
                    rows[scheme]['accuracy'] = errors(result=solver.get_conservative_state(), expected=exact,
                        wet_depth=controls['wet_depth'], dx=domain.dx)
                rows[scheme]['balance'] = solver.cumulative_accounting()
            graph = solver._graph
            # Restore only fields mutated during stepping. The forcing arrays
            # and graph allocations retain their original identities.
            for key, field in solver._fields.items():
                field.assign(initial['arrays'][key])
            solver.time, solver.steps = initial['time'], initial['steps']
            assert solver._graph is graph
            stats = timed_graph(solver=solver, device=device, duration=domain.duration)
            if policy == 'common' and not stats['dt_min_s'] == stats['dt_max_s'] == solver.maximum_dt:
                raise ArithmeticError('CFL reduced the controlled GPU timestep')
            stats['cell_steps_per_second'] = domain.bed.size * stats['steps'] / stats['wall_seconds']
            stats['wall_with_capture_seconds'] = first['wall_seconds']
            stats['capture_seconds'] = first['graph_capture_seconds']
            rows[scheme]['repetitions'].append(stats)
    del solver
    for scheme, row in rows.items():
        gc.collect()
        baseline = reset_device_allocation_peak(device=device)
        solver = make_solver(domain=domain, scheme=scheme, policy=policy, controls=controls, device=device)
        timed_graph(solver=solver, device=device, duration=domain.duration)
        row['peak_live_cuda_pool_bytes'] = int(wp.get_mempool_used_mem_high(device=device) - baseline)
        row['persistent_array_bytes'] = memory_layers(solver=solver)['total']
        if scheme in ('roe', 'hllc'):
            cap = solver.maximum_dt
            tracked = checked_configuration(domain=domain, scheme=scheme, cap=cap, device=device, controls=controls)
            observed = trial(solver=tracked, duration=domain.duration, device=device, track=True, snapshots=1)
            np.testing.assert_array_equal(actual=tracked.get_conservative_state(), desired=solver.get_conservative_state())
            row['flux_outcomes'] = observed['stats'][0]['flux_outcomes']
            del tracked
        del solver
        samples = row['repetitions']
        for key in ('wall_seconds', 'wall_with_capture_seconds', 'loop_seconds', 'cell_steps_per_second'):
            row['median_' + key] = float(np.median([trial[key] for trial in samples]))
        row['wall_seconds_quartiles'] = np.quantile([trial['wall_seconds'] for trial in samples], (.25, .75)).tolist()
    return list(rows.values())


def profile(*, schemes, domain, controls, repeats, device):
    rows = []
    for scheme in schemes:
        kernels = []
        for _ in range(repeats):
            solver = make_solver(domain=domain, scheme=scheme, policy='common', controls=controls, device=device)
            wp.synchronize_device(device=device)
            with wp.ScopedTimer(name='FV1 flux comparison', cuda_filter=wp.TIMING_KERNEL, print=False) as timing:
                stats = solver.run_device_loop(duration=domain.duration, use_cuda_graph=False)
            groups = {}
            for event in timing.timing_results:
                name = event.name.split('_cuda')[0]
                groups[name] = groups.get(name, 0.) + event.elapsed * 1000. / stats['steps']
            kernels.append(dict(steps=stats['steps'], kernel_us_per_step=groups))
        medians = {key: float(np.median([sample['kernel_us_per_step'][key] for sample in kernels]))
                   for key in kernels[0]['kernel_us_per_step']}
        rows.append(dict(scheme=scheme, repeats=kernels, median_kernel_us_per_step=medians,
            median_all_kernel_us_per_step=float(np.median([sum(sample['kernel_us_per_step'].values()) for sample in kernels]))))
    return rows


def run(*, output, schemes, spacings, controls, repeats, device, length, width, profile_dx, profile_repeats):
    rows = []
    for dx in spacings:
        name = 'stoker'
        domain = analytical_domain(name=name, dx=dx, length=length, width=width, gravity=controls['gravity'])
        x, _ = coordinates(length=length, width=width, dx=dx)
        exact = cell_averages(name=name, x=x, dx=dx, time=domain.duration, gravity=controls['gravity'])
        for policy in ('common', 'adaptive'):
            rows.extend(repetitions(domain=domain, schemes=schemes, policy=policy, controls=controls,
                                    device=device, repeats=repeats, exact=exact))
        print(f'GPU cost, Stoker {dx:g} m', flush=True)
        report_json(path=output, content=dict(case=name, rows=rows))
    domain = synthetic_domain(name='expansion', dx=profile_dx, length=length, width=width)
    kernels = profile(schemes=schemes, domain=domain, controls=controls, repeats=profile_repeats, device=device)
    report_json(path=output, content=dict(case=name, controls=controls, rows=rows, kernel_case='expansion',
                kernel_shape=domain.bed.shape, kernel_profile=kernels))


if __name__ == "__main__":
    output = Path('validation/fv1_riemann_review/corrected/gpu.json')
    device = 'cuda:0'
    schemes = ('legacy_hll', 'contact_hll', 'star_contact_hll', 'roe', 'hllc')
    controls = dict(gravity=9.81, dry=.0001, wet_depth=.01, cfl=.4, dt_scale=.015625, adaptive_cap=1.)
    spacings = (20., 10., 5., 2.5)
    length, width = 640., 320.
    repeats, profile_repeats = 7, 3
    profile_dx = 5.
    run(output=output, schemes=schemes, spacings=spacings, controls=controls, repeats=repeats, device=device,
        length=length, width=width, profile_dx=profile_dx, profile_repeats=profile_repeats)
