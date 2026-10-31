"""Bounded static block FV1 accuracy/runtime controls against dense FV1.

This benchmark measures spatial approximation separately from exact sparse
execution. All thresholds/budgets were recorded before these measurements.
"""

from pathlib import Path
from time import perf_counter
import gc
import hashlib
import json

import numpy as np
import warp as wp

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.static_fv1 import StaticBlockFV1Solver
from floodwarp.static_fv1_mesh import restrict_state
from validation.benchmark_restart import sample_targets


def build_control(*, name, shape, device, options):
    row, column = np.indices(dimensions=shape)
    bed = np.zeros(shape=shape)
    depth = np.full(shape=shape, fill_value=options['depth'])
    protected = np.zeros(shape=shape, dtype=bool)
    if name == 'lake':
        bed = options['bed_slope'] * (row + column)
        bed[:, shape[1] // 2:] += options['bed_step']
        depth = options['lake_stage'] - bed
        protected[:, shape[1] // 2 - 1:shape[1] // 2 + 1] = True
    elif name == 'front':
        depth[:, shape[1] // 2:] = 0.
        protected[:, shape[1] // 2 - options['front_protection_cells']:
                     shape[1] // 2 + options['front_protection_cells']] = True
    elif name in ('ponding', 'infiltrating', 'heterogeneous'):
        depth.fill(0.)
    elif name == 'channel':
        bed.fill(options['barrier_height'])
        bed[:, shape[1] // 2 - 1:shape[1] // 2 + 1] = 0.
        depth = np.maximum(options['depth'] - bed, 0.)
        protected[:, shape[1] // 2 - 2:shape[1] // 2 + 2] = True
    solver = FullSWESolver(ny=shape[0], nx=shape[1], dx=options['spacing'], dy=options['spacing'],
        bed_elevation=bed, manning_n=options['manning'], active_mask=np.ones(shape=shape, dtype=bool),
        device=device, maximum_dt=options['maximum_dt'])
    solver.set_water_depth(depth=depth)
    if name == 'front':
        solver.set_momentum(hu=options['velocity_x'] * depth, hv=options['velocity_y'] * depth)
    if name == 'channel':
        solver.set_momentum(hu=np.zeros_like(depth), hv=options['velocity_y'] * depth)
        by = np.zeros(shape=(shape[0] + 1, shape[1]), dtype=np.int32)
        by[-1, shape[1] // 2 - 1:shape[1] // 2 + 1] = 2
        solver.set_boundary_type_qy(boundary_type=by)
    if name in ('ponding', 'infiltrating', 'heterogeneous'):
        solver.set_uniform_rainfall_rate(rainfall_rate=options['rain'])
        capacity = options['high_infiltration'] if name == 'infiltrating' else options['low_infiltration']
        if name == 'heterogeneous':
            capacity = np.where(column % options['soil_period_cells'] == 0,
                               options['high_infiltration'], options['low_infiltration'])
        solver.set_exponential_infiltration(f0=capacity, fd=options['decay_depth'], conductivity=capacity)
    return solver, protected


def time_run(*, solver, duration, graph):
    began = perf_counter()
    stats = solver.run_device_loop(duration=duration, use_cuda_graph=graph)
    wp.synchronize_device(device=solver.device)
    stats['end_to_end_seconds'] = perf_counter() - began
    stats['seconds_per_step'] = stats['loop_seconds'] / stats['steps']
    return stats


def sampled_run(*, solver, duration, interval, graph):
    rows = []
    initial_time, previous_time = solver.time, solver.time
    initial_balance = solver.cumulative_accounting()
    previous_volume = initial_balance['boundary_outflow'] - initial_balance['boundary_inflow']
    for elapsed in sample_targets(duration=duration, interval=interval):
        target = initial_time + float(elapsed)
        solver.run_device_loop(duration=target - solver.time, use_cuda_graph=graph)
        balance = solver.cumulative_accounting()
        volume = balance['boundary_outflow'] - balance['boundary_inflow']
        rows.append(dict(time=solver.time, discharge=(volume - previous_volume) / (solver.time - previous_time), accounting=balance))
        previous_volume = volume
        previous_time = solver.time
    return rows


def accuracy(*, solver, reference, reference_state, rows, reference_rows, thresholds, wet_threshold):
    current = solver.get_conservative_state()
    target = reference_state['arrays']['q'].reshape(current.shape)
    error = current[..., 0] - target[..., 0]
    wet = (current[..., 0] >= wet_threshold) | (target[..., 0] >= wet_threshold)
    maximum = solver.get_max_water_depth()
    expected_maximum = reference_state['arrays']['maximum'].reshape(maximum.shape)
    norm = float(np.linalg.norm(target[..., 0]))
    wet_norm = float(np.linalg.norm(target[..., 0][wet]))
    maximum_norm = float(np.linalg.norm(expected_maximum))
    maximum_error = maximum - expected_maximum
    extent = {}
    for threshold in thresholds:
        a, b = maximum >= threshold, expected_maximum >= threshold
        union = a | b
        extent[str(threshold)] = dict(iou=float(np.count_nonzero(a & b) / union.sum()) if union.any() else 1.,
            mismatch_fraction=float(np.mean(a != b)))
    q = np.asarray([row['discharge'] for row in rows])
    q_ref = np.asarray([row['discharge'] for row in reference_rows])
    peak = float(q_ref.max())
    ledger, ref_ledger = rows[-1]['accounting'], reference_rows[-1]['accounting']
    return dict(depth_rmse_m=float(np.sqrt(np.mean(error**2))),
        wet_union_rmse_m=float(np.sqrt(np.mean(error[wet]**2))) if wet.any() else 0.,
        wet_union_relative_l2=float(np.linalg.norm(error[wet]) / wet_norm) if wet_norm > 0. else None,
        depth_relative_l2=float(np.linalg.norm(error) / norm) if norm > 0. else None,
        maximum_depth_rmse_m=float(np.sqrt(np.mean(maximum_error**2))),
        maximum_depth_relative_l2=float(np.linalg.norm(maximum_error) / maximum_norm) if maximum_norm > 0. else None,
        maximum_depth_absolute_error_m=float(np.max(np.abs(maximum_error))), maximum_event_inundation=extent,
        momentum_maximum_absolute_error=float(np.max(np.abs(current[..., 1:] - target[..., 1:]))),
        minimum_depth_m=float(current[..., 0].min()),
        hydrograph_nrmse=float(np.sqrt(np.mean((q - q_ref)**2)) / peak) if peak > 0. else None,
        peak_discharge_relative_error=float((q.max() - peak) / peak) if peak > 0. else None,
        peak_timing_error_s=float(rows[int(np.argmax(q))]['time'] - reference_rows[int(np.argmax(q_ref))]['time']) if peak > 0. else None,
        infiltration_difference_m3=ledger['infiltration'] - ref_ledger['infiltration'],
        cumulative_export_difference_m3=ledger['boundary_outflow'] - ref_ledger['boundary_outflow'],
        mass_residual_m3=ledger['residual'],
        mass_residual_limit_m3=1.e-8 * max(1., solver.initial_volume + ledger['rain']),
        accepted_step_count_difference=solver.steps - reference_state['steps'])


def run_benchmark(*, names, shape, device, options, mesh_options, block_sizes, duration, interval, repeats, graph, output):
    report = dict(device=str(wp.get_device(device)), shape=shape, options=options, mesh_options=mesh_options,
        duration_s=duration, sample_interval_s=interval, repeats=repeats, graph=graph, cases=[],
        budget_document='docs/STATIC_FV1_BUDGETS.md',
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in Path('src/floodwarp').glob('*fv1*.py')})
    for name in names:
        reference, protected = build_control(name=name, shape=shape, device=device, options=options)
        initial = reference.capture_state()
        time_run(solver=reference, duration=duration, graph=graph)
        dense_runs = []
        for repeat in range(repeats):
            reference.restore_state(state=initial)
            dense_runs.append(time_run(solver=reference, duration=duration, graph=graph))
        reference.restore_state(state=initial)
        ref_rows = sampled_run(solver=reference, duration=duration, interval=interval, graph=graph)
        ref_final = reference.capture_state()
        case = dict(name=name, dense_runs=dense_runs, dense_median_seconds=float(np.median([row['loop_seconds'] for row in dense_runs])), configurations=[])
        for block_size in block_sizes:
            reference.restore_state(state=initial)
            setup_started = perf_counter()
            solver = StaticBlockFV1Solver.from_solver(solver=reference, block_size=block_size, protected=protected, **mesh_options)
            wp.synchronize_device(device=solver.device)
            setup_seconds = perf_counter() - setup_started
            static_initial = solver.capture_state()
            time_run(solver=solver, duration=duration, graph=graph)
            runs = []
            for repeat in range(repeats):
                solver.restore_state(state=static_initial)
                runs.append(time_run(solver=solver, duration=duration, graph=graph))
            solver.restore_state(state=static_initial)
            rows = sampled_run(solver=solver, duration=duration, interval=interval, graph=graph)
            metrics = accuracy(solver=solver, reference=reference, reference_state=ref_final,
                rows=rows, reference_rows=ref_rows, thresholds=(.001, .01, .1), wet_threshold=.001)
            # Bounded one-step R(A_f(U)) vs A_c(R(U)); never a production shadow.
            solver.restore_state(state=static_initial)
            reference.restore_state(state=initial)
            reference.step(fixed_dt=options['maximum_dt'])
            solver.step(fixed_dt=options['maximum_dt'])
            mesh = solver._static_execution.mesh
            restricted_fine = restrict_state(q=reference.q.numpy(), mesh=mesh)
            restricted_coarse = restrict_state(q=solver.q.numpy(), mesh=mesh)
            commutator = np.max(np.abs(restricted_fine - restricted_coarse), axis=0).tolist()
            median = float(np.median([row['loop_seconds'] for row in runs]))
            physical_pass = (metrics['wet_union_rmse_m'] <= .001 or (metrics['wet_union_relative_l2'] is not None and metrics['wet_union_relative_l2'] <= .02)) and (
                metrics['maximum_depth_rmse_m'] <= .001 or (metrics['maximum_depth_relative_l2'] is not None and metrics['maximum_depth_relative_l2'] <= .02))
            row = dict(block_size=block_size, runs=runs, median_seconds=median,
                measured_speedup=case['dense_median_seconds'] / median, topology_setup_seconds=setup_seconds,
                diagnostics=solver.execution_diagnostics(),
                accuracy=metrics, depth_accuracy_budget_pass=bool(physical_pass),
                bounded_one_step_commutator_maximum=commutator,
                fine_projection_and_sources_retained=True, total_persistent_bytes=sum(array.capacity for array in solver._fields.values()) +
                    solver.execution_diagnostics()['extra_persistent_bytes'] + sum(array.capacity for array in
                        (solver.ctrl, solver.condition, solver._rain_times, solver._rain_rates)))
            case['configurations'].append(row)
            print(name, block_size, 'speedup', row['measured_speedup'], 'depth RMSE', metrics['depth_rmse_m'], flush=True)
            solver = None
            gc.collect()
        report['cases'].append(case)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')


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
    duration = 6.
    interval = .5
    repeats = 3
    graph = True
    output = Path('validation/static_review/benchmark.json')
    run_benchmark(names=names, shape=shape, device=device, options=options, mesh_options=mesh_options,
                  block_sizes=block_sizes, duration=duration, interval=interval, repeats=repeats, graph=graph, output=output)
