"""Aggregate existing FV1 statistics without changing accepted stepping."""
import math


def new_run_metrics(*, version, source, duration_s, device, numerical_profile):
    return dict(schema=1, version=version, source=source, solver='fv1', device=device,
        copyright='Lincoln Agritech Ltd', developed_by='Patrick Durney',
        numerical_profile=numerical_profile, requested_duration_s=float(duration_s),
        actual_time_s=0., accepted_steps=0, completed_intervals=0,
        hydraulic_seconds=0., advance_wall_seconds=0., compilation_seconds=0., graph_capture_seconds=0.,
        preparation_seconds=0., checkpoint_seconds=0., diagnostics_seconds=0., output_seconds=0.,
        worker_wall_seconds=0., dt_min_s=None, dt_mean_s=None, dt_max_s=None, dt_rms_s=None,
        weighted_dt_squared_s2=0., cuda_graph=None, complete=False,
        timing_scope='Hydraulic time sums successful native device-loop calls. Advance wall includes their setup. '
            'Worker wall starts at execute_run entry and excludes Python process startup and browser validation. '
            'On failure, step/dt/hydraulic totals cover completed advance calls only.')


def accumulate_advance(*, metrics, stats, wall_seconds):
    steps = int(stats['steps'])
    if steps <= 0:
        raise ValueError('A completed FV1 advance must contain accepted steps')
    metrics['accepted_steps'] += steps
    metrics['completed_intervals'] += 1
    metrics['actual_time_s'] = float(stats['time_s'])
    metrics['hydraulic_seconds'] += float(stats['loop_seconds'])
    metrics['advance_wall_seconds'] += float(wall_seconds)
    metrics['compilation_seconds'] += float(stats['compilation_seconds'])
    metrics['graph_capture_seconds'] += float(stats['graph_capture_seconds'])
    low, high = float(stats['dt_min_s']), float(stats['dt_max_s'])
    metrics['dt_min_s'] = low if metrics['dt_min_s'] is None else min(metrics['dt_min_s'], low)
    metrics['dt_max_s'] = high if metrics['dt_max_s'] is None else max(metrics['dt_max_s'], high)
    metrics['weighted_dt_squared_s2'] += steps * float(stats['dt_rms_s'])**2
    metrics['dt_mean_s'] = metrics['actual_time_s'] / metrics['accepted_steps']
    metrics['dt_rms_s'] = math.sqrt(metrics['weighted_dt_squared_s2'] / metrics['accepted_steps'])
    metrics['cuda_graph'] = bool(stats['cuda_graph'])


def finish_run_metrics(*, metrics, wall_seconds, complete):
    metrics['worker_wall_seconds'] = float(wall_seconds)
    metrics['complete'] = bool(complete)
    steps, hydraulic = metrics['accepted_steps'], metrics['hydraulic_seconds']
    metrics['hydraulic_microseconds_per_step'] = hydraulic * 1.e6 / steps if steps else None
    metrics['simulated_seconds_per_hydraulic_second'] = metrics['actual_time_s'] / hydraulic if hydraulic > 0. else None
    return metrics
