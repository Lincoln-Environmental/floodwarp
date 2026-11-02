"""Step statistics are aggregated from actual accepted native advances."""
import math

import pytest

from floodwarp.run_metrics import new_run_metrics, accumulate_advance, finish_run_metrics


def interval(*, steps, end, low, high, mean, rms, hydraulic):
    return dict(steps=steps,time_s=end,dt_min_s=low,dt_max_s=high,dt_mean_s=mean,dt_rms_s=rms,
        loop_seconds=hydraulic,compilation_seconds=.2,graph_capture_seconds=.1,cuda_graph=True)


def test_weighted_timestep_statistics_and_separate_runtime_scopes():
    metrics=new_run_metrics(version='test',source={'commit':'fixture'},duration_s=5.,device='cpu',numerical_profile='ordinary')
    accumulate_advance(metrics=metrics,stats=interval(steps=2,end=1.,low=.25,high=.75,mean=.5,
        rms=math.sqrt((.25**2+.75**2)/2.),hydraulic=.3),wall_seconds=.8)
    accumulate_advance(metrics=metrics,stats=interval(steps=4,end=5.,low=1.,high=1.,mean=1.,rms=1.,hydraulic=.7),wall_seconds=1.2)
    finish_run_metrics(metrics=metrics,wall_seconds=4.,complete=True)
    assert metrics['accepted_steps']==6
    assert metrics['dt_mean_s']==pytest.approx(5./6.)
    assert metrics['dt_rms_s']==pytest.approx(math.sqrt((.25**2+.75**2+4.)/6.))
    assert metrics['dt_min_s']==.25 and metrics['dt_max_s']==1.
    assert metrics['hydraulic_seconds']==1. and metrics['advance_wall_seconds']==2.
    assert metrics['worker_wall_seconds']==4.
    assert metrics['compilation_seconds']==.4 and metrics['graph_capture_seconds']==.2
    assert metrics['actual_time_s']==5. and metrics['complete']


def test_no_completed_steps_has_no_invented_dt_or_throughput():
    metrics=new_run_metrics(version='test',source={},duration_s=1.,device='cpu',numerical_profile='ordinary')
    finish_run_metrics(metrics=metrics,wall_seconds=.1,complete=False)
    assert metrics['accepted_steps']==0 and metrics['dt_min_s'] is None
    assert metrics['dt_mean_s'] is None and metrics['hydraulic_microseconds_per_step'] is None
    assert not metrics['complete']
