"""Benchmark interruption, integrity and exact-duration regressions."""
from dataclasses import replace
import os
import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from validation.benchmark_restart import (sample_targets, read_progress, publish_progress)
from validation.gpu_benchmark import analytical_case, run_floodwarp,run_reference,validate_completed_simulation


def benchmark_case():
    case=analytical_case(name='restart',shape=(2,3),spacing=1.,duration=1.37,
        sample_interval=.4,roughness=.035,depth=.02,rain=.00002,infiltration=.00001)
    return replace(case,times=np.array([0.,.3,.8]),rates=np.array([0.,.00003,.00001]))


def numerical_controls():
    return dict(cfl=.4,minimum_dt_s=.000001,maximum_dt_s=.1,gravity_m_s2=9.80665,
        dry_depth_m=.0001,fv1_stage_momentum='discharge_extrapolation')


def interrupt_after_first(*, time_s):
    raise InterruptedError('Simulated interruption after durable checkpoint')


@pytest.mark.parametrize('duration,interval',[(1.37,.4),(1.2,.4),(.1,1.)])
def test_sampling_finishes_at_requested_duration(duration,interval):
    samples=sample_targets(duration=duration,interval=interval)
    assert samples[-1]==duration
    assert np.all(np.diff(np.r_[0.,samples])>0.)
    assert np.all(samples<=duration)


def test_completion_gate_rejects_bad_time_balance_and_fields():
    case=benchmark_case()
    row=dict(seconds=case.duration,storage_m3=.12,rain_m3=0.,infiltration_m3=0.,
        outflow_m3=0.,inflow_m3=0.,residual_m3=0.)
    depth=np.full(case.bed.shape,.02)
    validate_completed_simulation(case=case,depth=depth,maximum=depth,ledger=[row],time_s=case.duration)
    with pytest.raises(ValueError,match='incomplete'):
        validate_completed_simulation(case=case,depth=depth,maximum=depth,ledger=[row],time_s=1.2)
    with pytest.raises(ValueError,match='conservation'):
        validate_completed_simulation(case=case,depth=depth,maximum=depth,
            ledger=[dict(row,residual_m3=.000001)],time_s=case.duration)
    invalid=depth.copy();invalid[0,0]=np.nan
    with pytest.raises(ValueError,match='invalid'):
        validate_completed_simulation(case=case,depth=invalid,maximum=depth,ledger=[row],time_s=case.duration)


@pytest.mark.parametrize('method',['li','fv1'])
@pytest.mark.parametrize('device,graph',[('cpu',False),('cuda:0',False),('cuda:0',None)])
def test_interrupted_benchmark_matches_continuous(tmp_path,method,device,graph):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    case=benchmark_case();numerical=numerical_controls();signature=dict(test='restart-v1')
    continuous=run_floodwarp(case=case,method=method,directory=tmp_path/'continuous',
        device=device,numerical=numerical,signature=signature,graph=graph)
    directory=tmp_path/'resumed'
    with pytest.raises(InterruptedError,match='durable checkpoint'):
        run_floodwarp(case=case,method=method,directory=directory,device=device,
            numerical=numerical,signature=signature,graph=graph,after_checkpoint=interrupt_after_first)
    progress=read_progress(directory=directory,signature=signature)
    assert progress['time_s']==.4
    resumed=run_floodwarp(case=case,method=method,directory=directory,device=device,
        numerical=numerical,signature=signature,graph=graph)
    assert resumed['time_s']==case.duration
    assert resumed['steps']==continuous['steps']
    assert resumed['restarts']==1
    for a,b in zip(resumed['ledger'],continuous['ledger'],strict=True):
        for name in ('storage_m3','rain_m3','infiltration_m3','outflow_m3','inflow_m3'):
            assert a[name]==pytest.approx(b[name],abs=1e-10,rel=1e-9)
        assert abs(a['residual_m3'])<1e-10
    with np.load(file=directory/'fields.npz') as a,np.load(file=tmp_path/'continuous/fields.npz') as b:
        np.testing.assert_allclose(a['depth'],b['depth'],atol=1e-5,rtol=1e-4)
        np.testing.assert_allclose(a['maximum'],b['maximum'],atol=1e-5,rtol=1e-4)
    assert run_floodwarp(case=case,method=method,directory=directory,device=device,
        numerical=numerical,signature=signature,graph=graph)==resumed
    with pytest.raises(ValueError,match='configuration/source'):
        run_floodwarp(case=case,method=method,directory=directory,device=device,
            numerical=numerical,signature=dict(test='changed-source'),graph=graph)
    original_record=(directory/'record.json').read_text()
    corrupted=json.loads(original_record);corrupted['ledger'][0]['storage_m3']+=1.
    (directory/'record.json').write_text(json.dumps(corrupted))
    with pytest.raises(ValueError,match='checksum'):
        run_floodwarp(case=case,method=method,directory=directory,device=device,
            numerical=numerical,signature=signature,graph=graph)
    (directory/'record.json').write_text(original_record)
    with (directory/'fields.npz').open('ab') as stream:stream.write(b'drift')
    with pytest.raises(ValueError,match='checksum'):
        run_floodwarp(case=case,method=method,directory=directory,device=device,
            numerical=numerical,signature=signature,graph=graph)


def test_failed_checkpoint_publication_preserves_previous(tmp_path):
    def save_good(*,path):
        path.write_bytes(b'valid solver state')

    def save_interrupted(*,path):
        path.write_bytes(b'incomplete')
        raise OSError('Interrupted state write')

    options=dict(directory=tmp_path,signature=dict(test=1),ledger=[],statistics=[],timings={},suffix='.bin')
    previous=publish_progress(time_s=0.,save_state=save_good,**options)
    with pytest.raises(OSError,match='Interrupted'):
        publish_progress(time_s=0.,save_state=save_interrupted,**options)
    assert read_progress(directory=tmp_path,signature=dict(test=1))==previous
    original_manifest=(tmp_path/'progress.json').read_text()
    corrupted=json.loads(original_manifest);corrupted['timings']['hydraulic_seconds']=100.
    (tmp_path/'progress.json').write_text(json.dumps(corrupted))
    with pytest.raises(ValueError,match='checksum'):
        read_progress(directory=tmp_path,signature=dict(test=1))
    (tmp_path/'progress.json').write_text(original_manifest)
    (tmp_path/previous['checkpoint']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        read_progress(directory=tmp_path,signature=dict(test=1))


def test_external_reference_resume_matches_continuous(tmp_path):
    binary_text=os.environ.get('FLOODWARP_REFERENCE_BINARY')
    if binary_text is None:pytest.skip('Explicit adapted FV1 GPU reference executable required')
    import warp as wp
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    binary=Path(binary_text)
    case=benchmark_case();numerical=numerical_controls()
    # A moving wet front exercises both native FV1 flow buffers, beyond
    # the uniform rainfall/storage case used for the driver interruption.
    initial=case.initial.copy();initial[:,0]+=.2
    case=replace(case,initial=initial)
    numerical['wall_elevation_m']=1000000.
    signature=dict(test='external-resume-v1')
    continuous=run_reference(case=replace(case,checkpoint_interval=case.duration),
        directory=tmp_path/'continuous',numerical=numerical,binary=binary,
        signature=signature,timeout=120.)
    directory=tmp_path/'resumed'
    with pytest.raises(InterruptedError):
        run_reference(case=case,directory=directory,numerical=numerical,binary=binary,
            signature=signature,timeout=120.,after_checkpoint=interrupt_after_first)
    resumed=run_reference(case=case,directory=directory,numerical=numerical,binary=binary,
        signature=signature,timeout=120.)
    assert resumed['time_s']==case.duration
    assert resumed['steps']==continuous['steps']
    for a,b in zip(resumed['ledger'],continuous['ledger'],strict=True):
        for name in ('storage_m3','rain_m3','infiltration_m3','outflow_m3','inflow_m3'):
            assert a[name]==pytest.approx(b[name],abs=1e-10,rel=1e-9)
        assert abs(a['residual_m3'])<1e-10
    with np.load(file=directory/'fields.npz') as a,np.load(file=tmp_path/'continuous/fields.npz') as b:
        np.testing.assert_allclose(a['depth'],b['depth'],atol=1e-5,rtol=1e-4)
        np.testing.assert_allclose(a['maximum'],b['maximum'],atol=1e-5,rtol=1e-4)
