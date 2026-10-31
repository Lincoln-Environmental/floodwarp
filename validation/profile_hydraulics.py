"""Repeat a saved wet hydraulic workload with optional graph execution/NVTX.

Run profiling separately from benchmark timing. Restoring an identical saved
state before each repetition isolates launch/kernel costs from hydrology drift.
"""
from pathlib import Path
import os

import numpy as np
import warp as wp

from floodwarp.model import load_checkpoint,diagnostics
from validation.gpu_benchmark import load_event,build_floodwarp,file_hash,write_json


def run_loop(*,model,method,duration,graph,label):
    with wp.ScopedTimer(name=label,print=False,use_nvtx=model.solver.device.is_cuda):
        if method=='li':
            result=model.solver.run_device_loop(duration=duration,rainfall_hyetograph=model.rainfall,use_cuda_graph=graph)
        else:result=model.solver.run_device_loop(duration=duration,use_cuda_graph=graph)
        wp.synchronize_device(device=model.solver.device)
    return result


def profile(*,input_directory,checkpoint,method,device,numerical,duration,graph,repeats,output):
    wp.init();output.mkdir(parents=True,exist_ok=True)
    case=load_event(directory=input_directory,duration=259200.,sample_interval=60.)
    model=build_floodwarp(case=case,method=method,device=device,numerical=numerical)
    load_checkpoint(model=model,path=checkpoint)
    initial=model.solver.capture_state()
    run_loop(model=model,method=method,duration=duration,graph=graph,label='profile_warmup')
    rows=[];balances=[]
    for repeat in range(repeats):
        model.solver.restore_state(state=initial)
        rows.append(run_loop(model=model,method=method,duration=duration,graph=graph,label='hydraulic_measure'))
        balances.append(diagnostics(model=model))
    write_json(path=output/'profile_run.json',value=dict(method=method,graph=graph,duration_s=duration,
        checkpoint=str(checkpoint),checkpoint_sha256=file_hash(path=checkpoint),
        numerical=numerical,runs=rows,balances=balances,
        median_hydraulic_seconds=float(np.median([row['loop_seconds'] for row in rows])),
        driver_sha256=file_hash(path=Path(__file__)),
        sources={path.name:file_hash(path=path) for path in Path('src/floodwarp').glob('*.py')}))


if __name__ == "__main__":
    input_directory=Path('/home/patrickdurney/PycharmProjects/Overlander/.external/purau_lisflood_fp64_20261023/event_5m_fp64_v3')
    method=os.environ.get('FLOODWARP_PROFILE_METHOD','fv1')
    checkpoint=Path('outputs/repair_benchmark/purau_event_72h')/(method+'_0')/'profiling_restart.npz'
    device='cuda:0'
    numerical=dict(cfl=.4,minimum_dt_s=.000001,maximum_dt_s=1.,gravity_m_s2=9.80665,
        dry_depth_m=.0001,fv1_stage_momentum='discharge_extrapolation')
    duration=60.
    graph=None if os.environ.get('FLOODWARP_PROFILE_GRAPH','1')=='1' else False
    repeats=int(os.environ.get('FLOODWARP_PROFILE_REPEATS','3'))
    output=Path(os.environ.get('FLOODWARP_PROFILE_OUTPUT','outputs/profile_'+method+'_'+str(graph)))
    profile(input_directory=input_directory,checkpoint=checkpoint,method=method,device=device,
        numerical=numerical,duration=duration,graph=graph,repeats=repeats,output=output)
