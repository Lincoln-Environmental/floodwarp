"""Recover the retained FV1 run; record continuation separately from old timing."""
from pathlib import Path
from time import perf_counter
import shutil

import numpy as np
import warp as wp

from floodwarp.model import load_checkpoint, diagnostics, save_checkpoint
from validation.gpu_benchmark import load_event, build_floodwarp, file_hash, write_json
from validation.benchmark_restart import sample_targets


def recover(*, input_directory, checkpoint, output, numerical, duration, interval, device):
    wp.init()
    output.mkdir(parents=True,exist_ok=True)
    event=load_event(directory=input_directory,duration=duration,sample_interval=interval)
    model=build_floodwarp(case=event,method='fv1',device=device,numerical=numerical)
    load_checkpoint(model=model,path=checkpoint)
    solver=model.solver
    initial_time=solver.time
    initial_steps=solver.steps
    initial_ledger=diagnostics(model=model)
    identity=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(path=checkpoint),
        starting_time_s=initial_time,starting_steps=initial_steps,initial_ledger=initial_ledger,
        note='Continuation only; historical pre-interruption timing and hourly samples remain separate.',
        source_sha256={name:file_hash(path=Path('src/floodwarp')/name) for name in
            ('full_swe_solver.py','full_swe_kernels.py')})
    write_json(path=output/'recovery_identity.json',value=identity)
    for name in identity['source_sha256']:
        shutil.copyfile(src=Path('src/floodwarp')/name,dst=output/name)
    records=[]
    began=perf_counter()
    for target in sample_targets(duration=duration,interval=interval):
        if target<=solver.time:continue
        stats=solver.run_device_loop(duration=float(target-solver.time),use_cuda_graph=None)
        wp.synchronize_device(device=device)
        ledger=diagnostics(model=model)
        records.append(dict(seconds=float(solver.time),statistics=stats,ledger=ledger))
        write_json(path=output/'continuation_samples.json',value=records)
        if target%3600.==0. or target==duration:
            save_checkpoint(model=model,path=output/'restart.npz')
            print(f'FV1 resumed: {solver.time/3600.:.0f} h; steps={solver.steps}; residual={ledger["residual_m3"]:.6g}',flush=True)
    if solver.time!=duration:raise ValueError('Incomplete recession recovery')
    np.savez_compressed(file=output/'fields.npz',depth=solver.get_water_depth(),maximum=solver.get_max_water_depth())
    write_json(path=output/'continuation_record.json',value=dict(identity=identity,
        final_time_s=solver.time,final_steps=solver.steps,final_ledger=diagnostics(model=model),
        continuation_steps=solver.steps-initial_steps,continuation_wall_seconds=perf_counter()-began,
        continuation_hydraulic_seconds=sum(row['statistics']['loop_seconds'] for row in records),
        fields_sha256=file_hash(path=output/'fields.npz')))


if __name__ == "__main__":
    input_directory=Path('/home/patrickdurney/PycharmProjects/Overlander/.external/purau_lisflood_fp64_20261023/event_5m_fp64_v3')
    checkpoint=Path('outputs/repair_benchmark/purau_event_72h/fv1_0/restart.npz')
    output=Path('outputs/repair_benchmark/purau_event_72h/fv1_recovery')
    numerical=dict(cfl=.4,minimum_dt_s=.000001,maximum_dt_s=1.,gravity_m_s2=9.80665,
        dry_depth_m=.0001,fv1_stage_momentum='discharge_extrapolation')
    duration=259200.
    interval=60.
    device='cuda:0'
    recover(input_directory=input_directory,checkpoint=checkpoint,output=output,numerical=numerical,
        duration=duration,interval=interval,device=device)
