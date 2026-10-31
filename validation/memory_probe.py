"""Untimed process-resident VRAM probes, separate from runtime measurements."""
from pathlib import Path
from time import sleep
import json
import hashlib
import os
import subprocess
import sys

import warp as wp
from floodwarp.model import save_checkpoint

from validation.gpu_benchmark import (load_event, build_floodwarp, prepare_reference,
    case_signature, device_array_bytes, write_json)


def gpu_process_memory(*, pid):
    completed=subprocess.run(args=['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    for line in completed.stdout.splitlines():
        fields=[value.strip() for value in line.split(',')]
        if len(fields)==2 and fields[0]==str(pid):return int(fields[1])
    return 0


def sample_child(*, command, cwd, log_path, environment, interval):
    observed=[]
    with log_path.open('w') as stream:
        child=subprocess.Popen(args=command,cwd=cwd,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        while child.poll() is None:
            observed.append(gpu_process_memory(pid=child.pid))
            sleep(interval)
        if child.returncode:raise RuntimeError('Memory probe failed: '+str(log_path))
    return dict(peak_sampled_resident_mib=max(observed,default=0),samples=observed,
        interval_s=interval,scope='Whole process including CUDA context, code and allocator reservations; sampling can miss brief peaks')


def worker(*, case, method, numerical, device, output):
    wp.init()
    model=build_floodwarp(case=case,method=method,device=device,numerical=numerical)
    if method=='li':
        model.solver.run_device_loop(duration=case.duration,rainfall_hyetograph=model.rainfall,use_cuda_graph=None)
    else:model.solver.run_device_loop(duration=case.duration,use_cuda_graph=None)
    wp.synchronize_device(device=device)
    save_checkpoint(model=model,path=output.with_suffix('.npz'))
    write_json(path=output,value=dict(resident_mib=gpu_process_memory(pid=os.getpid()),
        explicit_array_bytes=device_array_bytes(solver=model.solver)))


def run(*, case, numerical, binary, output, device, interval, project):
    output.mkdir(parents=True,exist_ok=True)
    records={}
    for method in ('reference','li','fv1'):
        directory=(output/method).resolve()
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        if method=='reference':
            prepare_reference(case=case,directory=directory,numerical=numerical)
            signature=case_signature(case=case,numerical=numerical,binary=binary)
            identity=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
            environment.update(PURAU_INPUT=str(directory/'inputs_f64.bin'),PURAU_CHECKPOINT_ID=identity,
                PURAU_CHECKPOINT_WRITE=str(directory/'checkpoint.bin'),PURAU_STOP_AT=str(case.duration),
                MATCHED_SOURCE_WET_DEPTH='-1')
            environment.pop('PURAU_CHECKPOINT_READ',None)
            command=[str(binary.resolve()),'purau.par']
        else:
            directory.mkdir(parents=True,exist_ok=False)
            environment.update(FLOODWARP_MEMORY_WORKER=method,FLOODWARP_MEMORY_OUTPUT=str(directory/'worker.json'))
            command=[sys.executable,'-m','validation.memory_probe']
        run_directory=directory if method=='reference' else project
        record=sample_child(command=command,cwd=run_directory,log_path=directory/'process.log',
            environment=environment,interval=interval)
        if method!='reference':record['worker']=json.loads((directory/'worker.json').read_text())
        records[method]=record
        write_json(path=output/'results.json',value=records)


if __name__ == "__main__":
    project=Path(__file__).resolve().parents[1]
    reference_root=Path('/home/patrickdurney/PycharmProjects/Overlander/.external')
    event_directory=reference_root/'purau_lisflood_fp64_20261023/event_5m_fp64_v3'
    binary=project/'outputs/qualified_reference/build/lisflood'
    output=project/'outputs/memory_probe_final'
    numerical=dict(cfl=.4,maximum_dt_s=1.,minimum_dt_s=.000001,gravity_m_s2=9.80665,
        dry_depth_m=.0001,wall_elevation_m=1000000.,fv1_stage_momentum='discharge_extrapolation')
    device='cuda:0'
    duration=3600.
    sample_interval=3600.
    interval=.1
    method=os.environ.get('FLOODWARP_MEMORY_WORKER')
    worker_output=Path(os.environ.get('FLOODWARP_MEMORY_OUTPUT','outputs/memory_probe_worker.json'))
    case=load_event(directory=event_directory,duration=duration,sample_interval=sample_interval)
    if method is None:run(case=case,numerical=numerical,binary=binary,output=output,device=device,interval=interval,project=project)
    else:worker(case=case,method=method,numerical=numerical,device=device,output=worker_output)
