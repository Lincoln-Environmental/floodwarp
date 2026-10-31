"""Retain compact qualification, profiling and failure evidence without flow runs."""
from pathlib import Path
from time import perf_counter
import gzip
import json
import shutil
import subprocess

import numpy as np

from validation.benchmark_restart import publish_json,digest_file


def compact_sgs(*,source,destination):
    value=json.loads(source.read_text())
    for result in value['results']:
        for run in result['runs']:run.pop('signature',None)
    value['original_summary_sha256']=digest_file(path=source)
    publish_json(path=destination,value=value)


def collect_profiles(*,pairs,csv_files,destination):
    destination.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    for method,previous,current in pairs:
        old=json.loads(previous.read_text());new=json.loads(current.read_text())
        if old['checkpoint_sha256']!=new['checkpoint_sha256'] or old['numerical']!=new['numerical'] or old['duration_s']!=new['duration_s']:
            raise ValueError('Before/after profiles are not the same workload')
        comparisons.append(dict(method=method,simulated_duration_s=new['duration_s'],
            initial_event_time_s=new['balances'][0]['time_s']-new['duration_s'],
            previous_median_s=old['median_hydraulic_seconds'],current_median_s=new['median_hydraulic_seconds'],
            reduction_percent=100.*(1.-new['median_hydraulic_seconds']/old['median_hydraulic_seconds']),
            previous_times_s=[row['loop_seconds'] for row in old['runs']],current_times_s=[row['loop_seconds'] for row in new['runs']],
            accepted_steps_equal=[row['steps'] for row in old['runs']]==[row['steps'] for row in new['runs']],
            ledger_exact_diagnostic=old['balances']==new['balances'],
            note='Compare within this method only; LI and FV1 use different saved event times.'))
        shutil.copyfile(src=previous,dst=destination/(method+'_before.json'))
        shutil.copyfile(src=current,dst=destination/(method+'_after.json'))
    for path in csv_files:shutil.copyfile(src=path,dst=destination/path.name)
    publish_json(path=destination/'optimisations.json',value=comparisons)


def retain_failures(*,paths,destination):
    destination.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for path in paths:
        if not path.exists():continue
        archive=destination/(path.name+'.gz')
        with path.open('rb') as source,gzip.GzipFile(filename=str(archive),mode='wb',mtime=0) as target:
            shutil.copyfileobj(fsrc=source,fdst=target)
        manifest.append(dict(original=str(path),raw_sha256=digest_file(path=path),
            archive=archive.name,archive_sha256=digest_file(path=archive)))
    publish_json(path=destination/'manifest.json',value=manifest)


def measure_shared_inputs(*,directory,repeats,output):
    from validation.gpu_benchmark import load_event
    measured=[]
    for _ in range(repeats):
        started=perf_counter()
        case=load_event(directory=directory,duration=3600.,sample_interval=60.)
        measured.append(perf_counter()-started)
    publish_json(path=output,value=dict(times_s=measured,median_s=float(np.median(measured)),
        scope='Shared warm-filesystem input loading; excluded from per-solver medians',
        shape=list(case.bed.shape),active_cells=int(case.active.sum()),
        input_packet_sha256=digest_file(path=directory/'inputs_f64.bin')))


def tool_versions(*,commands,output,profiler_download,profiler_url):
    values={name:subprocess.run(args=command,capture_output=True,text=True,check=True).stdout.strip()
        for name,command in commands.items()}
    values['nsight_cli_download_sha256']=digest_file(path=profiler_download)
    values['nsight_official_download_page']=profiler_url
    publish_json(path=output,value=values)


def archive_completed(*,root,destination,field_cases):
    """Keep original checksummed scalar records and selected field bytes."""
    from validation.qualification_report import load_records
    destination.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for summary in sorted(root.glob('*/summary.json')):
        directory=summary.parent
        records=load_records(case_directory=directory)
        for method,rows in records.items():
            for index,_ in enumerate(rows):
                run=directory/(method+'_'+str(index))
                names=('record.json','fields.npz') if directory.name in field_cases else ('record.json',)
                for name in names:
                    source=run/name
                    archive=destination/(directory.name+'-'+run.name+'-'+name+'.gz')
                    with source.open('rb') as stream,gzip.GzipFile(filename=str(archive),mode='wb',mtime=0) as target:
                        shutil.copyfileobj(fsrc=stream,fdst=target)
                    manifest.append(dict(source=str(source),source_sha256=digest_file(path=source),
                        archive=archive.name,archive_sha256=digest_file(path=archive)))
    publish_json(path=destination/'manifest.json',value=manifest)


if __name__ == "__main__":
    output=Path('validation/repair_review')
    sgs_summaries=((Path('outputs/sgs_qualification_cpu_final/summary.json'),output/'sgs_qualification_cpu.json'),
        (Path('outputs/sgs_qualification_gpu_final/summary.json'),output/'sgs_qualification_gpu.json'))
    pairs=(('fv1',Path('outputs/profile_fv1_None/profile_run.json'),Path('outputs/profile_fv1_optimized_None/profile_run.json')),
        ('li',Path('outputs/profile_li_None/profile_run.json'),Path('outputs/profile_li_optimized_None/profile_run.json')))
    csv_files=tuple(Path('outputs/profiles')/name for name in (
        'fv1_wet_windows_gpukernsum.csv','fv1_wet_windows_cudaapisum.csv',
        'fv1_wet_optimized_graph_cuda_gpu_kern_sum.csv','fv1_wet_optimized_graph_cuda_api_sum.csv',
        'li_wet_windows_modern_cuda_gpu_kern_sum.csv','li_wet_windows_modern_cuda_api_sum.csv',
        'sgs_wet_modern_cuda_gpu_kern_sum.csv','sgs_wet_modern_cuda_api_sum.csv',
        'fv1_flux_faces_metrics.csv','fv1_faces_metrics.csv','sgs_coefficients_metrics.csv'))
    performance_files=((Path('outputs/sgs_performance_cuda_0/summary.json'),output/'sgs_dry_gpu.json'),
        (Path('outputs/sgs_wet_performance_cuda_0/summary.json'),output/'sgs_wet_gpu.json'),
        (Path('outputs/memory_probe_final/results.json'),output/'resident_memory.json'))
    failures=tuple(Path('/tmp')/name for name in ('floodwarp-full-72h-benchmark.log',
        'floodwarp-followup-core-tests.log','floodwarp-followup-full-cuda.log',
        'floodwarp-followup-stage-restart-cuda.log','floodwarp-nsys-li.log',
        'floodwarp-final-mixed-cpu.log','floodwarp-final-fp32-cpu.log'))
    input_directory=Path('/home/patrickdurney/PycharmProjects/Overlander/.external/purau_lisflood_fp64_20261023/event_5m_fp64_v3')
    input_repeats=3
    commands=dict(nvcc=['/home/patrickdurney/PycharmProjects/Overlander/.external/synthetic_catchment_20261027/cuda_toolkit/bin/nvcc','--version'],
        cxx=['/usr/bin/g++','--version'],nsight_compute=['ncu','--version'],
        nsight_systems=['/tmp/nsight-systems-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys','--version'])
    profiler_download=Path('/tmp/nsight-systems-cli-2026.deb')
    profiler_url='https://developer.nvidia.com/nsight-systems/get-started'
    completed_root=Path('outputs/qualification_optimized')
    field_cases=('purau_event_72h',)
    for source,destination in sgs_summaries:compact_sgs(source=source,destination=destination)
    collect_profiles(pairs=pairs,csv_files=csv_files,destination=output/'profiles')
    for source,destination in performance_files:shutil.copyfile(src=source,dst=destination)
    retain_failures(paths=failures,destination=output/'failures')
    measure_shared_inputs(directory=input_directory,repeats=input_repeats,output=output/'shared_input_preparation.json')
    tool_versions(commands=commands,output=output/'toolchain.json',profiler_download=profiler_download,profiler_url=profiler_url)
    archive_completed(root=completed_root,destination=output/'completed_runs',field_cases=field_cases)
