"""Copy the retained FV1 reference and adapt host diagnostics only.

The original research repository is read-only. Retain its compiled hydraulic
objects byte-for-byte, rebuild the host simulation adapter and record hashes.
"""
from pathlib import Path
import json
import shutil
import subprocess
from time import perf_counter

from validation.benchmark_restart import digest_file,publish_json


def replace_once(*,text,old,new):
    if text.count(old)!=1:
        raise ValueError('Reference adapter source differs from the supported retained version')
    return text.replace(old,new,1)


def patch_diagnostics(*,source):
    simulation=source/'cuda/fv1/cuda_fv1_simulate.cu'
    original=simulation.read_text()
    text=replace_once(text=original,old='    double checkpoint_io_s=0.;',new='''    double checkpoint_io_s=0.;
    const char* field_text=std::getenv("PURAU_FIELD_INTERVAL");
    const double field_interval=field_text ? std::strtod(field_text,nullptr) : pars.MassInt;
    purau_restart::require(std::isfinite(field_interval) && field_interval>0.,"Invalid field output interval");''')
    text=replace_once(text=text,old='            capacity.diagnostic_s+=purau_capacity::elapsed(diagnostic_start);\n            {',
        new='''            const bool field_due=solver.t==solver.Sim_Time ||
                std::abs(solver.t/field_interval-std::round(solver.t/field_interval))<1.e-10;
            if(field_due) {''')
    text=replace_once(text=text,old='\t\t\tsampler.write_if_buffer_full();',
        new='\t\t\tsampler.write_if_buffer_full();\n            cuda::sync();\n            capacity.diagnostic_s+=purau_capacity::elapsed(diagnostic_start);')
    simulation.write_text(text)
    capacity=source/'cuda/capacity_profile.cuh'
    original_capacity=capacity.read_text()
    text=replace_once(text=original_capacity,
        old='\\"peak_explicit_cuda_allocation_bytes\\":%zu,\\"allocation_calls\\":%zu,',
        new='\\"peak_explicit_cuda_allocation_bytes\\":%zu,\\"persistent_explicit_cuda_allocation_bytes\\":%zu,\\"allocation_calls\\":%zu,')
    text=replace_once(text=text,old='timesteps.size(),peak_bytes,allocations_count);',
        new='timesteps.size(),peak_bytes,current_bytes,allocations_count);')
    capacity.write_text(text)
    return {str(path.relative_to(source)):digest_file(path=path) for path in (simulation,capacity)}


def build_reference(*,retained,output):
    if output.exists():
        raise ValueError('Reference build output already exists; preserve it or choose a new destination')
    output.mkdir(parents=True)
    report=json.loads((retained/'build_report.json').read_text())
    source=output/'source';build=output/'build'
    shutil.copytree(src=retained/'source',dst=source)
    shutil.copytree(src=retained/'build',dst=build)
    before={str(path.relative_to(build)):digest_file(path=path) for path in build.rglob('*.o')}
    modified=patch_diagnostics(source=source)
    commands=[[part.replace(str(retained/'source'),str(source.resolve())) for part in command]
        for command in report['commands']]
    started=perf_counter()
    with (output/'build.log').open('w') as stream:
        for command in commands:
            subprocess.run(args=command,cwd=build,stdout=stream,stderr=subprocess.STDOUT,check=True)
    changed={'CMakeFiles/lisflood.dir/cuda/fv1/cuda_fv1_simulate.cu.o',
        'CMakeFiles/lisflood.dir/cmake_device_link.o'}
    preserved={}
    for name,digest in before.items():
        if name in changed:continue
        if digest_file(path=build/name)!=digest:
            raise ValueError('Unexpected reference hydraulic object modification: '+name)
        preserved[name]=digest
    result=dict(retained_binary_sha256=digest_file(path=retained/'build/lisflood'),
        binary_sha256=digest_file(path=build/'lisflood'),modified_sources=modified,
        reused_object_sha256=preserved,commands=commands,build_seconds=perf_counter()-started,
        adapter_scope='Host full-field output cadence, complete diagnostic timing and persistent allocation accounting; hydraulic objects unchanged.',
        retained_report_sha256=digest_file(path=retained/'build_report.json'),
        build_driver_sha256=digest_file(path=Path(__file__)))
    publish_json(path=output/'build_report.json',value=result)


if __name__ == "__main__":
    retained=Path('/home/patrickdurney/PycharmProjects/Overlander/.external/synthetic_catchment_nn_matched_20261028_v1/fv1_build')
    output=Path('outputs/qualified_reference')
    build_reference(retained=retained,output=output)
