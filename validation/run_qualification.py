"""Serial benchmark execution with durable logs, heartbeat and stall detection."""
from pathlib import Path
from time import monotonic,sleep
from datetime import datetime,timezone
import os
import subprocess
import sys

from validation.benchmark_restart import digest_file,publish_json


def monitor_command(*,command,environment,log_path,status_path,poll_seconds,stall_seconds,timeout_seconds):
    started=monotonic();last_change=started;previous_size=0
    with log_path.open('w') as stream:
        process=subprocess.Popen(args=command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                size=log_path.stat().st_size
                if size!=previous_size:last_change=monotonic();previous_size=size
                elapsed=monotonic()-started
                publish_json(path=status_path,value=dict(pid=process.pid,command=command,
                    status='running',elapsed_wall_seconds=elapsed,log_bytes=size,
                    seconds_since_log_progress=monotonic()-last_change,
                    heartbeat_utc=datetime.now(tz=timezone.utc).isoformat(),log=str(log_path)))
                if elapsed>timeout_seconds or monotonic()-last_change>stall_seconds:
                    raise TimeoutError('Benchmark wall/stall timeout; last committed checkpoint remains valid')
                sleep(poll_seconds)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10.)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if process.returncode:
        raise subprocess.CalledProcessError(returncode=process.returncode,cmd=command)
    return monotonic()-started


def run_sequence(*,scopes,output,reference,poll_seconds,stall_seconds,timeout_seconds):
    output.mkdir(parents=True,exist_ok=True)
    provenance=dict(python=sys.version,reference=str(reference.resolve()),
        reference_sha256=digest_file(path=reference),
        benchmark_driver_sha256=digest_file(path=Path(__file__).with_name('gpu_benchmark.py')),
        monitor_driver_sha256=digest_file(path=Path(__file__)),
        scopes=list(scopes),poll_seconds=poll_seconds,stall_seconds=stall_seconds)
    publish_json(path=output/'monitor_provenance.json',value=provenance)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        FLOODWARP_BENCHMARK_OUTPUT=str(output.resolve()),
        FLOODWARP_REFERENCE_BINARY=str(reference.resolve()),PYTHONUNBUFFERED='1')
    for scope in scopes:
        environment['FLOODWARP_BENCHMARK_SCOPE']=scope
        command=[sys.executable,'-m','validation.gpu_benchmark']
        log=output/(scope+'-'+datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.log')
        print('Starting',scope,'log',log,flush=True)
        try:
            seconds=monitor_command(command=command,environment=environment,log_path=log,
                status_path=output/'monitor_status.json',poll_seconds=poll_seconds,
                stall_seconds=stall_seconds,timeout_seconds=timeout_seconds)
        except BaseException as error:
            publish_json(path=output/'monitor_status.json',value=dict(status='failed',scope=scope,
                error=type(error).__name__+': '+str(error),log=str(log),
                checkpoint_policy='Only validated, atomically published progress is resumed'))
            raise
        publish_json(path=output/('completed-'+scope+'.json'),value=dict(status='completed',scope=scope,
            wall_seconds=seconds,log=str(log),provenance=provenance))
        print('Completed',scope,'wall seconds',seconds,flush=True)
    publish_json(path=output/'monitor_status.json',value=dict(status='completed',scopes=list(scopes)))


if __name__ == "__main__":
    scopes=tuple(os.environ.get('FLOODWARP_QUALIFICATION_SCOPES','small,full').split(','))
    output=Path(os.environ.get('FLOODWARP_QUALIFICATION_OUTPUT','outputs/qualification_benchmark'))
    reference=Path('outputs/qualified_reference/build/lisflood')
    poll_seconds=10.
    stall_seconds=300.
    timeout_seconds=14400.
    run_sequence(scopes=scopes,output=output,reference=reference,poll_seconds=poll_seconds,
        stall_seconds=stall_seconds,timeout_seconds=timeout_seconds)
