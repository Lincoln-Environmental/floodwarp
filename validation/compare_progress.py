"""Compare measured, common simulated-time prefixes using saved checkpoints."""
from pathlib import Path
import json

from validation.benchmark_restart import read_progress,publish_json,digest_file


def compare_prefix(*,previous_path,current_directory,output):
    previous=json.loads(previous_path.read_text())
    header=json.loads((current_directory/'progress.json').read_text())
    current=read_progress(directory=current_directory,signature=header['signature'])
    for name in ('shape','spacing_m','fields','numerical'):
        if previous['signature'][name]!=current['signature'][name]:
            raise ValueError('Checkpoint comparisons require matching physical inputs and controls: '+name)
    common=[row['seconds'] for row in previous['ledger'] if row['seconds']<=current['time_s']]
    if not common:raise ValueError('No common completed simulated-time checkpoint')
    time=float(common[-1])
    old=[row for ledger,row in zip(previous['ledger'],previous['statistics'],strict=True) if ledger['seconds']<=time]
    new=[row for ledger,row in zip(current['ledger'],current['statistics'],strict=True) if ledger['seconds']<=time]
    old_seconds=sum(row['loop_seconds'] for row in old);new_seconds=sum(row['loop_seconds'] for row in new)
    old_steps=sum(row['steps'] for row in old);new_steps=sum(row['steps'] for row in new)
    result=dict(simulated_time_s=time,previous_hydraulic_seconds=old_seconds,current_hydraulic_seconds=new_seconds,
        reduction_percent=100.*(1.-new_seconds/old_seconds),previous_steps=old_steps,current_steps=new_steps,
        per_step_reduction_percent=100.*(1.-(new_seconds/new_steps)/(old_seconds/old_steps)),
        previous_sample_interval_s=previous['signature']['sample_interval_s'],
        current_sample_interval_s=current['signature']['sample_interval_s'],
        cadence_changed=previous['signature']['sample_interval_s']!=current['signature']['sample_interval_s'],
        comparison_scope='Single-run common event prefix; cadence changes can change accepted timestep counts. Controlled repeated microbenchmarks separately isolate the optimisation.',
        previous_record_sha256=digest_file(path=previous_path),current_checkpoint_sha256=current['checkpoint_sha256'],
        previous_sources=previous['signature']['sources'],current_sources=current['signature']['sources'],
        driver_sha256=digest_file(path=Path(__file__)))
    publish_json(path=output,value=result)
    print(f'{time/3600.:.0f} h: {old_seconds:.3f} -> {new_seconds:.3f} hydraulic s; {result["reduction_percent"]:.2f}% reduction',flush=True)
    return result


if __name__ == "__main__":
    previous_path=Path('outputs/repair_benchmark/purau_event_72h/li_0/record.json')
    current_directory=Path('outputs/qualification_optimized/purau_event_72h/li_0')
    output=Path('validation/repair_review/progress_speed_comparison.json')
    compare_prefix(previous_path=previous_path,current_directory=current_directory,output=output)
