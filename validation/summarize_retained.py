"""Validate and compare retained 72-hour fields without repeating simulation.

The recovered FV1 timing is explicitly excluded from the primary benchmark:
its earlier interval predates durable scalar records and recovery was staged.
"""
from pathlib import Path
import json

import numpy as np

from validation.gpu_benchmark import load_event,physical_metrics,file_hash,write_json


def checked_fields(*,path,shape,active):
    with np.load(file=path,allow_pickle=False) as saved:
        fields={name:saved[name].copy() for name in ('depth','maximum')}
    for name,field in fields.items():
        if field.shape!=shape or not np.isfinite(field).all() or np.any(field<0.):
            raise ValueError('Invalid retained physical field: '+name)
        if np.any(field[~active]!=0.):raise ValueError('Nonzero retained inactive-cell water')
    if np.any(fields['maximum']+1e-10<fields['depth']):
        raise ValueError('Retained maximum envelope is below final depth')
    return fields


def summarize(*,root,input_directory,output,duration,thresholds,indifference_m):
    case=load_event(directory=input_directory,duration=duration,sample_interval=3600.)
    reference=json.loads((root/'reference_0/record.json').read_text())
    local=json.loads((root/'li_0/record.json').read_text())
    recovered=json.loads((root/'fv1_recovery/continuation_record.json').read_text())
    if reference['time_s']!=duration or local['time_s']!=duration or recovered['final_time_s']!=duration:
        raise ValueError('Retained benchmark did not reach full requested duration')
    if recovered['fields_sha256']!=file_hash(path=root/'fv1_recovery/fields.npz'):
        raise ValueError('Retained recovered field checksum mismatch')
    fields={name:checked_fields(path=root/directory/'fields.npz',shape=case.bed.shape,active=case.active)
        for name,directory in (('reference','reference_0'),('li','li_0'),('fv1','fv1_recovery'))}
    final=recovered['final_ledger']
    fv_rows=[dict(seconds=duration,storage_m3=final['storage_m3'],rain_m3=final['rain'],
        infiltration_m3=final['infiltration'],outflow_m3=final['boundary_outflow'],
        inflow_m3=final['boundary_inflow'],residual_m3=final['residual_m3'])]
    records={}
    for method,rows in (('li',local['ledger']),('fv1',fv_rows)):
        reference_rows=reference['ledger'] if method=='li' else [reference['ledger'][-1]]
        metrics=physical_metrics(candidate=fields[method]['depth'],reference=fields['reference']['depth'],
            active=case.active,thresholds=thresholds,indifference_m=indifference_m,
            candidate_rows=rows,reference_rows=reference_rows)
        maximum_error=(fields[method]['maximum']-fields['reference']['maximum'])[case.active]
        metrics['maximum_envelope_rmse_m']=float(np.sqrt(np.mean(maximum_error**2)))
        metrics['maximum_envelope_max_error_m']=float(np.max(np.abs(maximum_error)))
        if method=='fv1':
            for name in list(metrics):
                if name.startswith(('hydrograph','peak_','reference_peak_')):metrics.pop(name)
            metrics['hydrograph_coverage']='Incomplete pre-interruption scalar records; no whole-event peak or timing comparison'
        else:metrics['hydrograph_sample_interval_s']=case.sample_interval
        records[method]=dict(physical=metrics,final_ledger=rows[-1],
            steps=local['steps'] if method=='li' else recovered['final_steps'],
            fields_sha256=file_hash(path=root/('li_0' if method=='li' else 'fv1_recovery')/'fields.npz'))
    for row in (reference['ledger'][-1],local['ledger'][-1],fv_rows[-1]):
        if abs(row['residual_m3'])>1e-8*max(1.,row['rain_m3']+row['inflow_m3']):
            raise ValueError('Retained benchmark failed normalised water balance')
    write_json(path=output,value=dict(duration_s=duration,shape=list(case.bed.shape),spacing_m=case.spacing,
        reference_binary_sha256=reference['signature']['external_binary_sha256'],
        status='Completed retained hydraulic fields; historical hourly hydrographs and recovered timing are not the final qualification protocol',
        reference_final_ledger=reference['ledger'][-1],results=records,
        hydraulic_seconds=dict(reference=reference['hydraulic_seconds'],li=local['hydraulic_seconds'],
            fv1_recovery_only=recovered['continuation_hydraulic_seconds']),
        fv1_whole_event_comparable_timing_available=False,
        driver_sha256=file_hash(path=Path(__file__))))


if __name__ == "__main__":
    root=Path('outputs/repair_benchmark/purau_event_72h')
    input_directory=Path('/home/patrickdurney/PycharmProjects/Overlander/.external/purau_lisflood_fp64_20261023/event_5m_fp64_v3')
    output=Path('validation/repair_review/retained_72h_comparison.json')
    duration=259200.
    thresholds=(.01,.10)
    indifference_m=.00001
    summarize(root=root,input_directory=input_directory,output=output,duration=duration,
        thresholds=thresholds,indifference_m=indifference_m)
