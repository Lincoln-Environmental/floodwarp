"""Integrity-checked tables for completed measurements, without rerunning flow."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
import subprocess

import numpy as np

from validation.benchmark_restart import read_completed,read_progress,publish_json,digest_file


def observed_publication_wall(*,directory):
    """Independent fresh-run wall interval, including Python publication work.

    Component timers omit some manifest/checksum work. Linux inode birth time
    and the final record write preserve that work for these uninterrupted runs.
    Keep this measurement distinct from component totals and restart downtime.
    """
    text=subprocess.run(args=['stat','--printf=%w',str(directory)],capture_output=True,text=True,check=True).stdout.strip()
    if text=='-':return None
    date,time,offset=text.split()
    born=datetime.fromisoformat(date+'T'+time+offset)
    ended=(directory/'record.json').stat().st_mtime_ns/1e9
    elapsed=ended-born.timestamp()
    if elapsed<=0.:return None
    return dict(seconds=elapsed,directory_birth=text,record_last_write_ns=(directory/'record.json').stat().st_mtime_ns,
        scope='Fresh directory creation through final record data write; includes manifest/checksum/field I/O, excludes shared input loading and final directory fsync')


def load_records(*,case_directory):
    records={}
    expected=dict(reference='LISFLOOD-FP FV1 GPU',li='FloodWarp LI',fv1='FloodWarp FV1')
    for method,label in expected.items():
        directories=sorted(path for path in case_directory.glob(method+'_*') if path.name.rsplit('_',1)[-1].isdigit())
        rows=[]
        for directory in directories:
            record_path=directory/'record.json'
            if not record_path.exists():continue
            header=json.loads(record_path.read_text())
            record=read_completed(directory=directory,signature=header['signature'],duration=header['signature']['duration_s'])
            if record['solver']!=label or not record.get('physical_completion_verified',False):
                raise ValueError('Incorrect solver identity or unverified physical completion')
            record['analysis_observed_wall']=observed_publication_wall(directory=directory)
            snapshot=case_directory/'source_snapshot'
            for name,digest in record['signature']['sources'].items():
                if digest_file(path=snapshot/name)!=digest:raise ValueError('Recorded runtime snapshot mismatch')
            for name,digest in record['signature']['benchmark_sources'].items():
                if digest_file(path=snapshot/('benchmark-'+name))!=digest:raise ValueError('Recorded benchmark snapshot mismatch')
            if method=='reference':
                progress=read_progress(directory=directory,signature=record['signature'])
                profiles=[entry['profile'] for entry in progress['statistics'] if 'profile' in entry]
                record['analysis_native_phases']={key:sum(profile[key] for profile in profiles)
                    for key in ('initialization_seconds','diagnostic_seconds','final_output_seconds')}
            rows.append(record)
        if not rows:raise ValueError('Incomplete benchmark case: '+str(case_directory))
        records[method]=rows
    return records


def median(*,rows,key,default=0.):
    return float(np.median([row.get(key,default) for row in rows]))


def runtime_metrics(*,records,duration):
    result={}
    reference=median(rows=records['reference'],key='hydraulic_seconds')
    for method,rows in records.items():
        measured=median(rows=rows,key='hydraulic_seconds')
        result[method]=dict(repetitions=len(rows),hydraulic_median_s=measured,
            hydraulic_range_s=[min(row['hydraulic_seconds'] for row in rows),max(row['hydraulic_seconds'] for row in rows)],
            runtime_ratio_to_external_fv1=measured/reference,throughput_simulated_s_per_wall_s=duration/measured,
            preprocessing_median_s=median(rows=rows,key='preparation_seconds'),
            compilation_median_s=median(rows=rows,key='compilation_seconds'),
            graph_capture_median_s=median(rows=rows,key='graph_capture_seconds'),
            diagnostics_io_median_s=median(rows=rows,key='diagnostics_and_checkpoint_seconds') if method!='reference' else
                float(np.median([row['analysis_native_phases']['diagnostic_seconds']+
                    row['analysis_native_phases']['final_output_seconds'] for row in rows])),
            native_initialization_median_s=float(np.median([row['analysis_native_phases']['initialization_seconds'] for row in rows])) if method=='reference' else None,
            end_to_end_median_s=median(rows=rows,key='end_to_end_seconds'),
            observed_publication_wall_median_s=float(np.median([row['analysis_observed_wall']['seconds'] for row in rows])) if
                all(row['analysis_observed_wall'] is not None and row.get('restarts',0)==0 for row in rows) else None,
            observed_wall_metadata=[row['analysis_observed_wall'] for row in rows],
            persistent_solver_array_mib=max(row.get('persistent_solver_array_bytes') or 0 for row in rows)/1024**2,
            peak_allocated_device_mib=max(row['peak_explicit_device_bytes'] for row in rows)/1024**2,
            steps=rows[0]['steps'],dt_min_s=rows[0]['dt_min_s'],dt_mean_s=rows[0]['dt_mean_s'],
            dt_max_s=rows[0]['dt_max_s'],restarts=max(row.get('restarts',0) for row in rows))
    return result


def assess_physics(*,case_directory,records,depth_atol_m,depth_rtol,outlet_nrmse_limit):
    summary=json.loads((case_directory/'summary.json').read_text())
    result={}
    for method in ('li','fv1'):
        metrics=summary['results'][method]['physical']
        reference=records['reference'][0]['ledger'][-1];candidate=records[method][0]['ledger'][-1]
        relative=metrics.get('depth_relative_l2')
        covered=metrics['meaningful_depth_coverage']
        depth_gate=metrics['depth_rmse_m']<=depth_atol_m or (relative is not None and relative<=depth_rtol)
        peak=metrics['reference_peak_discharge_m3_s']
        with np.load(file=case_directory/(method+'_0')/'fields.npz',allow_pickle=False) as candidate_fields, np.load(
                file=case_directory/'reference_0/fields.npz',allow_pickle=False) as reference_fields:
            maximum_norm=float(np.linalg.norm(reference_fields['maximum']))
            maximum_relative=float(np.linalg.norm(candidate_fields['maximum']-reference_fields['maximum'])/maximum_norm) if maximum_norm>0. else None
        result[method]=dict(metrics=metrics,meaningful_wet_coverage=covered,
            final_depth_gate=bool(depth_gate) if covered else None,
            hydrograph_gate=metrics['hydrograph_nrmse']<=outlet_nrmse_limit if metrics['hydrograph_coverage'] else None,
            storage_difference_m3=candidate['storage_m3']-reference['storage_m3'],
            rainfall_difference_m3=candidate['rain_m3']-reference['rain_m3'],
            infiltration_difference_m3=candidate['infiltration_m3']-reference['infiltration_m3'],
            peak_discharge_relative_error=(metrics['peak_discharge_m3_s']-peak)/peak if peak>0. else None,
            peak_timing_difference_s=metrics['peak_interval_end_s']-metrics['reference_peak_interval_end_s'] if peak>0. else None,
            maximum_envelope_relative_l2=maximum_relative,
            candidate_final_ledger=candidate,reference_final_ledger=reference)
    return result


def build_report(*,root,output,expected_cases,depth_atol_m,depth_rtol,outlet_nrmse_limit):
    output.mkdir(parents=True,exist_ok=True)
    measured=[]
    for path in sorted(root.glob('*/summary.json')):
        directory=path.parent;records=load_records(case_directory=directory)
        signature=records['reference'][0]['signature'];duration=signature['duration_s']
        if any(row['signature']!=signature for rows in records.values() for row in rows):
            raise ValueError('Different configurations/sources within comparison')
        measured.append(dict(case=directory.name,duration_s=duration,spacing_m=signature['spacing_m'],
            shape=signature['shape'],signature=signature,
            runtime=runtime_metrics(records=records,duration=duration),
            physical=assess_physics(case_directory=directory,records=records,depth_atol_m=depth_atol_m,
                depth_rtol=depth_rtol,outlet_nrmse_limit=outlet_nrmse_limit),
            source_root=str(directory),record_payload_sha256={method:[row['payload_sha256'] for row in rows] for method,rows in records.items()}))
    completed={row['case'] for row in measured}
    pending=sorted(set(expected_cases)-completed)
    result=dict(status='complete' if not pending else 'partial',pending=pending,completed_measurements=measured,
        depth_absolute_gate_m=depth_atol_m,depth_relative_l2_gate=depth_rtol,
        outlet_nrmse_gate=outlet_nrmse_limit,
        completion_note='Completed simulations and physical gates are distinct; missing wet coverage is not an accuracy pass.',
        report_driver_sha256=digest_file(path=Path(__file__)))
    publish_json(path=output/'measurements.json',value=result)
    lines=['# Completed GPU measurements','',
        'Status: '+result['status']+'. Pending cases: '+(', '.join(pending) if pending else 'none')+'.','',
        'Median hydraulic runtime uses three short-case repetitions and one 72-hour event. Ratios use external LISFLOOD-FP FV1 GPU.','',
        '| Case | Solver | Hydraulic s | End-to-end s | Ratio | Steps | dt mean s | Peak allocated MiB | Persistent arrays MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in measured:
        for method,row in case['runtime'].items():
            wall=row['observed_publication_wall_median_s'] or row['end_to_end_median_s']
            lines.append(f'| {case["case"]} | {method} | {row["hydraulic_median_s"]:.6g} | {wall:.6g} | {row["runtime_ratio_to_external_fv1"]:.4g} | {row["steps"]} | {row["dt_mean_s"]:.6g} | {row["peak_allocated_device_mib"]:.3f} | {row["persistent_solver_array_mib"]:.3f} |')
    lines.extend(['','Dry first-hour Purau fields provide timing evidence without meaningful wet or hydrograph coverage.',
        'CUDA allocated high-watermark and native explicit-allocation counters exclude context/reserved memory. Persistent capacity is separate.',
        'Warm compiled caches were used; compilation fields include module loading. External build provenance is recorded separately.',''])
    lines.extend(['End-to-end table values use fresh-directory birth through final record write, including Python checksum/manifest work missed by component totals. Shared input loading and final directory fsync are outside that interval. Native end-to-end includes its hourly process restarts; hydraulic timing excludes initialization.',''])
    (output/'MEASUREMENTS.md').write_text('\n'.join(lines))
    return result


if __name__ == "__main__":
    root=Path('outputs/qualification_optimized')
    output=Path('validation/repair_review/qualified_measurements')
    expected_cases=('lake','dam_break','rain_infiltration','boundary_outflow','purau_coast_crop',
        'purau_event','purau_coast_crop_grid_control','purau_coast_crop_refined','purau_coast_crop_cfl02','purau_event_72h')
    depth_atol_m=.001
    depth_rtol=.02
    outlet_nrmse_limit=.05
    build_report(root=root,output=output,expected_cases=expected_cases,depth_atol_m=depth_atol_m,
        depth_rtol=depth_rtol,outlet_nrmse_limit=outlet_nrmse_limit)
