"""Additive repair validation; frozen fixtures and v1/v2 evidence stay intact."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np

from validation.compare import run_comparisons
from validation.li_replay import replay
from validation.li_assess import assess_runs
from validation.li_engineering_v2 import assess, depth_assessment, extent_assessment
from validation.gpu_benchmark import write_json
from validation.sensitivity import assess as assess_sensitivity


def reproduce_reviews(*, roots, fixtures, output, thresholds, bands, percentiles, engineering, criteria):
    v1=output/'frozen_v1_reproduction';v1.mkdir(parents=True,exist_ok=True)
    assess_runs(roots=roots,fixtures=fixtures,output=v1,thresholds=thresholds,bands=bands,
        percentiles=percentiles,engineering=engineering)
    v2=output/'frozen_v2_reproduction'
    assess(roots={name:roots[name] for name in ('baseline','shipped_defaults')},fixtures=fixtures,
        previous=Path('validation/li_review_v1'),output=v2,criteria=criteria,thresholds=thresholds)
    checks={}
    for name,path in (('v1',v1/'engineering_results.json'),('v2',v2/'results.json')):
        frozen=Path('validation/li_review_v1/engineering_results.json') if name=='v1' else Path('validation/li_review_v2/results.json')
        checks[name]=json.loads(path.read_text())==json.loads(frozen.read_text())
    if not all(checks.values()):raise AssertionError('Frozen review analysis did not reproduce')
    return checks


def assess_repairs(*, fixtures, frozen_roots, output, profiles, closures, history_capacity, thresholds,
                   bands, percentiles, engineering, criteria):
    output.mkdir(parents=True,exist_ok=True)
    baseline=output/'current';defaults=output/'current_defaults'
    if not (baseline/'manifest.json').exists():
        replay(namespace='floodwarp',fixtures=fixtures,output=baseline,device='cpu',profiles=profiles,
            closures=closures,history_capacity=history_capacity)
    if not (defaults/'manifest.json').exists():
        replay(namespace='floodwarp',fixtures=fixtures,output=defaults,device='cpu',profiles=(None,),
            closures=(None,),history_capacity=history_capacity,shipped_defaults=True)
    current_v1=output/'current_v1';current_v1.mkdir(parents=True,exist_ok=True)
    assess_runs(roots=dict(baseline=baseline,shipped_defaults=defaults),fixtures=fixtures,output=current_v1,
        thresholds=thresholds,bands=bands,percentiles=percentiles,engineering=engineering)
    strict=[];practical=[]
    for row in json.loads((baseline/'manifest.json').read_text()):
        with np.load(file=baseline/(row['label']+'.npz')) as archive:
            arrays={name:archive[name].copy() for name in archive.files}
        with np.load(file=frozen_roots['baseline']/(row['label']+'.npz')) as archive:
            differences={}
            for name,array in arrays.items():
                old=archive[name]
                if array.shape!=old.shape:
                    differences[name]=dict(shape_before=list(old.shape),shape_after=list(array.shape))
                elif array.tobytes()!=old.tobytes():
                    differences[name]=dict(max_absolute=float(np.max(np.abs(array-old))))
        strict.append(dict(label=row['label'],bitwise=not differences,differences=differences,
            scope='Strict diagnostic against preserved pre-repair trajectory; numerical repairs can change accepted steps'))
        with np.load(file=fixtures/row['case']/'depth_series.npz') as archive:
            reference=archive['depth'].copy()
        weights=np.diff(np.r_[0.,arrays['times']])
        practical.append(dict(label=row['label'],finite=row['finite'],minimum_depth_m=row['minimum_depth'],
            max_mass_residual_m3=row['max_mass_residual_m3'],
            equivalent_boundary=row['numerical']['free_outflow_closure']=='local_or_prescribed_slope',
            depth=depth_assessment(actual=arrays['depth'],reference=reference,weights=weights,criteria=criteria),
            extent=[extent_assessment(actual=arrays['depth'],reference=reference,weights=weights,
                threshold=threshold,criteria=criteria) for threshold in (.01,.10)]))
    write_json(path=output/'strict_repair_differences.json',value=strict)
    write_json(path=output/'current_v2_depth_extent.json',value=practical)
    return practical


def validate(*, fixtures, output, frozen_roots, profiles, closures, history_capacity, thresholds,
             bands, percentiles, engineering, criteria, evidence, cfl_values):
    protected={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (fixtures,Path('validation/li_review_v1'),Path('validation/li_review_v2'))
        for path in directory.rglob('*') if path.is_file()}
    output.mkdir(parents=True,exist_ok=True)
    original=reproduce_reviews(roots=frozen_roots,fixtures=fixtures,output=output,thresholds=thresholds,
        bands=bands,percentiles=percentiles,engineering=engineering,criteria=criteria)
    report=run_comparisons(fixtures=fixtures,output=output/'strict_comparisons',device='cpu',
        profiles=('physical_candidate','lisflood_fp_acc'))
    assess_sensitivity(fixtures=fixtures,results=output/'strict_comparisons',output=output/'sensitivity.json',
        profiles=('physical_candidate','lisflood_fp_acc'),cfl_values=cfl_values)
    repaired=assess_repairs(fixtures=fixtures,frozen_roots=frozen_roots,output=output,profiles=profiles,
        closures=closures,history_capacity=history_capacity,thresholds=thresholds,bands=bands,
        percentiles=percentiles,engineering=engineering,criteria=criteria)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in protected.items()):
        raise AssertionError('Protected fixture/review changed')
    evidence.mkdir(parents=True,exist_ok=True)
    for source in (output/'strict_comparisons/report.json',output/'strict_repair_differences.json',
                   output/'current_v2_depth_extent.json',output/'current_v1/engineering_results.json',output/'sensitivity.json'):
        shutil.copyfile(src=source,dst=evidence/(source.parent.name+'_'+source.name))
    write_json(path=evidence/'validation.json',value=dict(frozen_reviews_reproduced=original,
        protected_sha256=protected,strict_comparisons_passed=report['passed'],strict_comparisons_total=report['total'],
        repaired_trajectories=len(repaired),finite_nonnegative=all(row['finite'] and row['minimum_depth_m']>=0. for row in repaired),
        maximum_mass_residual_m3=max(row['max_mass_residual_m3'] for row in repaired),
        notes=['Strict frozen acceptance remains diagnostic evidence; engineering depth and extent are evaluated separately.',
            'ACC fixtures are not the external FV1 performance baseline.']))


if __name__ == "__main__":
    fixtures=Path('validation/fixtures')
    output=Path('outputs/repair_validation')
    evidence=Path('validation/repair_review')
    frozen_roots=dict(baseline=Path('outputs/li_review_v1/baselines/target_results'),
        attribution=Path('outputs/li_review_v1/attribution_v2'),
        shipped_defaults=Path('outputs/li_review_v1/shipped_defaults_v2'))
    profiles=('physical_candidate','lisflood_fp_acc',None)
    closures=('local_or_prescribed_slope',None)
    history_capacity=200000
    cfl_values=(.4,.2,.1,.05)
    thresholds=(.0001,.001,.005,.01,.05,.1)
    bands=(.000001,.00001,.0001,.001)
    percentiles=(50,90,95,99)
    engineering=dict(rmse_m=.001,relative_l2=.02,p95_m=.005,max_m=.01,
        extent_thresholds=(.01,.05,.1),iou=.95,band_m=.001,q_nrmse=.05,volume_fraction=.02)
    criteria=dict(meaningful_depth_m=.05,absolute_depth_m=.01,relative_depth=.05,percentile=.95,
        local_review_m=.05,extent_thresholds_m=(.05,.1),extent_band_m=.01,
        extent_fraction=.05,instant_extent_fraction=.1,duration_fraction=.1,q_nrmse=.05,
        peak_fraction=.1,timing_fraction=.05,volume_fraction=.05,recharge_fraction=.1)
    validate(fixtures=fixtures,output=output,frozen_roots=frozen_roots,profiles=profiles,closures=closures,
        history_capacity=history_capacity,thresholds=thresholds,bands=bands,percentiles=percentiles,
        engineering=engineering,criteria=criteria,evidence=evidence,cfl_values=cfl_values)
