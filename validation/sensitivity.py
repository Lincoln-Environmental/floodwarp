"""Compute resolution/timestep differences from completed bounded comparisons."""
import json
from pathlib import Path
import numpy as np


def read_depth(*, root, label, reference):
    if reference:
        filename = 'arrays.npz' if label.startswith('acc_') else 'physical_final.npz'
        with np.load(root/label/filename, allow_pickle=False) as data:
            return data['final'].copy() if filename == 'arrays.npz' else data['state'][...,0].copy()
    with np.load(root/(label+'.npz'), allow_pickle=False) as data:
        depth = data['depth']
        return depth[-1].copy() if depth.ndim == 3 else depth.copy()


def rmse(*, first, second):
    return float(np.sqrt(np.mean((first-second)**2)))


def assess(*, fixtures, results, output, profiles, cfl_values):
    rows = []
    for profile in (*profiles, 'native'):
        reference = profile == 'native'
        root = fixtures if reference else results
        suffix = '' if reference else '_'+profile
        for coarse, fine in ((20,10),(10,5)):
            a = read_depth(root=root, label=f'acc_complex_{coarse}m_0mmh_dt1'+suffix, reference=reference)
            b = read_depth(root=root, label=f'acc_complex_{fine}m_0mmh_dt1'+suffix, reference=reference)
            factor = coarse//fine
            restricted = b.reshape(a.shape[0],factor,a.shape[1],factor).mean(axis=(1,3))
            rows.append(dict(kind='grid', model=profile, coarse_m=coarse, fine_m=fine,
                             final_depth_rmse_m=rmse(first=a,second=restricted)))
        a = read_depth(root=root, label='acc_heterogeneous_10m_1mmh_dt0.5'+suffix, reference=reference)
        b = read_depth(root=root, label='acc_heterogeneous_10m_1mmh_dt0.25'+suffix, reference=reference)
        rows.append(dict(kind='timestep', model=profile, caps_s=[.5,.25], final_depth_rmse_m=rmse(first=a,second=b)))
    for case in ('wet_dam','strong_shock','dry_dam','retained_outlet_crop'):
        for reference in (False,True):
            root = fixtures if reference else results
            final = read_depth(root=root,label=f'fv1_{case}_{cfl_values[-1]}',reference=reference)
            for cfl in cfl_values[:-1]:
                depth = read_depth(root=root,label=f'fv1_{case}_{cfl}',reference=reference)
                rows.append(dict(kind='FV1 CFL sensitivity',case=case,model='native' if reference else 'floodwarp',
                    cfl=cfl, reference_cfl=cfl_values[-1], final_depth_rmse_m=rmse(first=depth,second=final)))
    output.write_text(json.dumps(dict(rows=rows,limitations=[
        'Two timestep caps show sensitivity, not an asymptotic convergence order.',
        'ACC grid comparisons restrict simultaneous final depths, not maximum envelopes.',
        'FV1 independent grid-refinement and resolved time-dependent outlet/depth fixtures remain missing.']),indent=2)+'\n')


if __name__ == "__main__":
    fixtures = Path('validation/fixtures')
    results = Path('outputs/validation')
    output = Path('outputs/validation/sensitivity.json')
    profiles = ('physical_candidate','lisflood_fp_acc')
    cfl_values = (.4,.2,.1,.05)
    assess(fixtures=fixtures,results=results,output=output,profiles=profiles,cfl_values=cfl_values)
