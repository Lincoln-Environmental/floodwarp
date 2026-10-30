"""Evaluate frozen criteria; discrepancies are reported, never retuned away."""
import hashlib
import json
from pathlib import Path

import numpy as np

from floodwarp import LocalInertialSolver, FullSWESolver
from floodwarp.rainfall import Hyetograph


def field_metrics(*, actual, reference, area):
    difference = actual-reference
    rmse = float(np.sqrt(np.mean(difference**2)))
    norm = float(np.linalg.norm(reference))
    relative = float(np.linalg.norm(difference)/max(norm, 1.e-30))
    wet_a, wet_b = actual > .001, reference > .001
    union = np.count_nonzero(wet_a | wet_b)
    jaccard = float(np.count_nonzero(wet_a & wet_b)/union) if union else 1.
    mismatch = float(np.mean(wet_a != wet_b))
    volumes_a = actual.sum(axis=(-2, -1))*area
    volumes_b = reference.sum(axis=(-2, -1))*area
    volume_error = float(np.max(np.abs(volumes_a-volumes_b)))
    # On thin Riemann strips the boundary band is along propagation (x),
    # with transverse wall operators reported separately in provenance.
    interior = difference[..., 3:-3]
    boundary = np.concatenate((difference[..., :3].ravel(), difference[..., -3:].ravel()))
    return dict(rmse_m=rmse, relative_l2=relative, wet_jaccard=jaccard, wet_mismatch=mismatch,
        volume_error_m3=volume_error, interior_x_rmse_m=float(np.sqrt(np.mean(interior**2))),
        boundary_x_rmse_m=float(np.sqrt(np.mean(boundary**2))),
        depth_pass=rmse <= .001 or relative <= .02,
        wet_pass=jaccard >= .95 or mismatch <= .0005,
        volume_pass=bool(np.all(np.abs(volumes_a-volumes_b) <= 5.+.02*np.maximum(volumes_a,volumes_b))))


def compare_li(*, directory, case, profile, device):
    scenario = json.loads((directory/'scenario.json').read_text())
    with np.load(directory/'physical_inputs.npz', allow_pickle=False) as data:
        bed, roughness, times, rates = (data[key].copy() for key in ('bed','roughness','times','rates'))
    with np.load(directory/'arrays.npz', allow_pickle=False) as data:
        reference = {key:data[key].copy() for key in data.files}
    with np.load(directory/'depth_series.npz', allow_pickle=False) as data:
        sample_times, reference_depths = data['times_s'], data['depth']
    spacing = scenario['spacing']
    solver = LocalInertialSolver(ny=bed.shape[0], nx=bed.shape[1], dx=spacing, dy=spacing,
        bed_elevation=bed, manning_n=roughness, active_mask=np.ones_like(bed, dtype=np.int32),
        device=device, validation_profile=profile, free_outflow_closure='local_or_prescribed_slope',
        maximum_dt=case['cap'], minimum_dt=case['minimum_dt'])
    solver.gravity = type(solver.gravity)(case['gravity'])
    solver.cfl_number = type(solver.cfl_number)(case['cfl'])
    x, y = solver.boundary_type_qx.numpy(), solver.boundary_type_qy.numpy()
    x[:, (0,-1)] = 2
    y[(0,-1), :] = 2
    solver.set_boundary_type_qx(boundary_type=x)
    solver.set_boundary_type_qy(boundary_type=y)
    solver.set_water_depth(depth=np.full_like(bed, scenario['initial']))
    solver.set_source_rates(infil=np.full_like(bed, scenario['capacity']/3.6e6))
    rain = Hyetograph(node_times_s=times, node_rates_m_per_s=rates, storm_type='reference', metadata={})
    depths, hydrograph, volumes = [], [], []
    for target in sample_times:
        solver.run_device_loop(duration=float(target-solver.simulation_time), rainfall_hyetograph=rain,
                               use_cuda_graph=device!='cpu')
        depths.append(solver.get_water_depth().copy())
        qx, qy = solver.get_qx(), solver.get_qy()
        outward = np.r_[-qx[:,0]*spacing, qx[:,-1]*spacing, -qy[0]*spacing, qy[-1]*spacing]
        hydrograph.append(float(np.maximum(outward,0.).sum()))
        volumes.append(solver.cumulative_accounting()['boundary_outflow'])
    ledger = solver.cumulative_accounting()
    storage = float(depths[-1].sum()*spacing**2)
    residual = solver.initial_volume + ledger['rain'] + ledger['boundary_inflow'] - ledger['infiltration'] - ledger['boundary_outflow'] - storage
    expected_q = np.interp(sample_times, reference['times'], reference['qout'])
    q_rmse = float(np.sqrt(np.mean((np.asarray(hydrograph)-expected_q)**2)))
    peak = float(np.max(np.abs(expected_q)))
    nrmse = q_rmse/peak if peak else (0. if q_rmse == 0. else float('inf'))
    export_error = abs(volumes[-1]-float(reference['cumulative_out'][-1]))
    final = field_metrics(actual=depths[-1], reference=reference['final'], area=spacing**2)
    temporal = field_metrics(actual=np.asarray(depths), reference=reference_depths, area=spacing**2)
    passed = (final['depth_pass'] and final['wet_pass'] and final['volume_pass'] and temporal['depth_pass'] and
        temporal['wet_pass'] and nrmse <= .05 and export_error <= 5.+.02*reference['cumulative_out'][-1] and
        abs(residual) <= 1.e-8*max(1.,solver.initial_volume+ledger['rain']+ledger['boundary_inflow']))
    return dict(case=case['id'], profile=profile, numerical=solver.numerical_configuration(),
        final=final, temporal=temporal, outlet_nrmse=nrmse if np.isfinite(nrmse) else None,
        outlet_absolute_rmse_m3_s=q_rmse, cumulative_outlet_error_m3=export_error, mass_residual_m3=residual,
        budget=ledger, finite=bool(np.isfinite(depths).all()), minimum_depth=float(np.min(depths)),
        passed=bool(passed), samples=sample_times.tolist(), outlet_m3_s=hydrograph, export_m3=volumes), np.asarray(depths)


def compare_fv1(*, directory, case, device):
    with np.load(directory/'inputs.npz', allow_pickle=False) as data:
        inputs = {key:data[key].copy() for key in data.files}
    with np.load(directory/'physical_final.npz', allow_pickle=False) as data:
        expected = data['state']
    bed = inputs['bed']
    solver = FullSWESolver(ny=bed.shape[0], nx=bed.shape[1], dx=float(inputs['dx']), dy=float(inputs['dy']),
        bed_elevation=bed, manning_n=inputs['roughness'], active_mask=inputs['mask'], device=device,
        gravity=case['gravity'], cfl=case['cfl'], dry_depth=case['dry_depth'], maximum_dt=case['cap'])
    solver.set_water_depth(depth=inputs['depth'])
    solver.set_momentum(hu=inputs['hu'], hv=inputs['hv'])
    solver.set_boundary_type_qx(boundary_type=inputs['bx'])
    solver.set_boundary_type_qy(boundary_type=inputs['by'])
    solver.set_boundary_stage_qx(stage=inputs['sx'])
    solver.set_boundary_stage_qy(stage=inputs['sy'])
    solver.run_device_loop(duration=float(inputs['duration']), use_cuda_graph=device!='cpu')
    state = solver.get_conservative_state()
    ledger = solver.cumulative_accounting()
    final = field_metrics(actual=state[...,0], reference=expected[...,0], area=solver.dx*solver.dy)
    native = json.loads((directory/'report.json').read_text())
    native_export = native['final_budget']['outflow_m3']
    export_error = abs(ledger['boundary_outflow']-native_export)
    passed = (final['depth_pass'] and final['wet_pass'] and final['volume_pass'] and
        export_error <= 5.+.02*native_export and
        abs(ledger['residual']) <= 1.e-8*max(1.,solver.initial_volume+ledger['rain']+ledger['boundary_inflow']))
    return dict(case=case['id'], numerical=solver.numerical_configuration(), final=final,
        native_budget=native['final_budget'], cumulative_outlet_error_m3=export_error,
        mean_outlet_m3_s=ledger['boundary_outflow']/float(inputs['duration']),
        native_mean_outlet_m3_s=native_export/float(inputs['duration']),
        momentum_rmse=float(np.sqrt(np.mean((state[...,1:]-expected[...,1:])**2))),
        mass_residual_m3=ledger['residual'], budget=ledger,
        finite=bool(np.isfinite(state).all()), minimum_depth=float(state[...,0].min()), passed=bool(passed),
        limitations='Final-state fixture and cumulative outlet only; no resolved temporal depth or instantaneous hydrograph. Riemann cases have no meaningful outlet flow; the retained outlet crop supplies the nonzero-export gate. Stage/wall operators remain independent.'), state[...,0]


def run_comparisons(*, fixtures: Path, output: Path, device, profiles):
    manifest = json.loads((fixtures/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in manifest['cases']:
        directory = fixtures/case['id']
        for name, expected in case['files'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Reference checksum mismatch: '+str(directory/name))
        selected = profiles if case['engine'].endswith('ACC') else (None,)
        for profile in selected:
            if profile is None:
                row, depths = compare_fv1(directory=directory, case=case, device=device)
            else:
                row, depths = compare_li(directory=directory, case=case, profile=profile, device=device)
            label = case['id'] + ('' if profile is None else '_'+profile)
            np.savez_compressed(file=output/(label+'.npz'), depth=depths)
            row['passed'] = row['passed'] and row['finite'] and row['minimum_depth'] >= 0.
            rows.append(row)
            print(label, 'PASS' if row['passed'] else 'DISCREPANCY', row['final']['rmse_m'], flush=True)
    report = dict(device=device, acceptance='validation/ACCEPTANCE.md', passed=sum(row['passed'] for row in rows),
        total=len(rows), comparisons=rows,
        limitations=['FV1 grid refinement and temporal fields/outlet controls are not included in the retained small Riemann fixtures.',
        'ACC physical versus compatibility profiles have deliberate friction, source and wet-threshold differences.',
        'A passing field score does not establish universal hydraulic validity.'])
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == "__main__":
    fixtures = Path('validation/fixtures')
    output = Path('outputs/validation')
    device = 'cpu'
    profiles = ('physical_candidate', 'lisflood_fp_acc')
    run_comparisons(fixtures=fixtures, output=output, device=device, profiles=profiles)
