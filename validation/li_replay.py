"""Non-invasive baseline replay; settings belong to the execution guard."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np


def digest_array(*, value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def replay(*, namespace, fixtures, output, device, profiles, closures, history_capacity, case_ids=None, shipped_defaults=False):
    solver_type = importlib.import_module(name=namespace + '.local_inertial_solver').LocalInertialSolver
    rain_type = importlib.import_module(name=namespace + '.rainfall').Hyetograph
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in manifest['cases']:
        if not case['engine'].endswith('ACC') or (case_ids is not None and case['id'] not in case_ids):
            continue
        directory = fixtures / case['id']
        for name, expected in case['files'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        scenario = json.loads((directory / 'scenario.json').read_text())
        with np.load(file=directory / 'physical_inputs.npz') as data:
            bed, roughness, times, rates = (data[key].copy() for key in ('bed', 'roughness', 'times', 'rates'))
        with np.load(file=directory / 'depth_series.npz') as data:
            samples = data['times_s'].copy()
        assert max(bed.shape) <= 100
        for profile in profiles:
            for closure in closures:
                options = dict(ny=bed.shape[0], nx=bed.shape[1], dx=scenario['spacing'], dy=scenario['spacing'],
                    bed_elevation=bed, manning_n=roughness, active_mask=np.ones_like(bed, dtype=np.int32), device=device,
                    maximum_dt=case['cap'], minimum_dt=case['minimum_dt'])
                if profile is not None:
                    options['validation_profile'] = profile
                if closure is not None:
                    options['free_outflow_closure'] = closure
                if shipped_defaults and namespace == 'overland_flow':
                    options.pop('maximum_dt')
                    options.pop('minimum_dt')
                    solver = solver_type(**options)
                elif shipped_defaults:
                    inputs_api = importlib.import_module(name=namespace + '.inputs')
                    model_api = importlib.import_module(name=namespace + '.model')
                    terrain = inputs_api.terrain_from_array(elevation=bed, dx=scenario['spacing'], dy=scenario['spacing'],
                        row_order='south_to_north', vertical_datum='local')
                    solver = model_api.create_model(terrain=terrain, manning_n=roughness, device=device).solver
                else:
                    solver = solver_type(**options)
                    solver.gravity = type(solver.gravity)(case['gravity'])
                    solver.cfl_number = type(solver.cfl_number)(case['cfl'])
                bx, by = solver.boundary_type_qx.numpy().copy(), solver.boundary_type_qy.numpy().copy()
                bx[:, (0, -1)] = 2
                by[(0, -1), :] = 2
                solver.set_boundary_type_qx(boundary_type=bx)
                solver.set_boundary_type_qy(boundary_type=by)
                solver.set_water_depth(depth=np.full_like(bed, scenario['initial']))
                solver.set_source_rates(infil=np.full_like(bed, scenario['capacity'] / 3.6e6))
                rain = rain_type(node_times_s=times, node_rates_m_per_s=rates, storm_type='reference', metadata={})
                fields = {key: [] for key in ('depth', 'stage', 'qx', 'qy', 'applied_rain', 'applied_infil', 'applied_other', 'ledger', 'residual')}
                steps = []
                ledger_keys = None
                for target in samples:
                    trace = solver.run_device_loop(duration=float(target - solver.simulation_time), rainfall_hyetograph=rain,
                        use_cuda_graph=False, dt_history_capacity=history_capacity)
                    assert len(trace['dt_history']) == trace['steps']
                    steps.extend(trace['dt_history'].tolist())
                    depth = solver.get_water_depth().copy()
                    fields['depth'].append(depth)
                    fields['stage'].append(bed + depth)
                    fields['qx'].append(solver.get_qx().copy())
                    fields['qy'].append(solver.get_qy().copy())
                    for key in ('applied_rain', 'applied_infil', 'applied_other'):
                        fields[key].append(getattr(solver, key).numpy().copy())
                    ledger = solver.cumulative_accounting()
                    ledger_keys = list(ledger)
                    fields['ledger'].append(list(ledger.values()))
                    residual = solver.initial_volume + ledger['rain'] + ledger['other_sources'] + ledger['boundary_inflow'] - ledger['infiltration'] - ledger['boundary_outflow'] - float(depth.sum() * scenario['spacing']**2)
                    fields['residual'].append(residual)
                arrays = {key: np.asarray(value) for key, value in fields.items()}
                arrays['dt'] = np.asarray(steps)
                arrays['times'] = samples
                profile_name = getattr(profile, 'name', profile)
                label = case['id'] + '__' + str(profile_name) + '__' + str(closure)
                np.savez_compressed(file=output / (label + '.npz'), **arrays)
                row = dict(label=label, case=case['id'], profile=profile_name, closure=closure, numerical=solver.numerical_configuration(),
                    gravity=float(solver.gravity), cfl=float(solver.cfl_number), cap=float(solver.maximum_dt), minimum_dt=float(solver.minimum_dt),
                    device=device, dtype=str(depth.dtype), steps=len(steps), ledger_keys=ledger_keys,
                    hashes={key: digest_array(value=value) for key, value in arrays.items()},
                    finite=bool(np.isfinite(arrays['depth']).all()), minimum_depth=float(arrays['depth'].min()),
                    max_mass_residual_m3=float(np.max(np.abs(arrays['residual']))))
                rows.append(row)
                (output / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(label, len(steps), row['max_mass_residual_m3'], flush=True)


if __name__ == "__main__":
    namespace = os.environ['REPLAY_NAMESPACE']
    fixtures = Path(os.environ['REPLAY_FIXTURES'])
    output = Path(os.environ['REPLAY_OUTPUT'])
    device = 'cpu'
    shipped_defaults = os.environ.get('REPLAY_DEFAULTS', '0') == '1'
    profiles = (None,) if shipped_defaults else ('physical_candidate', 'lisflood_fp_acc', None)
    closures = (None,) if shipped_defaults else ('local_or_prescribed_slope', None)
    history_capacity = 200000
    replay(namespace=namespace, fixtures=fixtures, output=output, device=device, profiles=profiles,
           closures=closures, history_capacity=history_capacity, shipped_defaults=shipped_defaults)
