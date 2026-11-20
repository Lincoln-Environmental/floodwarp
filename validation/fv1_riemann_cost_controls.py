"""Paired CFL controls and cost measurements for shocks and a synthetic catchment."""

from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import warp as wp

from floodwarp import full_swe_kernels as kernels
from validation.fv1_riemann import report_json
from validation.fv1_riemann_cases import analytical_domain, cell_averages, coordinates, synthetic_domain
from validation.fv1_riemann_gpu import make_solver, profile, repetitions, timed_graph
from validation.static_review.source_snapshot import full_swe_kernels as archived


def cfl_parity(*, device, seed, size, trials):
    """Isolate both star reductions from other CFL constraints and timestep caps."""
    rng = np.random.default_rng(seed=seed)
    differences = []
    for i in range(trials):
        nx, ny = size
        count = nx*ny
        depth = 10.**rng.uniform(low=-7., high=1., size=count)
        depth[::9] = 0.
        momentum = depth[:, None]*rng.uniform(low=-20., high=20., size=(count, 2))
        q = wp.array(np.column_stack((depth, momentum)), dtype=wp.vec3d, device=device)
        bed = rng.uniform(low=-2., high=2., size=count)
        mask = rng.random(size=count) > .1
        mask[0] = True
        fields = [bed, rng.random(count)*.001, rng.random(count), rng.random(count)*.0001,
                  rng.random(count)*.001, .1+rng.random(count), rng.random(count)*.00001, rng.random(count)]
        faces = ((nx+1)*ny, nx*(ny+1))
        arrays = [q, wp.array(mask.astype(np.int32), device=device)]
        arrays.extend(wp.array(a, device=device) for a in fields)
        arrays.extend(wp.array(rng.integers(0, 4, size=n, dtype=np.int32), device=device) for n in faces)
        arrays.extend(wp.array(rng.random(n)*3., device=device) for n in faces)
        measured = []
        for module in (archived, kernels):
            initial = np.zeros(20)
            initial[[2, 14]] = np.inf
            initial[[8, 12]] = (.25, .001)
            ctrl = wp.array(initial, device=device)
            wp.launch(kernel=module.star_cfl_reduce, dim=count,
                      inputs=[*arrays, nx, ny, ctrl, 2., 5., 9.81, .0001, .4, i % 3], device=device)
            measured.append(ctrl.numpy()[[2, 14]])
        np.testing.assert_allclose(actual=measured[1], desired=measured[0], rtol=5.e-14, atol=0.)
        differences.append(float(np.max(np.abs(measured[1]-measured[0]))))
    return dict(device=device, trials=trials, seed=seed, max_absolute_dt_difference_s=max(differences))


def paired_profile(*, domain, controls, device, repeats):
    settings = dict(domain=domain, scheme='roe', policy='adaptive', controls=controls, device=device)
    versions = {'archived': archived.star_cfl_reduce, 'corrected': kernels.star_cfl_reduce}
    samples = {version: [] for version in versions}
    for i in range(repeats+1):
        order = list(versions) if i % 2 == 0 else list(reversed(versions))
        for version in order:
            with patch.object(kernels, 'star_cfl_reduce', versions[version]):
                solver = make_solver(**settings)
                graph_result = timed_graph(solver=solver, device=device, duration=domain.duration)
                state = solver.get_conservative_state()
                if i == 0 and version == 'archived':
                    baseline = state
                    steps = graph_result['steps']
                np.testing.assert_array_equal(actual=state, desired=baseline)
                assert graph_result['steps'] == steps
                if i > 0:
                    item = profile(schemes=('roe',), domain=domain, controls=controls, repeats=1, device=device)[0]
                    samples[version].append(dict(adaptive_graph=graph_result,
                        common_dt_kernel_us_per_step=item['median_kernel_us_per_step']))
    return samples


def additional_cost(*, directory, schemes, spacings, controls, device, repeats):
    data = json.loads((directory/'grid.json').read_text())
    rows = []
    for name in ('ritter', 'jump', 'jump_offset', 'irregular'):
        for dx in spacings:
            if name == 'irregular':
                domain = synthetic_domain(name=name, dx=dx, length=320., width=160.)
                exact = None
            else:
                domain = analytical_domain(name=name, dx=dx, length=640., width=320., gravity=controls['gravity'])
                x, _ = coordinates(length=640., width=320., dx=dx)
                exact = cell_averages(name=name, x=x, dx=dx, time=domain.duration, gravity=controls['gravity'])
            series = repetitions(domain=domain, schemes=schemes, policy='common', controls=controls, device=device,
                                 repeats=repeats, exact=exact)
            for row in series:
                row['case'] = name
                if name == 'irregular':
                    matching = next(r for r in data['rows'] if r['case'] == name and r['dx_m'] == dx
                                    and r['scheme'] == row['scheme'] and r['refinement'] == 1)
                    row['reference_errors'] = matching['deep_reference']
            rows.extend(series)
    return rows


if __name__ == "__main__":
    directory = Path('validation/fv1_riemann_review/corrected')
    controls = dict(gravity=9.81, dry=.0001, wet_depth=.01, cfl=.4, dt_scale=.015625, adaptive_cap=1.)
    gpu = 'cuda:0'
    spacings = (10., 5.)
    schemes = ('legacy_hll', 'contact_hll', 'star_contact_hll', 'roe', 'hllc')
    repeats = 7
    profile_domain = synthetic_domain(name='expansion', dx=5., length=640., width=320.)
    parity = [cfl_parity(device=device, seed=782, size=(9, 7), trials=128) for device in ('cpu', gpu)]
    paired = paired_profile(domain=profile_domain, controls=controls, device=gpu, repeats=repeats)
    rows = additional_cost(directory=directory, schemes=schemes, spacings=spacings,
                           controls=controls, device=gpu, repeats=repeats)
    report_json(path=directory/'cost_controls.json', content=dict(cfl_parity=parity, paired_roe=paired, rows=rows))
