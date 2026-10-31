"""Diagnostic constant/affine free-surface reconstruction; no mesh evolution."""

from pathlib import Path
import json

import numpy as np

from validation.sparse_fv1_benchmark import digest, select_checkpoints
from validation.static_fv1_assess import child_values


def reconstructed_depth(*, h, bed, affine, x, y):
    mean_h = h.mean(axis=-1, keepdims=True)
    relative_bed = bed - bed.mean(axis=-1, keepdims=True)
    result = mean_h - relative_bed
    if affine:
        relative_surface = h + relative_bed
        gx = np.sum(relative_surface * x, axis=-1, keepdims=True) / np.sum(x*x)
        gy = np.sum(relative_surface * y, axis=-1, keepdims=True) / np.sum(y*y)
        result = result + gx*x + gy*y
    # Correct only transfer roundoff in the state, never modify finest bed.
    return result + (mean_h - result.mean(axis=-1, keepdims=True))


def assess(*, paths, budgets, x, y, analytical_bed, lake_stage, film_depth, output):
    plane_bed = np.array([0., 1., 2., 3.]) * analytical_bed
    film = np.full(shape=plane_bed.shape, fill_value=film_depth)
    film_projected = reconstructed_depth(h=film, bed=plane_bed, affine=True, x=x, y=y)
    lake_bed = plane_bed.copy()
    lake_bed[-1] += analytical_bed
    lake = lake_stage - lake_bed
    lake_projected = reconstructed_depth(h=lake, bed=lake_bed, affine=True, x=x, y=y)
    assert np.max(np.abs(film_projected - film)) < 1.e-14
    assert np.max(np.abs(lake_projected - lake)) < 1.e-14
    rows = []
    for path in paths:
        with np.load(file=path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            config = metadata['configuration']
            shape = (config['ny'], config['nx'])
            h = child_values(values=saved['q'].reshape(*shape, 3)[..., 0])
            bed = child_values(values=saved['bed'].reshape(shape))
            wet = child_values(values=saved['active'].reshape(shape)).all(axis=-1) & np.all(h > 0., axis=-1)
            variants = {}
            for affine in (False, True):
                projected = reconstructed_depth(h=h, bed=bed, affine=affine, x=x, y=y)
                positive = wet & np.all(projected > 0., axis=-1)
                error = np.max(np.abs(projected - h), axis=-1)
                variants['affine_surface' if affine else 'constant_surface'] = dict(
                    positive_parents=int(positive.sum()),
                    parent_water_transfer_roundoff_m3=float(np.max(np.abs(projected[wet].sum(axis=-1) - h[wet].sum(axis=-1)), initial=0.) * config['dx'] * config['dy']),
                    screens=[dict(depth_budget_m=budget, parents=int(np.count_nonzero(positive & (error <= budget)))) for budget in budgets])
            rows.append(dict(time_s=metadata['time'], checkpoint_sha256=digest(path=path),
                fully_wet_physical_parents=int(wet.sum()), variants=variants))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Reconstruction diagnostic only. No topology/connectivity, boundary, source, momentum or dynamical-error gate is waived. '
        'An affine surface requires additional hydraulic degrees of freedom; it is not implemented in StaticBlockFV1Solver. '
        'Negative depths are rejected, never clipped or deleted.',
        analytical_controls=dict(sloping_film_error_m=float(np.max(np.abs(film_projected-film))),
            nonaffine_bed_lake_error_m=float(np.max(np.abs(lake_projected-lake)))),
        driver_sha256=digest(path=Path(__file__)), rows=rows), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 13, 25, 48, 60, 72)
    budgets = (.001, .01, .05)
    x = np.array([-.5, .5, -.5, .5])
    y = np.array([-.5, -.5, .5, .5])
    analytical_bed = .1
    lake_stage = 2.
    film_depth = .001
    output = Path('validation/static_reconstruction_review/affine_surface_screens.json')
    paths = select_checkpoints(directory=directory, hours=hours)
    assess(paths=paths, budgets=budgets, x=x, y=y, analytical_bed=analytical_bed,
        lake_stage=lake_stage, film_depth=film_depth, output=output)
