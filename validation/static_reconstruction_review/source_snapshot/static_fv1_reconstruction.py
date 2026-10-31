"""Bounded reconstruction and larger coarsenable controls; not production AMR."""

from pathlib import Path
import json

import numpy as np

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.static_fv1 import StaticBlockFV1Solver
from floodwarp.static_fv1_mesh import restrict_state
from validation.sparse_fv1_benchmark import digest, select_checkpoints
from validation.static_fv1_assess import child_values
from validation.static_fv1_benchmark import run_benchmark, time_run


def reconstruction_screens(*, paths, depth_budgets, output):
    rows = []
    for path in paths:
        with np.load(file=path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            config = metadata['configuration']
            shape = (config['ny'], config['nx'])
            q = saved['q'].reshape(*shape, 3)
            bed = child_values(values=saved['bed'].reshape(shape))
            h = child_values(values=q[..., 0])
            wet = child_values(values=saved['active'].reshape(shape)).all(axis=2) & np.all(h > 0., axis=2)
            mean_h = h.mean(axis=2)
            reconstructed = mean_h[..., None] + (bed.mean(axis=2)[..., None] - bed)
            representable = wet & np.all(reconstructed > 0., axis=2)
            depth_error = np.max(np.abs(reconstructed - h), axis=2)
            homogeneous = np.ones(shape=wet.shape, dtype=bool)
            for name in ('manning', 'soil', 'rain', 'infil', 'f0', 'fd', 'conductivity', 'multiplier'):
                homogeneous &= np.ptp(child_values(values=saved[name].reshape(shape)), axis=2) == 0.
            momentum_errors = []
            transfer_errors = [float(np.max(np.abs(reconstructed[representable].sum(axis=1) - h[representable].sum(axis=1)), initial=0.))]
            for component in (1, 2):
                values = child_values(values=q[..., component])
                velocity = np.divide(values.mean(axis=2), mean_h, out=np.zeros_like(mean_h), where=mean_h > 0.)
                projected = reconstructed * velocity[..., None]
                momentum_errors.append(float(np.max(np.abs(projected[representable] - values[representable]), initial=0.)))
                transfer_errors.append(float(np.max(np.abs(projected[representable].sum(axis=1) - values[representable].sum(axis=1)), initial=0.)))
            errors = depth_error[representable]
            rows.append(dict(time_s=metadata['time'], checkpoint_sha256=digest(path=path),
                fully_wet_physical_parents=int(wet.sum()), positive_reconstruction_parents=int(representable.sum()),
                depth_error_quantiles_m=np.quantile(errors, (.5, .9, .99)).tolist() if errors.size else [],
                maximum_momentum_reconstruction_error_m2_s=momentum_errors,
                maximum_parent_total_roundoff=transfer_errors,
                screens=[dict(depth_budget_m=budget,
                    positive_depth_candidates=int(np.count_nonzero(representable & (depth_error <= budget))),
                    also_exact_source_history=int(np.count_nonzero(representable & (depth_error <= budget) & homogeneous)))
                    for budget in depth_budgets]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Diagnostic complete 2x2 parents only; odd raster strips omitted. '
        'No topology/connectivity, boundary or dynamic-transport qualification is inferred from reconstruction error. '
        'No future inundation field is used and this does not define a production mask.',
        restriction='Area means of conservative water and momentum',
        prolongation='Common free surface and velocity; parent totals preserved to recorded roundoff', rows=rows), indent=2) + '\n')


def curved_controls(*, shape, device, options, geometry_tolerances, mesh_options,
                    duration, repeats, graph, output):
    y = np.linspace(start=-1., stop=1., num=shape[0])[:, None]
    x = np.linspace(start=-1., stop=1., num=shape[1])[None, :]
    bed = options['bed_slope_height'] * np.broadcast_to(x, shape) + options['curvature_height'] * (x*x + y*y)
    cases = []
    for moving in (False, True):
        stage = options['stage'] + (options['surface_amplitude'] * np.sin(2. * np.pi * x) if moving else 0.)
        h = stage - bed
        fine = FullSWESolver(ny=shape[0], nx=shape[1], dx=options['spacing'], dy=options['spacing'],
            bed_elevation=bed, manning_n=options['manning'], active_mask=np.ones(shape=shape, dtype=bool),
            maximum_dt=options['maximum_dt'], device=device, execution_mode='dense')
        fine.set_water_depth(depth=h)
        if moving:
            fine.set_momentum(hu=options['velocity_x'] * h, hv=options['velocity_y'] * h)
        initial = fine.capture_state()
        for geometry_tolerance in geometry_tolerances:
            fine.restore_state(state=initial)
            coarse = StaticBlockFV1Solver.from_solver(solver=fine,
                protected=np.zeros(shape=shape, dtype=bool), geometry_tolerance=geometry_tolerance, **mesh_options)
            coarse_initial = coarse.capture_state()
            mesh = coarse._static_execution.mesh
            fine.step(fixed_dt=options['maximum_dt'])
            coarse.step(fixed_dt=options['maximum_dt'])
            commutator = np.max(np.abs(restrict_state(q=fine.q.numpy(), mesh=mesh) -
                restrict_state(q=coarse.q.numpy(), mesh=mesh)), axis=0).tolist()
            timings = {}
            for name, solver, state in (('dense', fine, initial), ('static', coarse, coarse_initial)):
                solver.restore_state(state=state)
                time_run(solver=solver, duration=duration, graph=graph)
                runs = []
                for repeat in range(repeats):
                    solver.restore_state(state=state)
                    runs.append(time_run(solver=solver, duration=duration, graph=graph))
                timings[name] = runs
            difference = coarse.get_conservative_state() - fine.get_conservative_state()
            cases.append(dict(moving=moving, geometry_tolerance_m=geometry_tolerance,
                diagnostics=coarse.execution_diagnostics(), initial_projection_error=coarse.initial_projection_error,
                depth_rmse_m=float(np.sqrt(np.mean(difference[..., 0]**2))),
                maximum_absolute_by_component=np.max(np.abs(difference), axis=(0, 1)).tolist(),
                minimum_depth_m=float(coarse.get_water_depth().min()),
                accounting=coarse.cumulative_accounting(), one_step_commutator=commutator, timings=timings,
                finest_bed_unchanged=bool(np.array_equal(coarse._fields['bed'].numpy(), initial['arrays']['bed']))))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(shape=shape, options=options, mesh_options=mesh_options,
        duration_s=duration, repeats=repeats, driver_sha256=digest(path=Path(__file__)),
        scope='Known smooth, fully wet connected synthetic terrain only. Relaxed affine-fit tolerance is a bounded experiment, '
        'not a general connectivity safeguard or production AMR qualification.', cases=cases), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    hours = (1, 13, 25, 48, 60, 72)
    depth_budgets = (.001, .01, .05)
    device = 'cuda:0'
    graph = True
    repeats = 3
    output_root = Path('validation/static_reconstruction_review')
    curved_shape = (128, 128)
    curved_options = dict(bed_slope_height=.05, curvature_height=.05, stage=.8,
        surface_amplitude=.001, velocity_x=.1, velocity_y=.05, spacing=1., manning=.03, maximum_dt=.01)
    geometry_tolerances = (1.e-12, .001)
    curved_mesh = dict(block_size=4, surface_tolerance=.002, momentum_tolerance=.01, source_tolerance=0.)
    curved_duration = .3
    large_shape = (512, 512)
    large_names = ('lake', 'front', 'ponding', 'channel')
    large_options = dict(depth=.4, spacing=1., manning=.03, maximum_dt=.02, bed_slope=.0002,
        bed_step=.1, lake_stage=.8, barrier_height=2., front_protection_cells=8,
        velocity_x=.2, velocity_y=.1, rain=.001, high_infiltration=.002, low_infiltration=.0002,
        soil_period_cells=3, decay_depth=.01)
    large_mesh = dict(surface_tolerance=.001, momentum_tolerance=.001, geometry_tolerance=1.e-12, source_tolerance=0.)
    block_sizes = (4, 8)
    large_duration = 2.
    large_interval = .5
    paths = select_checkpoints(directory=directory, hours=hours)
    reconstruction_screens(paths=paths, depth_budgets=depth_budgets, output=output_root / 'saved_state_screens.json')
    curved_controls(shape=curved_shape, device=device, options=curved_options,
        geometry_tolerances=geometry_tolerances, mesh_options=curved_mesh,
        duration=curved_duration, repeats=repeats, graph=graph, output=output_root / 'curved_controls.json')
    run_benchmark(names=large_names, shape=large_shape, device=device, options=large_options,
        mesh_options=large_mesh, block_sizes=block_sizes, duration=large_duration,
        interval=large_interval, repeats=repeats, graph=graph, output=output_root / 'large_controls.json')
