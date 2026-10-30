"""Small direct-input orchestration layer; numerical operators stay in solvers."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from floodwarp import FullSWESolver, create_overland_solver
from floodwarp.inputs import Terrain, boundary_faces, field_array, validate_boundaries
from floodwarp.rainfall import Hyetograph
from floodwarp.checkpoints import durable_checkpoint, restore_complete


@dataclass
class Model:
    solver: object
    terrain: Terrain
    rainfall: Hyetograph | None
    stage_series_x: tuple | None
    stage_series_y: tuple | None
    identity: dict


def series_values(*, series, name):
    times, values = (np.asarray(value, dtype=np.float64) for value in series)
    if (times.ndim != 1 or times.size < 2 or values.shape != times.shape or
        not np.isfinite(times).all() or not np.isfinite(values).all() or
        times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError(f"{name} needs finite increasing nodes starting at zero")
    return times, values


def stage_field(*, value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0 and array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Stage must be a finite scalar or explicit face field")
    return np.broadcast_to(array, shape).copy()


def create_model(*, terrain: Terrain, solver_method="li", manning_n=.03,
                 initial_depth=None, initial_stage=None, rainfall=0., rainfall_series=None,
                 infiltration=0., boundary_qx=None, boundary_qy=None,
                 stage_qx=0., stage_qy=0., stage_series_qx=None, stage_series_qy=None,
                 stage_datum=None, device="cpu", **solver_options):
    if initial_depth is not None and initial_stage is not None:
        raise ValueError("Specify initial depth or initial stage, never both")
    boundaries = boundary_faces(terrain=terrain)
    x, y = validate_boundaries(terrain=terrain,
        qx=boundaries[0] if boundary_qx is None else boundary_qx,
        qy=boundaries[1] if boundary_qy is None else boundary_qy)
    has_stage = np.isin(x, (3, 5, 10)).any() or np.isin(y, (3, 4, 11)).any()
    if (has_stage or initial_stage is not None) and stage_datum != terrain.metadata['vertical_datum']:
        raise ValueError("Stage datum must explicitly match terrain vertical datum")
    roughness = field_array(values=manning_n, terrain=terrain, name='Manning n')
    if np.any(roughness[terrain.active] <= 0):
        raise ValueError("Manning n must be positive")
    bed = np.where(terrain.active, terrain.elevation, 0.)
    solver = create_overland_solver(solver_method=solver_method, ny=bed.shape[0], nx=bed.shape[1],
        dx=terrain.dx, dy=terrain.dy, bed_elevation=bed, manning_n=roughness,
        active_mask=terrain.active.astype(np.int32), device=device, **solver_options)
    # LI requires perimeter codes even where both sides are inactive; FV1
    # requires zero on such faces. This is representation-only, with no flux.
    if not isinstance(solver, FullSWESolver):
        active = terrain.active
        x[:, 1:-1][~(active[:, :-1] & active[:, 1:]) & (x[:, 1:-1] == 0)] = 1
        y[1:-1][~(active[:-1] & active[1:]) & (y[1:-1] == 0)] = 1
        x[:, 0] = np.where(x[:, 0] == 0, 1, x[:, 0])
        x[:, -1] = np.where(x[:, -1] == 0, 1, x[:, -1])
        y[0] = np.where(y[0] == 0, 1, y[0])
        y[-1] = np.where(y[-1] == 0, 1, y[-1])
    solver.set_boundary_type_qx(boundary_type=x)
    solver.set_boundary_type_qy(boundary_type=y)
    sx = stage_field(value=stage_qx, shape=x.shape)
    sy = stage_field(value=stage_qy, shape=y.shape)
    solver.set_boundary_stage_qx(stage=sx)
    solver.set_boundary_stage_qy(stage=sy)
    if initial_stage is not None:
        stage = field_array(values=initial_stage, terrain=terrain, name='initial stage', nonnegative=False)
        if getattr(solver, 'hydraulic_geometry_mode', 'coarse') != 'coarse':
            solver.set_water_surface_elevation(stage=stage)
        else:
            solver.set_water_depth(depth=np.maximum(stage-bed, 0.))
    else:
        solver.set_water_depth(depth=field_array(values=0. if initial_depth is None else initial_depth,
                                                  terrain=terrain, name='initial depth'))
    rain = field_array(values=rainfall, terrain=terrain, name='rainfall')
    infil = field_array(values=infiltration, terrain=terrain, name='infiltration')
    solver.set_source_rates(rain=rain, infil=infil)
    hyetograph = None
    if rainfall_series is not None:
        if np.any(rain):
            raise ValueError("Use rainfall_series with zero rainfall; set spatial multiplier on the solver explicitly")
        times, rates = series_values(series=rainfall_series, name='rainfall')
        hyetograph = Hyetograph(node_times_s=times, node_rates_m_per_s=rates, storm_type='direct', metadata={})
        if isinstance(solver, FullSWESolver):
            solver.set_rainfall_hyetograph(times_s=times, rates_m_s=rates)
    tx = None if stage_series_qx is None else series_values(series=stage_series_qx, name='qx stage')
    ty = None if stage_series_qy is None else series_values(series=stage_series_qy, name='qy stage')
    if tx is not None and not np.isin(x, (3, 5, 10)).any() or ty is not None and not np.isin(y, (3, 4, 11)).any():
        raise ValueError("Stage series requires prescribed-stage faces on its axis")
    identity = dict(numerical=solver.numerical_configuration(), dx=terrain.dx, dy=terrain.dy,
                    vertical_datum=terrain.metadata['vertical_datum'])
    fields = dict(bed=bed, roughness=roughness, active=terrain.active, bx=x, by=y, sx=sx, sy=sy, rain=rain, infil=infil)
    if hasattr(solver, '_native_bed'):
        fields['native_bed'] = solver._native_bed
        fields['native_manning'] = solver._native_manning
    if getattr(solver, '_subgrid_geometry', None) is not None:
        fields['native_input'] = np.asarray(solver_options.get('subgrid_bed_elevation'))
    identity['fields'] = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in fields.items()}
    identity['forcing'] = dict(rainfall=None if hyetograph is None else [times.tolist(), rates.tolist()],
        stage_x=None if tx is None else [a.tolist() for a in tx], stage_y=None if ty is None else [a.tolist() for a in ty])
    return Model(solver=solver, terrain=terrain, rainfall=hyetograph, stage_series_x=tx, stage_series_y=ty, identity=identity)


def model_time(*, model):
    return model.solver.time if isinstance(model.solver, FullSWESolver) else model.solver.simulation_time


def advance(*, model, end_time, use_cuda_graph=False):
    solver = model.solver
    start = model_time(model=model)
    if not np.isfinite(end_time) or end_time <= start:
        raise ValueError("End time must be finite and later than current time")
    for series in (model.stage_series_x, model.stage_series_y):
        if series is not None and end_time > series[0][-1]:
            raise ValueError("Stage series must cover the entire simulation")
    if model.rainfall is not None and end_time > model.rainfall.duration_s:
        raise ValueError("Rainfall series must cover simulation, including explicit zero-rain recession")
    if isinstance(solver, FullSWESolver):
        if model.stage_series_x is not None or model.stage_series_y is not None:
            if use_cuda_graph:
                raise ValueError("Time-varying FV1 stages require host stepping; constant stages support CUDA graphs")
            while solver.time < end_time:
                remaining = end_time - solver.time
                for axis, series, shape in (('x', model.stage_series_x, (solver.ny, solver.nx+1)),
                                             ('y', model.stage_series_y, (solver.ny+1, solver.nx))):
                    if series is not None:
                        times, stages = series
                        value = np.interp(solver.time, times, stages)
                        setter = solver.set_boundary_stage_qx if axis == 'x' else solver.set_boundary_stage_qy
                        setter(stage=np.full(shape, value))
                        later = times[times > solver.time]
                        if later.size:
                            remaining = min(remaining, later[0]-solver.time)
                solver.step(remaining_time=remaining)
        else:
            solver.run_device_loop(duration=end_time-start, use_cuda_graph=use_cuda_graph)
    else:
        solver.run_device_loop(duration=end_time-start, rainfall_hyetograph=model.rainfall,
            stage_series_qx=model.stage_series_x, stage_series_qy=model.stage_series_y,
            use_cuda_graph=use_cuda_graph)


def diagnostics(*, model):
    solver = model.solver
    ledger = solver.cumulative_accounting()
    storage = float(np.sum(solver.get_water_depth()[model.terrain.active])*model.terrain.dx*model.terrain.dy)
    residual = (solver.initial_volume + ledger['rain'] + ledger.get('other_sources', 0.) +
        ledger.get('additional_sources', 0.) + ledger['boundary_inflow'] -
        ledger['infiltration'] - ledger['boundary_outflow'] - storage)
    return dict(time_s=model_time(model=model), storage_m3=storage, residual_m3=residual,
                numerical=solver.numerical_configuration(), **ledger)


def save_checkpoint(*, model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model.solver, FullSWESolver):
        model.solver.save_checkpoint(path=path)
    else:
        durable_checkpoint(surface=model.solver, path=path, signature=checkpoint_identity(model=model), report={})
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(dict(source_commit='ba503b8bde51d3f212199e81efc7038d9fffecc8',
        identity=checkpoint_identity(model=model), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2)+'\n')


def load_checkpoint(*, model, path: Path):
    metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if metadata['identity'] != checkpoint_identity(model=model) or metadata['checkpoint_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Checkpoint provenance or direct inputs do not match")
    if isinstance(model.solver, FullSWESolver):
        model.solver.load_checkpoint(path=path)
    else:
        restore_complete(surface=model.solver, path=path, signature=checkpoint_identity(model=model))


def checkpoint_identity(*, model):
    """Bind restart to explicit numerical controls and configured soil physics."""
    solver = model.solver
    identity = dict(model.identity)
    controls = {}
    for name in ('gravity', 'cfl', 'cfl_number', 'maximum_dt', 'minimum_dt',
                 'dry_depth', 'adaptive_dt_growth_factor', 'infiltration_method'):
        value = getattr(solver, name, None)
        controls[name] = None if value is None else float(value)
    fields = {}
    names = ('infiltration_capacity', 'infiltration_f0', 'infiltration_fd', 'infiltration_k',
             'green_ampt_ks', 'green_ampt_psi', 'green_ampt_delta_theta')
    if isinstance(solver, FullSWESolver):
        candidates = {name:solver._fields[name] for name in ('infil', 'f0', 'fd', 'conductivity')}
    else:
        candidates = {name:getattr(solver, name) for name in names}
        if solver.infiltration_method in (1, 3):
            candidates.pop('infiltration_capacity')  # Derived live capacity, not an immutable soil parameter.
    for name, value in candidates.items():
        array = value.numpy()
        fields[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                           sha256=hashlib.sha256(array.tobytes()).hexdigest())
    identity.update(controls=controls, soil_parameters=fields)
    return identity
