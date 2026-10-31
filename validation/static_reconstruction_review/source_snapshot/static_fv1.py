"""Experimental, fixed two-level block FV1 with conservative shared subfaces.

Fine arrays are geometry, source history, diagnostic/projection scratch. Only
nonoverlapping leaf volumes are evolved hydraulically. This first correctness
implementation deliberately retains dense face scratch and the existing fine
source-aware global CFL bound. It makes no sparse-storage claim.
"""

from pathlib import Path
import hashlib
import json
import os

import numpy as np
import warp as wp

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.static_fv1_mesh import build_mesh, restrict_state, prolong_state
from floodwarp import static_fv1_kernels as kernels


class StaticBlockExecution:
    def __init__(self, *, solver, mesh, configuration):
        self.mesh = mesh
        self.configuration = configuration
        self.topology = {name: getattr(mesh, name) for name in
            ('owner', 'counts', 'mean_bed', 'manning', 'level', 'blocks', 'faces_x', 'faces_y', 'coarse_roots')}
        self.identity = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in self.topology.items()}
        self.fields = {}
        for name in ('owner', 'counts', 'faces_x', 'faces_y'):
            self.fields[name] = wp.array(self.topology[name].ravel(), dtype=wp.int32, device=solver.device)
        for name in ('mean_bed', 'manning'):
            self.fields[name] = wp.array(self.topology[name], dtype=wp.float64, device=solver.device)
        size = mesh.counts.size
        for name in ('leaves', 'delta'):
            self.fields[name] = wp.zeros(size, dtype=wp.vec3d, device=solver.device)
        for name in ('outgoing', 'theta'):
            self.fields[name] = wp.zeros(size, dtype=wp.float64, device=solver.device)

    def enqueue_projection(self, *, solver, record_history):
        f, s = solver._fields, self.fields
        wp.launch(kernels.project_leaves, dim=solver.nx * solver.ny,
            inputs=[s['leaves'], s['owner'], s['counts'], f['bed'], s['mean_bed'], solver.q,
                    f['theta'], s['theta'], f['maximum'], solver.ctrl, solver.dry_depth, int(record_history)], device=solver.device)

    def enqueue(self, *, solver):
        f, s = solver._fields, self.fields
        common = [solver.nx, solver.ny]
        s['leaves'].zero_()
        s['delta'].zero_()
        s['outgoing'].zero_()
        wp.launch(kernels.restrict_sources, dim=solver.nx * solver.ny,
            inputs=[solver.q, s['owner'], s['counts'], s['leaves']], device=solver.device)
        self.enqueue_projection(solver=solver, record_history=False)
        for axis, index in (('x', 0), ('y', 1)):
            f['flux_' + axis].zero_()
            f['pressure_' + axis].zero_()
            wp.launch(kernels.subfaces, dim=s['faces_' + axis].size,
                inputs=[s['faces_' + axis], solver.q, f['bed'], f['active'], f['types_' + axis],
                        f['stages_' + axis], f['flux_' + axis], f['pressure_' + axis], *common, index,
                        solver.gravity, solver.dry_depth, solver._stage_momentum_mode,
                        solver.ctrl, solver._flux_mode], device=solver.device)
        for axis, index, width in (('x', 0, solver.dy), ('y', 1, solver.dx)):
            wp.launch(kernels.collect_outflows, dim=s['faces_' + axis].size,
                inputs=[s['faces_' + axis], f['flux_' + axis], s['owner'], s['outgoing'],
                        *common, index, width], device=solver.device)
        wp.launch(kernels.donor_budgets, dim=s['leaves'].size,
            inputs=[s['leaves'], s['counts'], s['outgoing'], s['theta'], solver.ctrl,
                    solver.dx * solver.dy], device=solver.device)
        for axis, index, width in (('x', 0, solver.dy), ('y', 1, solver.dx)):
            wp.launch(kernels.accept_subfaces, dim=s['faces_' + axis].size,
                inputs=[s['faces_' + axis], f['flux_' + axis], f['pressure_' + axis], s['owner'],
                        s['theta'], s['delta'], f['volume_' + axis], f['inflow'], f['outflow'],
                        solver.ctrl, *common, index, width], device=solver.device)
        wp.launch(kernels.update_leaves, dim=s['leaves'].size,
            inputs=[s['leaves'], s['delta'], s['counts'], s['manning'], solver.ctrl,
                    solver.dx * solver.dy, solver.gravity, solver.dry_depth], device=solver.device)
        self.enqueue_projection(solver=solver, record_history=True)


class StaticBlockFV1Solver(FullSWESolver):
    """Opt-in experimental solver; construct from a configured uniform solver.

    The fixed mesh cannot encode a partly dry coarse cell over sloping terrain.
    Such evolution raises rather than silently deleting/redistributing water.
    Public face arrays describe represented subfaces; internal coarse-cell
    transport is unresolved and is not manufactured for diagnostics.
    """

    @classmethod
    def from_solver(cls, *, solver, block_size, protected, surface_tolerance,
                    momentum_tolerance, geometry_tolerance, source_tolerance, requested_roots=None):
        state = solver.capture_state()
        mesh = build_mesh(state=state, block_size=block_size, protected=protected,
            surface_tolerance=surface_tolerance, momentum_tolerance=momentum_tolerance,
            geometry_tolerance=geometry_tolerance, source_tolerance=source_tolerance, requested_roots=requested_roots)
        result = cls(**solver._configuration(), bed_elevation=solver._fields['bed'].numpy().reshape(solver._shape),
            manning_n=solver._fields['manning'].numpy().reshape(solver._shape), active_mask=solver._active,
            device=solver.device, execution_mode='dense')
        result.restore_state(state=state)
        configuration = dict(block_size=block_size, surface_tolerance=surface_tolerance,
            momentum_tolerance=momentum_tolerance, geometry_tolerance=geometry_tolerance,
            source_tolerance=source_tolerance, source_transfer='fine_common_surface_common_velocity_then_conservative_restriction',
            timestep='existing_finest_global_source_and_boundary_aware', levels=2)
        result._static_execution = StaticBlockExecution(solver=result, mesh=mesh, configuration=configuration)
        leaves = restrict_state(q=state['arrays']['q'], mesh=mesh)
        projected = prolong_state(leaves=leaves, mesh=mesh, bed=state['arrays']['bed'], dry_depth=result.dry_depth)
        result.initial_projection_error = float(np.max(np.abs(projected - state['arrays']['q'])))
        result.q.assign(projected)
        result._static_execution.fields['leaves'].assign(leaves)
        result._fields['maximum'].assign(np.maximum(state['arrays']['maximum'], projected[:, 0]))
        return result

    def configure_execution(self, *, mode, tile_size=8, dense_fraction=.65):
        if mode != 'dense':
            raise ValueError('Combined static/sparse execution is not qualified')
        super().configure_execution(mode=mode, tile_size=tile_size, dense_fraction=dense_fraction)

    def _validate_static_source(self, *, values):
        engine = getattr(self, '_static_execution', None)
        if engine is None:
            return
        field = self._field(values=values)
        width = 2 * engine.configuration['block_size']
        tolerance = engine.configuration['source_tolerance']
        for row, column in np.argwhere(engine.mesh.coarse_roots):
            selected = field[row * width:(row + 1) * width, column * width:(column + 1) * width]
            if np.ptp(selected) > tolerance:
                raise ValueError('Source heterogeneity requires a fine mesh; rebuild before changing these fields')

    def set_source_rates(self, *, rain=None, infil=None):
        for values in (rain, infil):
            if values is not None:
                self._validate_static_source(values=values)
        super().set_source_rates(rain=rain, infil=infil)

    def set_rainfall_spatial_multiplier(self, *, multiplier):
        self._validate_static_source(values=multiplier)
        super().set_rainfall_spatial_multiplier(multiplier=multiplier)

    def set_exponential_infiltration(self, *, f0, fd, conductivity):
        for values in (f0, fd, conductivity):
            self._validate_static_source(values=values)
        super().set_exponential_infiltration(f0=f0, fd=fd, conductivity=conductivity)

    def set_water_depth(self, *, depth):
        if getattr(self, '_static_execution', None) is not None:
            raise ValueError('Set initial state before building the static mesh')
        super().set_water_depth(depth=depth)

    def set_momentum(self, *, hu, hv):
        if getattr(self, '_static_execution', None) is not None:
            raise ValueError('Set initial state before building the static mesh')
        super().set_momentum(hu=hu, hv=hv)

    def _enqueue_hydraulics(self):
        self._static_execution.enqueue(solver=self)

    def run_device_loop(self, *, duration, use_cuda_graph=None, max_steps=10000000):
        wp.load_module(module=kernels, device=self.device)
        return super().run_device_loop(duration=duration, use_cuda_graph=use_cuda_graph, max_steps=max_steps)

    def numerical_configuration(self):
        return dict(super().numerical_configuration(), spatial_discretisation='experimental_static_block_fv1',
                    static_mesh=self._static_execution.configuration, topology=self._static_execution.identity)

    def execution_diagnostics(self):
        mesh = self._static_execution.mesh
        categories = dict(fine_fine=0, coarse_fine=0, coarse_coarse=0, exterior=0)
        for axis in ('x', 'y'):
            padded = np.pad(array=mesh.owner, pad_width=((0, 0), (1, 1)) if axis == 'x' else ((1, 1), (0, 0)), constant_values=-1)
            left, right = (padded[:, :-1], padded[:, 1:]) if axis == 'x' else (padded[:-1], padded[1:])
            ids = getattr(mesh, 'faces_' + axis)
            left, right = left.ravel()[ids], right.ravel()[ids]
            interior = (left >= 0) & (right >= 0)
            ll, rl = mesh.level[np.maximum(left, 0)], mesh.level[np.maximum(right, 0)]
            categories['fine_fine'] += int(np.count_nonzero(interior & (ll == 0) & (rl == 0)))
            categories['coarse_coarse'] += int(np.count_nonzero(interior & (ll == 1) & (rl == 1)))
            categories['coarse_fine'] += int(np.count_nonzero(interior & (ll != rl)))
            categories['exterior'] += int(np.count_nonzero(~interior))
        return dict(execution='experimental_static_block_fv1', resolution='fixed_2_to_1', storage='dense_fine_scratch',
            hydraulic_leaves=int(mesh.counts.size), fine_leaves=int(np.count_nonzero(mesh.level == 0)),
            coarse_leaves=int(np.count_nonzero(mesh.level == 1)),
            regular_blocks=int(mesh.blocks.shape[0]), coarse_roots=int(mesh.coarse_roots.sum()),
            evaluated_faces_x=int(mesh.faces_x.size), evaluated_faces_y=int(mesh.faces_y.size),
            evaluated_fine_subfaces_by_interface=categories,
            fine_source_cells=int(self._active.sum()), initial_projection_error=self.initial_projection_error,
            veto_counts=mesh.veto_counts,
            extra_persistent_bytes=sum(array.capacity for array in self._static_execution.fields.values()))

    def capture_state(self):
        state = super().capture_state()
        engine = getattr(self, '_static_execution', None)
        if engine is not None:
            state['static_mesh'] = dict(configuration=engine.configuration, identity=engine.identity,
                topology={name: values.copy() for name, values in engine.topology.items()},
                leaves=engine.fields['leaves'].numpy().copy(), initial_projection_error=self.initial_projection_error)
        return state

    def restore_state(self, *, state):
        engine = getattr(self, '_static_execution', None)
        if engine is not None:
            static = state.get('static_mesh')
            if static is None or static['configuration'] != engine.configuration or static['identity'] != engine.identity:
                raise ValueError('Static checkpoint topology/configuration mismatch')
            for name, values in engine.topology.items():
                if not np.array_equal(static['topology'][name], values):
                    raise ValueError('Static checkpoint topology array mismatch: ' + name)
            leaves = static['leaves']
            if leaves.shape != (engine.mesh.counts.size, 3) or not np.isfinite(leaves).all() or np.any(leaves[:, 0] < 0.):
                raise ValueError('Invalid static checkpoint conservative state')
        super().restore_state(state=state)
        if engine is not None:
            engine.fields['leaves'].assign(state['static_mesh']['leaves'])
            self.initial_projection_error = state['static_mesh']['initial_projection_error']

    def save_checkpoint(self, *, path: Path):
        state = self.capture_state()
        static = state.pop('static_mesh')
        arrays = dict(state.pop('arrays'), rain_times=state.pop('rain_times'),
                      rain_rates=state.pop('rain_rates'), control=state.pop('control'),
                      static_leaves=static.pop('leaves'))
        arrays.update({'static_' + name: values for name, values in static.pop('topology').items()})
        state['static_mesh'] = static
        state['sha256'] = {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in arrays.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        with temporary.open('wb') as stream:
            np.savez(file=stream, metadata=json.dumps(state, sort_keys=True), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target=path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def load_checkpoint(self, *, path: Path):
        with np.load(file=path, allow_pickle=False) as saved:
            state = json.loads(str(saved['metadata']))
            arrays = {name: saved[name].copy() for name in state['sha256']}
        for name, expected in state.pop('sha256').items():
            if hashlib.sha256(arrays[name].tobytes()).hexdigest() != expected:
                raise ValueError('Static checkpoint checksum failure: ' + name)
        static = state['static_mesh']
        static['leaves'] = arrays.pop('static_leaves')
        static['topology'] = {name: arrays.pop('static_' + name) for name in self._static_execution.topology}
        state.update(arrays=arrays, rain_times=arrays.pop('rain_times'),
                     rain_rates=arrays.pop('rain_rates'), control=arrays.pop('control'))
        self.restore_state(state=state)
