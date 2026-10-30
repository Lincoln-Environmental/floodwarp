"""GPU-native, independent first-order FP64 full shallow-water solver.

Hydraulic state is cell-centred (h, hu, hv), with south-to-north rows. Face
discharge getters return the last accepted HLL mass flux (m²/s), rather than
interpolated momentum. This solver has no SGS or local-inertial dependency.
"""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from floodwarp import full_swe_kernels as kernels


class FullSWESolver:
    def __init__(self, *, ny, nx, dx, dy, bed_elevation, manning_n, active_mask,
                 device="cpu", gravity=9.81, dry_depth=1.e-4, cfl=.4,
                 maximum_dt=1., minimum_dt=0., stage_boundary_momentum='velocity_extrapolation',
                 flux_scheme='contact_hll'):
        """Prescribed stages use an explicitly selected ghost momentum closure.

        Finite-velocity extrapolation is the default. Historical discharge
        extrapolation remains explicit; velocity mode keeps both components finite as exterior depth
        shrinks. It changes the stage boundary's HLL flux, including inflow;
        it is not a relaxation of the CFL bound or a reservoir-velocity model.

        Experimental characteristic mode prescribes stage and the outgoing
        invariant, uses critical free drainage and transparent supercritical
        outflow, and assumes zero external tangential velocity on inflow.
        Supercritical inflow and wet-stage inundation of dry boundary cells
        are rejected because their external momentum is unspecified.

        contact_hll retains endpoint HLL mass/normal fluxes and upwinds the
        transverse velocity. legacy_hll is exact historical compatibility;
        star_hll and star_contact_hll are explicit diagnostic alternatives.
        """
        modes = ('discharge_extrapolation','velocity_extrapolation','characteristic')
        if stage_boundary_momentum not in modes:
            raise ValueError('stage_boundary_momentum must be one of '+', '.join(modes))
        self.stage_boundary_momentum=stage_boundary_momentum
        self._stage_momentum_mode=modes.index(stage_boundary_momentum)
        flux_schemes = ('legacy_hll','contact_hll','star_hll','star_contact_hll')
        if flux_scheme not in flux_schemes:
            raise ValueError('flux_scheme must be one of '+', '.join(flux_schemes))
        self.flux_scheme = flux_scheme
        self._flux_mode = flux_schemes.index(flux_scheme)
        if not np.isfinite([nx, ny, dx, dy, gravity, dry_depth, cfl, maximum_dt, minimum_dt]).all():
            raise ValueError("Finite numerical parameters required")
        if nx < 1 or ny < 1 or min(dx, dy, gravity, maximum_dt) <= 0:
            raise ValueError("Positive grid dimensions, spacing, gravity and timestep required")
        if not 0 < cfl <= .45 or dry_depth <= 0 or not 0 <= minimum_dt <= maximum_dt:
            raise ValueError("Require 0 < CFL <= 0.45, positive dry threshold and nonnegative minimum_dt")
        self.ny, self.nx = int(ny), int(nx)
        self.dx, self.dy = float(dx), float(dy)
        self.gravity, self.dry_depth = float(gravity), float(dry_depth)
        self.cfl, self.maximum_dt, self.minimum_dt = float(cfl), float(maximum_dt), float(minimum_dt)
        self.device = wp.get_device(device)
        self.time = 0.
        self.initial_volume = 0.
        self.steps = 0
        self._graph = None
        self._fields = {}
        shape = (self.ny, self.nx)
        self._shape = shape
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != shape or not mask.any():
            raise ValueError("A nonempty active mask matching the grid is required")
        self._active = mask.copy()
        self._allocate(name="active", values=mask.astype(np.int32), dtype=wp.int32)
        self._allocate(name="bed", values=self._field(values=bed_elevation, nonnegative=False))
        self._allocate(name="manning", values=self._field(values=manning_n))
        for name in ("maximum", "soil", "rain_sum", "inflow", "outflow", "rain", "infil", "f0", "conductivity"):
            self._allocate(name=name, values=np.zeros(shape))
        for name in ("fd", "multiplier", "theta"):
            self._allocate(name=name, values=np.ones(shape))
        self.q = wp.zeros(self.nx * self.ny, dtype=wp.vec3d, device=self.device)
        self._fields["q"] = self.q
        self._fields["dt_histogram"] = wp.zeros(64, dtype=wp.int64, device=self.device)
        for axis, face_shape in (("x", (self.ny, self.nx + 1)), ("y", (self.ny + 1, self.nx))):
            self._allocate(name="types_" + axis, values=np.zeros(face_shape, dtype=np.int32), dtype=wp.int32)
            self._allocate(name="stages_" + axis, values=np.zeros(face_shape))
            self._allocate(name="volume_" + axis, values=np.zeros(face_shape))
            self._fields["flux_" + axis] = wp.zeros(int(np.prod(face_shape)), dtype=wp.vec3d, device=self.device)
            self._fields["pressure_" + axis] = wp.zeros(int(np.prod(face_shape)), dtype=wp.vec2d, device=self.device)
        self.ctrl = wp.zeros(15, dtype=wp.float64, device=self.device)
        self.condition = wp.ones(1, dtype=wp.int32, device=self.device)
        self.set_rainfall_hyetograph(times_s=np.array([0., 1.e100]), rates_m_s=np.zeros(2))

    def _field(self, *, values, nonnegative=True):
        array = np.broadcast_to(np.asarray(values, dtype=np.float64), self._shape).copy()
        if not np.isfinite(array[self._active]).all() or (nonnegative and np.any(array[self._active] < 0)):
            raise ValueError("Fields must be finite, and rates/depth/roughness nonnegative")
        return np.where(self._active, array, 0.)

    def _allocate(self, *, name, values, dtype=wp.float64):
        self._fields[name] = wp.array(np.asarray(values).ravel(), dtype=dtype, device=self.device)

    def _assign(self, *, name, values):
        self._fields[name].assign(np.asarray(values).ravel())

    def set_water_depth(self, *, depth):
        if self.steps:
            raise ValueError("Use a complete checkpoint to replace an evolved state")
        q = self.q.numpy()
        q[:, 0] = self._field(values=depth).ravel()
        q[q[:, 0] <= self.dry_depth, 1:] = 0.
        self.q.assign(q)
        self._assign(name="maximum", values=q[:, 0])
        self.initial_volume = float(q[:, 0].sum() * self.dx * self.dy)

    def set_momentum(self, *, hu, hv):
        if self.steps:
            raise ValueError("Initial momentum can only be set before stepping")
        q = self.q.numpy()
        q[:, 1] = self._field(values=hu, nonnegative=False).ravel()
        q[:, 2] = self._field(values=hv, nonnegative=False).ravel()
        if np.any((q[:, 0] <= self.dry_depth) & np.any(q[:, 1:] != 0., axis=1)):
            raise ValueError("Nonzero initial momentum in a dry cell")
        self.q.assign(q)

    def set_source_rates(self, *, rain=None, infil=None):
        for name, values in (("rain", rain), ("infil", infil)):
            if values is not None:
                self._assign(name=name, values=self._field(values=values))

    def set_uniform_rainfall_rate(self, *, rainfall_rate):
        self.set_rainfall_hyetograph(times_s=np.array([0., 1.e100]), rates_m_s=np.full(2, rainfall_rate))

    def set_rainfall_spatial_multiplier(self, *, multiplier):
        self._assign(name="multiplier", values=self._field(values=multiplier))

    def set_rainfall_hyetograph(self, *, times_s, rates_m_s):
        times, rates = np.asarray(times_s, dtype=np.float64), np.asarray(rates_m_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or rates.shape != times.shape:
            raise ValueError("At least two matching PWL rainfall nodes required")
        if not np.isfinite(times).all() or not np.isfinite(rates).all() or times[0] > self.time or np.any(np.diff(times) <= 0) or np.any(rates < 0):
            raise ValueError("Finite, increasing forcing times and nonnegative rain required")
        self._times, self._rates = times.copy(), rates.copy()
        self._rain_times = wp.array(times, dtype=wp.float64, device=self.device)
        self._rain_rates = wp.array(rates, dtype=wp.float64, device=self.device)
        self._graph = None

    def set_exponential_infiltration(self, *, f0, fd, conductivity):
        fields = {name: self._field(values=value) for name, value in (("f0", f0), ("fd", fd), ("conductivity", conductivity))}
        if np.any(fields["fd"][self._active] <= 0) or np.any(fields["f0"] < fields["conductivity"]):
            raise ValueError("Require positive decay depth and f0 >= conductivity")
        fields["fd"][~self._active] = 1.
        for name, values in fields.items():
            self._assign(name=name, values=values)

    def _boundary(self, *, axis, values, stage):
        shape = (self.ny, self.nx + 1) if axis == "x" else (self.ny + 1, self.nx)
        array = np.asarray(values)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Boundary array shape or values invalid")
        if stage:
            self._assign(name="stages_" + axis, values=array)
            return
        if np.any(array != array.astype(np.int32)) or not np.isin(array, np.arange(12)).all():
            raise ValueError("Unsupported boundary code")
        array = array.astype(np.int32)
        # Existing orientation-specific mask codes reduce to the same ghost law.
        array = np.where(np.isin(array, [4, 5, 10, 11]), 3, array)
        array = np.where(np.isin(array, [6, 7, 8, 9]), 2, array)
        padded = np.pad(self._active, ((0, 0), (1, 1)) if axis == "x" else ((1, 1), (0, 0)))
        left, right = (padded[:, :-1], padded[:, 1:]) if axis == "x" else (padded[:-1], padded[1:])
        if np.any((array != 0) & (left == right)):
            raise ValueError("Boundary types must occur on active/exterior interfaces")
        self._assign(name="types_" + axis, values=array)

    def set_boundary_type_qx(self, *, boundary_type):
        self._boundary(axis="x", values=boundary_type, stage=False)

    def set_boundary_type_qy(self, *, boundary_type):
        self._boundary(axis="y", values=boundary_type, stage=False)

    def set_boundary_stage_qx(self, *, stage):
        self._boundary(axis="x", values=stage, stage=True)

    def set_boundary_stage_qy(self, *, stage):
        self._boundary(axis="y", values=stage, stage=True)

    def _enqueue_step(self):
        f = self._fields
        common = [self.nx, self.ny]
        wp.launch(kernels.begin_step, dim=1, inputs=[self.ctrl, self._rain_times, self._rain_rates, len(self._times)], device=self.device)
        wp.launch(kernels.cfl_reduce, dim=self.nx * self.ny,
            inputs=[self.q, f["active"], f["rain"], f["multiplier"], f["bed"], f["types_x"], f["types_y"], f["stages_x"], f["stages_y"], self.nx, self.ctrl, self.dx, self.dy, self.gravity, self.dry_depth, self.cfl,self._stage_momentum_mode], device=self.device)
        if self._flux_mode >= 2:
            wp.launch(kernels.star_cfl_reduce,dim=self.nx*self.ny,
                inputs=[self.q,f['active'],f['bed'],f['rain'],f['multiplier'],f['infil'],f['f0'],f['fd'],
                    f['conductivity'],f['soil'],f['types_x'],f['types_y'],f['stages_x'],f['stages_y'],self.nx,self.ny,
                    self.ctrl,self.dx,self.dy,self.gravity,self.dry_depth,self.cfl,self._stage_momentum_mode],device=self.device)
        if self._stage_momentum_mode == 2:
            wp.launch(kernels.characteristic_source_bound, dim=self.nx*self.ny,
                inputs=[self.q, f['active'], f['rain'], f['multiplier'], f['infil'], f['f0'],
                    f['types_x'], f['types_y'], f['stages_x'], f['stages_y'], f['bed'], self.nx,
                    self.ctrl, self.dx, self.dy, self.gravity, self.dry_depth, self.cfl], device=self.device)
            wp.launch(kernels.validate_characteristic, dim=self.nx*self.ny,
                inputs=[self.q, f['active'], f['rain'], f['multiplier'], f['infil'], f['f0'], f['fd'],
                    f['conductivity'], f['soil'], f['bed'], f['types_x'], f['types_y'], f['stages_x'],
                    f['stages_y'], self.nx, self.ctrl, self.gravity, self.dry_depth], device=self.device)
        wp.launch(kernels.apply_sources, dim=self.nx * self.ny,
            inputs=[self.q, f["active"], f["rain"], f["multiplier"], f["infil"], f["f0"], f["fd"], f["conductivity"], f["soil"], f["rain_sum"], self.ctrl], device=self.device)
        for axis, index in (("x", 0), ("y", 1)):
            wp.launch(kernels.faces, dim=f["flux_" + axis].size,
                inputs=[self.q, f["bed"], f["active"], f["types_" + axis], f["stages_" + axis], f["flux_" + axis], f["pressure_" + axis], *common, index, self.gravity, self.dry_depth,self._stage_momentum_mode,self.ctrl,self._flux_mode], device=self.device)
        wp.launch(kernels.draining, dim=self.nx * self.ny,
            inputs=[self.q, f["flux_x"], f["flux_y"], f["theta"], self.ctrl, self.nx, self.dx, self.dy], device=self.device)
        for axis, index, width in (("x", 0, self.dy), ("y", 1, self.dx)):
            wp.launch(kernels.limit_faces, dim=f["flux_" + axis].size,
                inputs=[f["flux_" + axis], f["theta"], f["active"], f["volume_" + axis], self.ctrl, *common, index, width], device=self.device)
        wp.launch(kernels.update, dim=self.nx * self.ny,
            inputs=[self.q, f["flux_x"], f["flux_y"], f["pressure_x"], f["pressure_y"], f["active"], f["manning"], f["maximum"], f["inflow"], f["outflow"], self.ctrl, *common, self.dx, self.dy, self.gravity, self.dry_depth], device=self.device)
        wp.launch(kernels.finish_step, dim=1, inputs=[self.ctrl, self.condition, f["dt_histogram"], self.minimum_dt], device=self.device)

    def run_device_loop(self, *, duration, use_cuda_graph=True, max_steps=10000000):
        """Run to an exact requested time; CUDA uses a device conditional loop.

        No CFL/state readback occurs inside a CUDA graph. CPU fallback steps
        the identical kernels with host orchestration. Minimum dt is a failure
        threshold, never a floor that overrides the stability bound.
        """
        if not np.isfinite(duration) or duration < 0 or max_steps < 1:
            raise ValueError("Finite nonnegative duration and positive step budget required")
        target = self.time + duration
        if target > self._times[-1]:
            raise ValueError("Rainfall forcing does not cover requested interval")
        if duration == 0:
            return dict(steps=0, time_s=self.time, loop_seconds=0., compilation_seconds=0., graph_capture_seconds=0.)
        ctrl = np.zeros(self.ctrl.size, dtype=np.float64)
        ctrl[0:3] = [self.time, target, 0.]
        ctrl[3] = self.steps
        ctrl[4:7] = [np.inf, 0., 0.]
        ctrl[7:9] = [0., self.maximum_dt]
        ctrl[10] = 0.
        ctrl[13] = self.steps + max_steps
        ctrl[14] = self.maximum_dt
        self.ctrl.assign(ctrl)
        self.condition.fill_(1)
        began = perf_counter()
        wp.load_module(module=kernels, device=self.device)
        compilation = perf_counter() - began
        capture_seconds = 0.
        graph = use_cuda_graph and self.device.is_cuda
        if graph and self._graph is None:
            began = perf_counter()
            with wp.ScopedCapture(device=self.device, force_module_load=True) as body:
                self._enqueue_step()
            with wp.ScopedCapture(device=self.device) as loop:
                wp.capture_while(self.condition, while_body=body.graph)
            self._graph = loop.graph
            capture_seconds = perf_counter() - began
        began = perf_counter()
        if graph:
            wp.capture_launch(graph=self._graph)
            ctrl = self.ctrl.numpy()
        else:
            for _ in range(max_steps):
                self._enqueue_step()
                ctrl = self.ctrl.numpy()
                if ctrl[0] >= target or ctrl[7]:
                    break
        elapsed = perf_counter() - began
        done = int(ctrl[3]) - self.steps
        self.time, self.steps = float(ctrl[0]), int(ctrl[3])
        if ctrl[7] < 0:
            raise ArithmeticError('Characteristic stage boundary requires incoming momentum data for supercritical or dry-land inflow')
        if ctrl[14] < self.minimum_dt:
            raise ArithmeticError("FV1 stable timestep below configured failure threshold")
        if ctrl[7] or self.time != target:
            raise ArithmeticError(f"FV1 invalid state or step budget exhausted: t={self.time}, target={target}, errors={ctrl[7]}")
        return dict(steps=done, time_s=self.time, loop_seconds=elapsed, compilation_seconds=compilation,
                    graph_capture_seconds=capture_seconds, dt_min_s=float(ctrl[4]), dt_max_s=float(ctrl[5]),
                    dt_mean_s=duration / done, dt_rms_s=float(np.sqrt(ctrl[6] / done)),
                    host_state_syncs=1 if graph else done, cuda_graph=bool(graph))

    def step(self, *, remaining_time=None, fixed_dt=None):
        # A requested fixed dt remains a cap; never override velocity CFL.
        cap = self.maximum_dt if fixed_dt is None else float(fixed_dt)
        if remaining_time is not None:
            cap = min(cap, remaining_time)
        if not np.isfinite(cap) or cap <= 0 or self.time + cap > self._times[-1]:
            raise ValueError("Positive step duration required")
        ctrl = np.zeros(self.ctrl.size, dtype=np.float64)
        ctrl[0:3] = [self.time, self.time + cap, 0.]
        ctrl[3:9] = [self.steps, np.inf, 0., 0., 0., min(cap, self.maximum_dt)]
        ctrl[10], ctrl[13] = 0., self.steps + 1
        ctrl[14] = self.maximum_dt
        self.ctrl.assign(ctrl)
        self._enqueue_step()
        ctrl = self.ctrl.numpy()
        self.time, self.steps = float(ctrl[0]), int(ctrl[3])
        if ctrl[7] < 0:
            raise ArithmeticError('Characteristic stage boundary requires incoming momentum data for supercritical or dry-land inflow')
        if ctrl[14] < self.minimum_dt:
            raise ArithmeticError("FV1 stable timestep below configured failure threshold")
        if ctrl[7]:
            raise ArithmeticError("Invalid finite-volume step")
        return float(ctrl[2])

    def get_conservative_state(self):
        return self.q.numpy().reshape(self.ny, self.nx, 3).copy()

    def get_water_depth(self):
        return self.get_conservative_state()[..., 0]

    def get_momentum(self):
        state = self.get_conservative_state()
        return state[..., 1], state[..., 2]

    def get_qx(self):
        return self._fields["flux_x"].numpy()[:, 0].reshape(self.ny, self.nx + 1).copy()

    def get_qy(self):
        return self._fields["flux_y"].numpy()[:, 0].reshape(self.ny + 1, self.nx).copy()

    def get_max_water_depth(self):
        return self._fields["maximum"].numpy().reshape(self._shape).copy()

    def cumulative_infiltration_m(self):
        return self._fields["soil"].numpy().reshape(self._shape).copy()

    def get_timestep_histogram(self):
        """Exact accepted-step counts in powers-of-two seconds; end bins open."""
        edges = [2.**(index - 32) for index in range(65)]
        edges[0], edges[-1] = 0., None
        return dict(edges_s=edges, counts=self._fields["dt_histogram"].numpy().tolist())

    def cumulative_accounting(self):
        area = self.dx * self.dy
        result = {key: float(self._fields[name].numpy().sum() * factor) for key, name, factor in
            (("rain", "rain_sum", area), ("infiltration", "soil", area), ("boundary_inflow", "inflow", 1.), ("boundary_outflow", "outflow", 1.))}
        result["storage"] = float(self.get_water_depth().sum() * area)
        result["residual"] = self.initial_volume + result["rain"] + result["boundary_inflow"] - result["infiltration"] - result["boundary_outflow"] - result["storage"]
        return result

    def get_cumulative_face_volumes(self):
        return dict(qx=self._fields["volume_x"].numpy().reshape(self.ny, self.nx + 1).copy(),
                    qy=self._fields["volume_y"].numpy().reshape(self.ny + 1, self.nx).copy())

    def _configuration(self):
        return dict(nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, gravity=self.gravity,
                    dry_depth=self.dry_depth, cfl=self.cfl, maximum_dt=self.maximum_dt, minimum_dt=self.minimum_dt,
                    stage_boundary_momentum=self.stage_boundary_momentum,flux_scheme=self.flux_scheme)

    def numerical_configuration(self):
        """Resolved boundary semantics, including when the argument was omitted."""
        return dict(solver_method='fv1',**self._configuration())

    def capture_state(self):
        return dict(configuration=self._configuration(), time=self.time, steps=self.steps,
                    initial_volume=self.initial_volume, arrays={key: value.numpy().copy() for key, value in self._fields.items()},
                    rain_times=self._times.copy(), rain_rates=self._rates.copy(), control=self.ctrl.numpy().copy())

    def restore_state(self, *, state):
        recorded=dict(state['configuration'])
        # Before this option existed there was exactly one FV1 ghost law.
        # Its absence identifies legacy discharge extrapolation, never the new law.
        recorded.setdefault('stage_boundary_momentum','discharge_extrapolation')
        recorded.setdefault('flux_scheme','legacy_hll')
        if recorded != self._configuration() or set(state["arrays"]) != set(self._fields):
            raise ValueError("Checkpoint configuration/field set mismatch")
        for name, array in state["arrays"].items():
            if array.shape != self._fields[name].numpy().shape or not np.isfinite(array).all():
                raise ValueError("Checkpoint shape/nonfinite field: " + name)
        q = state["arrays"]["q"]
        if np.any(q[:, 0] < 0) or not np.array_equal(state["arrays"]["active"].astype(bool), self._active.ravel()):
            raise ValueError("Invalid checkpoint depths/mask")
        for name in ("bed", "manning"):
            if not np.array_equal(state["arrays"][name], self._fields[name].numpy()):
                raise ValueError("Checkpoint physical input mismatch: " + name)
        self.set_rainfall_hyetograph(times_s=state["rain_times"], rates_m_s=state["rain_rates"])
        for name, values in state["arrays"].items():
            self._fields[name].assign(values)
        self.ctrl.assign(state["control"])
        self.time, self.steps, self.initial_volume = state["time"], state["steps"], state["initial_volume"]

    def save_checkpoint(self, *, path: Path):
        state = self.capture_state()
        metadata = {name: value for name, value in state.items() if name not in ("arrays", "rain_times", "rain_rates", "control")}
        arrays = dict(state["arrays"], rain_times=state["rain_times"], rain_rates=state["rain_rates"], control=state["control"])
        metadata["sha256"] = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            np.savez(file=stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
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
            metadata = json.loads(str(saved["metadata"]))
            arrays = {name: saved[name].copy() for name in metadata["sha256"]}
        for name, digest in metadata.pop("sha256").items():
            if hashlib.sha256(arrays[name].tobytes()).hexdigest() != digest:
                raise ValueError("Checkpoint checksum failure: " + name)
        state = dict(metadata, rain_times=arrays.pop("rain_times"), rain_rates=arrays.pop("rain_rates"), control=arrays.pop("control"), arrays=arrays)
        self.restore_state(state=state)
