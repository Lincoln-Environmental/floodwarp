"""Experimental cell-centred full-SWE MUSCL/SSP-RK2 solver."""

import numpy as np
import warp as wp

from floodwarp import full_swe_kernels as fv1
from floodwarp import fv2_kernels as k
from floodwarp.full_swe_solver import FullSWESolver


class FV2Solver(FullSWESolver):
    method_name = "fv2"
    integration = "ssprk2_strang_drag"

    def __init__(
        self, *, limiter="mc", execution_mode="dense", structures=(), flux_scheme="contact_hll", **options
    ):
        if structures:
            raise ValueError("FV2 does not support stage-consistent culvert exchange")
        if limiter not in ("mc", "minmod", "adaptive") or (
            flux_scheme != "contact_hll"
            and not (self.method_name == "fv1_hancock" and flux_scheme == "low_froude_hll")
        ):
            raise ValueError("FV2 requires contact_hll and limiter mc, minmod or adaptive")
        self.limiter = limiter
        self._limiter_code = {"minmod": 0, "mc": 1, "adaptive": 2}[limiter]
        super().__init__(execution_mode=execution_mode, flux_scheme=flux_scheme, **options)
        self.ctrl = wp.zeros(18, dtype=wp.float64, device=self.device)
        count = self.nx * self.ny
        self._start = wp.zeros(count, dtype=wp.vec3d, device=self.device)
        self._stage = wp.zeros(count, dtype=wp.vec3d, device=self.device)
        # All neighbour reads finish in face kernels. The final Euler update
        # only reads its own cell and can overwrite the first-stage state.
        self._end = self._stage
        if self.method_name == "fv1_hancock":
            self._predicted = self._stage
            self._fuse_prediction = True
            self._predictor_fallback = wp.zeros(count, dtype=wp.int32, device=self.device)
        self._shock_flags = wp.zeros(
            count if limiter == "adaptive" else 0, dtype=wp.int32, device=self.device
        )
        self._inverse = wp.empty(count, dtype=wp.float64, device=self.device)
        self._slopes = [wp.zeros(count, dtype=wp.vec4d, device=self.device) for _ in range(2)]
        self._amounts = wp.zeros(count, dtype=wp.vec2d, device=self.device)
        self._flux = [
            [wp.zeros_like(self._fields["flux_" + axis]) for axis in ("x", "y")]
            for _ in range(2 if self.method_name == "fv2" else 1)
        ]
        self._fields["fv2_cfl_predictor"] = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._stage_series = {}
        self._stage_identity = {}
        self._samples = [wp.zeros(2, dtype=wp.float64, device=self.device) for _ in range(2)]
        self.set_stage_series(axis="x", series=None)
        self.set_stage_series(axis="y", series=None)
        self._diagnostics = None

    def configure_execution(self, *, mode, tile_size=8, dense_fraction=0.65):
        if mode not in ("dense", "sparse", "dynamic"):
            raise ValueError("FV2 execution mode must be dense, sparse or dynamic")
        self._omit_dry = int(mode != "dense")
        self._dynamic = mode == "dynamic"
        dense = wp.empty(0, dtype=wp.int32, device=self.device)
        self._work_indices = [dense, dense, dense]
        self._face_tiles = [dense, dense]
        if self._dynamic:
            self._face_tiles = [
                wp.zeros(
                    1 + (self._fields["flux_" + axis].size + k.TILE_FACES - 1) // k.TILE_FACES,
                    dtype=wp.int32,
                    device=self.device,
                )
                for axis in ("x", "y")
            ]
        if self._omit_dry and not self._active.all():
            mask = self._active
            live_x = np.pad(mask, ((0, 0), (1, 0))) | np.pad(mask, ((0, 0), (0, 1)))
            live_y = np.pad(mask, ((1, 0), (0, 0))) | np.pad(mask, ((0, 1), (0, 0)))
            self._work_indices = [
                wp.array(np.flatnonzero(live).astype(np.int32), dtype=wp.int32, device=self.device)
                for live in (mask, live_x, live_y)
            ]
        self._graph = None

    def execution_diagnostics(self):
        return dict(
            execution="dynamic_face_tiles"
            if self._dynamic
            else ("active_mask_and_dry_face_omission" if self._omit_dry else "dense_uniform"),
            storage="dense",
            scheduling_bytes=sum(a.capacity for a in (*self._work_indices, *self._face_tiles)),
            scheduled_indices=[a.size for a in self._work_indices],
        )

    def set_stage_series(self, *, axis, series):
        if axis not in ("x", "y"):
            raise ValueError("Unknown stage axis")
        if series is None:
            times = levels = np.zeros(1)
            self._samples[0 if axis == "x" else 1].zero_()
        else:
            times, levels = [np.asarray(v, dtype=float) for v in series]
            if (
                times.ndim != 1
                or levels.shape != times.shape
                or times.size < 2
                or np.any(np.diff(times) <= 0)
                or not np.isfinite(times).all()
                or not np.isfinite(levels).all()
            ):
                raise ValueError("FV2 stage series requires increasing times and finite levels")
        self._stage_identity[axis] = [times.tolist(), levels.tolist()] if series is not None else None
        self._stage_series[axis] = [
            wp.array(v, dtype=wp.float64, device=self.device) for v in (times, levels)
        ]
        self._graph = None

    def track_reconstruction(self, *, depth_edges):
        from floodwarp.fv2_diagnostics import ReconstructionDiagnostics

        if self.method_name != "fv2" and depth_edges is not None:
            raise ValueError("RK stage reconstruction diagnostics require fv2")
        if depth_edges is None:
            self._diagnostics = None
            self._end = self._stage
        else:
            probe = ReconstructionDiagnostics(solver=self, depth_edges=depth_edges)
            if self._end is self._stage:
                # Accepted-step diagnostics need both pre-Euler stage states.
                self._end = wp.empty_like(self._stage)
            self._diagnostics = probe
        self._graph = None
        return self._diagnostics

    def _configuration(self):
        configuration = dict(
            super()._configuration(),
            method=self.method_name,
            limiter=self.limiter,
            reconstruction="muscl_ordered_bed_velocity_v2",
            integration=self.integration,
            stage_forcing=self._stage_identity,
        )
        if self.limiter == "adaptive":
            configuration["shock_sensor"] = "compressive_froude_band_one_cell_halo_v1"
        return configuration

    def restore_state(self, *, state):
        if state["configuration"].get("reconstruction") == "muscl_eta_bed_velocity_v1":
            raise ValueError(
                "Checkpoint configuration uses superseded FV2 bed traces that can reverse a slope"
            )
        super().restore_state(state=state)

    def numerical_configuration(self):
        return dict(solver_method=self.method_name, **self._configuration())

    def _launch(self, kernel, count, inputs):
        wp.launch(kernel=kernel, dim=count, inputs=inputs, device=self.device)

    def _launch_grid(self, kernel, count, inputs, *, layout):
        indices = self._work_indices[layout]
        self._launch(kernel, indices.size or count, [*inputs, indices])

    def _stage_forcing(self, *, dest, stage):
        """Validation subclasses can add a prescribed manufactured residual here."""

    def _enqueue_step(self):
        f = self._fields
        ctrl = self.ctrl
        n = self.nx * self.ny
        self._launch(fv1.begin_step, 1, [ctrl, self._rain_times, self._rain_rates, len(self._times)])
        self._launch(k.predictor, 1, [ctrl, f["fv2_cfl_predictor"]])
        # Apply both boundary knot limits before any dt-dependent operation.
        for axis, sample in zip(("x", "y"), self._samples, strict=True):
            if self._stage_identity[axis] is not None:
                self._launch(k.forcing_time, 1, [*self._stage_series[axis], sample, ctrl, 0])
        source_args = [
            self.q,
            self._start,
            self._end,
            self._inverse,
            self._shock_flags,
            f["active"],
            f["manning"],
            f["soil"],
            f["rain"],
            f["multiplier"],
            f["infil"],
            f["f0"],
            f["fd"],
            f["conductivity"],
            self._amounts,
            f["rain_sum"],
            f["maximum"],
            ctrl,
            self.gravity,
            self.dry_depth,
        ]
        self._launch_grid(k.split_sources, n, [*source_args, 0], layout=0)
        stages = (
            ((self._start, self._stage), (self._stage, self._end))
            if self.method_name == "fv2"
            else ((self._start, self._end),)
        )
        for stage, (q, dest) in enumerate(stages):
            if stage and type(self)._stage_forcing is not FV2Solver._stage_forcing:
                # A manufactured residual may change depth after the Euler kernel.
                self._launch_grid(k.inverse_depth, n, [q, self._inverse, self.dry_depth], layout=0)
            if self._shock_flags.size:
                self._launch_grid(
                    k.mark_compressive_stencils,
                    n,
                    [q, f["active"], self._shock_flags, self.nx, self.ny, self.dry_depth, self.gravity],
                    layout=0,
                )
            if self.method_name != "fv1_hancock" or not self._fuse_prediction:
                self._launch_grid(
                    k.slopes,
                    n,
                    [
                        q,
                        self._inverse,
                        self._shock_flags,
                        f["bed"],
                        f["active"],
                        *self._slopes,
                        self.nx,
                        self.ny,
                        self.dry_depth,
                        self._limiter_code,
                    ],
                    layout=0,
                )
            face_inverse = self._inverse
            face_q = q
            forcing_stage = stage
            if self.method_name == "fv1_hancock":
                from floodwarp.fv1_plus_kernels import predict, slopes_predict

                face_q = self._predicted
                forcing_stage = 2
                # The donor budget is overwritten only after every midpoint face is read.
                face_inverse = f["theta"]
                arguments = [
                    q,
                    face_q,
                    self._inverse,
                    face_inverse,
                    *self._slopes,
                    f["active"],
                    self._predictor_fallback,
                    ctrl,
                    self.dx,
                    self.dy,
                    self.gravity,
                    self.dry_depth,
                    self.cfl,
                ]
                if self._fuse_prediction:
                    arguments.extend([self._shock_flags, f["bed"], self.nx, self.ny, self._limiter_code])
                    self._launch_grid(slopes_predict, n, arguments, layout=0)
                else:
                    self._launch_grid(predict, n, arguments, layout=0)
                if type(self)._stage_forcing is not FV2Solver._stage_forcing:
                    self._stage_forcing(dest=face_q, stage=-1)
                    self._launch_grid(k.inverse_depth, n, [face_q, face_inverse, self.dry_depth], layout=0)
            if self._dynamic:
                self._launch(k.reset_tiles, 1, self._face_tiles)
            for axis, name, spacing in ((0, "x", self.dx), (1, "y", self.dy)):
                if forcing_stage and self._stage_identity[name] is not None:
                    self._launch(
                        k.forcing_time,
                        1,
                        [*self._stage_series[name], self._samples[axis], ctrl, forcing_stage],
                    )
                flux = self._flux[stage][axis]
                tiles = self._face_tiles[axis]
                indices = self._work_indices[axis + 1]
                threads = indices.size or flux.size
                if self._dynamic:
                    self._launch(
                        k.schedule_faces,
                        tiles.size - 1,
                        [
                            face_q,
                            f["active"],
                            f["types_" + name],
                            flux,
                            f["pressure_" + name],
                            tiles,
                            self.nx,
                            self.ny,
                            axis,
                        ],
                    )
                    threads = (tiles.size - 1) * k.TILE_FACES
                self._launch(
                    k.faces,
                    threads,
                    [
                        face_q,
                        face_inverse,
                        f["bed"],
                        f["active"],
                        self._slopes[axis],
                        f["types_" + name],
                        f["stages_" + name],
                        self._samples[axis],
                        ctrl,
                        flux,
                        f["pressure_" + name],
                        self.nx,
                        self.ny,
                        axis,
                        spacing,
                        self.gravity,
                        self.dry_depth,
                        self.cfl,
                        self._stage_momentum_mode,
                        self._flux_mode,
                        self._omit_dry,
                        tiles,
                        indices,
                    ],
                )
            self._launch_grid(
                k.donor_budget,
                n,
                [q, f["theta"], *self._flux[stage], ctrl, self.nx, self.dx, self.dy],
                layout=0,
            )
            for axis, flux in enumerate(self._flux[stage]):
                self._launch_grid(
                    k.limit_faces,
                    flux.size,
                    [flux, f["theta"], f["active"], self.nx, self.ny, axis],
                    layout=axis + 1,
                )
            self._launch_grid(
                k.euler,
                n,
                [
                    q,
                    face_q,
                    dest,
                    self._inverse,
                    self._shock_flags,
                    stage,
                    f["active"],
                    *self._slopes,
                    *self._flux[stage],
                    f["pressure_x"],
                    f["pressure_y"],
                    ctrl,
                    self.nx,
                    self.dx,
                    self.dy,
                    self.gravity,
                    self.dry_depth,
                ],
                layout=0,
            )
            self._stage_forcing(dest=dest, stage=forcing_stage)
        self._launch(k.validate_trial, 1, [ctrl, self.minimum_dt])
        if self._diagnostics is not None:
            self._diagnostics.enqueue(solver=self)
        self._launch_grid(k.split_sources, n, [*source_args, 1 if self.method_name == "fv2" else 2], layout=0)
        for axis, name, width in ((0, "x", self.dy), (1, "y", self.dx)):
            flux = f["flux_" + name]
            self._launch_grid(
                k.accept_faces,
                flux.size,
                [
                    self._flux[0][axis],
                    self._flux[-1][axis],
                    flux,
                    f["volume_" + name],
                    f["active"],
                    f["inflow"],
                    f["outflow"],
                    ctrl,
                    self.nx,
                    self.ny,
                    axis,
                    width,
                ],
                layout=axis + 1,
            )
        self._launch(k.finish, 1, [ctrl, f["fv2_cfl_predictor"], self.condition, f["dt_histogram"]])

    def _load_extra_step_modules(self):
        wp.load_module(module=k, device=self.device)
        if self.method_name == "fv1_hancock":
            from floodwarp import fv1_plus_kernels

            wp.load_module(module=fv1_plus_kernels, device=self.device)
        if self._diagnostics is not None:
            from floodwarp import fv2_diagnostics

            wp.load_module(module=fv2_diagnostics, device=self.device)

    def _extra_loop_statistics(self, *, control):
        return dict(rejected_cfl_trials=int(control[16]))

    def step(self, *, remaining_time=None, fixed_dt=None):
        previous = self.steps
        while self.steps == previous:
            dt = super().step(remaining_time=remaining_time, fixed_dt=fixed_dt)
        return dt


class MUSCLEulerSolver(FV2Solver):
    method_name = "fv1_muscl_euler"
    integration = "muscl_euler_strang_drag"


class HancockSolver(FV2Solver):
    method_name = "fv1_hancock"
    integration = "unsplit_hancock_strang_drag"

    def predictor_fallback_cells(self):
        """Cells with an inadmissible half-time reconstruction in the last trial."""
        return self._predictor_fallback.numpy().reshape(self.ny, self.nx).astype(bool)
