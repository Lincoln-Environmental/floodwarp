"""Experimental cell-centred full-SWE MUSCL/SSP-RK2 solver."""

import numpy as np
import warp as wp

from floodwarp import hancock_tiles
from floodwarp import full_swe_kernels as fv1
from floodwarp import fv2_kernels as k
from floodwarp.solver_defaults import DEFAULT_FULL_SWE_FLUX, DEFAULT_HANCOCK_LIMITER
from floodwarp.full_swe_solver import FullSWESolver


class FV2Solver(FullSWESolver):
    method_name = "fv2"
    integration = "ssprk2_strang_drag"

    def __init__(
        self, *, limiter="mc", execution_mode="dense", structures=(), flux_scheme=DEFAULT_FULL_SWE_FLUX,
        free_boundary_reconstruction=None, **options
    ):
        if structures:
            raise ValueError(
                f"{self.method_name} does not support culvert exchange. "
                "Select solver_method='hancock' for full-SWE structures."
            )
        if limiter not in ("mc", "minmod", "adaptive"):
            raise ValueError(f"{self.method_name} requires limiter mc, minmod or adaptive")
        if flux_scheme != "contact_hll" and not (
            self.method_name == "fv1_hancock" and flux_scheme == "low_froude_hll"
        ):
            raise ValueError(f"{self.method_name} does not support {flux_scheme!r}. "
                             "Use contact_hll or explicitly select FV1 for its alternative fluxes.")
        self.limiter = limiter
        if free_boundary_reconstruction is None:
            free_boundary_reconstruction = "constant" if self.method_name == "fv1_muscl_euler" else "one_sided"
        if free_boundary_reconstruction not in ("constant", "one_sided"):
            raise ValueError("FREE boundary reconstruction requires constant or one_sided")
        self.free_boundary_reconstruction = free_boundary_reconstruction
        self._limiter_code = {"minmod": 0, "mc": 1, "adaptive": 2}[limiter]
        super().__init__(execution_mode=execution_mode, flux_scheme=flux_scheme, **options)
        directional = getattr(self, "cfl_policy", "legacy") == "directional"
        control_size = k.DIRECTIONAL_CONTROL_SIZE if directional else k.LEGACY_CONTROL_SIZE
        self.ctrl = wp.zeros(control_size, dtype=wp.float64, device=self.device)
        count = self.nx * self.ny
        self._start = wp.zeros(count, dtype=wp.vec3d, device=self.device)
        self._stage = wp.zeros(count, dtype=wp.vec3d, device=self.device)
        # All neighbour reads finish in face kernels. The final Euler update
        # only reads its own cell and can overwrite the first-stage state.
        self._end = self._stage
        if self.method_name == "fv1_hancock":
            self._predicted = self._stage
            self._fuse_prediction = True
            self._predictor_fallback = wp.zeros(count, dtype=wp.uint8, device=self.device)
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
        self._structures = None

    def configure_execution(self, *, mode, tile_size=8, dense_fraction=0.65):
        if mode == "tiles" and self.method_name != "fv1_hancock":
            raise ValueError("Tile scheduling requires Hancock")
        if mode not in ("dense", "sparse", "dynamic", "tiles"):
            raise ValueError("Unknown execution mode. Choose dense, sparse, dynamic or Hancock tiles.")
        if tile_size != 8 or dense_fraction != .65:
            raise ValueError(f"{self.method_name} does not use FV1 tile_size or dense_fraction tuning. "
                             "Omit these overrides or select solver_method='fv1'.")
        self._tiles = (
            hancock_tiles.HancockTiles(nx=self.nx, ny=self.ny, device=self.device)
            if mode == "tiles" else None
        )
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
        if self._tiles is None and self._omit_dry and not self._active.all():
            mask = self._active
            live_x = np.pad(mask, ((0, 0), (1, 0))) | np.pad(mask, ((0, 0), (0, 1)))
            live_y = np.pad(mask, ((1, 0), (0, 0))) | np.pad(mask, ((0, 1), (0, 0)))
            self._work_indices = [
                wp.array(np.flatnonzero(live).astype(np.int32), dtype=wp.int32, device=self.device)
                for live in (mask, live_x, live_y)
            ]
        self._graph = None

    def execution_diagnostics(self):
        if self._tiles is not None:
            return self._tiles.diagnostics()
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
        if self.free_boundary_reconstruction == "one_sided":
            configuration["free_boundary_reconstruction"] = "one_sided"
        return configuration

    def restore_state(self, *, state):
        if state["configuration"].get("reconstruction") == "muscl_eta_bed_velocity_v1":
            raise ValueError(
                "Checkpoint configuration uses superseded FV2 bed traces that can reverse a slope"
            )
        super().restore_state(state=state)
        if self._tiles is not None:
            self._tiles.previous.fill_(1)

    def numerical_configuration(self):
        return dict(solver_method=self.method_name, **self._configuration())

    def _launch(self, kernel, count, inputs):
        wp.launch(kernel=kernel, dim=count, inputs=inputs, device=self.device)

    def _launch_grid(self, kernel, count, inputs, *, layout):
        if self._tiles is not None:
            self._launch(
                kernel=kernel.tile_kernel, count=self._tiles.threads,
                inputs=[*inputs, self._tiles.grids[layout]],
            )
            return
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
        if self._tiles is not None:
            self._tiles.enqueue(
                solver=self,
                distributed_forcing=type(self)._stage_forcing is not FV2Solver._stage_forcing,
            )
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
                        f["types_x"], f["types_y"], int(self.free_boundary_reconstruction == "one_sided"),
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
                    arguments.extend([f["types_x"], f["types_y"], int(self.free_boundary_reconstruction == "one_sided")])
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
                if self._tiles is not None:
                    indices = self._tiles.grids[axis + 1]
                    threads = self._tiles.threads
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
                    k.faces.tile_kernel if self._tiles is not None else k.faces,
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
            if self.ctrl.size == k.DIRECTIONAL_CONTROL_SIZE:
                self._launch(k.directional_bound, 1, [ctrl, self.cfl])
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
        if self._structures is not None:
            self._structures.snapshot(solver=self)
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
        if self._structures is not None:
            self._structures.enqueue(solver=self)
        if self._tiles is not None:
            self._launch(
                kernel=hancock_tiles.commit_tiles, count=self._tiles.size,
                inputs=[self._tiles.current, self._tiles.previous, ctrl],
            )
        self._launch(k.finish, 1, [ctrl, f["fv2_cfl_predictor"], self.condition, f["dt_histogram"]])

    def _load_extra_step_modules(self):
        wp.load_module(module=k, device=self.device)
        if self._tiles is not None:
            wp.load_module(module=hancock_tiles, device=self.device)
        if self._structures is not None:
            from floodwarp import hancock_structures

            wp.load_module(module=hancock_structures, device=self.device)
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

    def __init__(self, *, limiter=DEFAULT_HANCOCK_LIMITER, cfl_policy="legacy", structures=(), **options):
        if cfl_policy not in ("legacy", "directional"):
            raise ValueError("Hancock CFL policy must be legacy or directional")
        self.cfl_policy = cfl_policy
        super().__init__(limiter=limiter, **options)
        if structures:
            from floodwarp.hancock_structures import HancockStructures

            self._structures = HancockStructures(solver=self, structures=structures)

    def get_structure_diagnostics(self):
        return self._structures.diagnostics() if self._structures is not None else []

    def _configuration(self):
        configuration = super()._configuration()
        if self._structures is not None:
            configuration["structures"] = dict(version="hancock_post_surface_links_v1", sha256=self._structures.identity)
        if self.cfl_policy == "directional":
            configuration["cfl_policy"] = self.cfl_policy
        return configuration

    def predictor_fallback_cells(self):
        """Cells with an inadmissible half-time reconstruction in the last trial."""
        return self._predictor_fallback.numpy().reshape(self.ny, self.nx).astype(bool)
