"""Experimental cell-centred full-SWE MUSCL/SSP-RK2 solver."""

import numpy as np
import warp as wp

from floodwarp import full_swe_kernels as fv1
from floodwarp import fv2_kernels as k
from floodwarp.full_swe_solver import FullSWESolver


class FV2Solver(FullSWESolver):
    def __init__(
        self, *, limiter="mc", execution_mode="dense", structures=(), flux_scheme="contact_hll", **options
    ):
        if structures:
            raise ValueError("FV2 does not support stage-consistent culvert exchange")
        if limiter not in ("mc", "minmod") or flux_scheme != "contact_hll":
            raise ValueError("FV2 requires contact_hll and limiter mc or minmod")
        self.limiter = limiter
        super().__init__(execution_mode=execution_mode, flux_scheme=flux_scheme, **options)
        self.ctrl = wp.zeros(18, dtype=wp.float64, device=self.device)
        count = self.nx * self.ny
        self._start, self._stage, self._end = [
            wp.zeros(count, dtype=wp.vec3d, device=self.device) for _ in range(3)
        ]
        self._slopes = [wp.zeros(count, dtype=wp.vec4d, device=self.device) for _ in range(2)]
        self._amounts = wp.zeros(count, dtype=wp.vec2d, device=self.device)
        self._flux = [[wp.empty_like(self._fields["flux_" + axis]) for axis in ("x", "y")] for _ in range(2)]
        self._fields["fv2_cfl_predictor"] = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._stage_series = {}
        self._stage_identity = {}
        self._samples = [wp.zeros(2, dtype=wp.float64, device=self.device) for _ in range(2)]
        self.set_stage_series(axis="x", series=None)
        self.set_stage_series(axis="y", series=None)
        self._diagnostics = None

    def configure_execution(self, *, mode, tile_size=8, dense_fraction=0.65):
        if mode not in ("dense", "sparse"):
            raise ValueError("FV2 execution mode must be dense or sparse")
        self._omit_dry = int(mode == "sparse")
        self._graph = None

    def execution_diagnostics(self):
        return dict(
            execution="exact_dry_face_omission" if self._omit_dry else "dense_uniform",
            storage="dense",
            scheduling_bytes=0,
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

        self._diagnostics = ReconstructionDiagnostics(solver=self, depth_edges=depth_edges)
        self._graph = None
        return self._diagnostics

    def _configuration(self):
        return dict(
            super()._configuration(),
            method="fv2",
            limiter=self.limiter,
            reconstruction="muscl_eta_bed_velocity_v1",
            integration="ssprk2_strang_drag",
            stage_forcing=self._stage_identity,
        )

    def numerical_configuration(self):
        return dict(solver_method="fv2", **self._configuration())

    def _launch(self, kernel, count, inputs):
        wp.launch(kernel=kernel, dim=count, inputs=inputs, device=self.device)

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
        self._launch(k.split_sources, n, [*source_args, 0])
        for stage, (q, dest) in enumerate(((self._start, self._stage), (self._stage, self._end))):
            self._launch(
                k.slopes,
                n,
                [
                    q,
                    f["bed"],
                    f["active"],
                    *self._slopes,
                    self.nx,
                    self.ny,
                    self.dry_depth,
                    int(self.limiter == "mc"),
                ],
            )
            for axis, name, spacing in ((0, "x", self.dx), (1, "y", self.dy)):
                if stage and self._stage_identity[name] is not None:
                    self._launch(
                        k.forcing_time, 1, [*self._stage_series[name], self._samples[axis], ctrl, stage]
                    )
                flux = self._flux[stage][axis]
                self._launch(
                    k.faces,
                    flux.size,
                    [
                        q,
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
                        self._omit_dry,
                    ],
                )
            self._launch(
                k.donor_budget, n, [q, f["theta"], *self._flux[stage], ctrl, self.nx, self.dx, self.dy]
            )
            for axis, flux in enumerate(self._flux[stage]):
                self._launch(k.limit_faces, flux.size, [flux, f["theta"], self.nx, self.ny, axis])
            self._launch(
                k.euler,
                n,
                [
                    q,
                    dest,
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
            )
            self._stage_forcing(dest=dest, stage=stage)
        self._launch(k.validate_trial, 1, [ctrl, self.minimum_dt])
        if self._diagnostics is not None:
            self._diagnostics.enqueue(solver=self)
        self._launch(k.split_sources, n, [*source_args, 1])
        for axis, name, width in ((0, "x", self.dy), (1, "y", self.dx)):
            flux = f["flux_" + name]
            self._launch(
                k.accept_faces,
                flux.size,
                [
                    self._flux[0][axis],
                    self._flux[1][axis],
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
            )
        self._launch(k.finish, 1, [ctrl, f["fv2_cfl_predictor"], self.condition, f["dt_histogram"]])

    def _load_extra_step_modules(self):
        wp.load_module(module=k, device=self.device)
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
