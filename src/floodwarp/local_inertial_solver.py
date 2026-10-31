from floodwarp.experimental import warn_experimental_sgs
import warp as wp
import numpy as np
import os
import time
import warnings
import inspect
from typing import Tuple

from floodwarp.config import (
    BED_FLOAT,
    NP_FLOAT,
    STAGE_FLOAT,
    USE_CONDITIONED_MIXED_KERNELS,
    WP_FLOAT,
)
from floodwarp.boundary_conditions import (
    BOUNDARY_INTERNAL, BOUNDARY_CLOSED,
    BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE,
    BOUNDARY_MASKED_NORTH_STAGE, BOUNDARY_MASKED_WEST_STAGE,
    BOUNDARY_MASKED_EAST_STAGE, BOUNDARY_MASKED_SOUTH_STAGE,
    BOUNDARY_MASKED_WEST_FREE, BOUNDARY_MASKED_EAST_FREE,
    BOUNDARY_MASKED_SOUTH_FREE, BOUNDARY_MASKED_NORTH_FREE,
)




from floodwarp.local_inertial_kernels import (
    apply_sources_committed_flux_kernel,
    apply_sources_kernel,
    accumulate_face_volume_qx_kernel,
    accumulate_face_volume_qy_kernel,
    accumulate_max_active_depth_kernel,
    device_loop_accumulate_face_volume_qx_kernel,
    device_loop_accumulate_face_volume_qy_kernel,
    reduce_hydraulic_diagnostic_state_kernel,
    predict_depth_for_cfl_kernel,
    calc_cfl_dt_x_kernel,
    calc_cfl_dt_y_kernel,
    calc_conditioned_cfl_dt_x_kernel,
    calc_conditioned_cfl_dt_y_kernel,
    store_cfl_timestep_control_kernel,
    update_qx_kernel,
    update_qy_kernel,
    update_momentum_kernel,
    override_free_outflow_arithmetic_kernel,
    update_conditioned_qx_kernel,
    update_conditioned_qy_kernel,
    update_conditioned_momentum_kernel,
    calc_outflow_limiter_kernel,
    apply_limiter_qx_kernel,
    apply_limiter_qy_kernel,
    apply_outflow_limiter_kernel,
    continuity_update_kernel,
    continuity_update_and_check_kernel,
    check_negative_depth_and_max_kernel,
    check_negative_depth_kernel,
    accumulate_mass_kernel,
    sum_array_kernel,
    set_uniform_field_kernel,
    set_scaled_rainfall_kernel,
    max_active_depth_kernel,
    device_loop_timestep_cap_kernel,
    device_loop_predict_depth_kernel,
    device_loop_prepare_forcing_kernel,
    device_loop_max_depth_kernel,
    device_loop_timestep_finalize_kernel,
    device_loop_stage_fill_kernel,
    device_loop_conditioned_stage_fill_kernel,
    device_loop_apply_sources_kernel,
    device_loop_update_momentum_kernel,
    device_loop_override_free_outflow_arithmetic_kernel,
    device_loop_update_conditioned_qx_kernel,
    device_loop_update_conditioned_qy_kernel,
    device_loop_apply_outflow_limiter_kernel,
    device_loop_continuity_kernel,
    device_loop_graph_condition_kernel,
    device_loop_continuity_checkpoint_kernel,
    device_loop_max_depth_partial_kernel,
    device_loop_max_depth_finalize_kernel,
    accumulate_free_outlet_trace_kernel,
    FREE_TRACE_COUNT,
    FREE_TRACE_QOUT,
    FREE_TRACE_EDGE_MIN,
    FREE_TRACE_EDGE_MAX,
    FREE_TRACE_EDGE_SUM,
    FREE_TRACE_INWARD_MIN,
    FREE_TRACE_INWARD_MAX,
    FREE_TRACE_INWARD_SUM,
    FREE_TRACE_DH_MIN,
    FREE_TRACE_DH_MAX,
    FREE_TRACE_DH_SUM,
    FREE_TRACE_ABS_SLOPE_SUM,
    FREE_TRACE_ABS_SLOPE_MAX,
    FREE_TRACE_ABS_QOLD_SUM,
    FREE_TRACE_ABS_QNEW_SUM,
    FREE_TRACE_GRAVITY_INCREMENT_SUM,
    FREE_TRACE_FRICTION_DENOMINATOR_SUM,
    FREE_TRACE_SURFACE_VOLUME,
    FREE_TRACE_SIZE,
    DL_TIME, DL_DT, DL_ERROR, DL_MAX_DEPTH, DL_RAIN_RATE, DL_COUNTER, DL_DONE,
    DL_LAST_ACCEPTED_DT,
    DL_GROWTH_REFERENCE, DL_DT_MIN, DL_DT_MAX, DL_DT_SUM, DL_SIZE,
    lts_predict_depth_kernel,
    lts_finalize_schedule_control_kernel,
    lts_classify_cell_levels_kernel,
    lts_wet_level_guard_kernel,
    lts_prepare_level_offsets_kernel,
    lts_schedule_snapshot_kernel,
    LTS_CTRL_BASE_DT, LTS_CTRL_MACRO_DT, LTS_CTRL_MACRO_LEVEL,
    LTS_CTRL_MACRO_TICKS, LTS_CTRL_ERROR, LTS_CTRL_SIZE,
    LTS_STATS_ACTIVE, LTS_STATS_CFL_LIMITED, LTS_STATS_ACCURACY_LIMITED,
    LTS_STATS_WET_GUARD_LIMITED, LTS_STATS_FRONT_GUARD_LIMITED,
    LTS_STATS_PROMOTIONS, LTS_STATS_DEMOTIONS, LTS_STATS_PROMOTION_HELD,
    LTS_STATS_DRAINAGE_LIMITED,
    LTS_STATS_SCALARS, LTS_STATS_HIST_BINS, LTS_STATS_SIZE,
    LTS_HIST_LOG_MIN, LTS_HIST_BINS_PER_DEX,
    lts_conditioned_face_dt_x_kernel,
    lts_conditioned_face_dt_y_kernel,
    lts_face_dt_x_kernel,
    lts_face_dt_y_kernel,
    lts_cell_level_kernel,
    lts_smooth_cell_levels_kernel,
    lts_face_levels_x_kernel,
    lts_face_levels_y_kernel,
    lts_limiter_level_kernel,
    lts_count_cell_levels_kernel,
    lts_count_face_levels_kernel,
    lts_compact_cell_levels_kernel,
    lts_compact_face_levels_kernel,
    lts_apply_sources_indexed_kernel,
    lts_update_conditioned_qx_indexed_kernel,
    lts_update_conditioned_qy_indexed_kernel,
    lts_update_qx_indexed_kernel,
    lts_update_qy_indexed_kernel,
    lts_limit_outflow_indexed_kernel,
    lts_commit_qx_volume_indexed_kernel,
    lts_commit_qy_volume_indexed_kernel,
    lts_continuity_indexed_kernel,
    lts_graph_init_kernel,
    lts_graph_apply_sources_indexed_kernel,
    lts_graph_update_conditioned_qx_indexed_kernel,
    lts_graph_update_conditioned_qy_indexed_kernel,
    lts_graph_update_qx_indexed_kernel,
    lts_graph_update_qy_indexed_kernel,
    lts_graph_accept_qx_indexed_kernel,
    lts_graph_accept_qy_indexed_kernel,
    lts_limit_held_outflow_indexed_kernel,
    lts_transport_qx_indexed_kernel,
    lts_transport_qy_indexed_kernel,
    lts_graph_continuity_indexed_kernel,
    lts_graph_advance_tick_kernel,
)
from floodwarp.subgrid_geometry import (
    HYDRAULIC_GEOMETRY_MODES,
    SubgridGeometry,
    build_subgrid_geometry,
    resolve_hydraulic_geometry_mode,
    storage_from_stage,
)
from floodwarp.subgrid_kernels import (
    lts_graph_update_subgrid_qx_indexed_kernel,
    lts_graph_update_subgrid_qy_indexed_kernel,
    lts_update_subgrid_qx_indexed_kernel,
    lts_update_subgrid_qy_indexed_kernel,
    subgrid_calc_cfl_dt_x_kernel,
    subgrid_calc_cfl_dt_y_kernel,
    subgrid_derived_state_kernel,
    subgrid_lts_face_dt_x_kernel,
    subgrid_lts_face_dt_y_kernel,
    subgrid_stage_kernel,
    subgrid_update_momentum_kernel,
    subgrid_update_qx_kernel,
    subgrid_update_qy_kernel,
    device_loop_subgrid_update_momentum_kernel,
)
from floodwarp.validation_profiles import LocalInertialValidationProfile, resolve_validation_profile
from floodwarp.structure_kernels import (
    snapshot_structure_endpoints, calculate_structure_requests, apply_structure_requests,
)


FREE_OUTFLOW_ARITHMETIC_MODES = {
    "native": 1,
    "lisflood_float_surface": 2,
    "lisflood_float_operator": 3,
    "lisflood_fp64_operator": 4,
}

class LocalInertialSolver:
    def __new__(cls,*args,**kwargs):
        if cls is LocalInertialSolver:
            if kwargs.get('surface_reconstruction', 'level_pool') in ('inclined', 'terrain_pressure'):
                raise ValueError('Inclined SGS is not included in floodwarp')
            if 'hydraulic_geometry_mode' in kwargs:mode=kwargs['hydraulic_geometry_mode']
            else:
                bound=inspect.signature(cls.__init__).bind_partial(None,*args)
                mode=bound.arguments.get('hydraulic_geometry_mode','coarse')
            if mode in ('subgrid','enhanced'):
                from floodwarp.enhanced_solver import EnhancedSGSSolver
                return object.__new__(EnhancedSGSSolver)
        return object.__new__(cls)

    def __init__(
        self,
        ny: int,
        nx: int,
        dx: float,
        dy: float,
        bed_elevation: np.ndarray,
        manning_n: np.ndarray,
        active_mask: np.ndarray,
        device: str = "cpu",
        validation_profile: str | LocalInertialValidationProfile | None = None,
        use_fused_kernels: bool = True,
        accounting_interval: int = 1,
        minimum_dt: float = 1.0e-3,
        maximum_dt: float = 60.0,
        adaptive_dt_growth_factor: float | None = 1.5,
        free_outflow_limiter_mode: int | None = None,
        free_outflow_arithmetic_mode: str = "native",
        hydraulic_geometry_mode: str = "coarse",
        subgrid_bed_elevation: np.ndarray | None = None,
        subgrid_resolution_m: float | None = None,
        subgrid_manning_n: np.ndarray | None = None,
        infiltration_method: str = "constant",
        infiltration_f0_m_per_s: float | np.ndarray | None = None,
        infiltration_fd_m: float | np.ndarray | None = None,
        infiltration_k_m_per_s: float | np.ndarray | None = None,
        infiltration_ks_m_per_s: float | np.ndarray | None = None,
        infiltration_psi_f_m: float | np.ndarray | None = None,
        infiltration_delta_theta: float | np.ndarray | None = None,
        track_cumulative_face_volumes: bool = False,
        track_absolute_face_volumes: bool = True,
        track_diagnostic_max_depth: bool = True,
        structures=(),
        track_max_water_depth: bool = False,
        boundary_type_qx: np.ndarray | None = None,
        boundary_type_qy: np.ndarray | None = None,
        surface_reconstruction: str = "level_pool",
        free_outflow_closure: str | None = None,
    ):
        if hydraulic_geometry_mode == "traditional":
            warn_experimental_sgs(formulation="traditional common-stage")
        if surface_reconstruction != "level_pool":
            raise ValueError("local-inertial surface_reconstruction must be level_pool")
        if ny <= 0 or nx <= 0:
            raise ValueError("ny and nx must be positive")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive")
        if bed_elevation.shape != (ny, nx):
            raise ValueError("bed_elevation shape must be (ny, nx)")
        if manning_n.shape != (ny, nx):
            raise ValueError("manning_n shape must be (ny, nx)")
        if active_mask.shape != (ny, nx):
            raise ValueError("active_mask shape must be (ny, nx)")

        if not np.all(np.isin(active_mask, [0, 1])):
            raise ValueError("active_mask must contain only 0 or 1")

        if not np.all(np.isfinite(bed_elevation[active_mask == 1])):
            raise ValueError("Active bed elevations must be finite")

        if not np.all((manning_n[active_mask == 1] > 0) & np.isfinite(manning_n[active_mask == 1])):
            raise ValueError("Active Manning n values must be finite and positive")

        self.ny = ny
        self.nx = nx
        self.dx = WP_FLOAT(dx)
        self.structures = tuple(dict(record) for record in structures)
        self.dy = WP_FLOAT(dy)
        self.device = wp.get_device(device)
        self._init_structures(self.structures, active_mask=active_mask, bed_elevation=bed_elevation)

        # Omission changes ordinary explicit LI, including ordinary subclasses.
        # Enhanced SGS pins its legacy profile before its internal coarse base.
        if validation_profile is None and hydraulic_geometry_mode == 'coarse':
            validation_profile = 'physical_candidate'
        profile = resolve_validation_profile(profile_name=validation_profile)
        self.validation_profile = profile.name
        self.use_fused_kernels = bool(use_fused_kernels)
        # Per-fusion overrides (default to the aggregate flag). Each is env-overridable
        # so a regression can be bisected at the kernel level (momentum / limiter /
        # continuity) without code changes: OVERLANDER_FUSED_<NAME>=0|1.
        def _flag(name: str, default: bool) -> bool:
            env = os.environ.get(f"OVERLANDER_FUSED_{name}")
            return default if env is None else (env not in ("0", "false", "False"))
        self.use_fused_momentum = _flag("MOMENTUM", self.use_fused_kernels)
        self.use_fused_limiter = _flag("LIMITER", self.use_fused_kernels)
        self.use_fused_continuity = _flag("CONTINUITY", self.use_fused_kernels)
        # Mass accounting: per-cell float64 atomics serialize on small grids but
        # dominate apply_sources on large grids. Default to a native tree-reduction
        # of the applied_* arrays when the grid is large enough that the reduction
        # beats the atomic cost; OVERLANDER_REDUCE_MASS=0|1 overrides.
        reduce_mass_default = (ny * nx) >= 50_000
        _rm = os.environ.get("OVERLANDER_REDUCE_MASS")
        self.use_reduced_mass_accounting = reduce_mass_default if _rm is None else (_rm not in ("0", "false", "False"))
        if accounting_interval < 1:
            raise ValueError("accounting_interval must be at least 1")
        if not np.isfinite(minimum_dt) or minimum_dt <= 0.0:
            raise ValueError("minimum_dt must be positive and finite")
        if not np.isfinite(maximum_dt) or maximum_dt < minimum_dt:
            raise ValueError("maximum_dt must be finite and at least minimum_dt")
        if adaptive_dt_growth_factor is not None:
            if not np.isfinite(adaptive_dt_growth_factor) or adaptive_dt_growth_factor < 1.0:
                raise ValueError("adaptive_dt_growth_factor must be at least 1.0 when enabled")
        self.accounting_interval = int(accounting_interval)
        # Deferred source-mass accounting: with accounting_interval > 1 the
        # per-step applied source depths accumulate on-device into pending_*
        # fields and are tree-summed into cumulative_totals only every
        # accounting_interval steps, on the final step of a run, and whenever
        # cumulative_accounting()/compute_diagnostics() is called. Totals stay
        # exact for every step (only the FP64 summation order changes). The
        # interval applies only to the tree-reduction path; the per-cell
        # atomic path (small grids) accounts every step regardless.
        self._deferred_mass_accounting = self.accounting_interval > 1 and self.use_reduced_mass_accounting
        self._steps_since_mass_accounting = 0

        self.gravity = WP_FLOAT(9.80665)
        self.dry_depth = WP_FLOAT(profile.dry_depth)
        self.depth_epsilon = WP_FLOAT(1.0e-6)
        self.cfl_number = WP_FLOAT(0.4)
        self.limiter_epsilon = WP_FLOAT(1e-12)
        self.negative_depth_tolerance = WP_FLOAT(1e-8)
        self.source_mode = int(profile.source_mode)
        self.manning_face_mode = int(profile.manning_face_mode)
        self.use_two_dimensional_friction = int(profile.use_two_dimensional_friction)
        if profile.include_boundary_transverse_friction:
            if not profile.use_two_dimensional_friction or hydraulic_geometry_mode != "coarse":
                raise ValueError("Boundary vector friction is supported for ordinary vector-friction LI only")
            # 0/1 preserve the historical component/interior-vector operators.
            self.use_two_dimensional_friction = 2
        if profile.manning_face_mode == 2 and hydraulic_geometry_mode != "coarse":
            raise ValueError("Resistance face roughness is supported for ordinary LI only")
        closures = {"dry_exterior_head": 0, "local_or_prescribed_slope": 1}
        if free_outflow_closure is not None and free_outflow_closure not in closures:
            raise ValueError(f"free_outflow_closure must be one of {tuple(closures)}")
        resolved_free_mode = (int(profile.free_outflow_mode) if free_outflow_closure is None
                              else closures[free_outflow_closure])
        if free_outflow_arithmetic_mode not in FREE_OUTFLOW_ARITHMETIC_MODES:
            raise ValueError(
                "free_outflow_arithmetic_mode must be one of "
                f"{tuple(FREE_OUTFLOW_ARITHMETIC_MODES)}"
            )
        if (
            free_outflow_arithmetic_mode != "native"
            and resolved_free_mode != 1
        ):
            raise ValueError(
                "LISFLOOD float FREE arithmetic requires the lisflood_fp_acc "
                "or lisflood_fp_acc_parity validation profile"
            )
        if free_outflow_arithmetic_mode != "native" and USE_CONDITIONED_MIXED_KERNELS:
            raise ValueError(
                "LISFLOOD float FREE arithmetic is currently defined only for "
                "the unconditioned FP64/FP32 boundary path"
            )
        self.free_outflow_arithmetic_mode = free_outflow_arithmetic_mode
        self.free_outflow_mode = resolved_free_mode
        if hydraulic_geometry_mode not in HYDRAULIC_GEOMETRY_MODES:
            raise ValueError(
                "hydraulic_geometry_mode must be one of "
                f"{HYDRAULIC_GEOMETRY_MODES}, got {hydraulic_geometry_mode!r}"
            )
        self.free_outflow_arithmetic_mode_code = FREE_OUTFLOW_ARITHMETIC_MODES[
            free_outflow_arithmetic_mode
        ]
        resolved_limiter_mode = (
            int(profile.free_outflow_limiter_mode)
            if free_outflow_limiter_mode is None
            else int(free_outflow_limiter_mode)
        )
        if resolved_limiter_mode not in (0, 1, 2):
            raise ValueError("free_outflow_limiter_mode must be 0 (standard), 1 (priority), or 2 (bypass)")
        # Validation-only control.  Mode 0 preserves the production donor-cell
        # limiter; modes 1 and 2 are explicit outlet-isolation experiments.
        self.free_outflow_limiter_mode = resolved_limiter_mode
        self.use_lisflood_acc_timestep = profile.use_lisflood_acc_timestep
        self.apply_sources_after_momentum = profile.apply_sources_after_momentum

        # ---- experimental subgrid hydraulic geometry (opt-in) -----------
        # Orthogonal to solver_method and to the time-integration profile:
        # "coarse" (default) keeps today's behaviour with zero extra work;
        # "subgrid" derives stage/face hydraulics from the native-resolution
        # terrain while water_depth remains storage-equivalent depth.
        self.hydraulic_geometry_mode = hydraulic_geometry_mode
        self._subgrid_geometry: SubgridGeometry | None = None
        if subgrid_manning_n is not None and hydraulic_geometry_mode not in ("subgrid","traditional"):
            raise ValueError(
                "subgrid_manning_n is only meaningful with "
                "hydraulic_geometry_mode='subgrid'; either supply native "
                "subgrid terrain or drop the native roughness field"
            )
        if hydraulic_geometry_mode == "enhanced":
            raise ValueError('Enhanced SGS requires the production explicit LocalInertialSolver dispatch')
        if hydraulic_geometry_mode in ("subgrid","traditional"):
            if subgrid_bed_elevation is None or subgrid_resolution_m is None:
                raise ValueError(
                    "hydraulic_geometry_mode='subgrid' requires "
                    "subgrid_bed_elevation and subgrid_resolution_m native "
                    "terrain arrays; none were supplied"
                )
            if USE_CONDITIONED_MIXED_KERNELS:
                raise ValueError(
                    "hydraulic_geometry_mode='subgrid' is not implemented for "
                    "the conditioned mixed-precision kernel path; use the "
                    "fp64 (default) or fp32 precision modes"
                )
            if self.use_lisflood_acc_timestep or self.free_outflow_mode != 0:
                raise ValueError(
                    "hydraulic_geometry_mode='subgrid' supports only the "
                    "native free-outflow convention (validation profile "
                    "'default'); the LISFLOOD parity profiles test the "
                    "coarse formulation"
                )
            if free_outflow_arithmetic_mode != "native":
                raise ValueError(
                    "hydraulic_geometry_mode='subgrid' supports only the "
                    "native free-outflow arithmetic mode"
                )
            self._subgrid_geometry = build_subgrid_geometry(
                ny=ny,
                nx=nx,
                dx=dx,
                dy=dy,
                subgrid_bed_elevation=subgrid_bed_elevation,
                subgrid_resolution_m=subgrid_resolution_m,
                active_mask=active_mask,
                subgrid_manning_n=subgrid_manning_n,
                manning_face_mode=self.manning_face_mode,
            )
        elif subgrid_bed_elevation is not None or subgrid_resolution_m is not None:
            raise ValueError(
                "subgrid_bed_elevation/subgrid_resolution_m were supplied but "
                "hydraulic_geometry_mode is 'coarse'; pass "
                "hydraulic_geometry_mode='subgrid' to enable the experimental "
                "subgrid model"
            )

        self.minimum_dt = float(minimum_dt)
        self.maximum_dt = float(maximum_dt)
        # The production solver may grow its global timestep only gradually.
        # The LISFLOOD-FP ACC validation profile deliberately disables this
        # controller so its documented timestep sequence is preserved.
        self.adaptive_dt_growth_factor = (
            None
            if self.use_lisflood_acc_timestep or adaptive_dt_growth_factor is None
            else float(adaptive_dt_growth_factor)
        )

        self.bed_elevation = wp.array(bed_elevation, dtype=BED_FLOAT, device=self.device)
        self.manning_n = wp.array(manning_n, dtype=WP_FLOAT, device=self.device)
        self.active_mask = wp.array(active_mask, dtype=wp.int32, device=self.device)

        infiltration_methods = {"constant": 0, "exponential": 1, "none": 2, "green_ampt": 3}
        if infiltration_method not in infiltration_methods:
            raise ValueError(
                "infiltration_method must be 'constant', 'exponential', 'none', or 'green_ampt'"
            )
        self.infiltration_method_name = infiltration_method
        self.infiltration_method = infiltration_methods[infiltration_method]
        self.track_cumulative_face_volumes = bool(track_cumulative_face_volumes)
        self.track_absolute_face_volumes = bool(track_absolute_face_volumes)
        self.track_diagnostic_max_depth = bool(track_diagnostic_max_depth)
        self.track_max_water_depth = bool(track_max_water_depth)

        self.water_depth = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.cfl_depth = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)

        # ---- subgrid hydraulic geometry (allocated only when enabled) ----
        # The stage field is derived once per step (global paths) or once per
        # LTS macrostep from storage-equivalent depth; the coarse default
        # never allocates or launches any of this.
        if self._subgrid_geometry is not None:
            geometry = self._subgrid_geometry
            self._subgrid_cell_count = geometry.subgrid_cell_count
            self._subgrid_factor = geometry.factor
            self._subgrid_cell_z = wp.array(
                geometry.cell_z, dtype=BED_FLOAT, device=self.device
            )
            self._subgrid_cell_breakpoints = wp.array(
                geometry.cell_storage_breakpoints,
                dtype=BED_FLOAT, device=self.device,
            )
            self._subgrid_face_z_x = wp.array(
                geometry.face_z_x, dtype=BED_FLOAT, device=self.device
            )
            self._subgrid_face_z_min_x = wp.array(
                geometry.face_z_min_x, dtype=BED_FLOAT, device=self.device
            )
            self._subgrid_face_z_y = wp.array(
                geometry.face_z_y, dtype=BED_FLOAT, device=self.device
            )
            self._subgrid_face_z_min_y = wp.array(
                geometry.face_z_min_y, dtype=BED_FLOAT, device=self.device
            )
            # Native face roughness drives the variable-roughness conveyance
            # reduction; without it the uniform coarse-face reduction runs
            # unchanged and the kernels receive inert placeholders.
            if geometry.face_n_x is not None:
                self._subgrid_face_n_x = wp.array(
                    geometry.face_n_x, dtype=WP_FLOAT, device=self.device
                )
                self._subgrid_face_n_y = wp.array(
                    geometry.face_n_y, dtype=WP_FLOAT, device=self.device
                )
                self._use_native_face_manning = 1
            else:
                self._subgrid_face_n_x = wp.zeros(
                    (1, 1, 1), dtype=WP_FLOAT, device=self.device
                )
                self._subgrid_face_n_y = wp.zeros(
                    (1, 1, 1), dtype=WP_FLOAT, device=self.device
                )
                self._use_native_face_manning = 0
            self._eta_stage = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
            self._eta_cfl = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
            self._source_subgrid_infiltration_enabled = 1
            self._source_subgrid_cell_z = self._subgrid_cell_z
            self._source_subgrid_cell_breakpoints = self._subgrid_cell_breakpoints
            self._source_subgrid_cell_count = self._subgrid_cell_count
        else:
            self._subgrid_cell_count = 0
            self._subgrid_factor = 1
            self._subgrid_cell_z = None
            self._subgrid_cell_breakpoints = None
            self._subgrid_face_z_x = None
            self._subgrid_face_z_min_x = None
            self._subgrid_face_z_y = None
            self._subgrid_face_z_min_y = None
            self._subgrid_face_n_x = None
            self._subgrid_face_n_y = None
            self._use_native_face_manning = 0
            self._eta_stage = None
            self._eta_cfl = None
            # Source kernels receive an array argument in both geometry modes.
            # Coarse mode keeps its source branch at a full wet fraction and
            # therefore never reads these one-element placeholders.
            self._source_subgrid_infiltration_enabled = 0
            self._source_subgrid_cell_z = wp.zeros(
                (1, 1, 1), dtype=BED_FLOAT, device=self.device
            )
            self._source_subgrid_cell_breakpoints = wp.zeros(
                (1, 1, 1), dtype=BED_FLOAT, device=self.device
            )
            self._source_subgrid_cell_count = 1
        # qx/qy expose the current post-step state. qx_next/qy_next are
        # persistent ping-pong work buffers: momentum reads the complete
        # current state (including transverse faces for 2-D friction) while
        # writing the next state, avoiding a full-field copy each step.
        self.qx = wp.zeros((ny, nx + 1), dtype=WP_FLOAT, device=self.device)
        self.qy = wp.zeros((ny + 1, nx), dtype=WP_FLOAT, device=self.device)
        self.qx_next = wp.zeros((ny, nx + 1), dtype=WP_FLOAT, device=self.device)
        self.qy_next = wp.zeros((ny + 1, nx), dtype=WP_FLOAT, device=self.device)
        if self.track_cumulative_face_volumes:
            self.cumulative_net_qx_volume_m3 = wp.zeros(
                (ny, nx + 1), dtype=wp.float64, device=self.device
            )
            self.cumulative_net_qy_volume_m3 = wp.zeros(
                (ny + 1, nx), dtype=wp.float64, device=self.device
            )
            if self.track_absolute_face_volumes:
                self.cumulative_abs_qx_volume_m3 = wp.zeros(
                    (ny, nx + 1), dtype=wp.float64, device=self.device
                )
                self.cumulative_abs_qy_volume_m3 = wp.zeros(
                    (ny + 1, nx), dtype=wp.float64, device=self.device
                )
                self._diagnostic_absolute_placeholder = None
            else:
                self.cumulative_abs_qx_volume_m3 = None
                self.cumulative_abs_qy_volume_m3 = None
                self._diagnostic_absolute_placeholder = wp.zeros(
                    (1, 1), dtype=wp.float64, device=self.device
                )
            self._diagnostic_peak_discharge_m2_per_s = wp.zeros(
                1, dtype=wp.float64, device=self.device
            )
            self._diagnostic_peak_qx_rows = wp.zeros(ny, dtype=wp.float64, device=self.device)
            self._diagnostic_peak_qy_rows = wp.zeros(ny + 1, dtype=wp.float64, device=self.device)
            self._diagnostic_scalar_state = wp.zeros(3, dtype=wp.float64, device=self.device)
            self._diagnostic_max_depth_m = (
                wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
                if self.track_diagnostic_max_depth else None
            )
        else:
            self.cumulative_net_qx_volume_m3 = None
            self.cumulative_net_qy_volume_m3 = None
            self.cumulative_abs_qx_volume_m3 = None
            self.cumulative_abs_qy_volume_m3 = None
            self._diagnostic_peak_discharge_m2_per_s = None
            self._diagnostic_max_depth_m = None
            self._diagnostic_absolute_placeholder = None
            self._diagnostic_peak_qx_rows = None
            self._diagnostic_peak_qy_rows = None
            self._diagnostic_scalar_state = None

        # Standalone per-cell maximum-depth accumulator (event inundation
        # labels).  Independent of the face-volume diagnostic: it owns its own
        # field and launch, so enabling it allocates/launches nothing else.
        self.max_water_depth_m = (
            wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
            if self.track_max_water_depth else None
        )

        self.rainfall_rate = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.rainfall_spatial_multiplier = wp.array(
            np.where(np.asarray(active_mask, dtype=bool), 1.0, 0.0),
            dtype=WP_FLOAT, device=self.device,
        )
        self._spatial_rainfall_enabled = False
        self.other_source_rate = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        # External additional surface-source source rate [m/s] (standalone surface input).
        # Behaves like other_source_rate in the source kernel but is accounted
        # separately (cumulative_totals slot 5) so emergence stays
        # distinguishable from any user-configured other sources.
        self.additional_source_rate = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        # Finite-soil-storage constraints on the infiltration sink. The mask
        # disables infiltration per cell; the persistent cap array is drawn
        # down by the applied sink inside the source kernel so the cumulative
        # infiltration accepted while a limit is active cannot exceed it.
        # Neither overwrites the configured soil properties.
        self.infiltration_exclusion_mask = wp.zeros((ny, nx), dtype=wp.int32, device=self.device)
        self.infiltration_cap_remaining_m = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.infiltration_cap_active = 0
        self._infiltration_exclusion_host = None
        self.infiltration_capacity = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.infiltration_f0 = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.infiltration_fd = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.infiltration_k = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.green_ampt_ks = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.green_ampt_psi = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.green_ampt_delta_theta = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.cumulative_infiltration = wp.zeros(
            (ny, nx), dtype=WP_FLOAT, device=self.device
        )

        # Accounting fields stay FP64 even when hydraulic state is FP32. This
        # preserves long-duration source totals while keeping the large state,
        # forcing, and flux arrays bandwidth-efficient.
        self.applied_rain = wp.zeros((ny, nx), dtype=wp.float64, device=self.device)
        self.applied_other = wp.zeros((ny, nx), dtype=wp.float64, device=self.device)
        self.applied_infil = wp.zeros((ny, nx), dtype=wp.float64, device=self.device)

        self.boundary_type_qx = wp.zeros((ny, nx + 1), dtype=wp.int32, device=self.device)
        self.boundary_stage_qx = wp.zeros((ny, nx + 1), dtype=STAGE_FLOAT, device=self.device)
        self.boundary_stage_depth_qx = wp.zeros((ny, nx + 1), dtype=WP_FLOAT, device=self.device)
        self.boundary_free_slope_qx = wp.zeros((ny, nx + 1), dtype=WP_FLOAT, device=self.device)
        self.boundary_stage_reference_qx = wp.zeros((ny, nx + 1), dtype=STAGE_FLOAT, device=self.device)

        self.boundary_type_qy = wp.zeros((ny + 1, nx), dtype=wp.int32, device=self.device)
        self.boundary_stage_qy = wp.zeros((ny + 1, nx), dtype=STAGE_FLOAT, device=self.device)
        self.boundary_stage_depth_qy = wp.zeros((ny + 1, nx), dtype=WP_FLOAT, device=self.device)
        self.boundary_free_slope_qy = wp.zeros((ny + 1, nx), dtype=WP_FLOAT, device=self.device)
        self.boundary_stage_reference_qy = wp.zeros((ny + 1, nx), dtype=STAGE_FLOAT, device=self.device)

        # Mixed precision never loads absolute FP64 DEM values in a timestep
        # kernel. These persistent fields hold FP64-precomputed, datum-relative
        # face differences and boundary-stage references in hydraulic FP32.
        self.bed_slope_qx = wp.zeros((ny, nx + 1), dtype=WP_FLOAT, device=self.device)
        self.bed_slope_qy = wp.zeros((ny + 1, nx), dtype=WP_FLOAT, device=self.device)
        self._conditioned_bed_array = None

        self.outflow_limiter = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        self.free_outflow_limiter = wp.zeros((ny, nx), dtype=WP_FLOAT, device=self.device)
        # [raw FREE volume, post-limiter FREE volume, samples, alpha sum,
        # limiter hits, minimum alpha].  Only updated in diagnostic runs.
        self.free_outflow_diagnostics = wp.zeros(6, dtype=wp.float64, device=self.device)
        self._reset_free_outflow_limiter_diagnostics()
        self._free_outlet_trace_stats = wp.zeros(
            FREE_TRACE_SIZE, dtype=wp.float64, device=self.device
        )

        self.dt_min_arr = wp.zeros(1, dtype=WP_FLOAT, device=self.device)
        self.maximum_depth_arr = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.total_vol_arr = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.error_flag = wp.zeros(1, dtype=wp.int32, device=self.device)
        # [maximum active depth after the accepted step, binary error flag].
        # Used only by the LISFLOOD-FP ACC host-stepped timestep path.
        self.lisflood_step_control = wp.zeros(2, dtype=wp.float64, device=self.device)

        # On-device mass accounting: [rain, infiltration, other_sources,
        # boundary_inflow, boundary_outflow, additional_sources]. Accumulated
        # by atomic adds inside the source and boundary-flux kernels so the
        # host does not have to read fields back every step. Synced only via
        # cumulative_accounting().
        self.cumulative_totals = wp.zeros(6, dtype=wp.float64, device=self.device)

        # Per-step source-volume reductions (mass-accounting path for large grids,
        # where ~3*N global float64 atomics dominate apply_sources). Each call to
        # wp.utils.array_sum writes one tree-sum here; accumulate_mass_kernel folds
        # them (scaled by cell area) into cumulative_totals.
        self._mass_rain_sum = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._mass_infil_sum = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._mass_other_sum = wp.zeros(1, dtype=wp.float64, device=self.device)

        # Deferred-accounting pending fields (allocated only when used).
        if self._deferred_mass_accounting:
            self._pending_rain = wp.zeros((ny, nx), dtype=wp.float64, device=self.device)
            self._pending_infil = wp.zeros((ny, nx), dtype=wp.float64, device=self.device)
            self._pending_other = wp.zeros((ny, nx), dtype=wp.float64, device=self.device)
        else:
            self._pending_rain = None
            self._pending_infil = None
            self._pending_other = None

        # Spatial local-timestep buffers are allocated lazily so the normal
        # global-step path pays no memory cost.
        self._lts_capacity_levels = 0
        self._lts_face_dt_qx = None
        self._lts_face_dt_qy = None
        self._lts_cell_levels = None
        self._lts_cell_levels_tmp = None
        self._lts_qx_levels = None
        self._lts_qy_levels = None
        self._lts_limiter_levels = None
        self._lts_pending_flux_volume = None
        self._lts_counts = None
        self._lts_cell_indices = None
        self._lts_limiter_indices = None
        self._lts_qx_indices = None
        self._lts_qy_indices = None
        self._lts_cell_write = None
        self._lts_limiter_write = None
        self._lts_qx_write = None
        self._lts_qy_write = None
        self._lts_schedule_control = None
        self._lts_schedule_snapshot = None
        # Accuracy-control history, retained between macrosteps and between
        # checkpointed run_spatially_adaptive_loop calls until the next depth
        # reset: previous cell levels, previous effective-depth samples, the
        # host-side sample timestamp, and run-accumulated classifier stats.
        self._lts_prev_cell_levels = None
        self._lts_prev_effective_depth = None
        self._lts_classify_stats = None
        self._lts_last_classify_time: float | None = None
        self._lts_graph_tick = None
        self._lts_graph_condition = None
        self._lts_graph_forcing = None
        self._lts_subcycle_graph = None
        self._lts_subcycle_graph_shape = None




        self.simulation_time = 0.0
        self.current_dt = 0.0
        self._adaptive_dt_reference = 0.0
        self.last_timestep_max_depth_m: float | None = None
        self._lisflood_cached_maximum_depth_m: float | None = None

        self._active_mask_host = active_mask.copy()
        self._configure_infiltration(
            method=infiltration_method,
            f0_m_per_s=infiltration_f0_m_per_s,
            fd_m=infiltration_fd_m,
            k_m_per_s=infiltration_k_m_per_s,
            green_ampt_ks_m_per_s=infiltration_ks_m_per_s,
            green_ampt_psi_f_m=infiltration_psi_f_m,
            green_ampt_delta_theta=infiltration_delta_theta,
        )

        self.initial_volume = 0.0

        self._refresh_conditioned_bed_fields()
        self._init_boundaries(active_mask)
        # Optional setup-time topology avoids constructing enhanced momentum
        # tables for boundaries which the model immediately replaces. Use the
        # same validators/setters without invoking subclass geometry callbacks
        # before that subclass has allocated its independent hydraulic state.
        if boundary_type_qx is not None:
            LocalInertialSolver.set_boundary_type_qx(self,boundary_type=boundary_type_qx)
        if boundary_type_qy is not None:
            LocalInertialSolver.set_boundary_type_qy(self,boundary_type=boundary_type_qy)

    def _refresh_conditioned_bed_fields(self) -> None:
        """Precompute datum-relative face differences from the current DEM.

        The subtraction is deliberately performed on the host in FP64 before
        the physically small elevation differences are stored as ``WP_FLOAT``.
        This keeps mixed-precision timestep kernels free of absolute-datum
        FP64 arithmetic. The object-identity marker also supports the legacy
        pattern of replacing ``solver.bed_elevation`` directly in tests.
        """
        bed = np.asarray(self.bed_elevation.numpy(), dtype=np.float64)
        slope_qx = np.zeros((self.ny, self.nx + 1), dtype=NP_FLOAT)
        slope_qy = np.zeros((self.ny + 1, self.nx), dtype=NP_FLOAT)
        if self.nx > 1:
            slope_qx[:, 1:self.nx] = np.diff(bed, axis=1)
            slope_qx[:, 0] = bed[:, 0] - bed[:, 1]
            slope_qx[:, self.nx] = bed[:, -1] - bed[:, -2]
        if self.ny > 1:
            slope_qy[1:self.ny, :] = np.diff(bed, axis=0)
            slope_qy[0, :] = bed[0, :] - bed[1, :]
            slope_qy[self.ny, :] = bed[-1, :] - bed[-2, :]

        stage_reference_qx = np.zeros((self.ny, self.nx + 1), dtype=np.float64)
        stage_reference_qx[:, 0] = bed[:, 0]
        stage_reference_qx[:, self.nx] = bed[:, -1]
        stage_reference_qy = np.zeros((self.ny + 1, self.nx), dtype=np.float64)
        stage_reference_qy[0, :] = bed[0, :]
        stage_reference_qy[self.ny, :] = bed[-1, :]

        wp.copy(self.bed_slope_qx, wp.array(slope_qx, dtype=WP_FLOAT, device=self.device))
        wp.copy(self.bed_slope_qy, wp.array(slope_qy, dtype=WP_FLOAT, device=self.device))
        wp.copy(
            self.boundary_stage_reference_qx,
            wp.array(stage_reference_qx, dtype=STAGE_FLOAT, device=self.device),
        )
        wp.copy(
            self.boundary_stage_reference_qy,
            wp.array(stage_reference_qy, dtype=STAGE_FLOAT, device=self.device),
        )
        self._conditioned_bed_array = self.bed_elevation
        self._set_conditioned_stage_qx(self.boundary_stage_qx.numpy(), bed)
        self._set_conditioned_stage_qy(self.boundary_stage_qy.numpy(), bed)

    def _ensure_conditioned_bed_fields(self) -> None:
        """Refresh conditioned fields if legacy code replaced the DEM array."""
        if USE_CONDITIONED_MIXED_KERNELS and self._conditioned_bed_array is not self.bed_elevation:
            self._refresh_conditioned_bed_fields()

    def _set_conditioned_stage_qx(self, stage: np.ndarray, bed: np.ndarray | None = None) -> None:
        """Update external-stage depth and FREE1 slope fields for x faces."""
        if bed is None:
            bed = np.asarray(self.bed_elevation.numpy(), dtype=np.float64)
        stage_depth = np.zeros((self.ny, self.nx + 1), dtype=NP_FLOAT)
        stage_depth[:, 0] = np.asarray(stage[:, 0], dtype=np.float64) - bed[:, 0]
        stage_depth[:, self.nx] = np.asarray(stage[:, self.nx], dtype=np.float64) - bed[:, -1]
        boundary_type = np.asarray(self.boundary_type_qx.numpy(), dtype=np.int32)
        masked_west = (
            boundary_type[:, 1:self.nx] == BOUNDARY_MASKED_WEST_STAGE
        )
        internal_depth = (
            np.asarray(stage[:, 1:self.nx], dtype=np.float64) - bed[:, 1:]
        )
        stage_depth[:, 1:self.nx][masked_west] = internal_depth[masked_west]
        masked_east = boundary_type[:, 1:self.nx] == BOUNDARY_MASKED_EAST_STAGE
        east_depth = np.asarray(stage[:, 1:self.nx],dtype=np.float64)-bed[:, :-1]
        stage_depth[:, 1:self.nx][masked_east] = east_depth[masked_east]
        reference=np.zeros_like(stage,dtype=np.float64)
        reference[:,0]=bed[:,0];reference[:,-1]=bed[:,-1]
        reference[:,1:self.nx][masked_west]=bed[:,1:][masked_west]
        reference[:,1:self.nx][masked_east]=bed[:,:-1][masked_east]
        wp.copy(self.boundary_stage_reference_qx,wp.array(reference,dtype=STAGE_FLOAT,device=self.device))
        wp.copy(self.boundary_stage_depth_qx, wp.array(stage_depth, dtype=WP_FLOAT, device=self.device))
        wp.copy(
            self.boundary_free_slope_qx,
            wp.array(stage, dtype=WP_FLOAT, device=self.device),
        )

    def _set_conditioned_stage_qy(self, stage: np.ndarray, bed: np.ndarray | None = None) -> None:
        """Update external-stage depth and FREE1 slope fields for y faces."""
        if bed is None:
            bed = np.asarray(self.bed_elevation.numpy(), dtype=np.float64)
        stage_depth = np.zeros((self.ny + 1, self.nx), dtype=NP_FLOAT)
        stage_depth[0, :] = np.asarray(stage[0, :], dtype=np.float64) - bed[0, :]
        stage_depth[self.ny, :] = np.asarray(stage[self.ny, :], dtype=np.float64) - bed[-1, :]
        boundary_type = np.asarray(self.boundary_type_qy.numpy(), dtype=np.int32)
        masked_north = (
            boundary_type[1:self.ny, :] == BOUNDARY_MASKED_NORTH_STAGE
        )
        internal_depth = (
            np.asarray(stage[1:self.ny, :], dtype=np.float64) - bed[:-1, :]
        )
        stage_depth[1:self.ny, :][masked_north] = internal_depth[masked_north]
        masked_south = boundary_type[1:self.ny, :] == BOUNDARY_MASKED_SOUTH_STAGE
        south_depth = np.asarray(stage[1:self.ny, :],dtype=np.float64)-bed[1:, :]
        stage_depth[1:self.ny, :][masked_south] = south_depth[masked_south]
        reference=np.zeros_like(stage,dtype=np.float64)
        reference[0,:]=bed[0,:];reference[-1,:]=bed[-1,:]
        reference[1:self.ny,:][masked_north]=bed[:-1,:][masked_north]
        reference[1:self.ny,:][masked_south]=bed[1:,:][masked_south]
        wp.copy(self.boundary_stage_reference_qy,wp.array(reference,dtype=STAGE_FLOAT,device=self.device))
        wp.copy(self.boundary_stage_depth_qy, wp.array(stage_depth, dtype=WP_FLOAT, device=self.device))
        wp.copy(
            self.boundary_free_slope_qy,
            wp.array(stage, dtype=WP_FLOAT, device=self.device),
        )

    def _init_boundaries(self, active_mask: np.ndarray):
        b_qx = np.full((self.ny, self.nx + 1), BOUNDARY_CLOSED, dtype=np.int32)
        b_qy = np.full((self.ny + 1, self.nx), BOUNDARY_CLOSED, dtype=np.int32)

        for j in range(self.ny):
            for i in range(self.nx - 1):
                if active_mask[j, i] == 1 and active_mask[j, i+1] == 1:
                    b_qx[j, i+1] = BOUNDARY_INTERNAL

        for j in range(self.ny - 1):
            for i in range(self.nx):
                if active_mask[j, i] == 1 and active_mask[j+1, i] == 1:
                    b_qy[j+1, i] = BOUNDARY_INTERNAL

        wp.copy(self.boundary_type_qx, wp.array(b_qx, dtype=wp.int32, device=self.device))
        wp.copy(self.boundary_type_qy, wp.array(b_qy, dtype=wp.int32, device=self.device))

    def _reset_free_outflow_limiter_diagnostics(self) -> None:
        """Reset optional FREE-outlet limiter counters without changing state."""
        initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        wp.copy(
            dest=self.free_outflow_diagnostics,
            src=wp.array(initial, dtype=wp.float64, device=self.device),
        )

    def free_outflow_limiter_diagnostics(self) -> dict[str, float]:
        """Return cumulative raw and limited FREE-outlet flux diagnostics."""
        values = np.asarray(self.free_outflow_diagnostics.numpy(), dtype=np.float64)
        samples = float(values[2])
        return {
            "raw_free_outflow_m3": float(values[0]),
            "limited_free_outflow_m3": float(values[1]),
            "free_outflow_limiter_removed_m3": float(values[0] - values[1]),
            "free_outflow_samples": samples,
            "free_outflow_limiter_hits": float(values[4]),
            "free_outflow_alpha_mean": float(values[3] / samples) if samples > 0.0 else 1.0,
            "free_outflow_alpha_min": float(values[5]) if samples > 0.0 else 1.0,
        }

    def free_outlet_trace_snapshot(self) -> dict[str, float]:
        """Return one compact diagnostic snapshot of all FREE boundary faces.

        This is intentionally opt-in and is suitable only at device-loop sync
        points. It reads one small reduction array, never the full hydraulic
        state, and does not modify the trajectory.
        """
        initial = np.zeros(FREE_TRACE_SIZE, dtype=np.float64)
        initial[FREE_TRACE_EDGE_MIN] = np.inf
        initial[FREE_TRACE_INWARD_MIN] = np.inf
        initial[FREE_TRACE_DH_MIN] = np.inf
        initial[FREE_TRACE_EDGE_MAX] = -np.inf
        initial[FREE_TRACE_INWARD_MAX] = -np.inf
        initial[FREE_TRACE_DH_MAX] = -np.inf
        wp.copy(
            dest=self._free_outlet_trace_stats,
            src=wp.array(initial, dtype=wp.float64, device=self.device),
        )
        wp.launch(
            kernel=accumulate_free_outlet_trace_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self.water_depth, self.qx, self.qy, self.qx_next, self.qy_next,
                self.bed_elevation, self.manning_n, self.active_mask,
                self.boundary_type_qx, self.boundary_type_qy,
                self.boundary_stage_qx, self.boundary_stage_qy,
                self.dx, self.dy, WP_FLOAT(self.current_dt), self.gravity,
                self.depth_epsilon, self.free_outflow_arithmetic_mode_code,
                self._free_outlet_trace_stats,
            ],
            device=self.device,
        )
        values = np.asarray(self._free_outlet_trace_stats.numpy(), dtype=np.float64)
        count = float(values[FREE_TRACE_COUNT])
        if count == 0.0:
            return {
                "outlet_face_count": 0.0,
                "overlander_surface_volume_m3": float(values[FREE_TRACE_SURFACE_VOLUME]),
            }

        def average(index: int) -> float:
            return float(values[index] / count)

        return {
            "outlet_face_count": count,
            "overlander_surface_volume_m3": float(values[FREE_TRACE_SURFACE_VOLUME]),
            "overlander_boundary_qout_m3_per_s": float(values[FREE_TRACE_QOUT]),
            "outlet_edge_depth_min_m": float(values[FREE_TRACE_EDGE_MIN]),
            "outlet_edge_depth_mean_m": average(FREE_TRACE_EDGE_SUM),
            "outlet_edge_depth_max_m": float(values[FREE_TRACE_EDGE_MAX]),
            "outlet_inward_depth_min_m": float(values[FREE_TRACE_INWARD_MIN]),
            "outlet_inward_depth_mean_m": average(FREE_TRACE_INWARD_SUM),
            "outlet_inward_depth_max_m": float(values[FREE_TRACE_INWARD_MAX]),
            "outlet_surface_dh_min_m": float(values[FREE_TRACE_DH_MIN]),
            "outlet_surface_dh_mean_m": average(FREE_TRACE_DH_SUM),
            "outlet_surface_dh_max_m": float(values[FREE_TRACE_DH_MAX]),
            "outlet_abs_surface_slope_mean": average(FREE_TRACE_ABS_SLOPE_SUM),
            "outlet_abs_surface_slope_max": float(values[FREE_TRACE_ABS_SLOPE_MAX]),
            "outlet_q_old_mean_m2_per_s": average(FREE_TRACE_ABS_QOLD_SUM),
            "outlet_q_new_mean_m2_per_s": average(FREE_TRACE_ABS_QNEW_SUM),
            "outlet_retained_abs_q_old_mean_m2_per_s": average(FREE_TRACE_ABS_QOLD_SUM),
            "outlet_gravity_slope_increment_mean_m2_per_s2": average(FREE_TRACE_GRAVITY_INCREMENT_SUM),
            "outlet_friction_denominator_mean": average(FREE_TRACE_FRICTION_DENOMINATOR_SUM),
        }

    def set_water_depth(self, depth: np.ndarray):
        """Reset hydraulic state and mass accounting to a new depth field.

        Fluxes, source totals, clocks, the LTS classifier history, and the
        adaptive-timestep growth reference all restart.  When
        ``track_cumulative_face_volumes`` is enabled the diagnostic transport
        state (cumulative net/absolute face volumes, peak discharge, and
        maximum depth) resets with the hydraulic state, so a new simulation
        never carries stale cumulative transport; use
        :meth:`reset_cumulative_face_volumes` for an explicit standalone
        reset.
        """
        if depth.shape != (self.ny, self.nx):
            raise ValueError("Depth shape must match (ny, nx)")
        if not np.all(np.isfinite(depth)):
            raise ValueError("Water depth must be finite")
        if np.any(depth < -float(self.negative_depth_tolerance)):
            raise ValueError("Water depth must be nonnegative")

        depth = np.where(self._active_mask_host == 1, depth, 0.0)
        wp.copy(self.water_depth, wp.array(depth, dtype=WP_FLOAT, device=self.device))
        self.qx.zero_()
        self.qy.zero_()
        self.qx_next.zero_()
        self.qy_next.zero_()
        self.applied_rain.zero_()
        self.applied_other.zero_()
        self.applied_infil.zero_()
        self.cumulative_infiltration.zero_()
        if self.infiltration_method == 1:
            wp.copy(self.infiltration_capacity, self.infiltration_f0)
        self.outflow_limiter.zero_()
        self.free_outflow_limiter.zero_()
        self._reset_free_outflow_limiter_diagnostics()
        if self.track_cumulative_face_volumes:
            # A depth reset starts a new simulation: stale cumulative
            # transport must not survive into it.
            self.reset_cumulative_face_volumes()
        if self.track_max_water_depth:
            # A depth reset starts a new simulation: a stale event maximum
            # must not survive into it.
            self.max_water_depth_m.zero_()
        if self._lts_pending_flux_volume is not None:
            self._lts_pending_flux_volume.zero_()
        if self._lts_prev_cell_levels is not None:
            # A depth reset invalidates the LTS accuracy-control history so no
            # stale level or effective-depth sample can survive into a new run.
            self._lts_prev_cell_levels.fill_(-1)
            self._lts_prev_effective_depth.zero_()
            self._lts_classify_stats.zero_()
            self._lts_last_classify_time = None

        self.simulation_time = 0.0
        self.current_dt = 0.0
        self._adaptive_dt_reference = 0.0
        self.cumulative_totals.zero_()
        self._steps_since_mass_accounting = 0
        if self._deferred_mass_accounting:
            self._pending_rain.zero_()
            self._pending_infil.zero_()
            self._pending_other.zero_()
        self.lisflood_step_control.zero_()
        self._lisflood_cached_maximum_depth_m = None
        self.reset_structure_diagnostics()

        self.total_vol_arr.zero_()
        wp.launch(
            kernel=sum_array_kernel,
            dim=(self.ny, self.nx),
            inputs=[self.water_depth, self.active_mask, self.dx, self.dy, self.total_vol_arr],
            device=self.device
        )
        self.initial_volume = float(self.total_vol_arr.numpy()[0])

    def reset(self) -> None:
        """Start a new event with zero surface storage and cumulative infiltration.

        Configured Green-Ampt ``delta_theta`` is retained as the antecedent
        scenario state. Continuation methods do not call this reset.
        """
        self.set_water_depth(np.zeros((self.ny, self.nx), dtype=np.float64))

    def set_face_fluxes(self, qx: np.ndarray, qy: np.ndarray) -> None:
        """Replace the current face-discharge state without resetting depth.

        Both persistent ping-pong buffers receive the supplied current state,
        so the next momentum update reads a complete and consistent old flux
        field.  This is intended for checkpoint/restart and exact surrogate
        data replay; callers remain responsible for providing physically
        compatible boundary-face values.
        """
        if qx.shape != (self.ny, self.nx + 1):
            raise ValueError("qx shape must match (ny, nx + 1)")
        if qy.shape != (self.ny + 1, self.nx):
            raise ValueError("qy shape must match (ny + 1, nx)")
        if not np.all(np.isfinite(qx)) or not np.all(np.isfinite(qy)):
            raise ValueError("Face fluxes must be finite")
        qx_values = wp.array(qx, dtype=WP_FLOAT, device=self.device)
        qy_values = wp.array(qy, dtype=WP_FLOAT, device=self.device)
        wp.copy(self.qx, qx_values)
        wp.copy(self.qy, qy_values)
        wp.copy(self.qx_next, qx_values)
        wp.copy(self.qy_next, qy_values)

    def set_bed_elevation(self, bed_elevation: np.ndarray) -> None:
        """Replace the DEM while preserving the external elevation datum.

        Mixed precision derives its FP32 face differences from this FP64 input
        before the next timestep; callers should prefer this method over direct
        replacement of ``bed_elevation``.
        """
        self._validate_field(name="bed_elevation", values=bed_elevation)
        if not np.all(np.isfinite(bed_elevation[self._active_mask_host == 1])):
            raise ValueError("Active bed elevations must be finite")
        self.bed_elevation = wp.array(bed_elevation, dtype=BED_FLOAT, device=self.device)
        self._refresh_conditioned_bed_fields()

    def _normalise_infiltration_parameter(
        self,
        name: str,
        values: float | np.ndarray,
        *,
        positive: bool,
    ) -> np.ndarray:
        if np.isscalar(values):
            array = np.full((self.ny, self.nx), float(values), dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (self.ny, self.nx):
                raise ValueError(
                    f"{name} must be a scalar or an array with shape "
                    f"({self.ny}, {self.nx})"
                )
            array = array.copy()
        active = self._active_mask_host == 1
        if not np.all(np.isfinite(array[active])):
            raise ValueError(f"{name} must be finite on active cells")
        valid = array[active] > 0.0 if positive else array[active] >= 0.0
        if not np.all(valid):
            requirement = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be {requirement} on active cells")
        array[~active] = 0.0
        return array

    def _configure_infiltration(
        self,
        *,
        method: str,
        f0_m_per_s: float | np.ndarray | None,
        fd_m: float | np.ndarray | None,
        k_m_per_s: float | np.ndarray | None,
        green_ampt_ks_m_per_s: float | np.ndarray | None = None,
        green_ampt_psi_f_m: float | np.ndarray | None = None,
        green_ampt_delta_theta: float | np.ndarray | None = None,
    ) -> None:
        if method == "exponential":
            if f0_m_per_s is None or fd_m is None or k_m_per_s is None:
                raise ValueError(
                    "exponential infiltration requires infiltration_f0_m_per_s, "
                    "infiltration_fd_m, and infiltration_k_m_per_s"
                )
            f0 = self._normalise_infiltration_parameter(
                "infiltration_f0_m_per_s", f0_m_per_s, positive=False
            )
            fd = self._normalise_infiltration_parameter(
                "infiltration_fd_m", fd_m, positive=True
            )
            k = self._normalise_infiltration_parameter(
                "infiltration_k_m_per_s", k_m_per_s, positive=False
            )
            active = self._active_mask_host == 1
            if np.any(f0[active] < k[active]):
                raise ValueError(
                    "infiltration_f0_m_per_s must be at least "
                    "infiltration_k_m_per_s on active cells"
                )
            wp.copy(self.infiltration_f0, wp.array(f0, dtype=WP_FLOAT, device=self.device))
            wp.copy(self.infiltration_fd, wp.array(fd, dtype=WP_FLOAT, device=self.device))
            wp.copy(self.infiltration_k, wp.array(k, dtype=WP_FLOAT, device=self.device))
            wp.copy(self.infiltration_capacity, self.infiltration_f0)
            self.cumulative_infiltration.zero_()
            self.green_ampt_ks.zero_()
            self.green_ampt_psi.zero_()
            self.green_ampt_delta_theta.zero_()
        elif method == "green_ampt":
            if (green_ampt_ks_m_per_s is None or green_ampt_psi_f_m is None
                    or green_ampt_delta_theta is None):
                raise ValueError(
                    "green_ampt infiltration requires Ks, wetting-front suction, "
                    "and delta_theta"
                )
            ks = self._normalise_infiltration_parameter(
                "infiltration_ks_m_per_s", green_ampt_ks_m_per_s, positive=False
            )
            psi = self._normalise_infiltration_parameter(
                "infiltration_psi_f_m", green_ampt_psi_f_m, positive=True
            )
            deficit = self._normalise_infiltration_parameter(
                "infiltration_delta_theta", green_ampt_delta_theta, positive=False
            )
            active = self._active_mask_host == 1
            if np.any(deficit[active] > 1.0):
                raise ValueError("infiltration_delta_theta must be in [0, 1]")
            wp.copy(self.green_ampt_ks, wp.array(ks, dtype=WP_FLOAT, device=self.device))
            wp.copy(self.green_ampt_psi, wp.array(psi, dtype=WP_FLOAT, device=self.device))
            wp.copy(self.green_ampt_delta_theta, wp.array(deficit, dtype=WP_FLOAT, device=self.device))
            self.infiltration_capacity.zero_()
            self.infiltration_f0.zero_()
            self.infiltration_fd.zero_()
            self.infiltration_k.zero_()
            self.cumulative_infiltration.zero_()
        else:
            self.green_ampt_ks.zero_()
            self.green_ampt_psi.zero_()
            self.green_ampt_delta_theta.zero_()
            self.infiltration_f0.zero_()
            self.infiltration_fd.zero_()
            self.infiltration_k.zero_()
            self.cumulative_infiltration.zero_()
            if method == "none":
                self.infiltration_capacity.zero_()

    def set_exponential_infiltration(
        self,
        infiltration_f0_m_per_s: float | np.ndarray,
        infiltration_fd_m: float | np.ndarray,
        infiltration_k_m_per_s: float | np.ndarray,
    ) -> None:
        'Enable exponential event-scale infiltration and reset cumulative F.'
        self.infiltration_method_name = "exponential"
        self.infiltration_method = 1
        self._configure_infiltration(
            method="exponential",
            f0_m_per_s=infiltration_f0_m_per_s,
            fd_m=infiltration_fd_m,
            k_m_per_s=infiltration_k_m_per_s,
        )

    def set_green_ampt_infiltration(
        self,
        infiltration_ks_m_per_s: float | np.ndarray,
        infiltration_psi_f_m: float | np.ndarray,
        infiltration_delta_theta: float | np.ndarray,
    ) -> None:
        """Enable Green-Ampt event infiltration and reset cumulative F.

        Parameters use SI units; ``delta_theta`` is a volumetric moisture
        deficit in [0, 1], supplied as a standalone soil parameter.
        """
        self.infiltration_method_name = "green_ampt"
        self.infiltration_method = 3
        self._configure_infiltration(
            method="green_ampt",
            f0_m_per_s=None,
            fd_m=None,
            k_m_per_s=None,
            green_ampt_ks_m_per_s=infiltration_ks_m_per_s,
            green_ampt_psi_f_m=infiltration_psi_f_m,
            green_ampt_delta_theta=infiltration_delta_theta,
        )

    @property
    def cumulative_infiltration_m(self) -> np.ndarray:
        'Return cumulative actual infiltration depth in metres.'
        return np.asarray(self.cumulative_infiltration.numpy())

    @property
    def current_infiltration_capacity_m_per_s(self) -> np.ndarray:
        'Return the current capacity field in m/s after a device sync.'
        if self.infiltration_method == 3:
            ks = self.green_ampt_ks.numpy()
            psi = self.green_ampt_psi.numpy()
            deficit = self.green_ampt_delta_theta.numpy()
            cumulative = np.maximum(self.cumulative_infiltration.numpy(), 0.0)
            a = psi * deficit
            capacity = ks.copy()
            positive_a = a > 0.0
            positive_f = cumulative > 0.0
            capacity[positive_a & positive_f] *= (1.0 + a[positive_a & positive_f] / cumulative[positive_a & positive_f])
            capacity[positive_a & ~positive_f & (ks > 0.0)] = np.inf
            return capacity
        if self.infiltration_method == 1:
            f0 = self.infiltration_f0.numpy()
            fd = np.maximum(self.infiltration_fd.numpy(), 1.0e-30)
            k = self.infiltration_k.numpy()
            cumulative = np.maximum(self.cumulative_infiltration.numpy(), 0.0)
            return k + (f0 - k) * np.exp(-cumulative / fd)
        return np.asarray(self.infiltration_capacity.numpy())

    def set_source_rates(self, rain: np.ndarray = None, other: np.ndarray = None, infil: np.ndarray = None):
        if rain is not None:
            self._validate_field(name="rain", values=rain)
            wp.copy(self.rainfall_rate, wp.array(rain, dtype=WP_FLOAT, device=self.device))
        if other is not None:
            self._validate_field(name="other", values=other)
            wp.copy(self.other_source_rate, wp.array(other, dtype=WP_FLOAT, device=self.device))
        if infil is not None:
            self._validate_field(name="infil", values=infil)
            wp.copy(self.infiltration_capacity, wp.array(infil, dtype=WP_FLOAT, device=self.device))

    def set_additional_source_rate(self, rate: np.ndarray) -> None:
        """Set the external additional surface-source source rate field [m/s].

        The rate must be finite and non-negative everywhere; it is ignored on
        inactive cells by the source kernel. The value persists across steps
        until replaced (or zeroed), so a surface input driver re-issues it
        once per synchronization interval. The volume is accounted in the
        dedicated ``additional_sources`` slot of :meth:`cumulative_accounting`
        and the source participates in ordinary routing, Green-Ampt supply,
        and max-depth tracking through accepted timesteps.
        """
        self._validate_field(name="additional_source_rate", values=rate)
        values = np.asarray(rate, dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError("additional_source_rate must be non-negative")
        wp.copy(self.additional_source_rate, wp.array(values, dtype=WP_FLOAT, device=self.device))

    def set_infiltration_exclusion_mask(self, mask: np.ndarray) -> None:
        """Disable infiltration on the masked cells until the mask is replaced.

        Boolean/array field of shape (ny, nx); truthy cells take zero
        infiltration sink in every source kernel while the mask is set.
        Configured soil properties are untouched; clearing via
        :meth:`clear_infiltration_limits` restores ordinary physics.
        """
        values = np.asarray(mask)
        if values.shape != (self.ny, self.nx):
            raise ValueError(f"mask shape must be {(self.ny, self.nx)}")
        self._infiltration_exclusion_host = (
            np.asarray(values, dtype=bool).astype(np.int32)
        )
        wp.copy(
            self.infiltration_exclusion_mask,
            wp.array(self._infiltration_exclusion_host, dtype=wp.int32, device=self.device),
        )

    def set_infiltration_storage_limit_m(self, limit_m: np.ndarray) -> None:
        """Set a per-cell remaining-storage cap on the infiltration sink [m].

        While active, the applied infiltration sink each step is clamped to
        the remaining cap and the cap is drawn down in-kernel, so the
        cumulative infiltration accepted after this call cannot exceed the
        supplied limit (subject to hydraulic-state precision). The limit must
        be re-issued per soil-cap interval; :meth:`clear_infiltration_limits`
        deactivates it.
        """
        values = np.asarray(limit_m, dtype=np.float64)
        if values.shape != (self.ny, self.nx):
            raise ValueError(f"limit_m shape must be {(self.ny, self.nx)}")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("limit_m must be finite and non-negative")
        wp.copy(self.infiltration_cap_remaining_m, wp.array(values, dtype=WP_FLOAT, device=self.device))
        self.infiltration_cap_active = 1

    def clear_infiltration_limits(self) -> None:
        """Clear the exclusion mask and storage cap (ordinary physics resume)."""
        self.infiltration_exclusion_mask.zero_()
        self.infiltration_cap_remaining_m.zero_()
        self.infiltration_cap_active = 0
        self._infiltration_exclusion_host = None

    @property
    def infiltration_limits_active(self) -> bool:
        """True while an exclusion mask or storage cap is set."""
        return bool(np.any(self._infiltration_exclusion_host)) or self.infiltration_cap_active == 1









































































































    def cumulative_accounting(self) -> dict[str, float]:
        """Return cumulative mass totals (one device->host sync).

        Reads the on-device accumulator so callers (diagnostics, tracing) get the
        fresh totals without per-step readbacks during the production step loop.
        """
        self._flush_pending_mass_accounting()
        values = self.cumulative_totals.numpy()
        return {
            "rain": float(values[0]),
            "infiltration": float(values[1]),
            "other_sources": float(values[2]),
            "boundary_inflow": float(values[3]),
            "boundary_outflow": float(values[4]),
            "additional_sources": float(values[5]),
        }

    def numerical_configuration(self) -> dict:
        """Resolved numerical identity, independent of an omitted input option."""
        return dict(validation_profile=self.validation_profile,
            hydraulic_geometry_mode=self.hydraulic_geometry_mode,
            dry_depth=float(self.dry_depth), source_mode=self.source_mode,
            manning_face_mode=self.manning_face_mode,
            use_two_dimensional_friction=self.use_two_dimensional_friction,
            apply_sources_after_momentum=bool(self.apply_sources_after_momentum),
            free_outflow_closure=('local_or_prescribed_slope' if self.free_outflow_mode == 1 else 'dry_exterior_head'),
            free_outflow_mode=self.free_outflow_mode,
            free_outflow_limiter_mode=self.free_outflow_limiter_mode)

    def capture_state(self, *, device_buffers: dict | None = None,
                      excluded_device_fields=()) -> dict:
        """Snapshot every mutable hydraulic, forcing, constraint and
        accounting field needed for an exact transactional rollback.

        The snapshot is deep (device arrays are cloned) and restore_state()
        returns the solver to bit-identical state, including the momentum
        ping-pong buffers, Green-Ampt cumulative infiltration, infiltration
        constraints, FP64 mass-accounting accumulators (with pending
        deferred-accounting fields when enabled), opt-in diagnostic
        accumulators, and the adaptive-timestep host history. Intended for
        restart drivers that retry failed substeps at smaller
        intervals; the spatially-adaptive (LTS) path keeps extra buffers
        that are not covered here (checkpoint stepping uses global stepping).

        External orchestrators may supply exclusively owned reusable device
        buffers. Exclusions are caller-declared immutable fields; the default
        remains the complete deep snapshot, including geometry/materials.
        """
        if self._lts_cell_levels is not None:
            raise RuntimeError("capture_state supports global stepping only; LTS state is not covered")
        state: dict = {
            "host": {
                "numerical_configuration": self.numerical_configuration(),
                "simulation_time": self.simulation_time,
                "current_dt": self.current_dt,
                "_adaptive_dt_reference": self._adaptive_dt_reference,
                "last_timestep_max_depth_m": self.last_timestep_max_depth_m,
                "_lisflood_cached_maximum_depth_m": self._lisflood_cached_maximum_depth_m,
                "infiltration_cap_active": self.infiltration_cap_active,
                "_infiltration_exclusion_host": (
                    None if self._infiltration_exclusion_host is None
                    else np.array(self._infiltration_exclusion_host, copy=True)
                ),
                "_spatial_rainfall_enabled": self._spatial_rainfall_enabled,
                "_steps_since_mass_accounting": self._steps_since_mass_accounting,
                "_conditioned_bed_array": self._conditioned_bed_array,
            },
            "device": {},
        }
        names = [
            # hydraulic state, including the momentum ping-pong pair
            "water_depth", "qx", "qy", "qx_next", "qy_next",
            # forcing fields (restored so a rollback also undoes any
            # forcing change made inside the failed transaction)
            "rainfall_rate", "rainfall_spatial_multiplier", "other_source_rate",
            "additional_source_rate",
            # Green-Ampt state and external infiltration constraints
            "cumulative_infiltration", "infiltration_cap_remaining_m", "infiltration_exclusion_mask",
            # FP64 mass accounting
            "cumulative_totals", "applied_rain", "applied_other", "applied_infil",
        ]
        if self.track_max_water_depth:
            names.append("max_water_depth_m")
        if self.track_cumulative_face_volumes:
            names.extend([
                "cumulative_net_qx_volume_m3", "cumulative_net_qy_volume_m3",
                "_diagnostic_peak_discharge_m2_per_s", "_diagnostic_peak_qx_rows",
                "_diagnostic_peak_qy_rows", "_diagnostic_scalar_state",
            ])
            if self.track_absolute_face_volumes:
                names.extend(["cumulative_abs_qx_volume_m3",
                              "cumulative_abs_qy_volume_m3"])
            if self.track_diagnostic_max_depth:
                names.append("_diagnostic_max_depth_m")
        if self._deferred_mass_accounting:
            names.extend(["_pending_rain", "_pending_infil", "_pending_other"])
        # Cover global-step scratch, CFL/boundary conditioning, structure
        # diagnostics and optional accumulators as well as primary fields.
        # Cloning static arrays too keeps the reference rollback contract
        # complete if a forcing/geometry field is mutated in a transaction.
        names.extend(name for name, value in vars(self).items()
                     if isinstance(value, wp.array) and not name.startswith("_lts_")
                     and name not in names)
        for name in names:
            if name in excluded_device_fields:
                continue
            field = getattr(self, name, None)
            if field is None:
                state["device"][name] = None
                continue
            if device_buffers is None:
                snapshot = wp.clone(field)
            else:
                snapshot = device_buffers.get(name)
                if (snapshot is None or snapshot.shape != field.shape
                        or snapshot.dtype != field.dtype or snapshot.device != field.device):
                    snapshot = wp.empty_like(field)
                    device_buffers[name] = snapshot
                wp.copy(snapshot, field)
            state["device"][name] = snapshot
        return state

    def restore_state(self, state: dict) -> None:
        """Restore an exact snapshot taken by :meth:`capture_state`."""
        host = state["host"]
        if host.get('numerical_configuration') != self.numerical_configuration():
            raise ValueError('Missing or mismatched resolved numerical provenance; ambiguous historical resumes are not supported')
        self.simulation_time = host["simulation_time"]
        self.current_dt = host["current_dt"]
        self._adaptive_dt_reference = host["_adaptive_dt_reference"]
        self.last_timestep_max_depth_m = host["last_timestep_max_depth_m"]
        self._lisflood_cached_maximum_depth_m = host["_lisflood_cached_maximum_depth_m"]
        self.infiltration_cap_active = host["infiltration_cap_active"]
        self._infiltration_exclusion_host = (
            None if host["_infiltration_exclusion_host"] is None
            else np.array(host["_infiltration_exclusion_host"], copy=True)
        )
        self._spatial_rainfall_enabled = host["_spatial_rainfall_enabled"]
        self._steps_since_mass_accounting = host["_steps_since_mass_accounting"]
        self._conditioned_bed_array = host["_conditioned_bed_array"]
        for name, snapshot in state["device"].items():
            if snapshot is None:
                continue
            wp.copy(getattr(self, name), snapshot)

    def restore_infiltration_constraints(self, *, state: dict) -> None:
        """Restore only temporary infiltration constraints after routing."""
        host = state["host"]
        self.infiltration_cap_active = host["infiltration_cap_active"]
        values = host["_infiltration_exclusion_host"]
        self._infiltration_exclusion_host = None if values is None else values.copy()
        for name in ("infiltration_exclusion_mask", "infiltration_cap_remaining_m"):
            wp.copy(getattr(self, name), state["device"][name])

    def set_uniform_rainfall_rate(self, rainfall_rate: float) -> None:
        """Update a spatially uniform rate without allocating a host field."""
        if not np.isfinite(rainfall_rate):
            raise ValueError("rainfall_rate must be finite")
        if self._spatial_rainfall_enabled:
            wp.launch(
                kernel=set_scaled_rainfall_kernel,
                dim=(self.ny, self.nx),
                inputs=[self.rainfall_rate, self.rainfall_spatial_multiplier,
                        WP_FLOAT(rainfall_rate)],
                device=self.device,
            )
            return
        wp.launch(
            kernel=set_uniform_field_kernel,
            dim=(self.ny, self.nx),
            inputs=[self.rainfall_rate, WP_FLOAT(rainfall_rate)],
            device=self.device,
        )

    def set_rainfall_spatial_multiplier(self, multiplier: np.ndarray) -> None:
        """Set the persistent dimensionless reference-rainfall multiplier."""
        values = np.asarray(multiplier, dtype=np.float64)
        if values.shape != (self.ny, self.nx):
            raise ValueError("rainfall_spatial_multiplier has incorrect grid shape")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("rainfall_spatial_multiplier must be finite and non-negative")
        active = self.active_mask.numpy().astype(bool)
        if np.any(values[~active] != 0.0):
            raise ValueError("rainfall_spatial_multiplier must be zero outside active cells")
        wp.copy(self.rainfall_spatial_multiplier, wp.array(values, dtype=WP_FLOAT, device=self.device))
        self._spatial_rainfall_enabled = True

    def set_boundary_type_qx(self, boundary_type: np.ndarray):
        self._validate_boundary_type_qx(boundary_type=boundary_type)
        wp.copy(self.boundary_type_qx, wp.array(boundary_type, dtype=wp.int32, device=self.device))
        if USE_CONDITIONED_MIXED_KERNELS:
            self._ensure_conditioned_bed_fields()
            self._set_conditioned_stage_qx(stage=self.boundary_stage_qx.numpy())

    def set_boundary_stage_qx(self, stage: np.ndarray):
        self._validate_boundary_stage(name="boundary_stage_qx", stage=stage, expected_shape=(self.ny, self.nx + 1))
        wp.copy(self.boundary_stage_qx, wp.array(stage, dtype=STAGE_FLOAT, device=self.device))
        if USE_CONDITIONED_MIXED_KERNELS:
            self._ensure_conditioned_bed_fields()
            self._set_conditioned_stage_qx(stage=stage)

    def set_free_outflow_slope_qx(self, slopes: np.ndarray) -> None:
        """Set FREE slope values, using NaN only as an explicit zero marker.

        The normal stage setter intentionally rejects NaNs.  For a FREE face,
        this diagnostic representation distinguishes LISFLOOD's explicit
        ``FREE 0`` from the historic Overlander zero value meaning "derive the
        local water-surface slope".  NaN is never valid on another boundary.
        """
        if slopes.shape != (self.ny, self.nx + 1):
            raise ValueError("free outflow slopes must match boundary_type_qx")
        boundary_type = self.boundary_type_qx.numpy()
        invalid_nan = np.isnan(slopes) & (boundary_type != BOUNDARY_FREE_OUTFLOW)
        if np.any(invalid_nan) or np.any(np.isinf(slopes)):
            raise ValueError("NaN FREE slopes are permitted only on FREE_OUTFLOW faces")
        wp.copy(
            self.boundary_stage_qx,
            wp.array(slopes, dtype=STAGE_FLOAT, device=self.device),
        )
        if USE_CONDITIONED_MIXED_KERNELS:
            self._ensure_conditioned_bed_fields()
            self._set_conditioned_stage_qx(stage=slopes)

    def set_boundary_type_qy(self, boundary_type: np.ndarray):
        self._validate_boundary_type_qy(boundary_type=boundary_type)
        wp.copy(self.boundary_type_qy, wp.array(boundary_type, dtype=wp.int32, device=self.device))
        if USE_CONDITIONED_MIXED_KERNELS:
            self._ensure_conditioned_bed_fields()
            self._set_conditioned_stage_qy(stage=self.boundary_stage_qy.numpy())

    def set_boundary_stage_qy(self, stage: np.ndarray):
        self._validate_boundary_stage(name="boundary_stage_qy", stage=stage, expected_shape=(self.ny + 1, self.nx))
        wp.copy(self.boundary_stage_qy, wp.array(stage, dtype=STAGE_FLOAT, device=self.device))
        if USE_CONDITIONED_MIXED_KERNELS:
            self._ensure_conditioned_bed_fields()
            self._set_conditioned_stage_qy(stage=stage)

    def set_free_outflow_slope_qy(self, slopes: np.ndarray) -> None:
        """Set FREE y-face slopes; NaN represents an explicit FREE zero slope."""
        if slopes.shape != (self.ny + 1, self.nx):
            raise ValueError("free outflow slopes must match boundary_type_qy")
        boundary_type = self.boundary_type_qy.numpy()
        invalid_nan = np.isnan(slopes) & (boundary_type != BOUNDARY_FREE_OUTFLOW)
        if np.any(invalid_nan) or np.any(np.isinf(slopes)):
            raise ValueError("NaN FREE slopes are permitted only on FREE_OUTFLOW faces")
        wp.copy(
            self.boundary_stage_qy,
            wp.array(slopes, dtype=STAGE_FLOAT, device=self.device),
        )
        if USE_CONDITIONED_MIXED_KERNELS:
            self._ensure_conditioned_bed_fields()
            self._set_conditioned_stage_qy(stage=slopes)

    def _validate_field(self, name: str, values: np.ndarray):
        if values.shape != (self.ny, self.nx):
            raise ValueError(f"{name} shape must match (ny, nx)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} values must be finite")

    def _validate_boundary_stage(self, name: str, stage: np.ndarray, expected_shape: Tuple[int, int]):
        if stage.shape != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}")
        if not np.all(np.isfinite(stage)):
            raise ValueError(f"{name} values must be finite")

    def _validate_boundary_values(self, boundary_type: np.ndarray, expected_shape: Tuple[int, int], name: str):
        if boundary_type.shape != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}")
        if not np.issubdtype(boundary_type.dtype, np.integer):
            raise ValueError(f"{name} values must be integer boundary codes")

        valid_types = [
            BOUNDARY_INTERNAL, BOUNDARY_CLOSED, BOUNDARY_FREE_OUTFLOW,
            BOUNDARY_SPECIFIED_STAGE, BOUNDARY_MASKED_NORTH_STAGE,
            BOUNDARY_MASKED_WEST_STAGE, BOUNDARY_MASKED_WEST_FREE,
            BOUNDARY_MASKED_EAST_FREE, BOUNDARY_MASKED_SOUTH_FREE,
            BOUNDARY_MASKED_NORTH_FREE,
            BOUNDARY_MASKED_EAST_STAGE, BOUNDARY_MASKED_SOUTH_STAGE,
        ]
        if not np.all(np.isin(boundary_type, valid_types)):
            raise ValueError(f"{name} contains an unknown boundary code")

    def _validate_boundary_type_qx(self, boundary_type: np.ndarray):
        self._validate_boundary_values(
            boundary_type=boundary_type,
            expected_shape=(self.ny, self.nx + 1),
            name="boundary_type_qx",
        )

        if np.any(np.isin(boundary_type, (BOUNDARY_MASKED_NORTH_STAGE, BOUNDARY_MASKED_SOUTH_STAGE))):
            raise ValueError("BOUNDARY_MASKED_NORTH_STAGE is only valid on qy faces")
        if np.any(np.isin(boundary_type, [BOUNDARY_MASKED_SOUTH_FREE, BOUNDARY_MASKED_NORTH_FREE])):
            raise ValueError("BOUNDARY_MASKED_SOUTH_FREE/BOUNDARY_MASKED_NORTH_FREE are only valid on qy faces")
        if np.any(np.isin(boundary_type[:, (0, self.nx)], (BOUNDARY_MASKED_WEST_STAGE, BOUNDARY_MASKED_EAST_STAGE))):
            raise ValueError("BOUNDARY_MASKED_WEST_STAGE is only valid on interior qx faces")
        if np.any(np.isin(boundary_type[:, (0, self.nx)], [BOUNDARY_MASKED_WEST_FREE, BOUNDARY_MASKED_EAST_FREE])):
            raise ValueError("Masked FREE boundary types are only valid on interior qx faces")
        if np.any(boundary_type[:, 0] == BOUNDARY_INTERNAL) or np.any(boundary_type[:, self.nx] == BOUNDARY_INTERNAL):
            raise ValueError("BOUNDARY_INTERNAL is only valid on interior qx faces")
        if np.any(np.isin(boundary_type[:, 1:self.nx], [BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE])):
            raise ValueError("Open qx boundary types are only valid on exterior faces")
        if np.any(boundary_type[:, 1:self.nx][self._active_mask_host[:, :-1] * self._active_mask_host[:, 1:] == 0] == BOUNDARY_INTERNAL):
            raise ValueError("BOUNDARY_INTERNAL qx faces must connect two active cells")
        masked_west = (
            boundary_type[:, 1:self.nx] == BOUNDARY_MASKED_WEST_STAGE
        )
        valid_masked_west = (
            (self._active_mask_host[:, :-1] == 0)
            & (self._active_mask_host[:, 1:] == 1)
        )
        if np.any(masked_west & ~valid_masked_west):
            raise ValueError(
                "BOUNDARY_MASKED_WEST_STAGE must have an inactive i-1 cell "
                "and an active i cell"
            )
        masked_west_free = (
            boundary_type[:, 1:self.nx] == BOUNDARY_MASKED_WEST_FREE
        )
        if np.any(masked_west_free & ~valid_masked_west):
            raise ValueError(
                "BOUNDARY_MASKED_WEST_FREE must have an inactive i-1 cell "
                "and an active i cell"
            )
        masked_east_free = (
            boundary_type[:, 1:self.nx] == BOUNDARY_MASKED_EAST_FREE
        )
        valid_masked_east_free = (
            (self._active_mask_host[:, :-1] == 1)
            & (self._active_mask_host[:, 1:] == 0)
        )
        masked_east_stage = boundary_type[:, 1:self.nx] == BOUNDARY_MASKED_EAST_STAGE
        if np.any(masked_east_stage & ~valid_masked_east_free):
            raise ValueError("BOUNDARY_MASKED_EAST_STAGE requires active west and inactive east cells")
        if np.any(masked_east_free & ~valid_masked_east_free):
            raise ValueError(
                "BOUNDARY_MASKED_EAST_FREE must have an active i-1 cell "
                "and an inactive i cell"
            )
        if np.any(np.isin(boundary_type[:, 0], [BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE]) & (self._active_mask_host[:, 0] == 0)):
            raise ValueError("Open west qx boundaries must adjoin active cells")
        if np.any(np.isin(boundary_type[:, self.nx], [BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE]) & (self._active_mask_host[:, -1] == 0)):
            raise ValueError("Open east qx boundaries must adjoin active cells")

    def _validate_boundary_type_qy(self, boundary_type: np.ndarray):
        self._validate_boundary_values(
            boundary_type=boundary_type,
            expected_shape=(self.ny + 1, self.nx),
            name="boundary_type_qy",
        )

        if np.any(np.isin(boundary_type, (BOUNDARY_MASKED_WEST_STAGE, BOUNDARY_MASKED_EAST_STAGE))):
            raise ValueError("BOUNDARY_MASKED_WEST_STAGE is only valid on qx faces")
        if np.any(np.isin(boundary_type, [BOUNDARY_MASKED_WEST_FREE, BOUNDARY_MASKED_EAST_FREE])):
            raise ValueError("BOUNDARY_MASKED_WEST_FREE/BOUNDARY_MASKED_EAST_FREE are only valid on qx faces")
        if np.any(boundary_type[0, :] == BOUNDARY_INTERNAL) or np.any(boundary_type[self.ny, :] == BOUNDARY_INTERNAL):
            raise ValueError("BOUNDARY_INTERNAL is only valid on interior qy faces")
        if np.any(np.isin(boundary_type[(0, self.ny), :], (BOUNDARY_MASKED_NORTH_STAGE, BOUNDARY_MASKED_SOUTH_STAGE))):
            raise ValueError("BOUNDARY_MASKED_NORTH_STAGE is only valid on interior qy faces")
        if np.any(np.isin(boundary_type[(0, self.ny), :], [BOUNDARY_MASKED_SOUTH_FREE, BOUNDARY_MASKED_NORTH_FREE])):
            raise ValueError("Masked FREE boundary types are only valid on interior qy faces")
        if np.any(np.isin(boundary_type[1:self.ny, :], [BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE])):
            raise ValueError("Ordinary open qy boundary types are only valid on exterior faces")
        masked_north = (
            boundary_type[1:self.ny, :] == BOUNDARY_MASKED_NORTH_STAGE
        )
        valid_masked_north = (
            (self._active_mask_host[:-1, :] == 1)
            & (self._active_mask_host[1:, :] == 0)
        )
        if np.any(masked_north & ~valid_masked_north):
            raise ValueError(
                "BOUNDARY_MASKED_NORTH_STAGE must have an active j-1 cell "
                "and an inactive j cell"
            )
        masked_north_free = (
            boundary_type[1:self.ny, :] == BOUNDARY_MASKED_NORTH_FREE
        )
        if np.any(masked_north_free & ~valid_masked_north):
            raise ValueError(
                "BOUNDARY_MASKED_NORTH_FREE must have an active j-1 cell "
                "and an inactive j cell"
            )
        masked_south_free = (
            boundary_type[1:self.ny, :] == BOUNDARY_MASKED_SOUTH_FREE
        )
        valid_masked_south_free = (
            (self._active_mask_host[:-1, :] == 0)
            & (self._active_mask_host[1:, :] == 1)
        )
        masked_south_stage = boundary_type[1:self.ny, :] == BOUNDARY_MASKED_SOUTH_STAGE
        if np.any(masked_south_stage & ~valid_masked_south_free):
            raise ValueError("BOUNDARY_MASKED_SOUTH_STAGE requires inactive south and active north cells")
        if np.any(masked_south_free & ~valid_masked_south_free):
            raise ValueError(
                "BOUNDARY_MASKED_SOUTH_FREE must have an inactive j-1 cell "
                "and an active j cell"
            )
        if np.any(boundary_type[1:self.ny, :][self._active_mask_host[:-1, :] * self._active_mask_host[1:, :] == 0] == BOUNDARY_INTERNAL):
            raise ValueError("BOUNDARY_INTERNAL qy faces must connect two active cells")
        if np.any(np.isin(boundary_type[0, :], [BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE]) & (self._active_mask_host[0, :] == 0)):
            raise ValueError("Open north qy boundaries must adjoin active cells")
        if np.any(np.isin(boundary_type[self.ny, :], [BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE]) & (self._active_mask_host[-1, :] == 0)):
            raise ValueError("Open south qy boundaries must adjoin active cells")

    def _accumulate_face_volume_diagnostics(self, qx, qy, dt_wp: WP_FLOAT) -> None:
        """Accumulate accepted global-step face volumes when explicitly enabled."""
        if not self.track_cumulative_face_volumes:
            return
        wp.launch(
            kernel=accumulate_face_volume_qx_kernel,
            dim=(self.ny, self.nx + 1),
            inputs=[
                qx,
                self.cumulative_net_qx_volume_m3,
                self.cumulative_abs_qx_volume_m3 if self.track_absolute_face_volumes else self._diagnostic_absolute_placeholder,
                self._diagnostic_peak_qx_rows,
                self.dy,
                dt_wp, 1 if self.track_absolute_face_volumes else 0,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=accumulate_face_volume_qy_kernel,
            dim=(self.ny + 1, self.nx),
            inputs=[
                qy,
                self.cumulative_net_qy_volume_m3,
                self.cumulative_abs_qy_volume_m3 if self.track_absolute_face_volumes else self._diagnostic_absolute_placeholder,
                self._diagnostic_peak_qy_rows,
                self.dx,
                dt_wp, 1 if self.track_absolute_face_volumes else 0,
            ],
            device=self.device,
        )

    def _accumulate_depth_diagnostic(self) -> None:
        """Track the post-continuity maximum depth for the opt-in diagnostic."""
        if not self.track_cumulative_face_volumes or not self.track_diagnostic_max_depth:
            return
        wp.launch(
            kernel=accumulate_max_active_depth_kernel,
            dim=(self.ny, self.nx),
            inputs=[self.water_depth, self.active_mask, self._diagnostic_max_depth_m],
            device=self.device,
        )

    def _accumulate_max_water_depth(self) -> None:
        """Track the standalone per-cell post-continuity maximum depth."""
        if not self.track_max_water_depth:
            return
        wp.launch(
            kernel=accumulate_max_active_depth_kernel,
            dim=(self.ny, self.nx),
            inputs=[self.water_depth, self.active_mask, self.max_water_depth_m],
            device=self.device,
        )

    def reset_cumulative_face_volumes(self) -> None:
        """Reset opt-in cumulative face-transport diagnostic fields."""
        if not self.track_cumulative_face_volumes:
            raise RuntimeError(
                "Face-volume diagnostics were not enabled; construct with "
                "track_cumulative_face_volumes=True"
            )
        self.cumulative_net_qx_volume_m3.zero_()
        self.cumulative_net_qy_volume_m3.zero_()
        if self.track_absolute_face_volumes:
            self.cumulative_abs_qx_volume_m3.zero_()
            self.cumulative_abs_qy_volume_m3.zero_()
        self._diagnostic_peak_discharge_m2_per_s.zero_()
        self._diagnostic_peak_qx_rows.zero_()
        self._diagnostic_peak_qy_rows.zero_()
        self._diagnostic_scalar_state.zero_()
        if self.track_diagnostic_max_depth:
            self._diagnostic_max_depth_m.zero_()

    def get_cumulative_face_volumes(self) -> dict[str, np.ndarray | float]:
        """Return opt-in cumulative net/absolute D4 transport after a run."""
        if not self.track_cumulative_face_volumes:
            raise RuntimeError(
                "Face-volume diagnostics were not enabled; construct with "
                "track_cumulative_face_volumes=True"
            )
        scalars = self.get_hydraulic_diagnostic_scalars()
        return {
            "cumulative_net_qx_volume_m3": self.cumulative_net_qx_volume_m3.numpy(),
            "cumulative_net_qy_volume_m3": self.cumulative_net_qy_volume_m3.numpy(),
            "cumulative_abs_qx_volume_m3": self.cumulative_abs_qx_volume_m3.numpy() if self.track_absolute_face_volumes else None,
            "cumulative_abs_qy_volume_m3": self.cumulative_abs_qy_volume_m3.numpy() if self.track_absolute_face_volumes else None,
            "peak_discharge_m2_per_s": scalars["peak_discharge_m2_per_s"],
            "diagnostic_max_depth_m": self._diagnostic_max_depth_m.numpy() if self.track_diagnostic_max_depth else None,
        }

    def get_max_water_depth(self) -> np.ndarray:
        """Read back the standalone per-cell maximum water depth field.

        The field holds ``max_t h(t)`` over the accepted post-continuity
        depths of the current run (zero where no step has run yet); one
        device-to-host transfer.  Construct with ``track_max_water_depth=True``.
        """
        if not self.track_max_water_depth:
            raise RuntimeError(
                "Maximum-depth tracking was not enabled; construct with "
                "track_max_water_depth=True"
            )
        return self.max_water_depth_m.numpy()

    def get_hydraulic_diagnostic_scalars(self) -> dict[str, float]:
        """Read current activity, active storage, and running peak as scalars."""
        if not self.track_cumulative_face_volumes:
            raise RuntimeError("Hydraulic diagnostics require track_cumulative_face_volumes=True")
        self._diagnostic_scalar_state.zero_()
        wp.launch(
            kernel=reduce_hydraulic_diagnostic_state_kernel,
            dim=(self.ny + 1, self.nx + 1),
            inputs=[
                self.water_depth, self.qx, self.qy, self.active_mask,
                wp.float64(float(self.dx) * float(self.dy)),
                self._diagnostic_peak_qx_rows,
                self._diagnostic_peak_qy_rows,
                self._diagnostic_scalar_state,
            ],
            device=self.device,
        )
        values = self._diagnostic_scalar_state.numpy()
        return {
            "activity_m2_per_s": float(values[0]),
            "surface_volume_m3": float(values[1]),
            "peak_discharge_m2_per_s": float(values[2]),
        }

    def get_diagnostic_peak_discharge(self) -> float:
        """Read only the cumulative accepted-flow peak-discharge scalar."""
        if not self.track_cumulative_face_volumes:
            raise RuntimeError("Hydraulic diagnostics require track_cumulative_face_volumes=True")
        return self.get_hydraulic_diagnostic_scalars()["peak_discharge_m2_per_s"]
    def get_water_depth(self) -> np.ndarray:
        return self.water_depth.numpy()

    def get_qx(self) -> np.ndarray:
        return self.qx.numpy()

    def get_qy(self) -> np.ndarray:
        return self.qy.numpy()

    # ---- subgrid output semantics ------------------------------------------
    # ``water_depth`` is deliberately NOT reinterpreted: it stays the
    # conservative storage-equivalent depth V/(dx*dy) in both modes, so mass
    # accounting is unchanged.  Physical depth diagnostics are explicit
    # accessors; in subgrid mode storage depth and local hydraulic depth can
    # differ by an order of magnitude in narrow channels.

    def is_subgrid(self) -> bool:
        """True when the experimental subgrid hydraulic geometry is active."""
        return self._subgrid_geometry is not None

    def _launch_stage_inversion(
        self, storage_array, eta_out,
    ) -> None:
        """Derive the water-surface elevation field from stored volumes."""
        wp.launch(
            kernel=subgrid_stage_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                storage_array, self._subgrid_cell_z,
                self._subgrid_cell_breakpoints, self._subgrid_cell_count,
                eta_out,
            ],
            device=self.device,
        )

    def set_water_surface_elevation(self, stage: np.ndarray) -> None:
        """Initialise storage from a water-surface elevation field.

        In subgrid mode the stage is converted through the forward cell
        hypsometry (exact inverse of :meth:`get_water_surface_elevation`); in
        coarse mode this is ``stage - bed_elevation``.  Fluxes, accounting,
        and clocks are reset exactly as in :meth:`set_water_depth`.
        """
        stage = np.asarray(stage, dtype=np.float64)
        if stage.shape != (self.ny, self.nx):
            raise ValueError("Stage shape must match (ny, nx)")
        if not np.all(np.isfinite(stage)):
            raise ValueError("Water surface elevation must be finite")
        if self.is_subgrid():
            storage = storage_from_stage(self._subgrid_geometry, stage)
        else:
            bed = np.asarray(self.bed_elevation.numpy(), dtype=np.float64)
            storage = stage - bed
        self.set_water_depth(np.maximum(storage, 0.0))

    def get_water_surface_elevation(self) -> np.ndarray:
        """Return the derived water-surface elevation field (one sync).

        Coarse mode returns ``bed_elevation + water_depth``; subgrid mode
        inverts the per-cell storage curve.
        """
        if not self.is_subgrid():
            bed = np.asarray(self.bed_elevation.numpy(), dtype=np.float64)
            return bed + np.asarray(self.water_depth.numpy(), dtype=np.float64)
        self._launch_stage_inversion(self.water_depth, self._eta_stage)
        wp.synchronize_device(self.device)
        return self._eta_stage.numpy()

    def get_subgrid_state(self) -> dict[str, np.ndarray]:
        """Derived subgrid physical state: stage, local depth, wet fraction.

        ``storage_depth`` is the raw conservative solver state
        (``water_depth``); ``maximum_local_depth``/``mean_wet_depth`` are the
        physical hydraulic depths inside each coarse cell.
        """
        if not self.is_subgrid():
            raise ValueError(
                "get_subgrid_state requires hydraulic_geometry_mode='subgrid'"
            )
        eta = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=self.device)
        max_local = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=self.device)
        wet_fraction = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=self.device)
        mean_wet = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=self.device)
        wp.launch(
            kernel=subgrid_derived_state_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self.water_depth, self._subgrid_cell_z,
                self._subgrid_cell_breakpoints, self._subgrid_cell_count,
                eta, max_local, wet_fraction, mean_wet,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return {
            "storage_depth": self.water_depth.numpy(),
            "water_surface_elevation": eta.numpy(),
            "maximum_local_depth": max_local.numpy(),
            "wet_fraction": wet_fraction.numpy(),
            "mean_wet_depth": mean_wet.numpy(),
        }

    def subgrid_memory_report(self) -> dict[str, int | str]:
        """Approximate extra memory held by the subgrid geometry."""
        if not self.is_subgrid():
            return {"enabled": False}
        sizes = self._subgrid_geometry.memory_bytes()
        stage_bytes = 2 * self.ny * self.nx * int(
            wp.types.type_size_in_bytes(WP_FLOAT)
        )
        return {
            "enabled": True,
            "factor": self._subgrid_factor,
            "device_bytes": sizes["total"] + stage_bytes,
            **sizes,
        }

    def calculate_cfl_dt(self, water_depth=None) -> float:
        self._ensure_conditioned_bed_fields()
        if water_depth is None:
            water_depth = self.water_depth
        self.dt_min_arr.fill_(WP_FLOAT(self.maximum_dt))

        if self.is_subgrid():
            # Wave speeds come from the maximum local face depth of the
            # derived stage field, never the storage-equivalent depth.
            self._launch_stage_inversion(water_depth, self._eta_cfl)
            wp.launch(
                kernel=subgrid_calc_cfl_dt_x_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    self._eta_cfl, self._subgrid_face_z_min_x,
                    self.boundary_type_qx, self.boundary_stage_qx,
                    self.dx, self.gravity, self.dry_depth, self.cfl_number,
                    self.dt_min_arr,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=subgrid_calc_cfl_dt_y_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    self._eta_cfl, self._subgrid_face_z_min_y,
                    self.boundary_type_qy, self.boundary_stage_qy,
                    self.dy, self.gravity, self.dry_depth, self.cfl_number,
                    self.dt_min_arr,
                ],
                device=self.device,
            )
        elif USE_CONDITIONED_MIXED_KERNELS:
            wp.launch(
                kernel=calc_conditioned_cfl_dt_x_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    water_depth, self.bed_slope_qx, self.boundary_type_qx,
                    self.boundary_stage_depth_qx, self.dx, self.gravity,
                    self.dry_depth, self.cfl_number, self.dt_min_arr,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=calc_conditioned_cfl_dt_y_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    water_depth, self.bed_slope_qy, self.boundary_type_qy,
                    self.boundary_stage_depth_qy, self.dy, self.gravity,
                    self.dry_depth, self.cfl_number, self.dt_min_arr,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=calc_cfl_dt_x_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    water_depth, self.bed_elevation, self.boundary_type_qx, self.boundary_stage_qx,
                    self.dx, self.gravity, self.dry_depth, self.cfl_number, self.dt_min_arr
                ],
                device=self.device
            )
            wp.launch(
                kernel=calc_cfl_dt_y_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    water_depth, self.bed_elevation, self.boundary_type_qy, self.boundary_stage_qy,
                    self.dy, self.gravity, self.dry_depth, self.cfl_number, self.dt_min_arr
                ],
                device=self.device
            )

        wp.launch(
            kernel=store_cfl_timestep_control_kernel,
            dim=(1,),
            inputs=[self.dt_min_arr, self.lisflood_step_control],
            device=self.device,
        )
        step_control = self.lisflood_step_control.numpy()
        if step_control[1] != 0.0:
            raise RuntimeError("Significant negative depth encountered.")
        dt_val = float(step_control[0])
        return min(self.maximum_dt, dt_val)

    def _has_prescribed_stage_boundaries(self) -> bool:
        return bool(np.isin(self.boundary_type_qx.numpy(),(3,5,10)).any()
            or np.isin(self.boundary_type_qy.numpy(),(3,4,11)).any())

    def calculate_stable_timestep(self, remaining_time: float = None) -> float:
        """Return the source-aware adaptive global timestep without advancing state.

        Timestep contraction is immediate because the hydraulic CFL/ACC candidate is
        recomputed every step. For the normal Overlander profile, timestep growth is
        limited relative to the previous unconstrained adaptive step. This prevents a
        rapidly receding wet front from jumping directly from a short hydraulic step
        to ``maximum_dt``. The LISFLOOD-FP ACC validation profile bypasses the growth
        controller so its external-comparison timestep convention remains unchanged.
        """
        max_dt = self.maximum_dt
        if remaining_time is not None:
            max_dt = min(max_dt, remaining_time)
        if self.adaptive_dt_growth_factor is not None and self._adaptive_dt_reference > 0.0:
            max_dt = min(
                max_dt,
                self._adaptive_dt_reference * self.adaptive_dt_growth_factor,
            )
        if self.use_lisflood_acc_timestep:
            if self._lisflood_cached_maximum_depth_m is None:
                self.maximum_depth_arr.zero_()
                wp.launch(
                    kernel=max_active_depth_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.active_mask, self.maximum_depth_arr],
                    device=self.device,
                )
                maximum_depth_m = float(self.maximum_depth_arr.numpy()[0])
            else:
                maximum_depth_m = self._lisflood_cached_maximum_depth_m
            self.last_timestep_max_depth_m = maximum_depth_m
            lisflood_dt=max_dt
            if maximum_depth_m > float(self.dry_depth):
                lisflood_dt = float(self.cfl_number) * float(self.dx) / np.sqrt(
                    float(self.gravity) * maximum_depth_m
                )
            if self._has_prescribed_stage_boundaries():
                wp.launch(kernel=predict_depth_for_cfl_kernel,dim=(self.ny,self.nx),
                    inputs=[self.water_depth,self.rainfall_rate,self.other_source_rate,
                        self.additional_source_rate,self.active_mask,WP_FLOAT(max_dt),self.cfl_depth],device=self.device)
                lisflood_dt=min(lisflood_dt,self.calculate_cfl_dt(water_depth=self.cfl_depth))
            return min(max_dt, lisflood_dt)
        wp.launch(
            kernel=predict_depth_for_cfl_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self.water_depth, self.rainfall_rate, self.other_source_rate,
                self.additional_source_rate,
                self.active_mask, WP_FLOAT(max_dt), self.cfl_depth,
            ],
            device=self.device
        )
        self.last_timestep_max_depth_m = None
        return min(max_dt, self.calculate_cfl_dt(water_depth=self.cfl_depth))

    def _apply_sources(
        self,
        dt_wp: WP_FLOAT,
        defer: bool,
        flush: bool,
        committed_qx=None,
        committed_qy=None,
    ) -> None:
        """Apply rain/other/infiltration sources for one step.

        With ``defer`` the per-cell applied depths accumulate into the pending
        fields (no per-step reduction); with ``flush`` the pending (or per-step)
        fields are tree-summed here and folded into cumulative_totals by the
        continuity kernel later in the step.

        ``committed_qx``/``committed_qy`` are the post-limiter committed face
        fluxes, passed only by profiles that apply sources after momentum
        (where the limiter has already budgeted outfluxes against pre-source
        depth): the infiltration sink is then additionally capped at the
        depth remaining after those committed outfluxes so a fully draining
        cell cannot overshoot into negative depth.
        """
        # On large grids skip the per-cell float64 mass atomics in the kernel and
        # reduce the applied_* arrays once instead (tree-sum accounting path).
        accumulate_accounting = 0 if self.use_reduced_mass_accounting else 1
        if defer:
            out_rain, out_other, out_infil = self._pending_rain, self._pending_other, self._pending_infil
        else:
            out_rain, out_other, out_infil = self.applied_rain, self.applied_other, self.applied_infil
        if committed_qx is not None:
            wp.launch(
                kernel=apply_sources_committed_flux_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self.water_depth,
                    self.rainfall_rate,
                    self.other_source_rate,
                    self.infiltration_capacity,
                    self.infiltration_f0,
                    self.infiltration_fd,
                    self.infiltration_k,
                    self.green_ampt_ks,
                    self.green_ampt_psi,
                    self.green_ampt_delta_theta,
                    self.cumulative_infiltration,
                    self.infiltration_method,
                    self._source_subgrid_infiltration_enabled,
                    self._source_subgrid_cell_z,
                    self._source_subgrid_cell_breakpoints,
                    self._source_subgrid_cell_count,
                    self.active_mask,
                    dt_wp,
                    self.source_mode,
                    self.dry_depth,
                    committed_qx,
                    committed_qy,
                    out_rain,
                    out_other,
                    out_infil,
                    self.dx,
                    self.dy,
                    self.cumulative_totals,
                    accumulate_accounting,
                    1 if defer else 0,
                    self.additional_source_rate,
                    self.infiltration_exclusion_mask,
                    self.infiltration_cap_active,
                    self.infiltration_cap_remaining_m,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=apply_sources_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self.water_depth,
                    self.rainfall_rate,
                    self.other_source_rate,
                    self.infiltration_capacity,
                    self.infiltration_f0,
                    self.infiltration_fd,
                    self.infiltration_k,
                    self.green_ampt_ks,
                    self.green_ampt_psi,
                    self.green_ampt_delta_theta,
                    self.cumulative_infiltration,
                    self.infiltration_method,
                    self._source_subgrid_infiltration_enabled,
                    self._source_subgrid_cell_z,
                    self._source_subgrid_cell_breakpoints,
                    self._source_subgrid_cell_count,
                    self.active_mask,
                    dt_wp,
                    self.source_mode,
                    self.dry_depth,
                    out_rain,
                    out_other,
                    out_infil,
                    self.dx,
                    self.dy,
                    self.cumulative_totals,
                    accumulate_accounting,
                    1 if defer else 0,
                    self.additional_source_rate,
                    self.infiltration_exclusion_mask,
                    self.infiltration_cap_active,
                    self.infiltration_cap_remaining_m,
                ],
                device=self.device,
            )
        if self.use_reduced_mass_accounting and flush:
            self._reduce_source_mass_arrays(out_rain, out_infil, out_other)

    def _reduce_source_mass_arrays(self, rain_arr, infil_arr, other_arr) -> None:
        """Tree-sum per-cell source depths into the one-element mass sums.

        The fold of these sums into cumulative_totals happens in the continuity
        kernel of the same step (thread (0, 0)), saving a launch per step; an
        off-cycle flush via cumulative_accounting() folds with
        accumulate_mass_kernel instead.
        """
        wp.utils.array_sum(rain_arr, out=self._mass_rain_sum)
        wp.utils.array_sum(infil_arr, out=self._mass_infil_sum)
        wp.utils.array_sum(other_arr, out=self._mass_other_sum)

    def _flush_pending_mass_accounting(self) -> None:
        """Off-cycle flush of a deferred accounting window (exact totals)."""
        if self._pending_rain is None or self._steps_since_mass_accounting == 0:
            return
        self._reduce_source_mass_arrays(self._pending_rain, self._pending_infil, self._pending_other)
        wp.launch(
            kernel=accumulate_mass_kernel,
            dim=(1,),
            inputs=[
                self._mass_rain_sum,
                self._mass_infil_sum,
                self._mass_other_sum,
                wp.float64(float(self.dx) * float(self.dy)),
                self.cumulative_totals,
            ],
            device=self.device,
        )
        self._pending_rain.zero_()
        self._pending_infil.zero_()
        self._pending_other.zero_()
        self._steps_since_mass_accounting = 0

    def _finish_mass_accounting_flush(self, flush: bool) -> None:
        """Zero the pending fields after a flush step's continuity fold."""
        if flush and self._deferred_mass_accounting:
            self._pending_rain.zero_()
            self._pending_infil.zero_()
            self._pending_other.zero_()
            self._steps_since_mass_accounting = 0

    def _swap_q_buffers(self, qx_next, qy_next) -> None:
        """Publish completed face fluxes and retain the old state as scratch."""
        self.qx, self.qx_next = qx_next, self.qx
        self.qy, self.qy_next = qy_next, self.qy





































































































    def _init_structures(self, structures, *, active_mask, bed_elevation) -> None:
        """Validate and pack solver-neutral structure records into sparse arrays."""
        self._structure_ids = tuple(str(item.get("id", "")) for item in structures)
        self._structure_count = len(structures)
        self._structure_diag = None
        if not structures:
            return
        if len(set(self._structure_ids)) != len(self._structure_ids) or any(not value for value in self._structure_ids):
            raise ValueError("structure IDs must be non-empty and unique")
        kind_codes = {"culvert": 0, "weir": 2, "bridge": 4}
        endpoints = []
        mapped = []
        for record in structures:
            kind_name = record.get("type")
            if kind_name not in kind_codes:
                raise ValueError(f"unsupported active hydraulic structure type {kind_name!r}")
            row_a, col_a = int(record["inlet_row"]), int(record["inlet_col"])
            row_b, col_b = int(record["outlet_row"]), int(record["outlet_col"])
            if not (0 <= row_a < self.ny and 0 <= row_b < self.ny and 0 <= col_a < self.nx and 0 <= col_b < self.nx):
                raise ValueError(f"structure {record['id']!r} endpoint lies outside solver grid")
            if active_mask[row_a, col_a] != 1 or active_mask[row_b, col_b] != 1:
                raise ValueError(f"structure {record['id']!r} endpoint maps to an inactive solver cell")
            if (row_a, col_a) == (row_b, col_b) and not record.get("allow_same_cell", False):
                raise ValueError(f"structure {record['id']!r} endpoints map to the same solver cell")
            item = dict(record)
            item["_flat_a"] = row_a * self.nx + col_a
            item["_flat_b"] = row_b * self.nx + col_b
            mapped.append(item)
            endpoints.extend((item["_flat_a"], item["_flat_b"]))
        endpoint_cells = sorted(set(endpoints))
        slots = {cell: slot for slot, cell in enumerate(endpoint_cells)}
        self._structure_endpoint_cells = wp.array(np.asarray(endpoint_cells, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._structure_slot_a = wp.array(np.asarray([slots[x["_flat_a"]] for x in mapped], dtype=np.int32), dtype=wp.int32, device=self.device)
        self._structure_slot_b = wp.array(np.asarray([slots[x["_flat_b"]] for x in mapped], dtype=np.int32), dtype=wp.int32, device=self.device)
        kind, one_way, arrays = [], [], {name: [] for name in ("width", "height", "invert_a", "invert_b", "length", "manning", "entrance", "exit", "crest", "weir_length", "weir_factor", "max_q")}
        for record in mapped:
            typ = record["type"]
            if typ == "culvert":
                circular = record.get("shape") == "circular"
                kind.append(0 if circular else 1)
                width = float(record["diameter_m"] if circular else record["width_m"])
                height = float(record["diameter_m"] if circular else record["height_m"])
                arrays["invert_a"].append(float(record["inlet_invert_m"]))
                arrays["invert_b"].append(float(record["outlet_invert_m"]))
                arrays["crest"].append(0.0); arrays["weir_length"].append(0.0); arrays["weir_factor"].append(0.0)
                arrays["max_q"].append(float(record.get("maximum_discharge_m3s") or 1.0e30))
            elif typ == "weir":
                broad = record["weir_type"] == "broad_crested"
                kind.append(3 if broad else 2)
                width = height = 0.0
                arrays["invert_a"].append(0.0); arrays["invert_b"].append(0.0)
                arrays["crest"].append(float(record["crest_elevation_m"]))
                arrays["weir_length"].append(float(record["crest_length_m"]))
                coefficient = float(record["discharge_coefficient"])
                factor = (8.0 * 9.80665 / 27.0) ** 0.5 if broad else (2.0 / 3.0) * (2.0 * 9.80665) ** 0.5
                arrays["weir_factor"].append(coefficient * factor)
                arrays["max_q"].append(float(record.get("maximum_discharge_m3s") or 1.0e30))
            else:
                kind.append(4)
                width, height = float(record["opening_width_m"]), float(record["opening_height_m"])
                arrays["invert_a"].append(float(record["opening_invert_m"]))
                arrays["invert_b"].append(float(record["opening_invert_m"]))
                arrays["crest"].append(0.0); arrays["weir_length"].append(0.0); arrays["weir_factor"].append(0.0)
                arrays["max_q"].append(float(record.get("maximum_discharge_m3s") or 1.0e30))
            arrays["width"].append(width); arrays["height"].append(height)
            arrays["length"].append(float(record.get("length_m", 0.0)))
            arrays["manning"].append(float(record.get("manning_n", 0.0)))
            arrays["entrance"].append(float(record.get("entrance_loss_coefficient", 0.0)))
            arrays["exit"].append(float(record.get("exit_loss_coefficient", 0.0)))
            one_way.append(int(bool(record.get("one_way", False))))
        def dev(values, dtype=WP_FLOAT):
            return wp.array(np.asarray(values, dtype=np.float64 if dtype == wp.float64 else np.float32 if dtype == wp.float32 else np.int32), dtype=dtype, device=self.device)
        self._structure_kind = dev(kind, wp.int32)
        self._structure_one_way = dev(one_way, wp.int32)
        self._structure_params = {name: dev(values) for name, values in arrays.items()}
        n, m = len(mapped), len(endpoint_cells)
        self._structure_stage_placeholder = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=self.device)
        self._structure_endpoint_stage = wp.zeros(m, dtype=WP_FLOAT, device=self.device)
        self._structure_endpoint_storage = wp.zeros(m, dtype=WP_FLOAT, device=self.device)
        self._structure_donor_requested = wp.zeros(m, dtype=WP_FLOAT, device=self.device)
        self._structure_request = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_actual = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_q = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_head = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_positive = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_negative = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_net = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_max_q = wp.zeros(n, dtype=WP_FLOAT, device=self.device)
        self._structure_limiter_count = wp.zeros(n, dtype=wp.int32, device=self.device)

    def _apply_structure_transfers(self, *, dt: float) -> None:
        if self.is_subgrid():
            self._launch_stage_inversion(self.water_depth, self._eta_stage)
        self._structure_donor_requested.zero_()
        wp.launch(
            snapshot_structure_endpoints,
            dim=len(self._structure_endpoint_cells),
            inputs=[self.water_depth, self.bed_elevation, self.active_mask,
                    self._eta_stage if self.is_subgrid() else self._structure_stage_placeholder,
                    self._structure_endpoint_cells, self._structure_endpoint_stage,
                    self._structure_endpoint_storage, self._structure_donor_requested,
                    self.nx, WP_FLOAT(float(self.dx) * float(self.dy)), int(self.is_subgrid())],
            device=self.device,
        )
        params = self._structure_params
        wp.launch(
            calculate_structure_requests,
            dim=self._structure_count,
            inputs=[self._structure_endpoint_stage, self._structure_endpoint_storage,
                    self._structure_endpoint_cells, self._structure_slot_a, self._structure_slot_b,
                    self._structure_kind, self._structure_one_way, params["width"], params["height"],
                    params["invert_a"], params["invert_b"], params["length"], params["manning"],
                    params["entrance"], params["exit"], params["crest"], params["weir_length"],
                    params["weir_factor"], params["max_q"], WP_FLOAT(dt), self._structure_donor_requested,
                    self._structure_request, self._structure_q, self._structure_head],
            device=self.device,
        )
        wp.launch(
            apply_structure_requests,
            dim=self._structure_count,
            inputs=[self.water_depth, self._structure_endpoint_cells, self._structure_slot_a,
                    self._structure_slot_b, self._structure_endpoint_storage,
                    self._structure_donor_requested, self._structure_request,
                    self._structure_positive, self._structure_negative, self._structure_net,
                    self._structure_max_q, self._structure_q, self._structure_limiter_count,
                    self._structure_actual, self.nx, WP_FLOAT(float(self.dx) * float(self.dy))],
            device=self.device,
        )

    def reset_structure_diagnostics(self) -> None:
        """Reset per-link cumulative and instantaneous diagnostics."""
        if not self.structures:
            return
        for array in (
            self._structure_q,
            self._structure_head,
            self._structure_positive,
            self._structure_negative,
            self._structure_net,
            self._structure_max_q,
        ):
            array.zero_()
        self._structure_limiter_count.zero_()

    def get_structure_mass_balance_diagnostics(self) -> dict:
        """Return paired internal debit/credit totals and their residual.

        Each accepted link volume is applied as one debit and one credit. The
        same cumulative gross moved volume forms both ledgers, so this
        structural residual is zero by construction; global water-storage
        diagnostics still include solver-precision accumulation roundoff.
        """
        if not self.structures:
            return {
                "debited_volume_m3": 0.0,
                "credited_volume_m3": 0.0,
                "internal_transfer_residual_m3": 0.0,
            }
        positive = np.asarray(self._structure_positive.numpy(), dtype=np.float64)
        negative = np.asarray(self._structure_negative.numpy(), dtype=np.float64)
        gross = float(np.sum(positive - negative, dtype=np.float64))
        return {
            "debited_volume_m3": gross,
            "credited_volume_m3": gross,
            "internal_transfer_residual_m3": 0.0,
        }

    def get_structure_diagnostics(self) -> list[dict]:
        """Return per-structure instantaneous and cumulative transfer diagnostics."""
        if not self.structures:
            return []
        q = self._structure_q.numpy(); head = self._structure_head.numpy()
        pos = self._structure_positive.numpy(); neg = self._structure_negative.numpy()
        net = self._structure_net.numpy(); peak = self._structure_max_q.numpy()
        limited = self._structure_limiter_count.numpy()
        return [
            {"structure_id": sid, "instantaneous_q_m3s": float(q[i]),
             "flow_direction": "A_to_B" if q[i] > 0 else "B_to_A" if q[i] < 0 else "none",
             "maximum_abs_q_m3s": float(peak[i]), "cumulative_positive_volume_m3": float(pos[i]),
             "cumulative_negative_volume_m3": float(neg[i]), "net_transferred_volume_m3": float(net[i]),
             "maximum_head_difference_m": float(head[i]), "limiter_activations": int(limited[i])}
            for i, sid in enumerate(self._structure_ids)
        ]

    def _update_custom_face_fluxes(self, *, qx_next, qy_next, dt: float) -> bool:
        """Optional host-step surface operator; False retains existing momentum.

        An alternative numerical approximation may supply candidate face fluxes
        while reusing the ordinary sources, donor limiter and continuity. It
        must provide its own stable timestep and reject unsupported schedulers.
        """
        return False

    def step(self, remaining_time: float = None, fixed_dt: float | None = None):
        """Perform one time step, optionally using a precomputed stable timestep."""
        self._ensure_conditioned_bed_fields()
        if fixed_dt is None:
            dt = self.calculate_stable_timestep(remaining_time=remaining_time)
        else:
            if fixed_dt <= 0.0:
                raise ValueError("fixed_dt must be positive")
            if remaining_time is not None and fixed_dt > remaining_time + 1.0e-12:
                raise ValueError("fixed_dt must not exceed remaining_time")
            dt = fixed_dt

        if dt < self.minimum_dt:
            if remaining_time is not None and remaining_time < self.minimum_dt:
                pass
            else:
                raise RuntimeError(f"Calculated stable time step ({dt}) is below minimum_dt ({self.minimum_dt}).")

        self.current_dt = dt
        # Precomputing a stable dt and passing it back must retain the same
        # controller history as ordinary adaptive stepping. Boundary-clipped
        # steps retain the preceding unconstrained reference in both paths.
        boundary_limited = (
            remaining_time is not None
            and dt >= float(remaining_time) - 1.0e-12
            and float(remaining_time) < self.maximum_dt - 1.0e-12
        )
        if not boundary_limited or self._adaptive_dt_reference <= 0.0:
            self._adaptive_dt_reference = dt
        dt_wp = WP_FLOAT(dt)

        # Deferred source-mass accounting: flush every accounting_interval
        # steps and always on the final step so end-of-run totals are exact.
        is_final_step = remaining_time is not None and dt >= remaining_time - 1.0e-12
        flush_mass = False
        if self.use_reduced_mass_accounting:
            if self._deferred_mass_accounting:
                self._steps_since_mass_accounting += 1
                flush_mass = is_final_step or self._steps_since_mass_accounting >= self.accounting_interval
            else:
                flush_mass = True

        if not self.apply_sources_after_momentum:
            self._apply_sources(dt_wp=dt_wp, defer=self._deferred_mass_accounting, flush=flush_mass)

        qx_old, qy_old = self.qx, self.qy
        qx_next, qy_next = self.qx_next, self.qy_next

        if self._update_custom_face_fluxes(qx_next=qx_next, qy_next=qy_next, dt=dt):
            pass
        elif self.is_subgrid():
            # Custom closures derive their own face geometry. Only the
            # ordinary SGS momentum kernels consume this level-pool stage.
            self._launch_stage_inversion(self.water_depth, self._eta_stage)
            if self.use_fused_momentum:
                wp.launch(
                    kernel=subgrid_update_momentum_kernel,
                    dim=(self.ny + 1, self.nx + 1),
                    inputs=[
                        qx_next, qy_next, qx_old, qy_old, self._eta_stage,
                        self._subgrid_face_z_x, self._subgrid_face_z_min_x, self._subgrid_face_n_x,
                        self._subgrid_face_z_y, self._subgrid_face_z_min_y, self._subgrid_face_n_y,
                        self.manning_n,
                        self.boundary_type_qx, self.boundary_stage_qx,
                        self.boundary_type_qy, self.boundary_stage_qy,
                        self.dx, self.dy, dt_wp, self.gravity, self.dry_depth,
                        self.depth_epsilon, self.manning_face_mode,
                        self.use_two_dimensional_friction, self._use_native_face_manning, self._subgrid_factor,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=subgrid_update_qx_kernel,
                    dim=(self.ny, self.nx + 1),
                    inputs=[
                        qx_next, qx_old, qy_old, self._eta_stage,
                        self._subgrid_face_z_x, self._subgrid_face_z_min_x, self._subgrid_face_n_x,
                        self.manning_n, self.boundary_type_qx,
                        self.boundary_stage_qx,
                        self.dx, dt_wp, self.gravity, self.dry_depth,
                        self.depth_epsilon, self.manning_face_mode,
                        self.use_two_dimensional_friction, self._use_native_face_manning, self._subgrid_factor,
                    ],
                    device=self.device,
                )
                wp.launch(
                    kernel=subgrid_update_qy_kernel,
                    dim=(self.ny + 1, self.nx),
                    inputs=[
                        qy_next, qy_old, qx_old, self._eta_stage,
                        self._subgrid_face_z_y, self._subgrid_face_z_min_y, self._subgrid_face_n_y,
                        self.manning_n, self.boundary_type_qy,
                        self.boundary_stage_qy,
                        self.dy, dt_wp, self.gravity, self.dry_depth,
                        self.depth_epsilon, self.manning_face_mode,
                        self.use_two_dimensional_friction, self._use_native_face_manning, self._subgrid_factor,
                    ],
                    device=self.device,
                )
        elif USE_CONDITIONED_MIXED_KERNELS and self.use_fused_momentum:
            wp.launch(
                kernel=update_conditioned_momentum_kernel,
                dim=(self.ny + 1, self.nx + 1),
                inputs=[
                    qx_next, qy_next, qx_old, qy_old, self.water_depth,
                    self.bed_slope_qx, self.bed_slope_qy, self.manning_n,
                    self.boundary_type_qx, self.boundary_stage_depth_qx,
                    self.boundary_free_slope_qx, self.boundary_type_qy,
                    self.boundary_stage_depth_qy, self.boundary_free_slope_qy,
                    self.dx, self.dy, dt_wp, self.gravity, self.dry_depth,
                    self.depth_epsilon, self.manning_face_mode,
                    self.use_two_dimensional_friction, self.free_outflow_mode,
                ],
                device=self.device,
            )
        elif USE_CONDITIONED_MIXED_KERNELS:
            wp.launch(
                kernel=update_conditioned_qx_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    qx_next, qx_old, qy_old, self.water_depth,
                    self.bed_slope_qx, self.manning_n, self.boundary_type_qx,
                    self.boundary_stage_depth_qx, self.boundary_free_slope_qx,
                    self.dx, dt_wp, self.gravity, self.dry_depth, self.depth_epsilon,
                    self.manning_face_mode, self.use_two_dimensional_friction,
                    self.free_outflow_mode,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=update_conditioned_qy_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    qy_next, qy_old, qx_old, self.water_depth,
                    self.bed_slope_qy, self.manning_n, self.boundary_type_qy,
                    self.boundary_stage_depth_qy, self.boundary_free_slope_qy,
                    self.dy, dt_wp, self.gravity, self.dry_depth, self.depth_epsilon,
                    self.manning_face_mode, self.use_two_dimensional_friction,
                    self.free_outflow_mode,
                ],
                device=self.device,
            )
        elif self.use_fused_momentum:
            wp.launch(
                kernel=update_momentum_kernel,
                dim=(self.ny + 1, self.nx + 1),
                inputs=[
                    qx_next, qy_next, qx_old, qy_old, self.water_depth, self.bed_elevation,
                    self.manning_n, self.boundary_type_qx, self.boundary_stage_qx,
                    self.boundary_type_qy, self.boundary_stage_qy,
                    self.dx, self.dy, dt_wp, self.gravity, self.dry_depth, self.depth_epsilon,
                    self.manning_face_mode, self.use_two_dimensional_friction, self.free_outflow_mode,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=update_qx_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    qx_next, qx_old, qy_old, self.water_depth, self.bed_elevation, self.manning_n,
                    self.boundary_type_qx, self.boundary_stage_qx,
                    self.dx, dt_wp, self.gravity, self.dry_depth, self.depth_epsilon,
                    self.manning_face_mode, self.use_two_dimensional_friction, self.free_outflow_mode,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=update_qy_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    qy_next, qy_old, qx_old, self.water_depth, self.bed_elevation, self.manning_n,
                    self.boundary_type_qy, self.boundary_stage_qy,
                    self.dy, dt_wp, self.gravity, self.dry_depth, self.depth_epsilon,
                    self.manning_face_mode, self.use_two_dimensional_friction, self.free_outflow_mode,
                ],
                device=self.device,
            )

        if self.free_outflow_arithmetic_mode_code > FREE_OUTFLOW_ARITHMETIC_MODES["native"]:
            wp.launch(
                kernel=override_free_outflow_arithmetic_kernel,
                dim=(self.ny + 1, self.nx + 1),
                inputs=[
                    qx_next, qy_next, qx_old, qy_old, self.water_depth,
                    self.bed_elevation, self.manning_n, self.boundary_type_qx,
                    self.boundary_stage_qx, self.boundary_type_qy,
                    self.boundary_stage_qy, self.dx, self.dy, dt_wp,
                    self.gravity, self.dry_depth, self.depth_epsilon,
                    self.free_outflow_arithmetic_mode_code,
                ],
                device=self.device,
            )


        if self.use_fused_limiter:
            wp.launch(
                kernel=apply_outflow_limiter_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self.water_depth, qx_next, qy_next, self.active_mask,
                    self.boundary_type_qx, self.boundary_type_qy,
                    self.dx, self.dy, dt_wp, self.limiter_epsilon,
                    self.free_outflow_limiter_mode, self.free_outflow_diagnostics, 0,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=calc_outflow_limiter_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self.water_depth, qx_next, qy_next, self.outflow_limiter,
                    self.free_outflow_limiter, self.active_mask,
                    self.boundary_type_qx, self.boundary_type_qy,
                    self.dx, self.dy, dt_wp, self.limiter_epsilon,
                    self.free_outflow_limiter_mode, self.free_outflow_diagnostics, 0,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=apply_limiter_qx_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[qx_next, self.outflow_limiter, self.free_outflow_limiter, self.boundary_type_qx],
                device=self.device,
            )
            wp.launch(
                kernel=apply_limiter_qy_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[qy_next, self.outflow_limiter, self.free_outflow_limiter, self.boundary_type_qy],
                device=self.device,
            )


        self._accumulate_face_volume_diagnostics(qx_next, qy_next, dt_wp)

        if self.apply_sources_after_momentum:
            self._apply_sources(
                dt_wp=dt_wp,
                defer=self._deferred_mass_accounting,
                flush=flush_mass,
                committed_qx=qx_next,
                committed_qy=qy_next,
            )

        if self.use_fused_continuity:
            reduce_max_depth = 1 if self.use_lisflood_acc_timestep else 0
            self.lisflood_step_control.zero_()
            wp.launch(
                kernel=continuity_update_and_check_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self.water_depth, qx_next, qy_next, self.active_mask,
                    self.dx, self.dy, dt_wp, self.negative_depth_tolerance,
                    self.lisflood_step_control, reduce_max_depth,
                    self.boundary_type_qx, self.boundary_type_qy, self.cumulative_totals,
                    self._mass_rain_sum, self._mass_infil_sum, self._mass_other_sum,
                    wp.float64(float(self.dx) * float(self.dy)),
                    1 if flush_mass else 0,
                ],
                device=self.device,
            )
            if self.use_lisflood_acc_timestep:
                step_control = self.lisflood_step_control.numpy()
                if step_control[1] != 0.0:
                    raise RuntimeError(
                        "Significant negative depth encountered after "
                        f"the timestep ending at {self.simulation_time + dt:.12g} s."
                    )
                self._lisflood_cached_maximum_depth_m = float(step_control[0])
            elif is_final_step:
                # The next timestep normally reports the fused error flag along
                # with its CFL limit. Read it now when this is the final step.
                step_control = self.lisflood_step_control.numpy()
                if step_control[1] != 0.0:
                    raise RuntimeError("Significant negative depth encountered.")
            elif fixed_dt is not None:
                # When a fixed dt is used, calculate_cfl_dt() is not called on the
                # next step, so the fused error flag would never be checked. Read
                # it now to preserve the original per-step safety guarantee.
                step_control = self.lisflood_step_control.numpy()
                if step_control[1] != 0.0:
                    raise RuntimeError("Significant negative depth encountered.")
        else:
            wp.launch(
                kernel=continuity_update_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self.water_depth, qx_next, qy_next, self.active_mask,
                    self.dx, self.dy, dt_wp,
                    self.boundary_type_qx, self.boundary_type_qy, self.cumulative_totals,
                    self._mass_rain_sum, self._mass_infil_sum, self._mass_other_sum,
                    wp.float64(float(self.dx) * float(self.dy)),
                    1 if flush_mass else 0,
                ],
                device=self.device,
            )

            if self.use_lisflood_acc_timestep:
                self.lisflood_step_control.zero_()
                wp.launch(
                    kernel=check_negative_depth_and_max_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[
                        self.water_depth,
                        self.active_mask,
                        self.negative_depth_tolerance,
                        self.lisflood_step_control,
                    ],
                    device=self.device,
                )
                step_control = self.lisflood_step_control.numpy()
                if step_control[1] != 0.0:
                    raise RuntimeError(
                        "Significant negative depth encountered after "
                        f"the timestep ending at {self.simulation_time + dt:.12g} s."
                    )
                self._lisflood_cached_maximum_depth_m = float(step_control[0])
            else:
                self.error_flag.zero_()
                wp.launch(
                    kernel=check_negative_depth_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.active_mask, self.negative_depth_tolerance, self.error_flag],
                    device=self.device,
                )
                if int(self.error_flag.numpy()[0]) > 0:
                    raise RuntimeError("Significant negative depth encountered.")

        if self.structures:
            self._apply_structure_transfers(dt=dt)
            self.error_flag.zero_()
            wp.launch(
                kernel=check_negative_depth_kernel,
                dim=(self.ny, self.nx),
                inputs=[self.water_depth, self.active_mask, self.negative_depth_tolerance, self.error_flag],
                device=self.device,
            )
            if int(self.error_flag.numpy()[0]) > 0:
                raise RuntimeError("Structure transfer produced a significant negative depth")

        self._accumulate_depth_diagnostic()
        self._accumulate_max_water_depth()

        # Boundary-flux volumes were folded into the continuity kernel above
        # (qx/qy are final after the limiter, matching the previous evaluation
        # order). Source volumes accumulate on device inside
        # apply_sources_kernel / the pending fields; no per-step host readback.
        self._finish_mass_accounting_flush(flush=flush_mass)

        # The queued kernels used qx_next/qy_next for every write in this step.
        # Exchange Python handles only after enqueueing them; no full-field copy
        # is needed and qx/qy remain the caller-visible current state.
        self._swap_q_buffers(qx_next=qx_next, qy_next=qy_next)

        self.simulation_time += dt
        return dt

    def _prepare_stage_variation_faces(self, *, use_x, use_y):
        from floodwarp import stage_forcing
        x_count=self.ny*(self.nx+1)
        x=np.flatnonzero(np.isin(self.boundary_type_qx.numpy(),(3,5,10))) if use_x else np.zeros(0,dtype=int)
        y=np.flatnonzero(np.isin(self.boundary_type_qy.numpy(),(3,4,11)))+x_count if use_y else np.zeros(0,dtype=int)
        values=np.r_[x,y].astype(np.int32)
        self._stage_variation_faces=wp.array(values,dtype=int,device=self.device)
        self._stage_variation_face_count=len(values)

    def _limit_stage_variation(self, *, control,minimum,times_x,values_x,use_x,times_y,values_y,use_y):
        if not (use_x or use_y):return
        from floodwarp import stage_forcing
        common=[self.nx,self.ny*(self.nx+1),self.boundary_type_qx,self.boundary_type_qy,
            times_x,values_x,use_x,times_y,values_y,use_y,control,minimum]
        if self.hydraulic_geometry_mode=='enhanced':
            if self._connection_count:
                wp.launch(kernel=stage_forcing.limit_connections,dim=self._connection_count,
                    inputs=[self._ec_face,self._ec_endpoints,self._ec_sill,self._ec_eta,*common[:2],
                        self._ec_adjacency_offsets,self._ec_adjacency,self._ec_q,self._ec_area,*common[2:]],device=self.device)
        elif self._stage_variation_face_count:
            traditional=self.hydraulic_geometry_mode=='traditional'
            wp.launch(kernel=stage_forcing.limit_faces,dim=self._stage_variation_face_count,
                inputs=[self._stage_variation_faces,self.nx,self.ny*(self.nx+1),
                    self.water_depth,self.bed_elevation,
                    self._subgrid_face_z_min_x if traditional else self.bed_elevation,
                    self._subgrid_face_z_min_y if traditional else self.bed_elevation,int(traditional),
                    self.qx,self.qy,self.dx,self.dy,self._source_subgrid_cell_z,
                    self._source_subgrid_cell_breakpoints,self._source_subgrid_cell_count,
                    *common[2:]],device=self.device)

    @staticmethod
    def _validate_time_series(name: str, series) -> tuple[np.ndarray, np.ndarray]:
        """Validate a (times, values) node series for device-resident forcing."""
        times, values = series
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
            raise ValueError(f"{name} must be a (times, values) pair of equal-length 1-D arrays")
        if times.size < 2 or times[0] != 0.0:
            raise ValueError(f"{name} times must start at zero and contain at least two nodes")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError(f"{name} times must be strictly increasing")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} values must be finite")
        return times, values

    def run_device_loop(
        self,
        duration: float,
        sync_interval: int = 32,
        rainfall_hyetograph=None,
        stage_series_qx=None,
        stage_series_qy=None,
        dt_history_capacity: int | None = None,
        on_sync=None,
        collect_free_outflow_diagnostics: bool = False,
        checkpoint_times_s=None,
        checkpoint_outlet_rows=None,
        field_checkpoint_indices=None,
        use_cuda_graph: bool | None = None,
        on_progress=None,
        progress_interval_s: float = 30.0,
    ) -> dict:
        """Advance ``duration`` seconds with the stepping loop on-device.

        ``use_cuda_graph=None`` (default) automatically selects conditional
        CUDA graphs on supported CUDA devices without per-sync callbacks or
        FREE-outflow traces; otherwise it retains the queued-window path.
        ``False`` explicitly selects windows. ``True`` requires graph support
        and raises if these execution constraints are not satisfied.
        Graph execution captures unchanged production kernels in a
        device-conditional two-step loop with a single completion readback
        unless optional live progress is requested.
        CUDA only; per-sync callbacks/traces keep the established window path.
        Optional ``on_progress(steps, simulation_time, execution)`` observes accepted state
        without disabling graphs. It must not mutate the solver. The same graph
        is replayed in even accepted-step batches targeting ``progress_interval_s``
        wall seconds; no simulated-time boundaries or timestep changes are added.
        Reports occur after the initial batch, approximately at that cadence and
        at completion. The cadence is approximate when hydraulic cost changes.
        Errors stop at the next two-step boundary. Capture cost is returned
        separately and is included in caller wall time. Physics, sources, output
        checkpoints, accounting and adaptive control are unchanged.

        Opt-in alternative to host ``step()`` looping: each window of
        ``sync_interval`` steps runs without any host synchronisation. The
        adaptive timestep uses the production CFL kernels with forcing-node
        caps and a local head-variation bound for changing prescribed stages.
        Host parity drivers apply these same forcing controls. Rainfall uses
        the configured source-splitting sample; prescribed stages use midpoint
        quadrature and a conservative trial head during CFL selection.

        The host syncs once per window to (a) check the device error flag and
        raise RuntimeError for a significant negative depth or a sub-minimum
        timestep, (b) flush deferred source-mass accounting, and (c) test
        termination. An error is therefore raised at most ``sync_interval``
        steps after the flagged step; larger windows trade failure latency for
        fewer synchronisations.

        Requirements/notes:
        - fused kernel path (use_fused_kernels=True and no per-fusion env off);
        - adaptive timestep only (no fixed_dt);
        - ``rainfall_hyetograph`` forcing is spatially uniform (the host
          dynamic-rainfall driver path); without it, the static
          rainfall/other/infiltration fields are used as-is;
        - ``stage_series_qx/qy`` are (times, values) node series applied
          uniformly to all SPECIFIED_STAGE faces of that array;
        - ``track_cumulative_face_volumes`` is supported: accepted post-limiter
          face transport, peak discharge, and optional max-depth diagnostics
          remain device-resident until an explicit readback;
        - ``track_max_water_depth`` is supported: the standalone per-cell
          maximum-depth label accumulator updates after every accepted step
          and remains device-resident until ``get_max_water_depth()``;
        - starts from the current solver state/clock; when continuing a run
          with a hyetograph, the CFL predictor's initial rate is re-sampled at
          the current time.

        Optional ``checkpoint_times_s`` are exact output boundaries: timesteps
        are capped at those times and scalar volume/outlet plus requested full
        depth fields are captured into device-resident buffers inside the
        continuity kernel. ``checkpoint_outlet_rows`` is a boolean row mask
        for positive east-boundary discharge. ``field_checkpoint_indices``
        selects rows of ``checkpoint_times_s`` for full-field capture.

        Controlled timestep growth shares its accepted-step reference with
        host stepping, including externally precomputed stable timesteps
        passed back through ``step(fixed_dt=...)``.

        Returns a dict with steps taken, final simulation_time, sync_count,
        wall-clock ``loop_seconds`` (stepping windows) and
        ``checkpoint_readback_seconds`` (final batched trace transfer), and
        (when dt_history_capacity is set) the per-step dt history.  When
        ``collect_free_outflow_diagnostics`` is true, it also returns sparse
        per-sync raw/post-limiter FREE-outlet flux counters.
        """
        if on_progress is not None:
            if not callable(on_progress):
                raise ValueError("on_progress must be callable")
            if not np.isfinite(progress_interval_s) or progress_interval_s <= 0.0:
                raise ValueError("progress_interval_s must be finite and positive")
        if use_cuda_graph is None or use_cuda_graph:
            graph_supported = (wp.get_device(self.device).is_cuda
                and on_sync is None and not collect_free_outflow_diagnostics)
            if graph_supported:
                with wp.ScopedDevice(self.device):
                    graph_supported = wp.is_conditional_graph_supported()
            if use_cuda_graph and not graph_supported:
                raise ValueError("conditional CUDA execution requires CUDA conditional-graph support without per-sync callbacks/traces; use the established window path for those features")
            use_cuda_graph = graph_supported
        self._ensure_conditioned_bed_fields()
        if self.structures:
            raise ValueError("Hydraulic structures currently require host-stepped step(); device-loop execution is not yet supported")
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if sync_interval < 1:
            raise ValueError("sync_interval must be at least 1")
        if not (self.use_fused_momentum and self.use_fused_limiter and self.use_fused_continuity):
            raise ValueError("run_device_loop requires the fused kernel path")
        if rainfall_hyetograph is not None:
            node_times = np.asarray(rainfall_hyetograph.node_times_s, dtype=np.float64)
            node_rates = np.asarray(rainfall_hyetograph.node_rates_m_per_s, dtype=np.float64)
            storm_duration = float(rainfall_hyetograph.duration_s)
        else:
            node_times = np.zeros(1, dtype=np.float64)
            node_rates = np.zeros(1, dtype=np.float64)
            storm_duration = 0.0
        if stage_series_qx is not None:
            stage_times_qx, stage_values_qx = self._validate_time_series("stage_series_qx", stage_series_qx)
        else:
            stage_times_qx = stage_values_qx = np.zeros(1, dtype=np.float64)
        if stage_series_qy is not None:
            stage_times_qy, stage_values_qy = self._validate_time_series("stage_series_qy", stage_series_qy)
        else:
            stage_times_qy = stage_values_qy = np.zeros(1, dtype=np.float64)
        if dt_history_capacity is not None and dt_history_capacity < 1:
            raise ValueError("dt_history_capacity must be positive when given")

        end_time = self.simulation_time + duration
        if checkpoint_times_s is None:
            checkpoint_times = np.zeros(0, dtype=np.float64)
        else:
            checkpoint_times = np.asarray(checkpoint_times_s, dtype=np.float64)
            if checkpoint_times.ndim != 1:
                raise ValueError("checkpoint_times_s must be one-dimensional")
            if not np.all(np.isfinite(checkpoint_times)):
                raise ValueError("checkpoint_times_s must be finite")
            if checkpoint_times.size and (
                checkpoint_times[0] <= self.simulation_time
                or checkpoint_times[-1] > end_time
                or np.any(np.diff(checkpoint_times) <= 0.0)
            ):
                raise ValueError(
                    "checkpoint_times_s must be strictly increasing within the run interval"
                )
        checkpoint_count = int(checkpoint_times.size)
        if checkpoint_outlet_rows is None:
            outlet_rows = np.zeros(self.ny, dtype=np.int32)
        else:
            outlet_rows = np.asarray(checkpoint_outlet_rows, dtype=np.int32)
            if outlet_rows.shape != (self.ny,):
                raise ValueError(f"checkpoint_outlet_rows must have shape ({self.ny},)")
            outlet_rows = np.ascontiguousarray(outlet_rows != 0, dtype=np.int32)
        if field_checkpoint_indices is None:
            field_indices = np.zeros(0, dtype=np.int64)
        else:
            field_indices = np.asarray(field_checkpoint_indices, dtype=np.int64)
            if field_indices.ndim != 1:
                raise ValueError("field_checkpoint_indices must be one-dimensional")
            if field_indices.size and (
                np.any(field_indices < 0)
                or np.any(field_indices >= checkpoint_count)
                or np.any(np.diff(field_indices) <= 0)
            ):
                raise ValueError(
                    "field_checkpoint_indices must be strictly increasing valid checkpoint indices"
                )
        field_row_for_checkpoint = np.full(max(checkpoint_count, 1), -1, dtype=np.int32)
        for field_row, checkpoint_index in enumerate(field_indices):
            field_row_for_checkpoint[int(checkpoint_index)] = field_row

        use_hyetograph = 1 if rainfall_hyetograph is not None else 0
        use_stage_qx = 1 if stage_series_qx is not None else 0
        use_stage_qy = 1 if stage_series_qy is not None else 0
        use_stage = use_stage_qx or use_stage_qy
        if use_stage:self._prepare_stage_variation_faces(use_x=use_stage_qx,use_y=use_stage_qy)
        use_acc = 1 if self.use_lisflood_acc_timestep else 0
        prescribed_stage_cfl = self.use_lisflood_acc_timestep and self._has_prescribed_stage_boundaries()
        sample_at_half_step = 0 if self.apply_sources_after_momentum else 1

        hyeto_times = wp.array(node_times, dtype=wp.float64, device=self.device)
        hyeto_rates = wp.array(node_rates, dtype=wp.float64, device=self.device)
        stage_times_qx_wp = wp.array(stage_times_qx, dtype=wp.float64, device=self.device)
        stage_values_qx_wp = wp.array(stage_values_qx, dtype=wp.float64, device=self.device)
        stage_times_qy_wp = wp.array(stage_times_qy, dtype=wp.float64, device=self.device)
        stage_values_qy_wp = wp.array(stage_values_qy, dtype=wp.float64, device=self.device)
        if checkpoint_count:
            checkpoint_times_wp = wp.array(
                checkpoint_times, dtype=wp.float64, device=self.device
            )
            checkpoint_outlet_rows_wp = wp.array(
                outlet_rows, dtype=wp.int32, device=self.device
            )
            checkpoint_field_indices_wp = wp.array(
                field_row_for_checkpoint, dtype=wp.int32, device=self.device
            )
            checkpoint_times_out = wp.zeros(
                checkpoint_count, dtype=wp.float64, device=self.device
            )
            checkpoint_outlet_out = wp.zeros(
                (checkpoint_count, self.ny), dtype=wp.float64, device=self.device
            )
            checkpoint_volume_out = wp.zeros(
                checkpoint_count, dtype=wp.float64, device=self.device
            )
            checkpoint_depth_out = wp.zeros(
                (max(int(field_indices.size), 1), self.ny, self.nx),
                dtype=WP_FLOAT,
                device=self.device,
            )
        else:
            # The timestep kernels share one signature for traced and ordinary
            # runs; this inert one-element input is never dereferenced when
            # checkpoint_count == 0.
            checkpoint_times_wp = wp.zeros(1, dtype=wp.float64, device=self.device)

        if self.hydraulic_geometry_mode == 'enhanced' and not checkpoint_count:
            checkpoint_outlet_rows_wp=wp.zeros(1,dtype=wp.int32,device=self.device)
            checkpoint_field_indices_wp=wp.zeros(1,dtype=wp.int32,device=self.device)
            checkpoint_times_out=wp.zeros(1,dtype=wp.float64,device=self.device)
            checkpoint_outlet_out=wp.zeros((1,1),dtype=wp.float64,device=self.device)
            checkpoint_volume_out=wp.zeros(1,dtype=wp.float64,device=self.device)
            checkpoint_depth_out=wp.zeros((1,1,1),dtype=WP_FLOAT,device=self.device)

        control_host = np.zeros(DL_SIZE, dtype=np.float64)
        control_host[DL_TIME] = self.simulation_time
        control_host[DL_DT] = self.current_dt
        control_host[DL_GROWTH_REFERENCE] = self._adaptive_dt_reference
        if use_hyetograph:
            # Hyetograph semantics: rainfall is zero at/after the storm end;
            # do not clamp onto the terminal rate.
            if self.simulation_time < storm_duration:
                control_host[DL_RAIN_RATE] = float(
                    rainfall_hyetograph.rate_at(self.simulation_time)
                )
        control_host[DL_DT_MIN] = np.inf
        control = wp.array(control_host, dtype=wp.float64, device=self.device)
        dt_min_scratch = wp.zeros(1, dtype=WP_FLOAT, device=self.device)

        record_history = 1 if dt_history_capacity is not None else 0
        history_capacity = int(dt_history_capacity or 0)
        dt_history = wp.zeros(max(history_capacity, 1), dtype=wp.float64, device=self.device)
        if collect_free_outflow_diagnostics:
            self._reset_free_outflow_limiter_diagnostics()
        free_outflow_trace: list[dict[str, float]] = []

        # Deferred source-mass accounting across each window (exact flush at
        # every sync); the per-cell atomic path accounts every step instead.
        defer_sources = self.use_reduced_mass_accounting and self.hydraulic_geometry_mode != 'enhanced'
        if defer_sources:
            if self._pending_rain is None:
                self._pending_rain = wp.zeros((self.ny, self.nx), dtype=wp.float64, device=self.device)
                self._pending_infil = wp.zeros((self.ny, self.nx), dtype=wp.float64, device=self.device)
                self._pending_other = wp.zeros((self.ny, self.nx), dtype=wp.float64, device=self.device)
            out_rain, out_other, out_infil = self._pending_rain, self._pending_other, self._pending_infil
            accumulate_accounting = 0
            accumulate_output = 1
        else:
            out_rain, out_other, out_infil = self.applied_rain, self.applied_other, self.applied_infil
            accumulate_accounting = 1
            accumulate_output = 0

        if self.use_lisflood_acc_timestep:
            # First-step max-depth reduction; later steps reuse the previous
            # step's post-continuity reduction, exactly like the host cache.
            wp.launch(
                kernel=device_loop_max_depth_kernel,
                dim=(self.ny, self.nx),
                inputs=[self.water_depth, self.active_mask, control],
                device=self.device,
            )
        # Bounded-contention per-step ACC max-depth reduction: coalesced
        # strided partial maxima (no atomics) folded by a small second launch.
        max_depth_reduce_threads = min(4096, self.ny * self.nx)
        max_depth_fold_threads = min(512, max_depth_reduce_threads)
        max_depth_partials = wp.zeros(
            max_depth_reduce_threads, dtype=wp.float64, device=self.device
        )

        if self.hydraulic_geometry_mode=='enhanced' and self._component_timestep in ('provisional','adaptive'):
            self._prepare_provisional_graph(control=control,end_time=end_time,minimum_dt=float(self.minimum_dt),
                hyeto_times=hyeto_times,hyeto_rates=hyeto_rates,use_hyetograph=use_hyetograph,storm_duration=storm_duration,
                stage_times_qx=stage_times_qx_wp,stage_values_qx=stage_values_qx_wp,use_stage_qx=use_stage_qx,
                stage_times_qy=stage_times_qy_wp,stage_values_qy=stage_values_qy_wp,use_stage_qy=use_stage_qy,
                checkpoint_times=checkpoint_times_wp,checkpoint_count=checkpoint_count)



        def enqueue_step() -> None:
            if self.hydraulic_geometry_mode == 'enhanced':
                self._enqueue_enhanced_device_step(control=control,dt_min_scratch=dt_min_scratch,
                    end_time=end_time,hyeto_times=hyeto_times,hyeto_rates=hyeto_rates,
                    use_hyetograph=use_hyetograph,storm_duration=storm_duration,
                    stage_times_qx=stage_times_qx_wp,stage_values_qx=stage_values_qx_wp,use_stage_qx=use_stage_qx,
                    stage_times_qy=stage_times_qy_wp,stage_values_qy=stage_values_qy_wp,use_stage_qy=use_stage_qy,
                    checkpoint_times=checkpoint_times_wp,checkpoint_count=checkpoint_count,
                    dt_history=dt_history,history_capacity=history_capacity,record_history=record_history,
                    checkpoint_outlet_rows=checkpoint_outlet_rows_wp,checkpoint_field_indices=checkpoint_field_indices_wp,
                    checkpoint_times_out=checkpoint_times_out,checkpoint_outlet_out=checkpoint_outlet_out,
                    checkpoint_volume_out=checkpoint_volume_out,checkpoint_depth_out=checkpoint_depth_out,
                    collect_free_outflow_diagnostics=collect_free_outflow_diagnostics)
                return
            wp.launch(
                kernel=device_loop_timestep_cap_kernel, dim=1,
                inputs=[control, dt_min_scratch, wp.float64(end_time), wp.float64(self.maximum_dt),
                        wp.float64(self.adaptive_dt_growth_factor or 0.0),
                        hyeto_times, use_hyetograph, wp.float64(storm_duration),
                        checkpoint_times_wp, checkpoint_count], device=self.device)
            wp.launch(kernel=device_loop_prepare_forcing_kernel, dim=1,
                inputs=[control,hyeto_times,hyeto_rates,use_hyetograph,wp.float64(storm_duration),
                        stage_times_qx_wp,stage_values_qx_wp,use_stage_qx,
                        stage_times_qy_wp,stage_values_qy_wp,use_stage_qy],device=self.device)
            if use_stage:
                from floodwarp import stage_forcing
                self._limit_stage_variation(control=control,minimum=dt_min_scratch,
                    times_x=stage_times_qx_wp,values_x=stage_values_qx_wp,use_x=use_stage_qx,
                    times_y=stage_times_qy_wp,values_y=stage_values_qy_wp,use_y=use_stage_qy)
                wp.launch(kernel=stage_forcing.forecast,dim=1,
                    inputs=[control,stage_times_qx_wp,stage_values_qx_wp,use_stage_qx,
                        stage_times_qy_wp,stage_values_qy_wp,use_stage_qy],device=self.device)
            if use_stage and USE_CONDITIONED_MIXED_KERNELS:
                wp.launch(kernel=device_loop_conditioned_stage_fill_kernel,
                    dim=(self.ny+1,self.nx+1),
                    inputs=[self.boundary_type_qx,self.boundary_stage_qx,
                        self.boundary_stage_depth_qx,self.boundary_stage_reference_qx,
                        self.boundary_type_qy,self.boundary_stage_qy,
                        self.boundary_stage_depth_qy,self.boundary_stage_reference_qy,
                        control,use_stage_qx,use_stage_qy],device=self.device)
            elif use_stage:
                wp.launch(kernel=device_loop_stage_fill_kernel,dim=(self.ny+1,self.nx+1),
                    inputs=[self.boundary_type_qx,self.boundary_stage_qx,
                        self.boundary_type_qy,self.boundary_stage_qy,control,use_stage_qx,use_stage_qy],device=self.device)
            if not self.use_lisflood_acc_timestep or prescribed_stage_cfl:
                wp.launch(
                    kernel=device_loop_predict_depth_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.rainfall_rate, self.rainfall_spatial_multiplier, self.other_source_rate,
                            self.additional_source_rate,
                            self.active_mask, control, use_hyetograph, self.cfl_depth],
                    device=self.device,
                )
                if self.is_subgrid():
                    self._launch_stage_inversion(self.cfl_depth, self._eta_cfl)
                    wp.launch(
                        kernel=subgrid_calc_cfl_dt_x_kernel,
                        dim=(self.ny, self.nx + 1),
                        inputs=[self._eta_cfl, self._subgrid_face_z_min_x,
                                self.boundary_type_qx, self.boundary_stage_qx,
                                self.dx, self.gravity, self.dry_depth,
                                self.cfl_number, dt_min_scratch],
                        device=self.device,
                    )
                    wp.launch(
                        kernel=subgrid_calc_cfl_dt_y_kernel,
                        dim=(self.ny + 1, self.nx),
                        inputs=[self._eta_cfl, self._subgrid_face_z_min_y,
                                self.boundary_type_qy, self.boundary_stage_qy,
                                self.dy, self.gravity, self.dry_depth,
                                self.cfl_number, dt_min_scratch],
                        device=self.device,
                    )
                elif USE_CONDITIONED_MIXED_KERNELS:
                    wp.launch(
                        kernel=calc_conditioned_cfl_dt_x_kernel,
                        dim=(self.ny, self.nx + 1),
                        inputs=[self.cfl_depth, self.bed_slope_qx, self.boundary_type_qx,
                                self.boundary_stage_depth_qx, self.dx, self.gravity,
                                self.dry_depth, self.cfl_number, dt_min_scratch],
                        device=self.device,
                    )
                    wp.launch(
                        kernel=calc_conditioned_cfl_dt_y_kernel,
                        dim=(self.ny + 1, self.nx),
                        inputs=[self.cfl_depth, self.bed_slope_qy, self.boundary_type_qy,
                                self.boundary_stage_depth_qy, self.dy, self.gravity,
                                self.dry_depth, self.cfl_number, dt_min_scratch],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        kernel=calc_cfl_dt_x_kernel,
                        dim=(self.ny, self.nx + 1),
                        inputs=[self.cfl_depth, self.bed_elevation, self.boundary_type_qx,
                                self.boundary_stage_qx, self.dx, self.gravity, self.dry_depth,
                                self.cfl_number, dt_min_scratch],
                        device=self.device,
                    )
                    wp.launch(
                        kernel=calc_cfl_dt_y_kernel,
                        dim=(self.ny + 1, self.nx),
                        inputs=[self.cfl_depth, self.bed_elevation, self.boundary_type_qy,
                                self.boundary_stage_qy, self.dy, self.gravity, self.dry_depth,
                                self.cfl_number, dt_min_scratch],
                        device=self.device,
                    )
            wp.launch(
                kernel=device_loop_timestep_finalize_kernel,
                dim=(1,),
                inputs=[control, dt_min_scratch, wp.float64(end_time),
                        wp.float64(self.minimum_dt), wp.float64(self.maximum_dt),
                        wp.float64(self.adaptive_dt_growth_factor or 0.0),
                        use_acc, self.cfl_number, self.dx, self.gravity, self.dry_depth,
                        hyeto_times, hyeto_rates, use_hyetograph, wp.float64(storm_duration),
                        sample_at_half_step,
                        stage_times_qx_wp, stage_values_qx_wp, use_stage_qx,
                        stage_times_qy_wp, stage_values_qy_wp, use_stage_qy,
                        checkpoint_times_wp, checkpoint_count],
                device=self.device,
            )
            if use_stage and USE_CONDITIONED_MIXED_KERNELS:
                wp.launch(kernel=device_loop_conditioned_stage_fill_kernel,dim=(self.ny+1,self.nx+1),
                    inputs=[self.boundary_type_qx,self.boundary_stage_qx,self.boundary_stage_depth_qx,
                        self.boundary_stage_reference_qx,self.boundary_type_qy,self.boundary_stage_qy,
                        self.boundary_stage_depth_qy,self.boundary_stage_reference_qy,control,use_stage_qx,use_stage_qy],device=self.device)
            elif use_stage:
                wp.launch(kernel=device_loop_stage_fill_kernel,dim=(self.ny+1,self.nx+1),
                    inputs=[self.boundary_type_qx,self.boundary_stage_qx,self.boundary_type_qy,
                        self.boundary_stage_qy,control,use_stage_qx,use_stage_qy],device=self.device)
            qx_old, qy_old = self.qx, self.qy
            qx_next, qy_next = self.qx_next, self.qy_next
            if not self.apply_sources_after_momentum:
                wp.launch(
                    kernel=device_loop_apply_sources_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.rainfall_rate, self.rainfall_spatial_multiplier, self.other_source_rate,
                            self.infiltration_capacity, self.infiltration_f0, self.infiltration_fd, self.infiltration_k, self.green_ampt_ks, self.green_ampt_psi, self.green_ampt_delta_theta, self.cumulative_infiltration, self.infiltration_method,
                            self._source_subgrid_infiltration_enabled, self._source_subgrid_cell_z,
                            self._source_subgrid_cell_breakpoints, self._source_subgrid_cell_count,
                            self.active_mask, control, use_hyetograph,
                            self.source_mode, self.dry_depth, qx_next, qy_next, 0,
                            out_rain, out_other, out_infil,
                            self.dx, self.dy, self.cumulative_totals,
                            accumulate_accounting, accumulate_output,
                            self.additional_source_rate, self.infiltration_exclusion_mask,
                            self.infiltration_cap_active, self.infiltration_cap_remaining_m],
                    device=self.device,
                )
            if self.is_subgrid():
                # Sources just changed storage; refresh the derived stage the
                # momentum kernels read.
                self._launch_stage_inversion(self.water_depth, self._eta_stage)
            if self.is_subgrid():
                wp.launch(
                    kernel=device_loop_subgrid_update_momentum_kernel,
                    dim=(self.ny + 1, self.nx + 1),
                    inputs=[qx_next, qy_next, qx_old, qy_old, self._eta_stage,
                            self._subgrid_face_z_x, self._subgrid_face_z_min_x, self._subgrid_face_n_x,
                            self._subgrid_face_z_y, self._subgrid_face_z_min_y, self._subgrid_face_n_y,
                            self.manning_n, self.boundary_type_qx,
                            self.boundary_stage_qx, self.boundary_type_qy,
                            self.boundary_stage_qy,
                            self.dx, self.dy, control, self.gravity, self.dry_depth,
                            self.depth_epsilon, self.manning_face_mode,
                            self.use_two_dimensional_friction, self._use_native_face_manning, self._subgrid_factor],
                    device=self.device,
                )
            elif USE_CONDITIONED_MIXED_KERNELS:
                wp.launch(
                    kernel=device_loop_update_conditioned_qx_kernel,
                    dim=(self.ny, self.nx + 1),
                    inputs=[qx_next, qx_old, qy_old, self.water_depth,
                            self.bed_slope_qx, self.manning_n, self.boundary_type_qx,
                            self.boundary_stage_depth_qx, self.boundary_free_slope_qx,
                            self.dx, control, self.gravity, self.dry_depth,
                            self.depth_epsilon, self.manning_face_mode,
                            self.use_two_dimensional_friction, self.free_outflow_mode],
                    device=self.device,
                )
                wp.launch(
                    kernel=device_loop_update_conditioned_qy_kernel,
                    dim=(self.ny + 1, self.nx),
                    inputs=[qy_next, qy_old, qx_old, self.water_depth,
                            self.bed_slope_qy, self.manning_n, self.boundary_type_qy,
                            self.boundary_stage_depth_qy, self.boundary_free_slope_qy,
                            self.dy, control, self.gravity, self.dry_depth,
                            self.depth_epsilon, self.manning_face_mode,
                            self.use_two_dimensional_friction, self.free_outflow_mode],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=device_loop_update_momentum_kernel,
                    dim=(self.ny + 1, self.nx + 1),
                    inputs=[qx_next, qy_next, qx_old, qy_old, self.water_depth, self.bed_elevation,
                            self.manning_n, self.boundary_type_qx, self.boundary_stage_qx,
                            self.boundary_type_qy, self.boundary_stage_qy,
                            self.dx, self.dy, control, self.gravity, self.dry_depth,
                            self.depth_epsilon, self.manning_face_mode,
                            self.use_two_dimensional_friction, self.free_outflow_mode],
                    device=self.device,
                )
            if self.free_outflow_arithmetic_mode_code > FREE_OUTFLOW_ARITHMETIC_MODES["native"]:
                wp.launch(
                    kernel=device_loop_override_free_outflow_arithmetic_kernel,
                    dim=(self.ny + 1, self.nx + 1),
                    inputs=[
                        qx_next, qy_next, qx_old, qy_old, self.water_depth,
                        self.bed_elevation, self.manning_n, self.boundary_type_qx,
                        self.boundary_stage_qx, self.boundary_type_qy,
                        self.boundary_stage_qy, self.dx, self.dy, control,
                        self.gravity, self.dry_depth, self.depth_epsilon,
                        self.free_outflow_arithmetic_mode_code,
                    ],
                    device=self.device,
                )
            wp.launch(
                kernel=device_loop_apply_outflow_limiter_kernel,
                dim=(self.ny, self.nx),
                inputs=[self.water_depth, qx_next, qy_next, self.active_mask,
                        self.boundary_type_qx, self.boundary_type_qy,
                        self.dx, self.dy, control, self.limiter_epsilon,
                        self.free_outflow_limiter_mode, self.free_outflow_diagnostics,
                        1 if collect_free_outflow_diagnostics else 0],
                device=self.device,
            )
            if self.track_cumulative_face_volumes:
                # Accepted fluxes only: this is deliberately after the donor
                # limiter and before continuity, matching global step().
                wp.launch(
                    kernel=device_loop_accumulate_face_volume_qx_kernel,
                    dim=(self.ny, self.nx + 1),
                    inputs=[
                        qx_next, self.cumulative_net_qx_volume_m3,
                        self.cumulative_abs_qx_volume_m3 if self.track_absolute_face_volumes else self._diagnostic_absolute_placeholder,
                        self._diagnostic_peak_qx_rows,
                        self.dy,
                        control, 1 if self.track_absolute_face_volumes else 0,
                    ],
                    device=self.device,
                )
                wp.launch(
                    kernel=device_loop_accumulate_face_volume_qy_kernel,
                    dim=(self.ny + 1, self.nx),
                    inputs=[
                        qy_next, self.cumulative_net_qy_volume_m3,
                        self.cumulative_abs_qy_volume_m3 if self.track_absolute_face_volumes else self._diagnostic_absolute_placeholder,
                        self._diagnostic_peak_qy_rows,
                        self.dx,
                        control, 1 if self.track_absolute_face_volumes else 0,
                    ],
                    device=self.device,
                )
            if self.apply_sources_after_momentum:
                wp.launch(
                    kernel=device_loop_apply_sources_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.rainfall_rate, self.rainfall_spatial_multiplier, self.other_source_rate,
                            self.infiltration_capacity, self.infiltration_f0, self.infiltration_fd, self.infiltration_k, self.green_ampt_ks, self.green_ampt_psi, self.green_ampt_delta_theta, self.cumulative_infiltration, self.infiltration_method,
                            self._source_subgrid_infiltration_enabled, self._source_subgrid_cell_z,
                            self._source_subgrid_cell_breakpoints, self._source_subgrid_cell_count,
                            self.active_mask, control, use_hyetograph,
                            self.source_mode, self.dry_depth, qx_next, qy_next, 1,
                            out_rain, out_other, out_infil,
                            self.dx, self.dy, self.cumulative_totals,
                            accumulate_accounting, accumulate_output,
                            self.additional_source_rate, self.infiltration_exclusion_mask,
                            self.infiltration_cap_active, self.infiltration_cap_remaining_m],
                    device=self.device,
                )
            if checkpoint_count:
                wp.launch(
                    kernel=device_loop_continuity_checkpoint_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, qx_next, qy_next, self.active_mask,
                            self.dx, self.dy, self.negative_depth_tolerance, control,
                            self.boundary_type_qx, self.boundary_type_qy, self.cumulative_totals,
                            dt_history, history_capacity, record_history,
                            checkpoint_outlet_rows_wp, checkpoint_field_indices_wp,
                            checkpoint_times_out, checkpoint_outlet_out,
                            checkpoint_volume_out, checkpoint_depth_out,
                            checkpoint_count],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=device_loop_continuity_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, qx_next, qy_next, self.active_mask,
                            self.dx, self.dy, self.negative_depth_tolerance, control,
                            self.boundary_type_qx, self.boundary_type_qy, self.cumulative_totals,
                            dt_history, history_capacity, record_history],
                    device=self.device,
                )
            if self.track_cumulative_face_volumes and self.track_diagnostic_max_depth:
                wp.launch(
                    kernel=accumulate_max_active_depth_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.active_mask, self._diagnostic_max_depth_m],
                    device=self.device,
                )
                # Per-step ACC max-depth reduction over the just-accepted
            if self.track_max_water_depth:
                wp.launch(
                    kernel=accumulate_max_active_depth_kernel,
                    dim=(self.ny, self.nx),
                    inputs=[self.water_depth, self.active_mask, self.max_water_depth_m],
                    device=self.device,
                )
            if use_acc:
                # depths: contention-free partials plus a single-thread fold
                # (order-exact FP64 max, bit-identical to a per-cell atomic
                # reduction).
                wp.launch(
                    kernel=device_loop_max_depth_partial_kernel,
                    dim=(max_depth_reduce_threads,),
                    inputs=[self.water_depth, self.active_mask,
                            max_depth_reduce_threads, control, max_depth_partials],
                    device=self.device,
                )
                wp.launch(
                    kernel=device_loop_max_depth_finalize_kernel,
                    dim=(max_depth_fold_threads,),
                    inputs=[max_depth_partials, max_depth_fold_threads, control],
                    device=self.device,
                )
            # The queued kernels retain their explicit array arguments. Swap
            # handles so the next queued step reads this completed q state.
            self._swap_q_buffers(qx_next=qx_next, qy_next=qy_next)

        graph = None
        graph_capture_seconds = 0.0
        if use_cuda_graph:
            capture_started = time.perf_counter()
            if USE_CONDITIONED_MIXED_KERNELS:
                # SGS is unsupported in mixed precision. Global eager loading
                # also compiles its imported modules, even for a coarse run.
                # Preload only the kernels used by this supported graph path.
                wp.load_module(module="floodwarp.local_inertial_kernels", device=self.device)
            condition = wp.ones(1, dtype=wp.int32, device=self.device)
            progress_budget = wp.zeros(1, dtype=wp.float64, device=self.device)
            # Two explicit buffer swaps return Python handles to their initial
            # identities. The last odd accepted step is repaired after replay.
            # Capture contains the very same source/CFL/momentum/limiter kernels.
            def capture_body():
                enqueue_step()
                enqueue_step()
                wp.launch(device_loop_graph_condition_kernel, dim=1,
                    inputs=[control, condition, wp.float64(end_time), progress_budget], device=self.device)
            if (self.hydraulic_geometry_mode=='enhanced' and self._component_timestep in ('provisional','adaptive')):
                # A conditional graph cannot be inserted as a child graph.
                # Capture the outer while body inline so the source-CFL while
                # nodes are genuinely nested, rather than conditional children.
                with wp.ScopedCapture(device=self.device,force_module_load=True) as loop:
                    wp.capture_while(condition,while_body=capture_body)
            else:
                with wp.ScopedCapture(device=self.device, force_module_load=not USE_CONDITIONED_MIXED_KERNELS) as body:
                    capture_body()
                with wp.ScopedCapture(device=self.device, force_module_load=False if USE_CONDITIONED_MIXED_KERNELS else None) as loop:
                    wp.capture_while(condition, while_body=body.graph)
            graph = loop.graph
            graph_capture_seconds = time.perf_counter()-capture_started

        steps_done = 0
        sync_count = 0
        loop_started = time.perf_counter()
        last_progress = None
        graph_steps = max(2, 2*((sync_interval+1)//2))
        while True:
            batch_started = time.perf_counter()
            if use_cuda_graph:
                if on_progress is not None:
                    progress_budget.fill_(float(steps_done+graph_steps))
                    wp.launch(device_loop_graph_condition_kernel, dim=1,
                        inputs=[control, condition, wp.float64(end_time), progress_budget], device=self.device)
                wp.capture_launch(graph)
            else:
                for _ in range(sync_interval):
                    enqueue_step()
            ctrl = control.numpy()  # the single blocking sync per window
            sync_count += 1
            new_steps = int(ctrl[DL_COUNTER]) - steps_done
            steps_done = int(ctrl[DL_COUNTER])
            # A final sync window may contain no-op kernels after DL_DONE.
            # Their host-side buffer-handle swaps must not select an old q
            # buffer for saved diagnostics. The kernels themselves return
            # before modifying state; restore the parity of accepted steps.
            queued_no_op_steps = (new_steps % 2 if use_cuda_graph else sync_interval - new_steps)
            if queued_no_op_steps % 2 == 1 and self.hydraulic_geometry_mode != 'enhanced':
                self._swap_q_buffers(qx_next=self.qx_next, qy_next=self.qy_next)
            if defer_sources:
                self._steps_since_mass_accounting += new_steps
            self._device_loop_dt_min_s = float(ctrl[DL_DT_MIN]) if steps_done else 0.0
            self._device_loop_dt_max_s = float(ctrl[DL_DT_MAX]) if steps_done else 0.0
            self._device_loop_dt_sum_s = float(ctrl[DL_DT_SUM])
            self.simulation_time = float(ctrl[DL_TIME])
            last_accepted_dt = float(ctrl[DL_LAST_ACCEPTED_DT])
            self.current_dt = (
                last_accepted_dt
                if last_accepted_dt > 0.0
                else float(ctrl[DL_DT])
            )
            self._adaptive_dt_reference = float(ctrl[DL_GROWTH_REFERENCE])
            # Progress reads only the small control record. Keep deferred
            # ledgers/coarse fields untouched until completion or error, so
            # reporting does not change reduction order or reconstruct grids.
            finished = ctrl[DL_DONE] != 0.0 or self.simulation_time >= end_time
            if not use_cuda_graph or on_progress is None or finished or ctrl[DL_ERROR] != 0.0:
                self._flush_pending_mass_accounting()
            if collect_free_outflow_diagnostics:
                trace_row = self.free_outflow_limiter_diagnostics()
                trace_row["time_s"] = self.simulation_time
                trace_row["steps"] = float(steps_done)
                free_outflow_trace.append(trace_row)
            if self.hydraulic_geometry_mode == 'enhanced':
                try:
                    self._check_component_status()
                except BaseException:
                    self._flush_pending_mass_accounting()
                    raise
            error = ctrl[DL_ERROR]
            if error == 1.0:
                raise RuntimeError(
                    "Significant negative depth encountered during device-loop run "
                    f"(detected at the sync after t={self.simulation_time:.12g} s; the "
                    f"flagged step is at most {sync_interval} steps earlier)."
                )
            if error == 2.0:
                raise RuntimeError(
                    f"Calculated stable time step ({self.current_dt}) is below "
                    f"minimum_dt ({self.minimum_dt}) during device-loop run."
                )
            if on_sync is not None:
                on_sync(steps_done, self.simulation_time)
            finished = ctrl[DL_DONE] != 0.0 or self.simulation_time >= end_time
            now = time.perf_counter()
            if on_progress is not None:
                if last_progress is None or now-last_progress >= progress_interval_s or finished:
                    try:
                        on_progress(steps_done, self.simulation_time,
                            "conditional_cuda" if use_cuda_graph else "queued_windows")
                    except BaseException:
                        self._flush_pending_mass_accounting()
                        raise
                    last_progress = now
                if use_cuda_graph and new_steps > 0:
                    # Scheduling observation work only; the hydraulic end time,
                    # forcing nodes and accepted-step controller are unchanged.
                    estimate = int(new_steps*progress_interval_s/max(now-batch_started, 1.e-9))
                    graph_steps = max(2, min(1_000_000, 2*((estimate+1)//2)))
            if finished:
                break
        loop_seconds = time.perf_counter() - loop_started

        # The final readback phase (small control reads plus the batched
        # checkpoint trace transfer) is timed separately so ensemble
        # throughput studies can attribute wall time to stepping vs readback.
        readback_started = time.perf_counter()
        if self.use_lisflood_acc_timestep:
            final_max_depth = float(control.numpy()[DL_MAX_DEPTH])
            self._lisflood_cached_maximum_depth_m = final_max_depth
            self.last_timestep_max_depth_m = final_max_depth

        checkpoint_trace = None
        if checkpoint_count:
            trace_times = checkpoint_times_out.numpy()[:checkpoint_count]
            if not np.all(np.isfinite(trace_times)) or not np.all(
                np.abs(trace_times - checkpoint_times) <= 1.0e-9
            ):
                raise RuntimeError(
                    "device-loop checkpoint trace is incomplete or landed away "
                    "from a requested checkpoint time"
                )
            checkpoint_trace = {
                "times_s": trace_times,
                "outlet_m3_per_s": (
                    np.sum(
                        checkpoint_outlet_out.numpy()[:checkpoint_count],
                        axis=1,
                        dtype=np.float64,
                    )
                    * float(self.dy)
                ),
                "volume_m3": checkpoint_volume_out.numpy()[:checkpoint_count],
                "field_checkpoint_indices": field_indices.copy(),
                "field_depths": (
                    checkpoint_depth_out.numpy()[:field_indices.size]
                    if field_indices.size
                    else None
                ),
            }
        checkpoint_readback_seconds = time.perf_counter() - readback_started
        return {
            "steps": steps_done,
            "simulation_time": self.simulation_time,
            "sync_count": sync_count,
            "loop_seconds": loop_seconds,
            "graph_capture_seconds": graph_capture_seconds,
            "graph_capture_count": int(use_cuda_graph),
            "execution": "conditional_cuda" if use_cuda_graph else "queued_windows",
            "dt_min_s": float(ctrl[DL_DT_MIN]) if steps_done else 0.0,
            "dt_mean_s": float(ctrl[DL_DT_SUM] / float(steps_done)) if steps_done else 0.0,
            "dt_max_s": float(ctrl[DL_DT_MAX]) if steps_done else 0.0,
            "checkpoint_readback_seconds": checkpoint_readback_seconds,
            "dt_history": dt_history.numpy()[:steps_done] if record_history else None,
            "checkpoint_trace": checkpoint_trace,
            "free_outflow_limiter": (
                self.free_outflow_limiter_diagnostics()
                if collect_free_outflow_diagnostics
                else None
            ),
            "free_outflow_limiter_trace": free_outflow_trace if collect_free_outflow_diagnostics else None,
        }

    def _ensure_lts_buffers(self, max_levels: int) -> None:
        """Allocate persistent buffers for conservative spatial local timestepping."""
        if max_levels < 0:
            raise ValueError("max_levels must be nonnegative")
        required_capacity = max_levels + 1
        if self._lts_capacity_levels >= required_capacity:
            return

        self._lts_capacity_levels = required_capacity
        self._lts_face_dt_qx = wp.zeros(
            (self.ny, self.nx + 1), dtype=WP_FLOAT, device=self.device
        )
        self._lts_face_dt_qy = wp.zeros(
            (self.ny + 1, self.nx), dtype=WP_FLOAT, device=self.device
        )
        self._lts_cell_levels = wp.zeros(
            (self.ny, self.nx), dtype=wp.int32, device=self.device
        )
        self._lts_cell_levels_tmp = wp.zeros(
            (self.ny, self.nx), dtype=wp.int32, device=self.device
        )
        self._lts_qx_levels = wp.zeros(
            (self.ny, self.nx + 1), dtype=wp.int32, device=self.device
        )
        self._lts_qy_levels = wp.zeros(
            (self.ny + 1, self.nx), dtype=wp.int32, device=self.device
        )
        self._lts_limiter_levels = wp.zeros(
            (self.ny, self.nx), dtype=wp.int32, device=self.device
        )
        self._lts_pending_flux_volume = wp.zeros(
            (self.ny, self.nx), dtype=wp.float64, device=self.device
        )

        # One compact index array per entity type. Each array is partitioned by
        # level using host-computed offsets from a tiny device count vector.
        self._lts_cell_indices = wp.zeros(
            self.ny * self.nx, dtype=wp.int32, device=self.device
        )
        self._lts_limiter_indices = wp.zeros(
            self.ny * self.nx, dtype=wp.int32, device=self.device
        )
        self._lts_qx_indices = wp.zeros(
            self.ny * (self.nx + 1), dtype=wp.int32, device=self.device
        )
        self._lts_qy_indices = wp.zeros(
            (self.ny + 1) * self.nx, dtype=wp.int32, device=self.device
        )

        self._lts_counts = wp.zeros(
            4 * required_capacity, dtype=wp.int32, device=self.device
        )
        self._lts_cell_write = wp.zeros(
            required_capacity, dtype=wp.int32, device=self.device
        )
        self._lts_limiter_write = wp.zeros(
            required_capacity, dtype=wp.int32, device=self.device
        )
        self._lts_qx_write = wp.zeros(
            required_capacity, dtype=wp.int32, device=self.device
        )
        self._lts_qy_write = wp.zeros(
            required_capacity, dtype=wp.int32, device=self.device
        )
        self._lts_schedule_control = wp.zeros(
            LTS_CTRL_SIZE, dtype=wp.float64, device=self.device
        )
        self._lts_schedule_snapshot = wp.zeros(
            LTS_CTRL_SIZE + 1 + 4 * required_capacity,
            dtype=wp.float64,
            device=self.device,
        )
        # Accuracy-control history: -1 marks "no previous level", which makes
        # the first classification after allocation/reset accept the fresh
        # CFL/accuracy target without hysteresis.
        self._lts_prev_cell_levels = wp.full(
            (self.ny, self.nx), -1, dtype=wp.int32, device=self.device
        )
        self._lts_prev_effective_depth = wp.zeros(
            (self.ny, self.nx), dtype=WP_FLOAT, device=self.device
        )
        self._lts_classify_stats = wp.zeros(
            LTS_STATS_SIZE, dtype=wp.int64, device=self.device
        )
        self._lts_last_classify_time = None
        if self._lts_graph_tick is None:
            self._lts_graph_tick = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._lts_graph_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._lts_graph_forcing = wp.zeros(2, dtype=wp.float64, device=self.device)
        # Level/index arrays above were reallocated, so any previously captured
        # CUDA graph would hold stale array pointers and must be rebuilt.
        self._lts_subcycle_graph = None
        self._lts_subcycle_graph_shape = None

    @staticmethod
    def _lts_offsets(counts: np.ndarray, capacity: int) -> np.ndarray:
        offsets = np.zeros(capacity, dtype=np.int32)
        if counts.size > 1:
            offsets[1:counts.size] = np.cumsum(counts[:-1], dtype=np.int64).astype(np.int32)
        return offsets

    @staticmethod
    def _lts_accuracy_summary(
        stats: np.ndarray,
        *,
        accuracy_fraction: float,
        accuracy_depth_scale: float,
        promotion_margin: float,
    ) -> dict[str, object]:
        """Summarise run-accumulated accuracy-classification statistics.

        ``stats`` is the device counter block accumulated since the last depth
        reset (or buffer allocation).  Indicator quantiles come from the
        fixed log-spaced histogram, so no full-grid host copy is ever needed.
        """
        scalars = stats[:LTS_STATS_SCALARS]
        histogram = stats[LTS_STATS_SCALARS:LTS_STATS_SIZE]
        active = int(scalars[LTS_STATS_ACTIVE])

        def limited_fraction(value: int) -> float:
            return float(value) / float(active) if active > 0 else 0.0

        samples = int(histogram.sum())
        quantiles: dict[str, float | None] = {}
        cumulative = np.cumsum(histogram)
        for label, fraction in (
            ("p50", 0.50),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
        ):
            value: float | None = None
            if samples > 0:
                bin_index = int(np.searchsorted(cumulative, fraction * samples))
                bin_index = min(bin_index, LTS_STATS_HIST_BINS - 1)
                log_value = LTS_HIST_LOG_MIN + (bin_index + 0.5) / LTS_HIST_BINS_PER_DEX
                value = float(10.0 ** log_value)
            quantiles[label] = value
        indicator_max: float | None = None
        if samples > 0:
            highest_bin = LTS_STATS_HIST_BINS - 1 - int(
                np.argmax(histogram[::-1] > 0)
            )
            indicator_max = float(
                10.0 ** (LTS_HIST_LOG_MIN + (highest_bin + 1) / LTS_HIST_BINS_PER_DEX)
            )

        return {
            "accuracy_fraction": float(accuracy_fraction),
            "accuracy_depth_scale": float(accuracy_depth_scale),
            "promotion_margin": float(promotion_margin),
            "classifications": active,
            "cfl_limited": int(scalars[LTS_STATS_CFL_LIMITED]),
            "accuracy_limited": int(scalars[LTS_STATS_ACCURACY_LIMITED]),
            "wet_guard_limited": int(scalars[LTS_STATS_WET_GUARD_LIMITED]),
            "front_guard_limited": int(scalars[LTS_STATS_FRONT_GUARD_LIMITED]),
            "drainage_limited": int(scalars[LTS_STATS_DRAINAGE_LIMITED]),
            "cfl_limited_fraction": limited_fraction(
                int(scalars[LTS_STATS_CFL_LIMITED])
            ),
            "accuracy_limited_fraction": limited_fraction(
                int(scalars[LTS_STATS_ACCURACY_LIMITED])
            ),
            "wet_guard_limited_fraction": limited_fraction(
                int(scalars[LTS_STATS_WET_GUARD_LIMITED])
            ),
            "front_guard_limited_fraction": limited_fraction(
                int(scalars[LTS_STATS_FRONT_GUARD_LIMITED])
            ),
            "drainage_limited_fraction": limited_fraction(
                int(scalars[LTS_STATS_DRAINAGE_LIMITED])
            ),
            "promotions": int(scalars[LTS_STATS_PROMOTIONS]),
            "demotions": int(scalars[LTS_STATS_DEMOTIONS]),
            "promotion_holds": int(scalars[LTS_STATS_PROMOTION_HELD]),
            "indicator_relative_change": {
                "samples": samples,
                "p50": quantiles["p50"],
                "p90": quantiles["p90"],
                "p95": quantiles["p95"],
                "p99": quantiles["p99"],
                "max": indicator_max,
            },
        }

    @staticmethod
    def _lts_hyetograph_rate(hyetograph, time_s: float) -> float:
        if hyetograph is None:
            return 0.0
        if time_s < 0.0 or time_s > hyetograph.duration_s:
            return 0.0
        return hyetograph.rate_at(time_s=time_s)

    @classmethod
    def _lts_hyetograph_max_rate(cls, hyetograph, start_s: float, end_s: float) -> float:
        if hyetograph is None:
            return 0.0
        return max(
            cls._lts_hyetograph_rate(hyetograph, start_s),
            cls._lts_hyetograph_rate(hyetograph, end_s),
        )

    @staticmethod
    def _lts_hyetograph_mean_rate(hyetograph, start_s: float, end_s: float) -> float:
        if hyetograph is None:
            return 0.0
        return hyetograph.mean_rate_between(
            start_time_s=start_s,
            end_time_s=end_s,
        )

    def _prepare_lts_schedule(
        self,
        cap_dt: float,
        rainfall_hyetograph,
        max_levels: int,
        wet_max_level: int,
        accuracy_fraction: float,
        accuracy_depth_scale: float,
        promotion_margin: float,
        guard_depth: float,
        drainage_max_level: int,
    ) -> dict[str, object]:
        """Build one compact power-of-two LTS schedule with device-side control.

        Base-dt finalisation, level assignment, wet-front guarding, short prefix
        sums, and compact-write offsets remain on the GPU.  A single tiny
        snapshot is copied to the host at the end because Warp launch dimensions
        for the compact per-level kernels are still host scalars.  This replaces
        the previous dt-min read plus count-vector read and removes four host to
        device offset uploads per macrostep.
        """
        self._ensure_lts_buffers(max_levels=max_levels)
        self._ensure_conditioned_bed_fields()

        start_time = float(self.simulation_time)
        rain_max = self._lts_hyetograph_max_rate(
            rainfall_hyetograph,
            start_time,
            start_time + cap_dt,
        )
        wp.launch(
            kernel=lts_predict_depth_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self.water_depth,
                self.rainfall_rate,
                self.rainfall_spatial_multiplier,
                self.other_source_rate,
                self.additional_source_rate,
                self.active_mask,
                WP_FLOAT(cap_dt),
                WP_FLOAT(rain_max),
                1 if rainfall_hyetograph is not None else 0,
                self.cfl_depth,
            ],
            device=self.device,
        )

        self.dt_min_arr.fill_(WP_FLOAT(cap_dt))
        if self.hydraulic_geometry_mode == 'enhanced':
            self._enhanced_lts_face_timesteps(cap_dt=cap_dt,rain_max=rain_max,
                use_hyetograph=rainfall_hyetograph is not None)
        elif self.is_subgrid():
            # Face-dt selection uses the source-predicted storage in
            # cfl_depth; invert it once for the subgrid face kernels.
            self._launch_stage_inversion(self.cfl_depth, self._eta_cfl)
            wp.launch(
                kernel=subgrid_lts_face_dt_x_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    self._eta_cfl, self._subgrid_face_z_min_x,
                    self.boundary_type_qx, self.boundary_stage_qx,
                    self.dx, self.gravity, self.dry_depth, self.cfl_number,
                    WP_FLOAT(cap_dt), self._lts_face_dt_qx, self.dt_min_arr,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=subgrid_lts_face_dt_y_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    self._eta_cfl, self._subgrid_face_z_min_y,
                    self.boundary_type_qy, self.boundary_stage_qy,
                    self.dy, self.gravity, self.dry_depth, self.cfl_number,
                    WP_FLOAT(cap_dt), self._lts_face_dt_qy, self.dt_min_arr,
                ],
                device=self.device,
            )
        elif USE_CONDITIONED_MIXED_KERNELS:
            wp.launch(
                kernel=lts_conditioned_face_dt_x_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    self.cfl_depth,
                    self.bed_slope_qx,
                    self.boundary_type_qx,
                    self.boundary_stage_depth_qx,
                    self.dx,
                    self.gravity,
                    self.dry_depth,
                    self.cfl_number,
                    WP_FLOAT(cap_dt),
                    self._lts_face_dt_qx,
                    self.dt_min_arr,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=lts_conditioned_face_dt_y_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    self.cfl_depth,
                    self.bed_slope_qy,
                    self.boundary_type_qy,
                    self.boundary_stage_depth_qy,
                    self.dy,
                    self.gravity,
                    self.dry_depth,
                    self.cfl_number,
                    WP_FLOAT(cap_dt),
                    self._lts_face_dt_qy,
                    self.dt_min_arr,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=lts_face_dt_x_kernel,
                dim=(self.ny, self.nx + 1),
                inputs=[
                    self.cfl_depth,
                    self.bed_elevation,
                    self.boundary_type_qx,
                    self.boundary_stage_qx,
                    self.dx,
                    self.gravity,
                    self.dry_depth,
                    self.cfl_number,
                    WP_FLOAT(cap_dt),
                    self._lts_face_dt_qx,
                    self.dt_min_arr,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=lts_face_dt_y_kernel,
                dim=(self.ny + 1, self.nx),
                inputs=[
                    self.cfl_depth,
                    self.bed_elevation,
                    self.boundary_type_qy,
                    self.boundary_stage_qy,
                    self.dy,
                    self.gravity,
                    self.dry_depth,
                    self.cfl_number,
                    WP_FLOAT(cap_dt),
                    self._lts_face_dt_qy,
                    self.dt_min_arr,
                ],
                device=self.device,
            )

        if self.hydraulic_geometry_mode == 'enhanced':
            self._enhanced_lts_finalize(cap_dt=cap_dt,max_levels=max_levels)
        else:
            wp.launch(kernel=lts_finalize_schedule_control_kernel,dim=1,
                inputs=[self.dt_min_arr,wp.float64(cap_dt),wp.float64(self.minimum_dt),
                    max_levels,self._lts_schedule_control],device=self.device)
        if self._lts_last_classify_time is None:
            sample_interval = -1.0
        else:
            sample_interval = start_time - self._lts_last_classify_time
        wp.launch(
            kernel=lts_classify_cell_levels_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self._lts_face_dt_qx,
                self._lts_face_dt_qy,
                self.water_depth,
                self._lts_pending_flux_volume,
                self.cfl_depth,
                self.active_mask,
                self._lts_prev_cell_levels,
                self._lts_prev_effective_depth,
                self._lts_schedule_control,
                WP_FLOAT(sample_interval),
                WP_FLOAT(accuracy_fraction),
                WP_FLOAT(accuracy_depth_scale),
                WP_FLOAT(promotion_margin),
                WP_FLOAT(guard_depth),
                wet_max_level,
                drainage_max_level,
                WP_FLOAT(float(self.dx) * float(self.dy)),
                self._lts_cell_levels,
                self._lts_classify_stats,
            ],
            device=self.device,
        )

        # Scale guard: the 1000x1000 test showed that allowing shallow active
        # cells to occupy very coarse levels can preserve wet extent while
        # badly distorting depth/volume.  Truly dry/quiescent cells may still
        # use every configured level.
        if wet_max_level >= 0:
            wp.launch(
                kernel=lts_wet_level_guard_kernel,
                dim=(self.ny, self.nx),
                inputs=[
                    self._lts_cell_levels,
                    self.cfl_depth,
                    self.active_mask,
                    WP_FLOAT(guard_depth),
                    wet_max_level,
                    self._lts_classify_stats,
                ],
                device=self.device,
            )

        # The actual macro level is device-resident, so launch the configured
        # maximum number of inexpensive relaxation passes. Extra passes simply
        # leave an already 2:1-smoothed field unchanged.
        levels_current = self._lts_cell_levels
        levels_next = self._lts_cell_levels_tmp
        for _ in range(max_levels):
            wp.launch(
                kernel=lts_smooth_cell_levels_kernel,
                dim=(self.ny, self.nx),
                inputs=[levels_current, self.active_mask, levels_next],
                device=self.device,
            )
            levels_current, levels_next = levels_next, levels_current
        if levels_current is not self._lts_cell_levels:
            self._lts_cell_levels, self._lts_cell_levels_tmp = levels_current, levels_next

        wp.launch(
            kernel=lts_face_levels_x_kernel,
            dim=(self.ny, self.nx + 1),
            inputs=[self._lts_cell_levels, self.boundary_type_qx, self._lts_qx_levels],
            device=self.device,
        )
        wp.launch(
            kernel=lts_face_levels_y_kernel,
            dim=(self.ny + 1, self.nx),
            inputs=[self._lts_cell_levels, self.boundary_type_qy, self._lts_qy_levels],
            device=self.device,
        )
        wp.launch(
            kernel=lts_limiter_level_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self._lts_qx_levels,
                self._lts_qy_levels,
                self.active_mask,
                self._lts_limiter_levels,
            ],
            device=self.device,
        )

        capacity = self._lts_capacity_levels
        self._lts_counts.zero_()
        wp.launch(
            kernel=lts_count_cell_levels_kernel,
            dim=(self.ny, self.nx),
            inputs=[self._lts_cell_levels, self._lts_counts, 0],
            device=self.device,
        )
        wp.launch(
            kernel=lts_count_cell_levels_kernel,
            dim=(self.ny, self.nx),
            inputs=[self._lts_limiter_levels, self._lts_counts, capacity],
            device=self.device,
        )
        wp.launch(
            kernel=lts_count_face_levels_kernel,
            dim=(self.ny, self.nx + 1),
            inputs=[self._lts_qx_levels, self._lts_counts, 2 * capacity],
            device=self.device,
        )
        wp.launch(
            kernel=lts_count_face_levels_kernel,
            dim=(self.ny + 1, self.nx),
            inputs=[self._lts_qy_levels, self._lts_counts, 3 * capacity],
            device=self.device,
        )
        wp.launch(
            kernel=lts_prepare_level_offsets_kernel,
            dim=1,
            inputs=[
                self._lts_counts,
                capacity,
                self._lts_cell_write,
                self._lts_limiter_write,
                self._lts_qx_write,
                self._lts_qy_write,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=lts_compact_cell_levels_kernel,
            dim=(self.ny, self.nx),
            inputs=[self._lts_cell_levels, self._lts_cell_write, self._lts_cell_indices],
            device=self.device,
        )
        wp.launch(
            kernel=lts_compact_cell_levels_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self._lts_limiter_levels,
                self._lts_limiter_write,
                self._lts_limiter_indices,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=lts_compact_face_levels_kernel,
            dim=(self.ny, self.nx + 1),
            inputs=[self._lts_qx_levels, self._lts_qx_write, self._lts_qx_indices],
            device=self.device,
        )
        wp.launch(
            kernel=lts_compact_face_levels_kernel,
            dim=(self.ny + 1, self.nx),
            inputs=[self._lts_qy_levels, self._lts_qy_write, self._lts_qy_indices],
            device=self.device,
        )

        snapshot_size = LTS_CTRL_SIZE + 1 + 4 * capacity
        wp.launch(
            kernel=lts_schedule_snapshot_kernel,
            dim=snapshot_size,
            inputs=[
                self._lts_schedule_control,
                self._lts_counts,
                self.error_flag,
                self._lts_schedule_snapshot,
            ],
            device=self.device,
        )
        snapshot = np.asarray(self._lts_schedule_snapshot.numpy(), dtype=np.float64)

        schedule_error = int(round(snapshot[LTS_CTRL_ERROR]))
        if schedule_error == 1:
            raise RuntimeError("Invalid LTS base timestep selected on device.")
        if schedule_error == 2:
            raise RuntimeError(
                f"Calculated LTS base timestep is below minimum_dt ({self.minimum_dt})."
            )
        if int(round(snapshot[LTS_CTRL_SIZE])) != 0:
            raise RuntimeError(
                "Significant negative depth encountered during the previous LTS macrostep."
            )

        # The classification at this macrostep start is now the reference for
        # the next change-rate estimate.
        self._lts_last_classify_time = start_time

        base_dt = float(snapshot[LTS_CTRL_BASE_DT])
        macro_dt = float(snapshot[LTS_CTRL_MACRO_DT])
        macro_level = int(round(snapshot[LTS_CTRL_MACRO_LEVEL]))
        macro_ticks = int(round(snapshot[LTS_CTRL_MACRO_TICKS]))
        nlevels = macro_level + 1

        counts_start = LTS_CTRL_SIZE + 1
        all_counts = np.rint(
            snapshot[counts_start:counts_start + 4 * capacity]
        ).astype(np.int64)
        cell_counts = all_counts[0:nlevels]
        limiter_counts = all_counts[capacity:capacity + nlevels]
        qx_counts = all_counts[2 * capacity:2 * capacity + nlevels]
        qy_counts = all_counts[3 * capacity:3 * capacity + nlevels]

        return {
            "base_dt": base_dt,
            "macro_dt": macro_dt,
            "macro_level": macro_level,
            "macro_ticks": macro_ticks,
            "cell_counts": cell_counts,
            "limiter_counts": limiter_counts,
            "qx_counts": qx_counts,
            "qy_counts": qy_counts,
            "cell_offsets": self._lts_offsets(cell_counts, capacity),
            "limiter_offsets": self._lts_offsets(limiter_counts, capacity),
            "qx_offsets": self._lts_offsets(qx_counts, capacity),
            "qy_offsets": self._lts_offsets(qy_counts, capacity),
        }

    def _lts_launch_sources_all(
        self,
        *,
        schedule: dict[str, object],
        interval_start: float,
        interval_end: float,
        rainfall_hyetograph,
    ) -> None:
        """Advance sources for every active cell on the LTS base clock.

        Hydraulic momentum and continuity are multirate, but rainfall, other
        local sources, and state-dependent infiltration are not.  Keeping the
        source clock at ``base_dt`` preserves the source discretisation used by
        the globally stepped explicit solver while still avoiding most coarse
        hydraulic work.

        Hyetograph sampling matches the global device-loop convention: when
        sources follow momentum (parity profile) the rate is sampled at the
        interval start; when sources lead, the interval mean (exactly the
        midpoint for the piecewise-linear hyetograph) matches the half-step
        sampling of the sources-leading global path.
        """
        count = int(np.sum(schedule["cell_counts"]))
        if count == 0:
            return
        dt = interval_end - interval_start
        if self.apply_sources_after_momentum:
            rain_mean = self._lts_hyetograph_rate(rainfall_hyetograph, interval_start)
        else:
            rain_mean = self._lts_hyetograph_mean_rate(
                rainfall_hyetograph,
                interval_start,
                interval_end,
            )
        wp.launch(
            kernel=lts_apply_sources_indexed_kernel,
            dim=count,
            inputs=[
                self._lts_cell_indices,
                0,
                self.water_depth,
                self._lts_pending_flux_volume,
                self.cfl_depth,
                self.rainfall_rate,
                self.rainfall_spatial_multiplier,
                self.other_source_rate,
                self.infiltration_capacity,
                self.infiltration_f0,
                self.infiltration_fd,
                self.infiltration_k,
                self.green_ampt_ks,
                self.green_ampt_psi,
                self.green_ampt_delta_theta,
                self.cumulative_infiltration,
                self.infiltration_method,
                self._source_subgrid_infiltration_enabled,
                self._source_subgrid_cell_z,
                self._source_subgrid_cell_breakpoints,
                self._source_subgrid_cell_count,
                self.active_mask,
                WP_FLOAT(dt),
                WP_FLOAT(rain_mean),
                1 if rainfall_hyetograph is not None else 0,
                self.source_mode,
                self.dry_depth,
                self.applied_rain,
                self.applied_other,
                self.applied_infil,
                self.dx,
                self.dy,
                self.negative_depth_tolerance,
                self.error_flag,
            ],
            device=self.device,
        )

    def _lts_refresh_stage_from_effective_depth(self) -> None:
        """Re-derive the subgrid stage from the source-updated effective state.

        Inside an LTS macrostep the coarse momentum reads ``cfl_depth``, which
        the base-clock source kernels keep at ``water_depth + pending source
        depth``.  The subgrid kernels read the derived stage of that same
        effective state, so the inversion is refreshed right after each
        sources launch.
        """
        if self.is_subgrid():
            self._launch_stage_inversion(self.cfl_depth, self._eta_stage)

    def _lts_launch_momentum(self, schedule: dict[str, object], level: int, dt: float) -> None:
        qx_count = int(schedule["qx_counts"][level])
        qy_count = int(schedule["qy_counts"][level])
        if self.is_subgrid():
            if qx_count > 0:
                wp.launch(
                    kernel=lts_update_subgrid_qx_indexed_kernel,
                    dim=qx_count,
                    inputs=[
                        self._lts_qx_indices,
                        int(schedule["qx_offsets"][level]),
                        self.qx_next,
                        self.qx,
                        self.qy,
                        self._eta_stage,
                        self._subgrid_face_z_x,
                        self._subgrid_face_z_min_x, self._subgrid_face_n_x,
                        self.manning_n,
                        self.boundary_type_qx,
                        self.boundary_stage_qx,
                        self.dx,
                        WP_FLOAT(dt),
                        self.gravity,
                        self.dry_depth,
                        self.depth_epsilon,
                        self.manning_face_mode,
                        self.use_two_dimensional_friction,
                        self._use_native_face_manning,
                        self._subgrid_factor,
                    ],
                    device=self.device,
                )
            if qy_count > 0:
                wp.launch(
                    kernel=lts_update_subgrid_qy_indexed_kernel,
                    dim=qy_count,
                    inputs=[
                        self._lts_qy_indices,
                        int(schedule["qy_offsets"][level]),
                        self.qy_next,
                        self.qy,
                        self.qx,
                        self._eta_stage,
                        self._subgrid_face_z_y,
                        self._subgrid_face_z_min_y, self._subgrid_face_n_y,
                        self.manning_n,
                        self.boundary_type_qy,
                        self.boundary_stage_qy,
                        self.dy,
                        WP_FLOAT(dt),
                        self.gravity,
                        self.dry_depth,
                        self.depth_epsilon,
                        self.manning_face_mode,
                        self.use_two_dimensional_friction,
                        self._use_native_face_manning,
                        self._subgrid_factor,
                    ],
                    device=self.device,
                )
            return
        if USE_CONDITIONED_MIXED_KERNELS:
            if qx_count > 0:
                wp.launch(
                    kernel=lts_update_conditioned_qx_indexed_kernel,
                    dim=qx_count,
                    inputs=[
                        self._lts_qx_indices,
                        int(schedule["qx_offsets"][level]),
                        self.qx_next,
                        self.qx,
                        self.qy,
                        self.cfl_depth,
                        self.bed_slope_qx,
                        self.manning_n,
                        self.boundary_type_qx,
                        self.boundary_stage_depth_qx,
                        self.boundary_free_slope_qx,
                        self.dx,
                        WP_FLOAT(dt),
                        self.gravity,
                        self.dry_depth,
                        self.depth_epsilon,
                        self.manning_face_mode,
                        self.use_two_dimensional_friction,
                        self.free_outflow_mode,
                    ],
                    device=self.device,
                )
            if qy_count > 0:
                wp.launch(
                    kernel=lts_update_conditioned_qy_indexed_kernel,
                    dim=qy_count,
                    inputs=[
                        self._lts_qy_indices,
                        int(schedule["qy_offsets"][level]),
                        self.qy_next,
                        self.qy,
                        self.qx,
                        self.cfl_depth,
                        self.bed_slope_qy,
                        self.manning_n,
                        self.boundary_type_qy,
                        self.boundary_stage_depth_qy,
                        self.boundary_free_slope_qy,
                        self.dy,
                        WP_FLOAT(dt),
                        self.gravity,
                        self.dry_depth,
                        self.depth_epsilon,
                        self.manning_face_mode,
                        self.use_two_dimensional_friction,
                        self.free_outflow_mode,
                    ],
                    device=self.device,
                )
        else:
            if qx_count > 0:
                wp.launch(
                    kernel=lts_update_qx_indexed_kernel,
                    dim=qx_count,
                    inputs=[
                        self._lts_qx_indices,
                        int(schedule["qx_offsets"][level]),
                        self.qx_next,
                        self.qx,
                        self.qy,
                        self.cfl_depth,
                        self.bed_elevation,
                        self.manning_n,
                        self.boundary_type_qx,
                        self.boundary_stage_qx,
                        self.dx,
                        WP_FLOAT(dt),
                        self.gravity,
                        self.dry_depth,
                        self.depth_epsilon,
                        self.manning_face_mode,
                        self.use_two_dimensional_friction,
                        self.free_outflow_mode,
                    ],
                    device=self.device,
                )
            if qy_count > 0:
                wp.launch(
                    kernel=lts_update_qy_indexed_kernel,
                    dim=qy_count,
                    inputs=[
                        self._lts_qy_indices,
                        int(schedule["qy_offsets"][level]),
                        self.qy_next,
                        self.qy,
                        self.qx,
                        self.cfl_depth,
                        self.bed_elevation,
                        self.manning_n,
                        self.boundary_type_qy,
                        self.boundary_stage_qy,
                        self.dy,
                        WP_FLOAT(dt),
                        self.gravity,
                        self.dry_depth,
                        self.depth_epsilon,
                        self.manning_face_mode,
                        self.use_two_dimensional_friction,
                        self.free_outflow_mode,
                    ],
                    device=self.device,
                )

    def _lts_accept_level_momentum(self, schedule: dict[str, object], level: int) -> None:
        """Copy q_next to held q for one compact level."""
        qx_count = int(schedule["qx_counts"][level])
        qy_count = int(schedule["qy_counts"][level])
        if qx_count > 0:
            wp.launch(
                kernel=lts_commit_qx_volume_indexed_kernel,
                dim=qx_count,
                inputs=[
                    self._lts_qx_indices,
                    int(schedule["qx_offsets"][level]),
                    self.qx,
                    self.qx_next,
                    self.boundary_type_qx,
                    self._lts_pending_flux_volume,
                    self.dy,
                    WP_FLOAT(0.0),
                    self.cumulative_totals,
                ],
                device=self.device,
            )
        if qy_count > 0:
            wp.launch(
                kernel=lts_commit_qy_volume_indexed_kernel,
                dim=qy_count,
                inputs=[
                    self._lts_qy_indices,
                    int(schedule["qy_offsets"][level]),
                    self.qy,
                    self.qy_next,
                    self.boundary_type_qy,
                    self._lts_pending_flux_volume,
                    self.dx,
                    WP_FLOAT(0.0),
                    self.cumulative_totals,
                ],
                device=self.device,
            )

    def _lts_limit_held_outflow_all(self, schedule: dict[str, object]) -> None:
        """Apply positivity limiting to held q over one base transport interval."""
        count = int(np.sum(schedule["cell_counts"]))
        if count == 0:
            return
        wp.launch(
            kernel=lts_limit_held_outflow_indexed_kernel,
            dim=count,
            inputs=[
                self._lts_cell_indices,
                self.water_depth,
                self._lts_pending_flux_volume,
                self.qx,
                self.qy,
                self.active_mask,
                self.dx,
                self.dy,
                self._lts_schedule_control,
                self.limiter_epsilon,
            ],
            device=self.device,
        )

    def _lts_transport_held_faces_all(self, schedule: dict[str, object]) -> None:
        """Move held face discharge for exactly one base interval."""
        qx_count = int(np.sum(schedule["qx_counts"]))
        qy_count = int(np.sum(schedule["qy_counts"]))
        if qx_count > 0:
            wp.launch(
                kernel=lts_transport_qx_indexed_kernel,
                dim=qx_count,
                inputs=[
                    self._lts_qx_indices,
                    self.qx,
                    self.boundary_type_qx,
                    self._lts_pending_flux_volume,
                    self.dy,
                    self._lts_schedule_control,
                    self.cumulative_totals,
                ],
                device=self.device,
            )
        if qy_count > 0:
            wp.launch(
                kernel=lts_transport_qy_indexed_kernel,
                dim=qy_count,
                inputs=[
                    self._lts_qy_indices,
                    self.qy,
                    self.boundary_type_qy,
                    self._lts_pending_flux_volume,
                    self.dx,
                    self._lts_schedule_control,
                    self.cumulative_totals,
                ],
                device=self.device,
            )

    def _lts_launch_continuity(self, schedule: dict[str, object], level: int) -> None:
        count = int(schedule["cell_counts"][level])
        if count == 0:
            return
        wp.launch(
            kernel=lts_continuity_indexed_kernel,
            dim=count,
            inputs=[
                self._lts_cell_indices,
                int(schedule["cell_offsets"][level]),
                self.water_depth,
                self._lts_pending_flux_volume,
                self.active_mask,
                self.dx,
                self.dy,
                self.negative_depth_tolerance,
                self.error_flag,
            ],
            device=self.device,
        )

    def _finish_lts_source_accounting(self) -> None:
        """Commit the accepted macrostep source ledger once."""
        self._reduce_source_mass_arrays(self.applied_rain,self.applied_infil,self.applied_other)
        wp.launch(kernel=accumulate_mass_kernel,dim=(1,),
            inputs=[self._mass_rain_sum,self._mass_infil_sum,self._mass_other_sum,
                wp.float64(float(self.dx)*float(self.dy)),self.cumulative_totals],device=self.device)

    def _lts_capture_signature(self):
        """Identity of captured buffers and launch-time numerical constants.

        In-place array updates are read by a graph on replay; replacing a
        buffer or changing a launch scalar requires recapture.
        """
        names=('dx','dy','gravity','dry_depth','depth_epsilon','limiter_epsilon',
            'negative_depth_tolerance','infiltration_method','source_mode',
            'manning_face_mode','use_two_dimensional_friction','free_outflow_mode',
            'free_outflow_limiter_mode','apply_sources_after_momentum',
            '_source_subgrid_infiltration_enabled','_subgrid_factor',
            '_use_native_face_manning','_kinetic_transport','_exact_momentum',
            '_ec_has_boundary_override','track_max_water_depth')
        scalars=tuple((name,str(getattr(self,name,None))) for name in names)
        buffers=tuple(sorted((name,value.ptr,tuple(value.shape),str(value.dtype))
            for name,value in vars(self).items() if isinstance(value,wp.array)))
        return scalars,buffers

    def _lts_build_subcycle_graph(
        self,
        schedule: dict[str, object],
        use_uniform_rain: bool,
    ) -> bool:
        """Capture one fully device-resident LTS macrostep subcycle loop.

        The compact entity arrays are re-ordered each macrostep but their total
        lengths are invariant for a fixed domain/boundary configuration. The
        captured graph therefore launches over the full compact arrays and each
        momentum/continuity kernel reads the current device level field and
        current device tick to decide whether work is due.
        """
        if not str(self.device).startswith("cuda"):
            return False
        try:
            with wp.ScopedDevice(self.device):
                if not wp.is_conditional_graph_supported():
                    return False
        except (AttributeError, RuntimeError):
            return False

        cell_count = int(np.sum(schedule["cell_counts"]))
        qx_count = int(np.sum(schedule["qx_counts"]))
        qy_count = int(np.sum(schedule["qy_counts"]))
        graph_shape = (
            self._lts_capture_signature(),
            cell_count,
            qx_count,
            qy_count,
            bool(USE_CONDITIONED_MIXED_KERNELS),
            bool(self.apply_sources_after_momentum),
            bool(use_uniform_rain),
        )
        if self._lts_subcycle_graph is not None and self._lts_subcycle_graph_shape == graph_shape:
            return True

        if cell_count <= 0:
            return False

        if USE_CONDITIONED_MIXED_KERNELS:
            wp.load_module(module="floodwarp.local_inertial_kernels", device=self.device)
        with wp.ScopedCapture(device=self.device, force_module_load=not USE_CONDITIONED_MIXED_KERNELS) as body_capture:
            if not self.apply_sources_after_momentum:
                wp.launch(
                    kernel=lts_graph_apply_sources_indexed_kernel,
                    dim=cell_count,
                    inputs=[
                        self._lts_cell_indices,
                        self.water_depth,
                        self._lts_pending_flux_volume,
                        self.cfl_depth,
                        self.rainfall_rate,
                        self.rainfall_spatial_multiplier,
                        self.other_source_rate,
                        self.infiltration_capacity,
                        self.infiltration_f0,
                        self.infiltration_fd,
                        self.infiltration_k,
                        self.green_ampt_ks,
                        self.green_ampt_psi,
                        self.green_ampt_delta_theta,
                        self.cumulative_infiltration,
                        self.infiltration_method,
                        self._source_subgrid_infiltration_enabled,
                        self._source_subgrid_cell_z,
                        self._source_subgrid_cell_breakpoints,
                        self._source_subgrid_cell_count,
                        self.active_mask,
                        self._lts_schedule_control,
                        self._lts_graph_tick,
                        self._lts_graph_forcing,
                        1 if use_uniform_rain else 0,
                        self.source_mode,
                        self.dry_depth,
                        self.applied_rain,
                        self.applied_other,
                        self.applied_infil,
                        self.dx,
                        self.dy,
                        self.negative_depth_tolerance,
                        self.error_flag,
                    ],
                    device=self.device,
                )
                # Keep the derived subgrid stage aligned with the
                # source-updated effective depth the momentum kernels read.
                if self.is_subgrid():
                    self._launch_stage_inversion(self.cfl_depth, self._eta_stage)

            if self.is_subgrid():
                if qx_count > 0:
                    wp.launch(
                        kernel=lts_graph_update_subgrid_qx_indexed_kernel,
                        dim=qx_count,
                        inputs=[
                            self._lts_qx_indices,
                            self.qx_next,
                            self.qx,
                            self.qy,
                            self._eta_stage,
                            self._subgrid_face_z_x,
                            self._subgrid_face_z_min_x, self._subgrid_face_n_x,
                            self.manning_n,
                            self.boundary_type_qx,
                            self.boundary_stage_qx,
                            self._lts_qx_levels,
                            self._lts_schedule_control,
                            self._lts_graph_tick,
                            self.dx,
                            self.gravity,
                            self.dry_depth,
                            self.depth_epsilon,
                            self.manning_face_mode,
                            self.use_two_dimensional_friction,
                            self._use_native_face_manning,
                            self._subgrid_factor,
                        ],
                        device=self.device,
                    )
                if qy_count > 0:
                    wp.launch(
                        kernel=lts_graph_update_subgrid_qy_indexed_kernel,
                        dim=qy_count,
                        inputs=[
                            self._lts_qy_indices,
                            self.qy_next,
                            self.qy,
                            self.qx,
                            self._eta_stage,
                            self._subgrid_face_z_y,
                            self._subgrid_face_z_min_y, self._subgrid_face_n_y,
                            self.manning_n,
                            self.boundary_type_qy,
                            self.boundary_stage_qy,
                            self._lts_qy_levels,
                            self._lts_schedule_control,
                            self._lts_graph_tick,
                            self.dy,
                            self.gravity,
                            self.dry_depth,
                            self.depth_epsilon,
                            self.manning_face_mode,
                            self.use_two_dimensional_friction,
                            self._use_native_face_manning,
                            self._subgrid_factor,
                        ],
                        device=self.device,
                    )
            elif USE_CONDITIONED_MIXED_KERNELS:
                if qx_count > 0:
                    wp.launch(
                        kernel=lts_graph_update_conditioned_qx_indexed_kernel,
                        dim=qx_count,
                        inputs=[
                            self._lts_qx_indices,
                            self.qx_next,
                            self.qx,
                            self.qy,
                            self.cfl_depth,
                            self.bed_slope_qx,
                            self.manning_n,
                            self.boundary_type_qx,
                            self.boundary_stage_depth_qx,
                            self.boundary_free_slope_qx,
                            self._lts_qx_levels,
                            self._lts_schedule_control,
                            self._lts_graph_tick,
                            self.dx,
                            self.gravity,
                            self.dry_depth,
                            self.depth_epsilon,
                            self.manning_face_mode,
                            self.use_two_dimensional_friction,
                            self.free_outflow_mode,
                        ],
                        device=self.device,
                    )
                if qy_count > 0:
                    wp.launch(
                        kernel=lts_graph_update_conditioned_qy_indexed_kernel,
                        dim=qy_count,
                        inputs=[
                            self._lts_qy_indices,
                            self.qy_next,
                            self.qy,
                            self.qx,
                            self.cfl_depth,
                            self.bed_slope_qy,
                            self.manning_n,
                            self.boundary_type_qy,
                            self.boundary_stage_depth_qy,
                            self.boundary_free_slope_qy,
                            self._lts_qy_levels,
                            self._lts_schedule_control,
                            self._lts_graph_tick,
                            self.dy,
                            self.gravity,
                            self.dry_depth,
                            self.depth_epsilon,
                            self.manning_face_mode,
                            self.use_two_dimensional_friction,
                            self.free_outflow_mode,
                        ],
                        device=self.device,
                    )
            else:
                if qx_count > 0:
                    wp.launch(
                        kernel=lts_graph_update_qx_indexed_kernel,
                        dim=qx_count,
                        inputs=[
                            self._lts_qx_indices,
                            self.qx_next,
                            self.qx,
                            self.qy,
                            self.cfl_depth,
                            self.bed_elevation,
                            self.manning_n,
                            self.boundary_type_qx,
                            self.boundary_stage_qx,
                            self._lts_qx_levels,
                            self._lts_schedule_control,
                            self._lts_graph_tick,
                            self.dx,
                            self.gravity,
                            self.dry_depth,
                            self.depth_epsilon,
                            self.manning_face_mode,
                            self.use_two_dimensional_friction,
                            self.free_outflow_mode,
                        ],
                        device=self.device,
                    )
                if qy_count > 0:
                    wp.launch(
                        kernel=lts_graph_update_qy_indexed_kernel,
                        dim=qy_count,
                        inputs=[
                            self._lts_qy_indices,
                            self.qy_next,
                            self.qy,
                            self.qx,
                            self.cfl_depth,
                            self.bed_elevation,
                            self.manning_n,
                            self.boundary_type_qy,
                            self.boundary_stage_qy,
                            self._lts_qy_levels,
                            self._lts_schedule_control,
                            self._lts_graph_tick,
                            self.dy,
                            self.gravity,
                            self.dry_depth,
                            self.depth_epsilon,
                            self.manning_face_mode,
                            self.use_two_dimensional_friction,
                            self.free_outflow_mode,
                        ],
                        device=self.device,
                    )

            if qx_count > 0:
                wp.launch(
                    kernel=lts_graph_accept_qx_indexed_kernel,
                    dim=qx_count,
                    inputs=[
                        self._lts_qx_indices,
                        self.qx,
                        self.qx_next,
                        self._lts_qx_levels,
                        self._lts_graph_tick,
                    ],
                    device=self.device,
                )
            if qy_count > 0:
                wp.launch(
                    kernel=lts_graph_accept_qy_indexed_kernel,
                    dim=qy_count,
                    inputs=[
                        self._lts_qy_indices,
                        self.qy,
                        self.qy_next,
                        self._lts_qy_levels,
                        self._lts_graph_tick,
                    ],
                    device=self.device,
                )

            wp.launch(
                kernel=lts_limit_held_outflow_indexed_kernel,
                dim=cell_count,
                inputs=[
                    self._lts_cell_indices,
                    self.water_depth,
                    self._lts_pending_flux_volume,
                    self.qx,
                    self.qy,
                    self.active_mask,
                    self.dx,
                    self.dy,
                    self._lts_schedule_control,
                    self.limiter_epsilon,
                ],
                device=self.device,
            )

            if qx_count > 0:
                wp.launch(
                    kernel=lts_transport_qx_indexed_kernel,
                    dim=qx_count,
                    inputs=[
                        self._lts_qx_indices,
                        self.qx,
                        self.boundary_type_qx,
                        self._lts_pending_flux_volume,
                        self.dy,
                        self._lts_schedule_control,
                        self.cumulative_totals,
                    ],
                    device=self.device,
                )
            if qy_count > 0:
                wp.launch(
                    kernel=lts_transport_qy_indexed_kernel,
                    dim=qy_count,
                    inputs=[
                        self._lts_qy_indices,
                        self.qy,
                        self.boundary_type_qy,
                        self._lts_pending_flux_volume,
                        self.dx,
                        self._lts_schedule_control,
                        self.cumulative_totals,
                    ],
                    device=self.device,
                )

            if self.apply_sources_after_momentum:
                wp.launch(
                    kernel=lts_graph_apply_sources_indexed_kernel,
                    dim=cell_count,
                    inputs=[
                        self._lts_cell_indices,
                        self.water_depth,
                        self._lts_pending_flux_volume,
                        self.cfl_depth,
                        self.rainfall_rate,
                        self.rainfall_spatial_multiplier,
                        self.other_source_rate,
                        self.infiltration_capacity,
                        self.infiltration_f0,
                        self.infiltration_fd,
                        self.infiltration_k,
                        self.green_ampt_ks,
                        self.green_ampt_psi,
                        self.green_ampt_delta_theta,
                        self.cumulative_infiltration,
                        self.infiltration_method,
                        self._source_subgrid_infiltration_enabled,
                        self._source_subgrid_cell_z,
                        self._source_subgrid_cell_breakpoints,
                        self._source_subgrid_cell_count,
                        self.active_mask,
                        self._lts_schedule_control,
                        self._lts_graph_tick,
                        self._lts_graph_forcing,
                        1 if use_uniform_rain else 0,
                        self.source_mode,
                        self.dry_depth,
                        self.applied_rain,
                        self.applied_other,
                        self.applied_infil,
                        self.dx,
                        self.dy,
                        self.negative_depth_tolerance,
                        self.error_flag,
                    ],
                    device=self.device,
                )

            wp.launch(
                kernel=lts_graph_continuity_indexed_kernel,
                dim=cell_count,
                inputs=[
                    self._lts_cell_indices,
                    self.water_depth,
                    self._lts_pending_flux_volume,
                    self._lts_cell_levels,
                    self.active_mask,
                    self._lts_graph_tick,
                    self.dx,
                    self.dy,
                    self.negative_depth_tolerance,
                    self.error_flag,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=lts_graph_advance_tick_kernel,
                dim=1,
                inputs=[self._lts_graph_tick, self._lts_graph_condition],
                device=self.device,
            )

        with wp.ScopedCapture(device=self.device) as loop_capture:
            wp.capture_while(
                self._lts_graph_condition,
                while_body=body_capture.graph,
            )

        self._lts_subcycle_graph = loop_capture.graph
        self._lts_subcycle_graph_shape = graph_shape
        return True

    def _lts_launch_subcycle_graph(
        self,
        schedule: dict[str, object],
        rainfall_hyetograph,
        macro_start: float,
    ) -> None:
        """Launch one captured device-side macrostep subcycle tree."""
        macro_dt = float(schedule["macro_dt"])
        if rainfall_hyetograph is None:
            rain_start = 0.0
            rain_slope = 0.0
        else:
            rain_start = self._lts_hyetograph_rate(rainfall_hyetograph, macro_start)
            rain_end = self._lts_hyetograph_rate(
                rainfall_hyetograph,
                macro_start + macro_dt,
            )
            rain_slope = (
                (rain_end - rain_start) / macro_dt
                if macro_dt > 0.0
                else 0.0
            )
            if self.apply_sources_after_momentum:
                # The graph sources kernel samples forcing at (tick + 0.5);
                # bias the intercept so the effective sample lands on the
                # interval start, matching the global device-loop convention
                # for profiles where sources follow momentum.
                rain_start -= 0.5 * rain_slope * float(schedule["base_dt"])
        wp.launch(
            kernel=lts_graph_init_kernel,
            dim=1,
            inputs=[
                self._lts_graph_tick,
                self._lts_graph_condition,
                self._lts_graph_forcing,
                int(schedule["macro_ticks"]),
                wp.float64(rain_start),
                wp.float64(rain_slope),
            ],
            device=self.device,
        )
        wp.capture_launch(self._lts_subcycle_graph)

    def run_spatially_adaptive_loop(
        self,
        duration: float,
        rainfall_hyetograph=None,
        max_levels: int = 10,
        wet_max_level: int = 2,
        use_device_graph: bool = False,
        accuracy_fraction: float = 0.25,
        accuracy_depth_scale: float = 1.0e-3,
        promotion_margin: float = 2.0,
        front_guard_depth: float | None = None,
        drainage_max_level: int = 0,
        on_macro_sync=None,
    ) -> dict[str, object]:
        """Advance with conservative power-of-two spatial local timesteps.

        The finest hydraulic region advances at ``base_dt``. Cells with larger
        local CFL limits use ``2**level * base_dt`` up to ``max_levels``.
        Adjacent active cells are smoothed to differ by at most one level.

        Level selection is accuracy-controlled: besides the face-CFL limit,
        each cell's local timestep must satisfy a temporal accuracy limit
        ``accuracy_fraction * tau_h`` where ``tau_h`` is the characteristic
        change time of the effective cell storage (water depth plus the
        pending conservative volume register) estimated between schedule
        rebuilds.  Demotion is immediate; promotion is limited to one level
        per schedule rebuild and requires ``promotion_margin`` headroom.
        ``accuracy_depth_scale`` floors the depth scale so dry and near-dry
        cells remain robust, and dry cells beside a wetting front are
        additionally clamped to ``wet_max_level``.  ``front_guard_depth``
        (default: the profile dry depth) sets the depth at which the wet and
        wetting-front guards engage; choosing a few times the dry depth also
        protects sub-threshold but hydraulically active fringe cells.

        Shared-face momentum is evaluated at the faster neighbour's cadence.
        The resulting discharge is held between momentum updates but transports
        water on every base tick. Base-clock transport increments accumulate in
        a float64 cell register and are committed when each cell reaches its
        local synchronization time. This avoids moving a coarse interval's
        future water at the interval start while preserving equal-and-opposite
        internal face volumes exactly.

        ``use_device_graph`` selects the experimental conditional-CUDA-graph
        subcycle replay.  It was measured slower than the default
        device-assisted compact loop (500x500: ~24.2 s vs ~22.3 s; 1000x1000:
        ~205 s vs ~149 s) and remains available only for opt-in experiments.

        ``track_cumulative_face_volumes`` is rejected: the face-volume
        diagnostic is wired into global host-stepped ``step()`` only.
        """
        if self.structures:
            raise ValueError("Hydraulic structures currently require host-stepped step(); spatial local timestepping is not yet supported")
        if duration <= 0.0 or not np.isfinite(duration):
            raise ValueError("duration must be positive and finite")
        if self.track_cumulative_face_volumes:
            raise ValueError(
                "run_spatially_adaptive_loop does not support "
                "track_cumulative_face_volumes; the cumulative face-volume "
                "diagnostic is supported only by global host-stepped step() "
                "execution, not the device loop or spatial local timestepping"
            )
        if self.track_max_water_depth and not getattr(self, '_lts_tracks_max_water_depth', False):
            raise ValueError(
                "run_spatially_adaptive_loop does not support "
                "track_max_water_depth; the per-cell maximum-depth label "
                "accumulator is supported only by global host-stepped step() "
                "and run_device_loop() execution, not spatial local timestepping"
            )
        if np.any(self.additional_source_rate.numpy() != 0.0):
            raise ValueError(
                "run_spatially_adaptive_loop does not support a nonzero "
                "additional_source_rate; the external additional surface-source "
                "source requires host-stepped step() or run_device_loop() "
                "execution"
            )
        if self.infiltration_limits_active:
            raise ValueError(
                "run_spatially_adaptive_loop does not support the external "
                "infiltration exclusion mask or storage cap; clear them or "
                "use host-stepped step() or run_device_loop() execution"
            )
        if max_levels < 0 or max_levels > 12:
            raise ValueError("max_levels must be between 0 and 12")
        if wet_max_level < 0 or wet_max_level > max_levels:
            raise ValueError("wet_max_level must be between 0 and max_levels")
        if accuracy_fraction <= 0.0 or not np.isfinite(accuracy_fraction):
            raise ValueError("accuracy_fraction must be positive and finite")
        if accuracy_depth_scale <= 0.0 or not np.isfinite(accuracy_depth_scale):
            raise ValueError("accuracy_depth_scale must be positive and finite")
        if promotion_margin < 1.0 or not np.isfinite(promotion_margin):
            raise ValueError("promotion_margin must be at least 1 and finite")
        if front_guard_depth is None:
            guard_depth = float(self.dry_depth)
        else:
            if front_guard_depth <= 0.0 or not np.isfinite(front_guard_depth):
                raise ValueError("front_guard_depth must be positive and finite")
            guard_depth = float(front_guard_depth)
        if drainage_max_level >= 0 and drainage_max_level > max_levels:
            raise ValueError("drainage_max_level must be negative or at most max_levels")
        if self.use_lisflood_acc_timestep:
            raise ValueError(
                "spatial local timestepping requires an Overlander face-CFL profile; "
                "use the default or lisflood_fp_acc_parity profile"
            )

        self._ensure_lts_buffers(max_levels=max_levels)
        self._lts_pending_flux_volume.zero_()
        self.error_flag.zero_()
        end_time = float(self.simulation_time) + float(duration)

        macro_steps = 0
        graph_macro_steps = 0
        base_steps = 0
        cell_updates = 0
        source_cell_updates = 0
        qx_updates = 0
        qy_updates = 0
        active_cells = int(np.count_nonzero(self._active_mask_host))
        level_cell_updates = np.zeros(max_levels + 1, dtype=np.int64)
        level_qx_updates = np.zeros(max_levels + 1, dtype=np.int64)
        level_qy_updates = np.zeros(max_levels + 1, dtype=np.int64)
        level_presence = np.zeros(max_levels + 1, dtype=np.int64)

        while self.simulation_time < end_time - 1.0e-12:
            remaining = end_time - float(self.simulation_time)
            cap_dt = min(self.maximum_dt, remaining)
            if rainfall_hyetograph is not None and self.simulation_time < rainfall_hyetograph.duration_s:
                breakpoint = rainfall_hyetograph.next_breakpoint_after(
                    time_s=float(self.simulation_time)
                )
                cap_dt = min(cap_dt, breakpoint - float(self.simulation_time))
            if cap_dt <= 0.0:
                raise RuntimeError("non-positive LTS macrostep cap")

            schedule = self._prepare_lts_schedule(
                cap_dt=cap_dt,
                rainfall_hyetograph=rainfall_hyetograph,
                max_levels=max_levels,
                wet_max_level=wet_max_level,
                accuracy_fraction=accuracy_fraction,
                accuracy_depth_scale=accuracy_depth_scale,
                promotion_margin=promotion_margin,
                guard_depth=guard_depth,
                drainage_max_level=drainage_max_level,
            )

            # Schedule construction uses cfl_depth as a source-predicted CFL
            # scratch field.  LTS momentum reuses it as the effective hydraulic
            # depth predictor, so reset it to the actual synchronized state at
            # the start of each macrostep.  Pending flux is zero here because
            # all levels synchronize at the preceding macrostep boundary.
            wp.copy(self.cfl_depth, self.water_depth)
            if self.is_subgrid() and self.hydraulic_geometry_mode != 'enhanced':
                # Subgrid momentum reads the derived stage of the
                # synchronized storage; held coarse fluxes freeze it for the
                # macrostep exactly like the coarse cfl_depth read.
                self._launch_stage_inversion(self.water_depth, self._eta_stage)

            base_dt = float(schedule["base_dt"])
            macro_ticks = int(schedule["macro_ticks"])
            macro_level = int(schedule["macro_level"])
            macro_start = float(self.simulation_time)

            self.applied_rain.zero_()
            self.applied_other.zero_()
            self.applied_infil.zero_()

            for level in range(macro_level + 1):
                level_presence[level] += int(schedule["cell_counts"][level])

            graph_used = False
            if use_device_graph:
                graph_used = self._lts_build_subcycle_graph(
                    schedule=schedule,
                    use_uniform_rain=rainfall_hyetograph is not None,
                )
            if graph_used:
                graph_macro_steps += 1
                self._lts_launch_subcycle_graph(
                    schedule=schedule,
                    rainfall_hyetograph=rainfall_hyetograph,
                    macro_start=macro_start,
                )
            else:
                for tick in range(macro_ticks):
                    event_start = macro_start + float(tick) * base_dt

                    # Sources run on the base clock for every active cell and
                    # accumulate into the same conservative pending-volume register
                    # as hydraulic fluxes.
                    if not self.apply_sources_after_momentum:
                        self._lts_launch_sources_all(
                            schedule=schedule,
                            interval_start=event_start,
                            interval_end=event_start + base_dt,
                            rainfall_hyetograph=rainfall_hyetograph,
                        )
                        self._lts_refresh_stage_from_effective_depth()

                    # Compute every face interval that starts at this event time.
                    # qx/qy candidates are kept in the next buffers until both
                    # directions have been evaluated, preserving lagged 2-D friction.
                    for level in range(macro_level + 1):
                        period = 1 << level
                        if tick % period != 0:
                            continue
                        dt_level = base_dt * float(period)
                        self._lts_launch_momentum(
                            schedule=schedule,
                            level=level,
                            dt=dt_level,
                        )

                    # Accept only the momentum states that were recomputed at this
                    # tick.  Crucially, do not move q*local_dt into storage here.
                    # The held discharge is transported one base interval at a time
                    # below, so future coarse-interval water cannot appear early in
                    # the effective state.
                    for level in range(macro_level + 1):
                        period = 1 << level
                        if tick % period == 0:
                            self._lts_accept_level_momentum(
                                schedule=schedule,
                                level=level,
                            )

                    # Positivity and transport both live on the physical base clock.
                    # Momentum is the multirate quantity; water movement is not.
                    self._lts_limit_held_outflow_all(schedule=schedule)
                    self._lts_transport_held_faces_all(schedule=schedule)

                    event_end = event_start + base_dt
                    if self.apply_sources_after_momentum:
                        self._lts_launch_sources_all(
                            schedule=schedule,
                            interval_start=event_start,
                            interval_end=event_end,
                            rainfall_hyetograph=rainfall_hyetograph,
                        )
                        self._lts_refresh_stage_from_effective_depth()

                    end_tick = tick + 1
                    for level in range(macro_level + 1):
                        period = 1 << level
                        if end_tick % period != 0:
                            continue
                        self._lts_launch_continuity(schedule=schedule, level=level)

            # One source-volume reduction per macrostep, regardless of the
            # number of local cell updates.
            self._finish_lts_source_accounting()

            for level in range(macro_level + 1):
                updates = macro_ticks // (1 << level)
                cells = int(schedule["cell_counts"][level]) * updates
                qxs = int(schedule["qx_counts"][level]) * updates
                qys = int(schedule["qy_counts"][level]) * updates
                cell_updates += cells
                qx_updates += qxs
                qy_updates += qys
                level_cell_updates[level] += cells
                level_qx_updates[level] += qxs
                level_qy_updates[level] += qys

            source_cell_updates += active_cells * macro_ticks
            base_steps += macro_ticks
            macro_steps += 1
            self.current_dt = base_dt
            self.simulation_time = min(end_time, macro_start + float(schedule["macro_dt"]))
            if on_macro_sync is not None:
                on_macro_sync(macro_steps, self.simulation_time, schedule)

        wp.synchronize_device(self.device)
        if int(self.error_flag.numpy()[0]) != 0:
            raise RuntimeError("Significant negative depth encountered during LTS run.")

        accuracy_control = self._lts_accuracy_summary(
            np.asarray(self._lts_classify_stats.numpy(), dtype=np.int64),
            accuracy_fraction=accuracy_fraction,
            accuracy_depth_scale=accuracy_depth_scale,
            promotion_margin=promotion_margin,
        )

        global_cell_updates = active_cells * base_steps
        cell_work_fraction = (
            float(cell_updates) / float(global_cell_updates)
            if global_cell_updates > 0
            else 0.0
        )
        source_plus_continuity_work_fraction = (
            float(source_cell_updates + cell_updates) / float(2 * global_cell_updates)
            if global_cell_updates > 0
            else 0.0
        )
        return {
            "macro_steps": macro_steps,
            "base_steps": base_steps,
            "schedule_mode": (
                "device_graph_subcycle"
                if graph_macro_steps == macro_steps and macro_steps > 0
                else "device_assisted_compact"
            ),
            "schedule_host_syncs": macro_steps,
            "device_graph_macrosteps": graph_macro_steps,
            "wet_max_level": wet_max_level,
            "simulation_time": self.simulation_time,
            "current_base_dt": self.current_dt,
            "cell_updates": cell_updates,
            "source_cell_updates": source_cell_updates,
            "qx_updates": qx_updates,
            "qy_updates": qy_updates,
            "equivalent_global_cell_updates": global_cell_updates,
            # Hydraulic continuity work only. Sources intentionally remain on
            # the base clock, so this is no longer an estimate of total cell
            # work; retain it for comparison with earlier LTS runs.
            "cell_work_fraction": cell_work_fraction,
            "ideal_cell_work_speedup": (
                1.0 / cell_work_fraction if cell_work_fraction > 0.0 else float("inf")
            ),
            "source_plus_continuity_work_fraction": source_plus_continuity_work_fraction,
            "ideal_source_plus_continuity_speedup": (
                1.0 / source_plus_continuity_work_fraction
                if source_plus_continuity_work_fraction > 0.0
                else float("inf")
            ),
            "level_cell_updates": level_cell_updates.tolist(),
            "level_qx_updates": level_qx_updates.tolist(),
            "level_qy_updates": level_qy_updates.tolist(),
            "level_presence": level_presence.tolist(),
            "accuracy_control": accuracy_control,
        }
