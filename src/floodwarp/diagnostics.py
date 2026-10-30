import warp as wp
import numpy as np
from floodwarp.config import WP_FLOAT
from floodwarp.boundary_conditions import (
    BOUNDARY_INTERNAL, BOUNDARY_CLOSED,
    BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE
)

@wp.kernel
def compute_volume_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    vol_out: wp.array(dtype=wp.float64)
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        wp.atomic_add(
            vol_out,
            0,
            wp.float64(water_depth[j, i]) * wp.float64(dx) * wp.float64(dy),
        )

@wp.kernel
def compute_extrema_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    min_depth: wp.array(dtype=wp.float64),
    max_depth: wp.array(dtype=wp.float64),
    max_q: wp.array(dtype=wp.float64),
    max_v: wp.array(dtype=wp.float64),
    wet_cells: wp.array(dtype=wp.int32),
    dry_depth: WP_FLOAT
):
    j, i = wp.tid()
    ny = water_depth.shape[0]
    nx = water_depth.shape[1]

    if active_mask[j, i] == 1:
        h = wp.float64(water_depth[j, i])
        wp.atomic_min(min_depth, 0, h)
        wp.atomic_max(max_depth, 0, h)

        if h > dry_depth:
            wp.atomic_add(wet_cells, 0, 1)

            q_abs_x = wp.max(wp.abs(wp.float64(qx[j, i])), wp.abs(wp.float64(qx[j, i+1])))
            q_abs_y = wp.max(wp.abs(wp.float64(qy[j, i])), wp.abs(wp.float64(qy[j+1, i])))
            v_x = q_abs_x / h
            v_y = q_abs_y / h
            v = wp.max(v_x, v_y)
            wp.atomic_max(max_v, 0, v)
            wp.atomic_max(max_q, 0, wp.max(q_abs_x, q_abs_y))

def compute_diagnostics(solver) -> dict:
    device = solver.device

    vol_out = wp.zeros(1, dtype=wp.float64, device=device)
    min_depth = wp.zeros(1, dtype=wp.float64, device=device)
    min_depth.fill_(wp.float64(1e9))
    max_depth = wp.zeros(1, dtype=wp.float64, device=device)
    max_depth.fill_(wp.float64(-1e9))
    max_q = wp.zeros(1, dtype=wp.float64, device=device)
    max_v = wp.zeros(1, dtype=wp.float64, device=device)
    wet_cells = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        kernel=compute_volume_kernel,
        dim=(solver.ny, solver.nx),
        inputs=[solver.water_depth, solver.active_mask, solver.dx, solver.dy, vol_out],
        device=device
    )

    wp.launch(
        kernel=compute_extrema_kernel,
        dim=(solver.ny, solver.nx),
        inputs=[
            solver.water_depth, solver.active_mask,
            solver.qx, solver.qy,
            min_depth, max_depth, max_q, max_v, wet_cells,
            solver.dry_depth
        ],
        device=device
    )

    current_volume = float(vol_out.numpy()[0])
    min_d = float(min_depth.numpy()[0])
    if min_d > 1e8:
        min_d = 0.0
    max_d = float(max_depth.numpy()[0])
    if max_d < -1e8:
        max_d = 0.0
    max_q_val = float(max_q.numpy()[0])
    max_v_val = float(max_v.numpy()[0])
    wet_c = int(wet_cells.numpy()[0])

    accounting = solver.cumulative_accounting()
    additional_sources = accounting.get("additional_sources", 0.0)

    mass_residual = (
        current_volume
        - solver.initial_volume
        - accounting["rain"]
        - accounting["other_sources"]
        - additional_sources
        - accounting["boundary_inflow"]
        + accounting["infiltration"]
        + accounting["boundary_outflow"]
    )

    ref_vol = max(
        solver.initial_volume + accounting["rain"] + accounting["other_sources"]
        + additional_sources + accounting["boundary_inflow"],
        1e-12
    )
    rel_error = mass_residual / ref_vol

    result = {
        "simulation_time": solver.simulation_time,
        "current_time_step": solver.current_dt,
        "min_water_depth": min_d,
        "max_water_depth": max_d,
        "max_absolute_face_discharge": max_q_val,
        "max_estimated_velocity": max_v_val,
        "number_of_wet_cells": wet_c,
        "total_surface_water_volume": current_volume,
        "cumulative_rainfall_volume": accounting["rain"],
        "cumulative_infiltration_volume": accounting["infiltration"],
        "cumulative_boundary_inflow": accounting["boundary_inflow"],
        "cumulative_boundary_outflow": accounting["boundary_outflow"],
        "cumulative_externally_imposed_source_volume": accounting["other_sources"],
        "cumulative_additional_source_source_volume": additional_sources,
        "initial_water_volume": solver.initial_volume,
        "mass_balance_residual": mass_residual,
        "relative_mass_balance_error": rel_error,
        "total_infiltration_volume_m3": accounting["infiltration"],
    }
    result['validation_profile'] = solver.validation_profile
    if hasattr(solver, 'numerical_configuration'):
        result['numerical_configuration'] = solver.numerical_configuration()

    cumulative = np.asarray(getattr(solver, "cumulative_infiltration_m", np.zeros((solver.ny, solver.nx))))
    active_cumulative = cumulative[solver._active_mask_host == 1]
    result["maximum_cumulative_infiltration_m"] = (
        float(active_cumulative.max()) if active_cumulative.size else 0.0
    )
    result["mean_cumulative_infiltration_m"] = (
        float(active_cumulative.mean()) if active_cumulative.size else 0.0
    )
    if getattr(solver, "infiltration_method", 0) in (1, 3):
        capacity = np.asarray(solver.current_infiltration_capacity_m_per_s)
        active_capacity = capacity[solver._active_mask_host == 1]
        result["min_infiltration_capacity_m_per_s"] = (
            float(active_capacity.min()) if active_capacity.size else 0.0
        )
        result["max_infiltration_capacity_m_per_s"] = (
            float(active_capacity.max()) if active_capacity.size else 0.0
        )

    # In subgrid mode the storage-equivalent depth is NOT the physical depth:
    # report both semantics. Mass balance intentionally keeps using storage
    # because V = storage_depth * dx * dy remains exact.
    if getattr(solver, "is_subgrid", None) is not None and solver.is_subgrid():
        state = solver.get_subgrid_state()
        storage = np.asarray(state["storage_depth"], dtype=np.float64)[
            solver._active_mask_host == 1
        ]
        local_depth = np.asarray(state["maximum_local_depth"], dtype=np.float64)[
            solver._active_mask_host == 1
        ]
        wet_fraction = np.asarray(state["wet_fraction"], dtype=np.float64)[
            solver._active_mask_host == 1
        ]
        result["hydraulic_geometry_mode"] = "subgrid"
        result["depth_semantics"] = "water_depth is storage-equivalent depth V/(dx*dy)"
        result["max_storage_depth"] = float(storage.max()) if storage.size else 0.0
        result["max_local_water_depth"] = (
            float(local_depth.max()) if local_depth.size else 0.0
        )
        result["max_mean_wet_depth"] = float(
            np.max(
                np.asarray(state["mean_wet_depth"], dtype=np.float64)[
                    solver._active_mask_host == 1
                ]
            )
        )
        result["mean_wet_fraction"] = (
            float(wet_fraction.mean()) if wet_fraction.size else 0.0
        )
    return result
