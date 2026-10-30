import numpy as np
from floodwarp.boundary_conditions import (
    BOUNDARY_INTERNAL, BOUNDARY_CLOSED,
    BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE
)

def step_local_inertial_numpy(
    water_depth: np.ndarray,
    bed_elevation: np.ndarray,
    manning_n: np.ndarray,
    active_mask: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    rainfall_rate: np.ndarray,
    other_source_rate: np.ndarray,
    infiltration_capacity: np.ndarray,
    boundary_type_qx: np.ndarray,
    boundary_stage_qx: np.ndarray,
    boundary_type_qy: np.ndarray,
    boundary_stage_qy: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    gravity: float = 9.80665,
    dry_depth: float = 1.0e-4,
    depth_epsilon: float = 1.0e-6,
    cfl_number: float = 0.4,
    limiter_epsilon: float = 1e-12,
    negative_depth_tolerance: float = 1e-8,
):
    ny, nx = water_depth.shape

    r_rate = np.maximum(0.0, rainfall_rate)
    o_rate = np.maximum(0.0, other_source_rate)
    i_cap = np.maximum(0.0, infiltration_capacity)

    r_depth = r_rate * dt
    o_depth = o_rate * dt

    h_after_sources = water_depth + r_depth + o_depth
    i_max_depth = np.minimum(i_cap * dt, h_after_sources)

    water_depth = np.where(active_mask == 1, h_after_sources - i_max_depth, 0.0)

    applied_rain = np.where(active_mask == 1, r_depth, 0.0)
    applied_other = np.where(active_mask == 1, o_depth, 0.0)
    applied_infil = np.where(active_mask == 1, i_max_depth, 0.0)

    dt_cfl = np.inf

    # Calculate CFL X
    for j in range(ny):
        for i in range(nx + 1):
            btype = boundary_type_qx[j, i]
            if btype == BOUNDARY_CLOSED: continue

            z_L, z_R, h_L, h_R, eta_L, eta_R = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if btype == BOUNDARY_INTERNAL:
                z_L = bed_elevation[j, i-1]
                z_R = bed_elevation[j, i]
                h_L = water_depth[j, i-1]
                h_R = water_depth[j, i]
                eta_L = z_L + h_L
                eta_R = z_R + h_R
            elif btype in (BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE):
                if i == 0:
                    z_R = bed_elevation[j, 0]
                    h_R = water_depth[j, 0]
                    eta_R = z_R + h_R
                    z_L = z_R
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_L = 0.0
                        eta_L = z_L + h_L
                    else:
                        eta_L = max(boundary_stage_qx[j, 0], z_L)
                        h_L = max(0.0, eta_L - z_L)
                else:
                    z_L = bed_elevation[j, nx-1]
                    h_L = water_depth[j, nx-1]
                    eta_L = z_L + h_L
                    z_R = z_L
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_R = 0.0
                        eta_R = z_R + h_R
                    else:
                        eta_R = max(boundary_stage_qx[j, nx], z_R)
                        h_R = max(0.0, eta_R - z_R)

            hf = max(0.0, max(eta_L, eta_R) - max(z_L, z_R))
            if hf > dry_depth:
                cel = np.sqrt(gravity * hf)
                if cel > 0.0:
                    dt_cfl = min(dt_cfl, cfl_number * dx / cel)

    # Calculate CFL Y
    for j in range(ny + 1):
        for i in range(nx):
            btype = boundary_type_qy[j, i]
            if btype == BOUNDARY_CLOSED: continue

            z_U, z_D, h_U, h_D, eta_U, eta_D = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if btype == BOUNDARY_INTERNAL:
                z_U = bed_elevation[j-1, i]
                z_D = bed_elevation[j, i]
                h_U = water_depth[j-1, i]
                h_D = water_depth[j, i]
                eta_U = z_U + h_U
                eta_D = z_D + h_D
            elif btype in (BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE):
                if j == 0:
                    z_D = bed_elevation[0, i]
                    h_D = water_depth[0, i]
                    eta_D = z_D + h_D
                    z_U = z_D
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_U = 0.0
                        eta_U = z_U + h_U
                    else:
                        eta_U = max(boundary_stage_qy[0, i], z_U)
                        h_U = max(0.0, eta_U - z_U)
                else:
                    z_U = bed_elevation[ny-1, i]
                    h_U = water_depth[ny-1, i]
                    eta_U = z_U + h_U
                    z_D = z_U
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_D = 0.0
                        eta_D = z_D + h_D
                    else:
                        eta_D = max(boundary_stage_qy[ny, i], z_D)
                        h_D = max(0.0, eta_D - z_D)

            hf = max(0.0, max(eta_U, eta_D) - max(z_U, z_D))
            if hf > dry_depth:
                cel = np.sqrt(gravity * hf)
                if cel > 0.0:
                    dt_cfl = min(dt_cfl, cfl_number * dy / cel)

    if np.isinf(dt_cfl):
        dt_cfl = dt
    dt = min(dt, dt_cfl)

    qx_new = np.zeros_like(qx)
    qy_new = np.zeros_like(qy)

    for j in range(ny):
        for i in range(nx + 1):
            btype = boundary_type_qx[j, i]
            if btype == BOUNDARY_CLOSED:
                qx_new[j, i] = 0.0
                continue

            z_L, z_R, h_L, h_R, eta_L, eta_R, nf = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if btype == BOUNDARY_INTERNAL:
                z_L = bed_elevation[j, i-1]
                z_R = bed_elevation[j, i]
                h_L = water_depth[j, i-1]
                h_R = water_depth[j, i]
                eta_L = z_L + h_L
                eta_R = z_R + h_R
                nf = max(manning_n[j, i-1], manning_n[j, i])
            elif btype in (BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE):
                if i == 0:
                    z_R = bed_elevation[j, 0]
                    h_R = water_depth[j, 0]
                    eta_R = z_R + h_R
                    z_L = z_R
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_L = 0.0
                        eta_L = z_L + h_L
                    else:
                        eta_L = max(boundary_stage_qx[j, 0], z_L)
                        h_L = max(0.0, eta_L - z_L)
                    nf = manning_n[j, 0]
                else:
                    z_L = bed_elevation[j, nx-1]
                    h_L = water_depth[j, nx-1]
                    eta_L = z_L + h_L
                    z_R = z_L
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_R = 0.0
                        eta_R = z_R + h_R
                    else:
                        eta_R = max(boundary_stage_qx[j, nx], z_R)
                        h_R = max(0.0, eta_R - z_R)
                    nf = manning_n[j, nx-1]

            hf = max(0.0, max(eta_L, eta_R) - max(z_L, z_R))
            if hf < dry_depth:
                qx_new[j, i] = 0.0
            else:
                Sf = (eta_R - eta_L) / dx
                q_old = qx[j, i]
                denom = 1.0 + gravity * nf**2 * dt * abs(q_old) / max(hf, depth_epsilon)**(7/3)
                num = q_old - gravity * hf * dt * Sf
                q_n = num / denom

                if btype == BOUNDARY_FREE_OUTFLOW:
                    if i == 0:
                        q_n = min(q_n, 0.0)
                    else:
                        q_n = max(q_n, 0.0)
                qx_new[j, i] = q_n

    for j in range(ny + 1):
        for i in range(nx):
            btype = boundary_type_qy[j, i]
            if btype == BOUNDARY_CLOSED:
                qy_new[j, i] = 0.0
                continue

            z_U, z_D, h_U, h_D, eta_U, eta_D, nf = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if btype == BOUNDARY_INTERNAL:
                z_U = bed_elevation[j-1, i]
                z_D = bed_elevation[j, i]
                h_U = water_depth[j-1, i]
                h_D = water_depth[j, i]
                eta_U = z_U + h_U
                eta_D = z_D + h_D
                nf = max(manning_n[j-1, i], manning_n[j, i])
            elif btype in (BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE):
                if j == 0:
                    z_D = bed_elevation[0, i]
                    h_D = water_depth[0, i]
                    eta_D = z_D + h_D
                    z_U = z_D
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_U = 0.0
                        eta_U = z_U + h_U
                    else:
                        eta_U = max(boundary_stage_qy[0, i], z_U)
                        h_U = max(0.0, eta_U - z_U)
                    nf = manning_n[0, i]
                else:
                    z_U = bed_elevation[ny-1, i]
                    h_U = water_depth[ny-1, i]
                    eta_U = z_U + h_U
                    z_D = z_U
                    if btype == BOUNDARY_FREE_OUTFLOW:
                        h_D = 0.0
                        eta_D = z_D + h_D
                    else:
                        eta_D = max(boundary_stage_qy[ny, i], z_D)
                        h_D = max(0.0, eta_D - z_D)
                    nf = manning_n[ny-1, i]

            hf = max(0.0, max(eta_U, eta_D) - max(z_U, z_D))
            if hf < dry_depth:
                qy_new[j, i] = 0.0
            else:
                Sf = (eta_D - eta_U) / dy
                q_old = qy[j, i]
                denom = 1.0 + gravity * nf**2 * dt * abs(q_old) / max(hf, depth_epsilon)**(7/3)
                num = q_old - gravity * hf * dt * Sf
                q_n = num / denom

                if btype == BOUNDARY_FREE_OUTFLOW:
                    if j == 0:
                        q_n = min(q_n, 0.0)
                    else:
                        q_n = max(q_n, 0.0)
                qy_new[j, i] = q_n

    qx = qx_new
    qy = qy_new

    q_out = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            if active_mask[j, i] == 1:
                qx_east = qx[j, i+1]
                if qx_east > 0: q_out[j, i] += qx_east * dy
                qx_west = qx[j, i]
                if qx_west < 0: q_out[j, i] -= qx_west * dy
                qy_south = qy[j+1, i]
                if qy_south > 0: q_out[j, i] += qy_south * dx
                qy_north = qy[j, i]
                if qy_north < 0: q_out[j, i] -= qy_north * dx

    alpha = np.ones((ny, nx))
    V_avail = water_depth * dx * dy
    mask_out = q_out > 0.0
    alpha[mask_out] = np.minimum(1.0, V_avail[mask_out] / (dt * q_out[mask_out] + limiter_epsilon))
    alpha = np.where(active_mask == 1, alpha, 1.0)

    for j in range(ny):
        for i in range(nx + 1):
            q = qx[j, i]
            btype = boundary_type_qx[j, i]
            if q != 0.0 and btype != BOUNDARY_CLOSED:
                if q > 0:
                    if i > 0:
                        qx[j, i] = q * alpha[j, i-1]
                elif q < 0:
                    if i < nx:
                        qx[j, i] = q * alpha[j, i]

    for j in range(ny + 1):
        for i in range(nx):
            q = qy[j, i]
            btype = boundary_type_qy[j, i]
            if q != 0.0 and btype != BOUNDARY_CLOSED:
                if q > 0:
                    if j > 0:
                        qy[j, i] = q * alpha[j-1, i]
                elif q < 0:
                    if j < ny:
                        qy[j, i] = q * alpha[j, i]

    inflow = 0.0
    outflow = 0.0
    for i in [0, nx]:
        for j in range(ny):
            b_type = boundary_type_qx[j, i]
            if b_type != BOUNDARY_CLOSED and b_type != BOUNDARY_INTERNAL:
                q_val = qx[j, i]
                if i == 0:
                    if q_val > 0: inflow += q_val * dy * dt
                    elif q_val < 0: outflow -= q_val * dy * dt
                else:
                    if q_val < 0: inflow -= q_val * dy * dt
                    elif q_val > 0: outflow += q_val * dy * dt

    for j in [0, ny]:
        for i in range(nx):
            b_type = boundary_type_qy[j, i]
            if b_type != BOUNDARY_CLOSED and b_type != BOUNDARY_INTERNAL:
                q_val = qy[j, i]
                if j == 0:
                    if q_val > 0: inflow += q_val * dx * dt
                    elif q_val < 0: outflow -= q_val * dx * dt
                else:
                    if q_val < 0: inflow -= q_val * dx * dt
                    elif q_val > 0: outflow += q_val * dx * dt

    dqx = (qx[:, 1:] - qx[:, :-1]) / dx
    dqy = (qy[1:, :] - qy[:-1, :]) / dy

    water_depth = np.where(active_mask == 1, water_depth - dt * (dqx + dqy), 0.0)

    mask_neg = (active_mask == 1) & (water_depth < 0.0)
    if np.any(mask_neg):
        if np.any(water_depth[mask_neg] < -negative_depth_tolerance):
            raise RuntimeError("Significant negative depth encountered.")
        water_depth[mask_neg] = 0.0

    return {
        "water_depth": water_depth,
        "qx": qx,
        "qy": qy,
        "accepted_dt": dt,
        "outflow_limiter": alpha,
        "applied_rain": applied_rain,
        "applied_other": applied_other,
        "applied_infil": applied_infil,
        "boundary_inflow_volume": inflow,
        "boundary_outflow_volume": outflow
    }
