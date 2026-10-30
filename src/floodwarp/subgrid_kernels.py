"""Subgrid hydraulic-geometry Warp kernels (experimental explicit mode).

This is a dedicated kernel family for ``hydraulic_geometry_mode="subgrid"``;
the production coarse kernels are never modified.  The coarse-cell storage
state (``water_depth`` = V/(dx*dy)) drives everything: a stage-inversion
kernel derives the water-surface elevation field once per step (or once per
LTS macrostep), and the face kernels below consume that stage field together
with the precomputed :class:`~floodwarp.subgrid_geometry.SubgridGeometry`
arrays.  Storage semantics (continuity, sources, limiter, pending flux
volumes, mass accounting) stay on the unchanged coarse kernels.

Face hydraulics per coarse face with segment elevations z_{f,k}:

    eta_f = max(eta_L, eta_R)                     (existing max-stage rule)
    h_A   = mean_k max(eta_f - z_{f,k}, 0)        (area depth)
    M     = mean_k max(eta_f - z_{f,k}, 0)^(5/3)  (conveyance moment)
    n_eff = n_f * h_A^(5/3) / M                   (M > 0)
    h_max = max(eta_f - min_k z_{f,k}, 0)         (wet/dry + CFL)

Specified-stage faces reconstruct both heads above each native sill before
integrating pressure: a sea stage below a dry lip exerts no pressure through
the air gap. FREE outflow faces use the face thalweg ``min_k z_{f,k}`` as
the dry exterior stage, which collapses to the coarse ``eta_ext = z_edge``
convention on a flat subgrid.  For a flat subgrid every quantity reduces to
the coarse kernel expression to floating-point tolerance.
"""
import warp as wp

from floodwarp.config import BED_FLOAT, STAGE_FLOAT, WP_FLOAT

from floodwarp.subgrid_hypsometry import subgrid_stage_from_storage
from floodwarp.local_inertial_kernels import (
    BOUNDARY_CLOSED,
    BOUNDARY_FREE_OUTFLOW,
    BOUNDARY_INTERNAL,
    BOUNDARY_MASKED_EAST_FREE,
    BOUNDARY_MASKED_NORTH_FREE,
    BOUNDARY_MASKED_NORTH_STAGE,
    BOUNDARY_MASKED_SOUTH_FREE,
    BOUNDARY_MASKED_WEST_FREE,
    BOUNDARY_MASKED_WEST_STAGE,
    BOUNDARY_SPECIFIED_STAGE,
    DL_DT,
    DL_DONE,
    LTS_CTRL_BASE_DT,
    _lts_face_is_scheduled,
)

# Subgrid mode supports the production native FREE convention only.  The
# LISFLOOD ACC/float FREE arithmetic oracles are validation-only paths of the
# coarse formulation and are rejected at solver construction.
FREE_OUTFLOW_MODE_NATIVE = 0


@wp.func
def subgrid_stage_pressure(
    face_z: wp.array3d(dtype=BED_FLOAT),
    j: int,
    i: int,
    eta_left: BED_FLOAT,
    eta_right: BED_FLOAT,
    factor: int,
) -> BED_FLOAT:
    """Mean depth times hydrostatic head jump at a specified-stage face.

    Each segment retains its own physical sill. Clipping to a face minimum
    would still transmit pressure through the dry portion of a higher bank.
    """
    pressure = BED_FLOAT(0.0)
    for k in range(factor):
        sill = face_z[j, i, k]
        left = wp.max(eta_left, sill)
        right = wp.max(eta_right, sill)
        depth = wp.max(left, right) - sill
        pressure += depth * (right - left)
    return pressure / BED_FLOAT(factor)


@wp.kernel
def subgrid_stage_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    cell_z: wp.array3d(dtype=BED_FLOAT),
    cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    eta_out: wp.array2d(dtype=WP_FLOAT),
):
    """Derive the water-surface elevation field from storage-equivalent depth."""
    j, i = wp.tid()
    eta_out[j, i] = WP_FLOAT(
        subgrid_stage_from_storage(
            water_depth[j, i], cell_z, cell_storage_breakpoints,
            j, i, subgrid_cell_count,
        )
    )


@wp.kernel
def subgrid_derived_state_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    cell_z: wp.array3d(dtype=BED_FLOAT),
    cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    eta_out: wp.array2d(dtype=WP_FLOAT),
    max_local_depth_out: wp.array2d(dtype=WP_FLOAT),
    wet_fraction_out: wp.array2d(dtype=WP_FLOAT),
    mean_wet_depth_out: wp.array2d(dtype=WP_FLOAT),
):
    """Derive stage plus the physical local-depth diagnostics in one pass."""
    j, i = wp.tid()
    eta = subgrid_stage_from_storage(
        water_depth[j, i], cell_z, cell_storage_breakpoints,
        j, i, subgrid_cell_count,
    )
    eta_out[j, i] = WP_FLOAT(eta)
    wet_count = int(0)
    depth_sum = BED_FLOAT(0.0)
    for k in range(subgrid_cell_count):
        depth = wp.max(BED_FLOAT(0.0), eta - cell_z[j, i, k])
        depth_sum += depth
        if depth > BED_FLOAT(0.0):
            wet_count += 1
    max_local_depth_out[j, i] = wp.max(
        WP_FLOAT(0.0), WP_FLOAT(eta - cell_z[j, i, 0])
    )
    wet_fraction_out[j, i] = (
        WP_FLOAT(wet_count) / WP_FLOAT(subgrid_cell_count)
    )
    mean_wet_depth_out[j, i] = (
        WP_FLOAT(depth_sum / BED_FLOAT(wet_count))
        if wet_count > 0
        else WP_FLOAT(0.0)
    )


@wp.func
def subgrid_face_area_depth(
    face_z: wp.array3d(dtype=BED_FLOAT),
    j: int,
    i: int,
    face_water_surface: BED_FLOAT,
    factor: int,
) -> BED_FLOAT:
    """Native-segment mean depth: preserves wetted cross-sectional area."""
    total = BED_FLOAT(0.0)
    for k in range(factor):
        total += wp.max(BED_FLOAT(0.0), face_water_surface - face_z[j, i, k])
    return total / BED_FLOAT(factor)


@wp.func
def subgrid_face_conveyance_moment(
    face_z: wp.array3d(dtype=BED_FLOAT),
    j: int,
    i: int,
    face_water_surface: BED_FLOAT,
    factor: int,
) -> BED_FLOAT:
    """Width-normalised sum of h_k^(5/3) over the native face segments."""
    total = BED_FLOAT(0.0)
    for k in range(factor):
        depth = wp.max(BED_FLOAT(0.0), face_water_surface - face_z[j, i, k])
        total += wp.pow(depth, BED_FLOAT(5.0 / 3.0))
    return total / BED_FLOAT(factor)


@wp.func
def subgrid_face_conveyance_conductance(
    face_z: wp.array3d(dtype=BED_FLOAT),
    face_n: wp.array3d(dtype=WP_FLOAT),
    j: int,
    i: int,
    face_water_surface: BED_FLOAT,
    factor: int,
) -> BED_FLOAT:
    """Width-normalised sum of h_k^(5/3) / n_k over the native segments.

    Variable-roughness counterpart of :func:`subgrid_face_conveyance_moment`.
    Dry/blocked segments have h_k = 0 and contribute zero; n_k > 0 always, so
    the division is safe. For uniform n_k = n this reduces to the moment
    divided by n, making the two reductions equivalent.
    """
    total = BED_FLOAT(0.0)
    for k in range(factor):
        depth = wp.max(BED_FLOAT(0.0), face_water_surface - face_z[j, i, k])
        total += wp.pow(depth, BED_FLOAT(5.0 / 3.0)) / BED_FLOAT(
            face_n[j, i, k]
        )
    return total / BED_FLOAT(factor)


@wp.func
def _subgrid_update_qx_face(
    i: int,
    j: int,
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_x: wp.array3d(dtype=BED_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    face_n_x: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
) -> WP_FLOAT:
    """Subgrid counterpart of ``_update_qx_face`` (native FREE mode only)."""
    ny = qx_old.shape[0]
    nx = qx_old.shape[1] - 1

    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_CLOSED:
        return WP_FLOAT(0.0)

    eta_L = WP_FLOAT(0.0)
    eta_R = WP_FLOAT(0.0)
    nf = WP_FLOAT(0.0)

    if btype == BOUNDARY_INTERNAL:
        eta_L = eta_stage[j, i - 1]
        eta_R = eta_stage[j, i]
        if manning_face_mode == 1:
            nf = WP_FLOAT(0.5) * (manning_n[j, i - 1] + manning_n[j, i])
        else:
            nf = wp.max(manning_n[j, i - 1], manning_n[j, i])
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        eta_L = boundary_stage_qx[j, i]
        eta_R = eta_stage[j, i]
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        eta_L = WP_FLOAT(face_z_min_x[j, i])
        eta_R = eta_stage[j, i]
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        eta_L = eta_stage[j, i - 1]
        eta_R = WP_FLOAT(face_z_min_x[j, i])
        nf = manning_n[j, i - 1]
    elif btype == BOUNDARY_FREE_OUTFLOW or btype == BOUNDARY_SPECIFIED_STAGE:
        if i == 0:
            eta_R = eta_stage[j, 0]
            if btype == BOUNDARY_FREE_OUTFLOW:
                eta_L = WP_FLOAT(face_z_min_x[j, 0])
            else:
                eta_L = boundary_stage_qx[j, 0]
            nf = manning_n[j, 0]
        else:
            eta_L = eta_stage[j, nx - 1]
            if btype == BOUNDARY_FREE_OUTFLOW:
                eta_R = WP_FLOAT(face_z_min_x[j, nx])
            else:
                eta_R = boundary_stage_qx[j, nx]
            nf = manning_n[j, nx - 1]

    eta_f = wp.max(BED_FLOAT(eta_L), BED_FLOAT(eta_R))
    max_local_depth = wp.max(
        BED_FLOAT(0.0), eta_f - face_z_min_x[j, i]
    )
    if max_local_depth < BED_FLOAT(dry_depth):
        return WP_FLOAT(0.0)

    h_a = subgrid_face_area_depth(face_z_x, j, i, eta_f, factor)
    h_a_hydraulic = WP_FLOAT(h_a)
    Sf = WP_FLOAT((BED_FLOAT(eta_R) - BED_FLOAT(eta_L)) / BED_FLOAT(dx))

    q_old = qx_old[j, i]
    friction_discharge = wp.abs(q_old)
    if use_two_dimensional_friction == 1 and i > 0 and i < nx:
        transverse_discharge = WP_FLOAT(0.25) * (
            qy_old[j, i - 1] + qy_old[j, i] + qy_old[j + 1, i - 1] + qy_old[j + 1, i]
        )
        friction_discharge = wp.sqrt(q_old * q_old + transverse_discharge * transverse_discharge)

    if use_native_manning == 1:
        # Variable-roughness reduction: K = mean(h_k^(5/3) / n_k) over the
        # native face segments, so n_eff = h_A^(5/3) / K independent of the
        # coarse face value nf. Dry segments contribute zero.
        conductance = subgrid_face_conveyance_conductance(
            face_z_x, face_n_x, j, i, eta_f, factor
        )
        n_eff = nf
        if conductance > BED_FLOAT(0.0):
            n_eff = WP_FLOAT(
                wp.pow(h_a, BED_FLOAT(5.0 / 3.0)) / conductance
            )
    else:
        # Uniform coarse-face-roughness reduction, unchanged when no native
        # Manning raster is supplied.
        moment = subgrid_face_conveyance_moment(face_z_x, j, i, eta_f, factor)
        n_eff = nf
        if moment > BED_FLOAT(0.0):
            n_eff = nf * WP_FLOAT(
                wp.pow(h_a, BED_FLOAT(5.0 / 3.0)) / moment
            )

    denominator = WP_FLOAT(1.0) + gravity * n_eff * n_eff * dt * friction_discharge / wp.pow(
        wp.max(h_a_hydraulic, depth_epsilon), WP_FLOAT(7.0 / 3.0)
    )
    numerator = q_old - gravity * h_a_hydraulic * dt * Sf
    if btype == BOUNDARY_SPECIFIED_STAGE or btype == BOUNDARY_MASKED_WEST_STAGE:
        pressure = subgrid_stage_pressure(
            face_z_x, j, i, BED_FLOAT(eta_L), BED_FLOAT(eta_R), factor
        )
        numerator = q_old - gravity * dt * WP_FLOAT(pressure / BED_FLOAT(dx))
    q_new = numerator / denominator

    if btype == BOUNDARY_MASKED_WEST_FREE:
        q_new = wp.min(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        q_new = wp.max(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_FREE_OUTFLOW:
        if i == 0:
            q_new = wp.min(q_new, WP_FLOAT(0.0))
        else:
            q_new = wp.max(q_new, WP_FLOAT(0.0))

    return q_new


@wp.func
def _subgrid_update_qy_face(
    i: int,
    j: int,
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_y: wp.array3d(dtype=BED_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    face_n_y: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
) -> WP_FLOAT:
    """Subgrid counterpart of ``_update_qy_face`` (native FREE mode only)."""
    ny = qy_old.shape[0] - 1
    nx = qy_old.shape[1]

    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_CLOSED:
        return WP_FLOAT(0.0)

    eta_U = WP_FLOAT(0.0)
    eta_D = WP_FLOAT(0.0)
    nf = WP_FLOAT(0.0)

    if btype == BOUNDARY_INTERNAL:
        eta_U = eta_stage[j - 1, i]
        eta_D = eta_stage[j, i]
        if manning_face_mode == 1:
            nf = WP_FLOAT(0.5) * (manning_n[j - 1, i] + manning_n[j, i])
        else:
            nf = wp.max(manning_n[j - 1, i], manning_n[j, i])
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        eta_U = eta_stage[j - 1, i]
        eta_D = boundary_stage_qy[j, i]
        nf = manning_n[j - 1, i]
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        eta_U = WP_FLOAT(face_z_min_y[j, i])
        eta_D = eta_stage[j, i]
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        eta_U = eta_stage[j - 1, i]
        eta_D = WP_FLOAT(face_z_min_y[j, i])
        nf = manning_n[j - 1, i]
    elif btype == BOUNDARY_FREE_OUTFLOW or btype == BOUNDARY_SPECIFIED_STAGE:
        if j == 0:
            eta_D = eta_stage[0, i]
            if btype == BOUNDARY_FREE_OUTFLOW:
                eta_U = WP_FLOAT(face_z_min_y[0, i])
            else:
                eta_U = boundary_stage_qy[0, i]
            nf = manning_n[0, i]
        else:
            eta_U = eta_stage[ny - 1, i]
            if btype == BOUNDARY_FREE_OUTFLOW:
                eta_D = WP_FLOAT(face_z_min_y[ny, i])
            else:
                eta_D = boundary_stage_qy[ny, i]
            nf = manning_n[ny - 1, i]

    eta_f = wp.max(BED_FLOAT(eta_U), BED_FLOAT(eta_D))
    max_local_depth = wp.max(
        BED_FLOAT(0.0), eta_f - face_z_min_y[j, i]
    )
    if max_local_depth < BED_FLOAT(dry_depth):
        return WP_FLOAT(0.0)

    h_a = subgrid_face_area_depth(face_z_y, j, i, eta_f, factor)
    h_a_hydraulic = WP_FLOAT(h_a)
    Sf = WP_FLOAT((BED_FLOAT(eta_D) - BED_FLOAT(eta_U)) / BED_FLOAT(dy))

    q_old = qy_old[j, i]
    friction_discharge = wp.abs(q_old)
    if use_two_dimensional_friction == 1 and j > 0 and j < ny:
        transverse_discharge = WP_FLOAT(0.25) * (
            qx_old[j - 1, i] + qx_old[j - 1, i + 1] + qx_old[j, i] + qx_old[j, i + 1]
        )
        friction_discharge = wp.sqrt(q_old * q_old + transverse_discharge * transverse_discharge)

    if use_native_manning == 1:
        # Variable-roughness reduction: K = mean(h_k^(5/3) / n_k) over the
        # native face segments, so n_eff = h_A^(5/3) / K independent of the
        # coarse face value nf. Dry segments contribute zero.
        conductance = subgrid_face_conveyance_conductance(
            face_z_y, face_n_y, j, i, eta_f, factor
        )
        n_eff = nf
        if conductance > BED_FLOAT(0.0):
            n_eff = WP_FLOAT(
                wp.pow(h_a, BED_FLOAT(5.0 / 3.0)) / conductance
            )
    else:
        # Uniform coarse-face-roughness reduction, unchanged when no native
        # Manning raster is supplied.
        moment = subgrid_face_conveyance_moment(face_z_y, j, i, eta_f, factor)
        n_eff = nf
        if moment > BED_FLOAT(0.0):
            n_eff = nf * WP_FLOAT(
                wp.pow(h_a, BED_FLOAT(5.0 / 3.0)) / moment
            )

    denominator = WP_FLOAT(1.0) + gravity * n_eff * n_eff * dt * friction_discharge / wp.pow(
        wp.max(h_a_hydraulic, depth_epsilon), WP_FLOAT(7.0 / 3.0)
    )
    numerator = q_old - gravity * h_a_hydraulic * dt * Sf
    if btype == BOUNDARY_SPECIFIED_STAGE or btype == BOUNDARY_MASKED_NORTH_STAGE:
        pressure = subgrid_stage_pressure(
            face_z_y, j, i, BED_FLOAT(eta_U), BED_FLOAT(eta_D), factor
        )
        numerator = q_old - gravity * dt * WP_FLOAT(pressure / BED_FLOAT(dy))
    q_new = numerator / denominator

    if btype == BOUNDARY_MASKED_SOUTH_FREE:
        q_new = wp.min(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        q_new = wp.max(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_FREE_OUTFLOW:
        if j == 0:
            q_new = wp.min(q_new, WP_FLOAT(0.0))
        else:
            q_new = wp.max(q_new, WP_FLOAT(0.0))

    return q_new


@wp.kernel
def subgrid_update_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_x: wp.array3d(dtype=BED_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    face_n_x: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    j, i = wp.tid()
    qx[j, i] = _subgrid_update_qx_face(
        i, j, qx_old, qy_old, eta_stage, face_z_x, face_z_min_x, face_n_x, manning_n,
        boundary_type_qx, boundary_stage_qx,
        dx, dt, gravity, dry_depth, depth_epsilon,
        manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
    )


@wp.kernel
def subgrid_update_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_y: wp.array3d(dtype=BED_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    face_n_y: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    j, i = wp.tid()
    qy[j, i] = _subgrid_update_qy_face(
        i, j, qy_old, qx_old, eta_stage, face_z_y, face_z_min_y, face_n_y, manning_n,
        boundary_type_qy, boundary_stage_qy,
        dy, dt, gravity, dry_depth, depth_epsilon,
        manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
    )


@wp.kernel
def subgrid_update_momentum_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_x: wp.array3d(dtype=BED_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    face_n_x: wp.array3d(dtype=WP_FLOAT),
    face_z_y: wp.array3d(dtype=BED_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    face_n_y: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    """Fused subgrid qx + qy momentum update over the union face grid."""
    j, i = wp.tid()
    if j < qx.shape[0]:
        qx[j, i] = _subgrid_update_qx_face(
            i, j, qx_old, qy_old, eta_stage, face_z_x, face_z_min_x, face_n_x, manning_n,
            boundary_type_qx, boundary_stage_qx,
            dx, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
        )
    if i < qy.shape[1]:
        qy[j, i] = _subgrid_update_qy_face(
            i, j, qy_old, qx_old, eta_stage, face_z_y, face_z_min_y, face_n_y, manning_n,
            boundary_type_qy, boundary_stage_qy,
            dy, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
        )


@wp.func
def _subgrid_face_max_local_depth(
    eta_L: WP_FLOAT,
    eta_R: WP_FLOAT,
    face_z_min: wp.array2d(dtype=BED_FLOAT),
    j: int,
    i: int,
) -> BED_FLOAT:
    eta_f = wp.max(BED_FLOAT(eta_L), BED_FLOAT(eta_R))
    return wp.max(BED_FLOAT(0.0), eta_f - face_z_min[j, i])


@wp.kernel
def subgrid_calc_cfl_dt_x_kernel(
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_min_out: wp.array(dtype=WP_FLOAT),
):
    """CFL from the maximum local face depth (conservative wave speed)."""
    j, i = wp.tid()
    nx = eta_stage.shape[1]

    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_CLOSED:
        return

    eta_L = WP_FLOAT(0.0)
    eta_R = WP_FLOAT(0.0)
    if btype == BOUNDARY_INTERNAL:
        eta_L = eta_stage[j, i - 1]
        eta_R = eta_stage[j, i]
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        eta_L = boundary_stage_qx[j, i]
        eta_R = eta_stage[j, i]
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        eta_L = WP_FLOAT(face_z_min_x[j, i])
        eta_R = eta_stage[j, i]
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        eta_L = eta_stage[j, i - 1]
        eta_R = WP_FLOAT(face_z_min_x[j, i])
    elif i == 0:
        eta_R = eta_stage[j, 0]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            eta_L = boundary_stage_qx[j, 0]
        else:
            eta_L = WP_FLOAT(face_z_min_x[j, 0])
    else:
        eta_L = eta_stage[j, nx - 1]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            eta_R = boundary_stage_qx[j, nx]
        else:
            eta_R = WP_FLOAT(face_z_min_x[j, nx])

    h_max = _subgrid_face_max_local_depth(eta_L, eta_R, face_z_min_x, j, i)
    if h_max > BED_FLOAT(dry_depth):
        celerity = wp.sqrt(BED_FLOAT(gravity) * h_max)
        if celerity > WP_FLOAT(0.0):
            dt = WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dx) / celerity)
            wp.atomic_min(dt_min_out, 0, dt)


@wp.kernel
def subgrid_calc_cfl_dt_y_kernel(
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_min_out: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    ny = eta_stage.shape[0]

    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_CLOSED:
        return

    eta_U = WP_FLOAT(0.0)
    eta_D = WP_FLOAT(0.0)
    if btype == BOUNDARY_INTERNAL:
        eta_U = eta_stage[j - 1, i]
        eta_D = eta_stage[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        eta_U = eta_stage[j - 1, i]
        eta_D = boundary_stage_qy[j, i]
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        eta_U = WP_FLOAT(face_z_min_y[j, i])
        eta_D = eta_stage[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        eta_U = eta_stage[j - 1, i]
        eta_D = WP_FLOAT(face_z_min_y[j, i])
    elif j == 0:
        eta_D = eta_stage[0, i]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            eta_U = boundary_stage_qy[0, i]
        else:
            eta_U = WP_FLOAT(face_z_min_y[0, i])
    else:
        eta_U = eta_stage[ny - 1, i]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            eta_D = boundary_stage_qy[ny, i]
        else:
            eta_D = WP_FLOAT(face_z_min_y[ny, i])

    h_max = _subgrid_face_max_local_depth(eta_U, eta_D, face_z_min_y, j, i)
    if h_max > BED_FLOAT(dry_depth):
        celerity = wp.sqrt(BED_FLOAT(gravity) * h_max)
        if celerity > WP_FLOAT(0.0):
            dt = WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dy) / celerity)
            wp.atomic_min(dt_min_out, 0, dt)


@wp.kernel
def subgrid_lts_face_dt_x_kernel(
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_cap: WP_FLOAT,
    face_dt: wp.array2d(dtype=WP_FLOAT),
    global_dt_min: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    nx = eta_stage.shape[1]
    btype = boundary_type_qx[j, i]
    dt = dt_cap
    if btype != BOUNDARY_CLOSED:
        eta_L = WP_FLOAT(0.0)
        eta_R = WP_FLOAT(0.0)
        if btype == BOUNDARY_INTERNAL:
            eta_L = eta_stage[j, i - 1]
            eta_R = eta_stage[j, i]
        elif btype == BOUNDARY_MASKED_WEST_STAGE:
            eta_L = boundary_stage_qx[j, i]
            eta_R = eta_stage[j, i]
        elif btype == BOUNDARY_MASKED_WEST_FREE:
            eta_L = WP_FLOAT(face_z_min_x[j, i])
            eta_R = eta_stage[j, i]
        elif btype == BOUNDARY_MASKED_EAST_FREE:
            eta_L = eta_stage[j, i - 1]
            eta_R = WP_FLOAT(face_z_min_x[j, i])
        elif i == 0:
            eta_R = eta_stage[j, 0]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_L = boundary_stage_qx[j, 0]
            else:
                eta_L = WP_FLOAT(face_z_min_x[j, 0])
        else:
            eta_L = eta_stage[j, nx - 1]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_R = boundary_stage_qx[j, nx]
            else:
                eta_R = WP_FLOAT(face_z_min_x[j, nx])
        h_max = _subgrid_face_max_local_depth(eta_L, eta_R, face_z_min_x, j, i)
        if h_max > BED_FLOAT(dry_depth):
            celerity = wp.sqrt(BED_FLOAT(gravity) * h_max)
            if celerity > WP_FLOAT(0.0):
                dt = wp.min(
                    dt_cap,
                    WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dx) / celerity),
                )
                wp.atomic_min(global_dt_min, 0, dt)
    face_dt[j, i] = dt


@wp.kernel
def subgrid_lts_face_dt_y_kernel(
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_cap: WP_FLOAT,
    face_dt: wp.array2d(dtype=WP_FLOAT),
    global_dt_min: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    ny = eta_stage.shape[0]
    btype = boundary_type_qy[j, i]
    dt = dt_cap
    if btype != BOUNDARY_CLOSED:
        eta_U = WP_FLOAT(0.0)
        eta_D = WP_FLOAT(0.0)
        if btype == BOUNDARY_INTERNAL:
            eta_U = eta_stage[j - 1, i]
            eta_D = eta_stage[j, i]
        elif btype == BOUNDARY_MASKED_NORTH_STAGE:
            eta_U = eta_stage[j - 1, i]
            eta_D = boundary_stage_qy[j, i]
        elif btype == BOUNDARY_MASKED_SOUTH_FREE:
            eta_U = WP_FLOAT(face_z_min_y[j, i])
            eta_D = eta_stage[j, i]
        elif btype == BOUNDARY_MASKED_NORTH_FREE:
            eta_U = eta_stage[j - 1, i]
            eta_D = WP_FLOAT(face_z_min_y[j, i])
        elif j == 0:
            eta_D = eta_stage[0, i]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_U = boundary_stage_qy[0, i]
            else:
                eta_U = WP_FLOAT(face_z_min_y[0, i])
        else:
            eta_U = eta_stage[ny - 1, i]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_D = boundary_stage_qy[ny, i]
            else:
                eta_D = WP_FLOAT(face_z_min_y[ny, i])
        h_max = _subgrid_face_max_local_depth(eta_U, eta_D, face_z_min_y, j, i)
        if h_max > BED_FLOAT(dry_depth):
            celerity = wp.sqrt(BED_FLOAT(gravity) * h_max)
            if celerity > WP_FLOAT(0.0):
                dt = wp.min(
                    dt_cap,
                    WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dy) / celerity),
                )
                wp.atomic_min(global_dt_min, 0, dt)
    face_dt[j, i] = dt


@wp.kernel
def lts_update_subgrid_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_x: wp.array3d(dtype=BED_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    face_n_x: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qx_old.shape[1]
    j = flat // width
    i = flat - j * width
    qx_next[j, i] = _subgrid_update_qx_face(
        i, j, qx_old, qy_old, eta_stage, face_z_x, face_z_min_x, face_n_x, manning_n,
        boundary_type_qx, boundary_stage_qx, dx, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
    )


@wp.kernel
def lts_update_subgrid_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_y: wp.array3d(dtype=BED_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    face_n_y: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qy_old.shape[1]
    j = flat // width
    i = flat - j * width
    qy_next[j, i] = _subgrid_update_qy_face(
        i, j, qy_old, qx_old, eta_stage, face_z_y, face_z_min_y, face_n_y, manning_n,
        boundary_type_qy, boundary_stage_qy, dy, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
    )


@wp.kernel
def lts_graph_update_subgrid_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_x: wp.array3d(dtype=BED_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    face_n_x: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    face_levels: wp.array2d(dtype=wp.int32),
    schedule_control: wp.array(dtype=wp.float64),
    tick: wp.array(dtype=wp.int32),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    tid = wp.tid()
    flat = indices[tid]
    width = qx_old.shape[1]
    j = flat // width
    i = flat - j * width
    level = face_levels[j, i]
    if _lts_face_is_scheduled(level, tick[0]) == 0:
        return
    dt = WP_FLOAT(schedule_control[LTS_CTRL_BASE_DT]) * WP_FLOAT(1 << level)
    qx_next[j, i] = _subgrid_update_qx_face(
        i, j, qx_old, qy_old, eta_stage, face_z_x, face_z_min_x, face_n_x, manning_n,
        boundary_type_qx, boundary_stage_qx, dx, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
    )


@wp.kernel
def lts_graph_update_subgrid_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_y: wp.array3d(dtype=BED_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    face_n_y: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    face_levels: wp.array2d(dtype=wp.int32),
    schedule_control: wp.array(dtype=wp.float64),
    tick: wp.array(dtype=wp.int32),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    tid = wp.tid()
    flat = indices[tid]
    width = qy_old.shape[1]
    j = flat // width
    i = flat - j * width
    level = face_levels[j, i]
    if _lts_face_is_scheduled(level, tick[0]) == 0:
        return
    dt = WP_FLOAT(schedule_control[LTS_CTRL_BASE_DT]) * WP_FLOAT(1 << level)
    qy_next[j, i] = _subgrid_update_qy_face(
        i, j, qy_old, qx_old, eta_stage, face_z_y, face_z_min_y, face_n_y, manning_n,
        boundary_type_qy, boundary_stage_qy, dy, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
    )


@wp.kernel
def device_loop_subgrid_update_momentum_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    eta_stage: wp.array2d(dtype=WP_FLOAT),
    face_z_x: wp.array3d(dtype=BED_FLOAT),
    face_z_min_x: wp.array2d(dtype=BED_FLOAT),
    face_n_x: wp.array3d(dtype=WP_FLOAT),
    face_z_y: wp.array3d(dtype=BED_FLOAT),
    face_z_min_y: wp.array2d(dtype=BED_FLOAT),
    face_n_y: wp.array3d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    use_native_manning: int,
    factor: int,
):
    """Fused subgrid momentum update with the device-loop control dt."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    dt = WP_FLOAT(control[DL_DT])
    if j < qx.shape[0]:
        qx[j, i] = _subgrid_update_qx_face(
            i, j, qx_old, qy_old, eta_stage, face_z_x, face_z_min_x, face_n_x, manning_n,
            boundary_type_qx, boundary_stage_qx,
            dx, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
        )
    if i < qy.shape[1]:
        qy[j, i] = _subgrid_update_qy_face(
            i, j, qy_old, qx_old, eta_stage, face_z_y, face_z_min_y, face_n_y, manning_n,
            boundary_type_qy, boundary_stage_qy,
            dy, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, use_native_manning, factor,
        )
