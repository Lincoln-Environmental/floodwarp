import warp as wp
from floodwarp.config import (
    BED_FLOAT,
    STAGE_FLOAT,
    USE_CONDITIONED_MIXED_KERNELS,
    USE_DIRECT_GRADIENT,
    WP_FLOAT,
)
from floodwarp.subgrid_hypsometry import subgrid_wet_fraction_from_storage

# We can import constants, but they need to be available in Warp compilation.
# Hardcode the constants here for warp kernels to avoid import issues.
BOUNDARY_INTERNAL = 0
BOUNDARY_CLOSED = 1
BOUNDARY_FREE_OUTFLOW = 2
BOUNDARY_SPECIFIED_STAGE = 3
# Interior qy face with active j-1 cell, inactive j cell, and a
# prescribed external stage on the north-facing mask boundary.
BOUNDARY_MASKED_NORTH_STAGE = 4
# Interior qx face with inactive i-1 cell, active i cell, and a
# prescribed external stage on a west-facing mask-boundary step.
BOUNDARY_MASKED_WEST_STAGE = 5
# One-way free outflow across an interior mask boundary against an
# inactive (nodata) cell: a ghost cell at the active bed elevation with
# zero depth (free drainage over a sill at bed level), then the ordinary
# local-inertial face update and an outward-only clamp.  The default
# profile formulation is used under every validation profile — the
# ``lisflood_fp_acc`` normal-flow FREE branch is edge-only and is
# deliberately not extended to masked faces.
# WEST: interior qx face, inactive i-1 / active i; outward is q < 0.
BOUNDARY_MASKED_WEST_FREE = 6
# EAST: interior qx face, active i-1 / inactive i; outward is q > 0.
BOUNDARY_MASKED_EAST_FREE = 7
# SOUTH: interior qy face, inactive j-1 / active j; outward is q < 0.
BOUNDARY_MASKED_SOUTH_FREE = 8
# NORTH: interior qy face, active j-1 / inactive j; outward is q > 0.
BOUNDARY_MASKED_NORTH_FREE = 9
# Fixed stage on the remaining two mask-edge orientations.
BOUNDARY_MASKED_EAST_STAGE = 10
BOUNDARY_MASKED_SOUTH_STAGE = 11

# ``free_outflow_mode`` values above one remain deliberately local to the
# opt-in LISFLOOD validation path.  They retain the ACC FREE equation but
# reproduce the reference executable's float arithmetic at its unstable
# no-slope outlet.  Generic production FREE remains mode zero.
FREE_OUTFLOW_MODE_LISFLOOD_ACC = 1
FREE_OUTFLOW_MODE_LISFLOOD_FLOAT_SURFACE = 2
FREE_OUTFLOW_MODE_LISFLOOD_FLOAT_OPERATOR = 3
FREE_OUTFLOW_MODE_LISFLOOD_FP64_OPERATOR = 4

@wp.kernel
def set_uniform_field_kernel(
    field: wp.array2d(dtype=WP_FLOAT),
    value: WP_FLOAT,
):
    j, i = wp.tid()
    field[j, i] = value


@wp.kernel
def set_scaled_rainfall_kernel(
    field: wp.array2d(dtype=WP_FLOAT),
    multiplier: wp.array2d(dtype=WP_FLOAT),
    reference_rate: WP_FLOAT,
):
    j, i = wp.tid()
    field[j, i] = reference_rate * multiplier[j, i]


@wp.kernel
def max_active_depth_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    maximum_depth: wp.array(dtype=wp.float64),
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        wp.atomic_max(maximum_depth, 0, wp.float64(water_depth[j, i]))

@wp.kernel
def predict_depth_for_cfl_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    additional_source_rate: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dt: WP_FLOAT,
    predicted_depth: wp.array2d(dtype=WP_FLOAT)
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        source_rate = wp.max(WP_FLOAT(0.0), rainfall_rate[j, i]) + wp.max(WP_FLOAT(0.0), other_source_rate[j, i]) + wp.max(WP_FLOAT(0.0), additional_source_rate[j, i])
        predicted_depth[j, i] = water_depth[j, i] + source_rate * dt
    else:
        predicted_depth[j, i] = WP_FLOAT(0.0)

@wp.func
def calc_face_h(eta_L: BED_FLOAT, eta_R: BED_FLOAT, z_L: BED_FLOAT, z_R: BED_FLOAT) -> BED_FLOAT:
    """Calculate water depth at cell face using maximum upwind elevation."""
    return wp.max(BED_FLOAT(0.0), wp.max(eta_L, eta_R) - wp.max(z_L, z_R))


@wp.func
def _lisflood_float_surface_dh(
    edge_bed: BED_FLOAT,
    edge_depth: WP_FLOAT,
    inward_bed: BED_FLOAT,
    inward_depth: WP_FLOAT,
    spacing: WP_FLOAT,
) -> WP_FLOAT:
    """Return boundary.cpp's left-associative float ``dh``.

    The local LISFLOOD build uses ``NUMERIC_TYPE=float`` and evaluates
    ``dh = z0 + h0 - z1 - h1``.  The sequence of three rounded operations is
    intentionally different from subtracting two pre-formed water surfaces.
    """
    z0 = wp.float32(edge_bed)
    h0 = wp.float32(edge_depth)
    z1 = wp.float32(inward_bed)
    h1 = wp.float32(inward_depth)
    dh = wp.float32(z0 + h0)
    dh = wp.float32(dh - z1)
    dh = wp.float32(dh - h1)
    return WP_FLOAT(dh)


@wp.func
def _lisflood_float_surface_slope_magnitude(
    edge_bed: BED_FLOAT,
    edge_depth: WP_FLOAT,
    inward_bed: BED_FLOAT,
    inward_depth: WP_FLOAT,
    spacing: WP_FLOAT,
) -> WP_FLOAT:
    """Return ``abs(Sf)`` with boundary.cpp's left-associative float dh."""
    dh = _lisflood_float_surface_dh(
        edge_bed, edge_depth, inward_bed, inward_depth, spacing,
    )
    return wp.abs(WP_FLOAT(wp.float32(-wp.float32(dh) / wp.float32(spacing))))


@wp.func
def _lisflood_float_free_unit_discharge(
    q_old: WP_FLOAT,
    edge_depth: WP_FLOAT,
    edge_bed: BED_FLOAT,
    inward_depth: WP_FLOAT,
    inward_bed: BED_FLOAT,
    manning_n: WP_FLOAT,
    spacing: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    direction: WP_FLOAT,
) -> WP_FLOAT:
    """Mirror the float ``FREE`` assignment in LISFLOOD boundary.cpp.

    Every named intermediate corresponds to a ``NUMERIC_TYPE`` temporary or
    the final ``*qoldptr`` assignment in the reference source.
    """
    z0 = wp.float32(edge_bed)
    h0 = wp.float32(edge_depth)
    z1 = wp.float32(inward_bed)
    h1 = wp.float32(inward_depth)
    q0 = wp.float32(q_old)
    fn = wp.float32(manning_n)
    g = wp.float32(gravity)
    tstep = wp.float32(dt)
    dx = wp.float32(spacing)

    dh = wp.float32(z0 + h0)
    dh = wp.float32(dh - z1)
    dh = wp.float32(dh - h1)
    sf = wp.float32(-dh / dx)

    gravity_increment = wp.float32(g * tstep)
    gravity_increment = wp.float32(gravity_increment * h0)
    gravity_increment = wp.float32(gravity_increment * sf)
    numerator = wp.float32(wp.abs(q0) + wp.abs(gravity_increment))
    h_power = wp.pow(h0, wp.float32(10.0) / wp.float32(3.0))
    friction = wp.float32(g * tstep)
    friction = wp.float32(friction * h0)
    friction = wp.float32(friction * fn)
    friction = wp.float32(friction * fn)
    friction = wp.float32(friction * wp.abs(q0))
    denominator = wp.float32(1.0 + friction / h_power)
    q_new = wp.float32(numerator / denominator)
    return WP_FLOAT(wp.float32(direction) * q_new)


@wp.func
def _lisflood_fp64_free_unit_discharge(
    q_old: WP_FLOAT,
    edge_depth: WP_FLOAT,
    edge_bed: BED_FLOAT,
    inward_depth: WP_FLOAT,
    inward_bed: BED_FLOAT,
    manning_n: WP_FLOAT,
    spacing: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    direction: WP_FLOAT,
) -> WP_FLOAT:
    """Evaluate the complete FREE recurrence in FP64 from the stored state."""
    z0 = wp.float64(edge_bed)
    h0 = wp.float64(edge_depth)
    z1 = wp.float64(inward_bed)
    h1 = wp.float64(inward_depth)
    q0 = wp.float64(q_old)
    fn = wp.float64(manning_n)
    g = wp.float64(gravity)
    tstep = wp.float64(dt)
    dx = wp.float64(spacing)
    dh = z0 + h0
    dh = dh - z1
    dh = dh - h1
    sf = -dh / dx
    numerator = wp.abs(q0) + wp.abs(g * tstep * h0 * sf)
    denominator = wp.float64(1.0) + (
        g * tstep * h0 * fn * fn * wp.abs(q0)
        / wp.pow(h0, wp.float64(10.0) / wp.float64(3.0))
    )
    return WP_FLOAT(wp.float64(direction) * numerator / denominator)

@wp.kernel
def calc_cfl_dt_x_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_min_out: wp.array(dtype=WP_FLOAT)
):
    j, i = wp.tid()
    ny = water_depth.shape[0]
    nx = water_depth.shape[1]

    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_CLOSED:
        return

    z_L = BED_FLOAT(0.0)
    z_R = BED_FLOAT(0.0)
    h_L = WP_FLOAT(0.0)
    h_R = WP_FLOAT(0.0)
    eta_L = BED_FLOAT(0.0)
    eta_R = BED_FLOAT(0.0)

    if btype == BOUNDARY_INTERNAL:
        z_L = bed_elevation[j, i-1]
        z_R = bed_elevation[j, i]
        h_L = water_depth[j, i-1]
        h_R = water_depth[j, i]
        eta_L = z_L + BED_FLOAT(h_L)
        eta_R = z_R + BED_FLOAT(h_R)
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        z_R = bed_elevation[j, i]
        z_L = z_R
        h_R = water_depth[j, i]
        eta_R = z_R + BED_FLOAT(h_R)
        eta_L = wp.max(boundary_stage_qx[j, i], z_L)
        h_L = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_L - z_L))
    elif btype == BOUNDARY_MASKED_EAST_STAGE:
        z_L = bed_elevation[j, i - 1]
        z_R = z_L
        h_L = water_depth[j, i - 1]
        eta_L = z_L + BED_FLOAT(h_L)
        eta_R = wp.max(boundary_stage_qx[j, i], z_R)
        h_R = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_R - z_R))
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        z_R = bed_elevation[j, i]
        z_L = z_R
        h_R = water_depth[j, i]
        eta_R = z_R + BED_FLOAT(h_R)
        eta_L = z_L
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        z_L = bed_elevation[j, i - 1]
        z_R = z_L
        h_L = water_depth[j, i - 1]
        eta_L = z_L + BED_FLOAT(h_L)
        eta_R = z_R
    elif btype == BOUNDARY_FREE_OUTFLOW or btype == BOUNDARY_SPECIFIED_STAGE:
        if i == 0: # Western boundary
            z_R = bed_elevation[j, 0]
            h_R = water_depth[j, 0]
            eta_R = z_R + BED_FLOAT(h_R)
            z_L = z_R
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_L = WP_FLOAT(0.0)
                eta_L = z_L + BED_FLOAT(h_L)
            else:
                eta_L = wp.max(boundary_stage_qx[j, 0], z_L)
                h_L = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_L - z_L))
        else: # Eastern boundary
            z_L = bed_elevation[j, nx-1]
            h_L = water_depth[j, nx-1]
            eta_L = z_L + BED_FLOAT(h_L)
            z_R = z_L
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_R = WP_FLOAT(0.0)
                eta_R = z_R + BED_FLOAT(h_R)
            else:
                eta_R = wp.max(boundary_stage_qx[j, nx], z_R)
                h_R = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_R - z_R))

    hf = calc_face_h(eta_L, eta_R, z_L, z_R)
    if hf > dry_depth:
        celerity = wp.sqrt(BED_FLOAT(gravity) * hf)
        if celerity > WP_FLOAT(0.0):
            dt = WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dx) / celerity)
            wp.atomic_min(dt_min_out, 0, dt)

@wp.kernel
def calc_cfl_dt_y_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_min_out: wp.array(dtype=WP_FLOAT)
):
    j, i = wp.tid()
    ny = water_depth.shape[0]
    nx = water_depth.shape[1]

    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_CLOSED:
        return

    z_U = BED_FLOAT(0.0)
    z_D = BED_FLOAT(0.0)
    h_U = WP_FLOAT(0.0)
    h_D = WP_FLOAT(0.0)
    eta_U = BED_FLOAT(0.0)
    eta_D = BED_FLOAT(0.0)

    if btype == BOUNDARY_INTERNAL:
        z_U = bed_elevation[j-1, i]
        z_D = bed_elevation[j, i]
        h_U = water_depth[j-1, i]
        h_D = water_depth[j, i]
        eta_U = z_U + BED_FLOAT(h_U)
        eta_D = z_D + BED_FLOAT(h_D)
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        z_U = bed_elevation[j - 1, i]
        z_D = z_U
        h_U = water_depth[j - 1, i]
        eta_U = z_U + BED_FLOAT(h_U)
        eta_D = wp.max(boundary_stage_qy[j, i], z_D)
        h_D = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_D - z_D))
    elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
        z_D = bed_elevation[j, i]
        z_U = z_D
        h_D = water_depth[j, i]
        eta_D = z_D + BED_FLOAT(h_D)
        eta_U = wp.max(boundary_stage_qy[j, i], z_U)
        h_U = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_U - z_U))
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        z_D = bed_elevation[j, i]
        z_U = z_D
        h_D = water_depth[j, i]
        eta_D = z_D + BED_FLOAT(h_D)
        eta_U = z_U
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        z_U = bed_elevation[j - 1, i]
        z_D = z_U
        h_U = water_depth[j - 1, i]
        eta_U = z_U + BED_FLOAT(h_U)
        eta_D = z_D
    elif btype == BOUNDARY_FREE_OUTFLOW or btype == BOUNDARY_SPECIFIED_STAGE:
        if j == 0: # Northern boundary
            z_D = bed_elevation[0, i]
            h_D = water_depth[0, i]
            eta_D = z_D + BED_FLOAT(h_D)
            z_U = z_D
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_U = WP_FLOAT(0.0)
                eta_U = z_U + BED_FLOAT(h_U)
            else:
                eta_U = wp.max(boundary_stage_qy[0, i], z_U)
                h_U = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_U - z_U))
        else: # Southern boundary
            z_U = bed_elevation[ny-1, i]
            h_U = water_depth[ny-1, i]
            eta_U = z_U + BED_FLOAT(h_U)
            z_D = z_U
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_D = WP_FLOAT(0.0)
                eta_D = z_D + BED_FLOAT(h_D)
            else:
                eta_D = wp.max(boundary_stage_qy[ny, i], z_D)
                h_D = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_D - z_D))

    hf = calc_face_h(eta_U, eta_D, z_U, z_D)
    if hf > dry_depth:
        celerity = wp.sqrt(BED_FLOAT(gravity) * hf)
        if celerity > WP_FLOAT(0.0):
            dt = WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dy) / celerity)
            wp.atomic_min(dt_min_out, 0, dt)


@wp.kernel
def store_cfl_timestep_control_kernel(
    dt_min: wp.array(dtype=WP_FLOAT),
    step_control: wp.array(dtype=wp.float64),
):
    """Store the default-profile CFL result beside the prior error flag.

    The fused continuity kernel writes ``step_control[1]`` at the end of an
    accepted step. The next timestep selection writes its CFL limit to element
    zero, allowing the host to read both values with one transfer.
    """
    step_control[0] = wp.float64(dt_min[0])


# ---------------------------------------------------------------------------
# Conditioned mixed-precision face path
# ---------------------------------------------------------------------------
#
# These kernels deliberately use only WP_FLOAT fields and arithmetic.  The
# solver precomputes dz = z_right - z_left (or z_down - z_up) in FP64 on the
# host, then stores the physically small difference in FP32.  With z_left as
# the local datum, eta_left=h_left and eta_right=dz+h_right, hence
#
#   h_face = max(0, max(h_left, dz + h_right) - max(0, dz))
#   S_f    = (dz + h_right - h_left) / d.
#
# This is algebraically identical to the absolute-elevation formulation while
# avoiding high-datum cancellation and FP32<->FP64 conversions in hot kernels.


@wp.func
def calc_conditioned_face_h(h_left: WP_FLOAT, h_right: WP_FLOAT, dz: WP_FLOAT) -> WP_FLOAT:
    return wp.max(
        WP_FLOAT(0.0),
        wp.max(h_left, dz + h_right) - wp.max(WP_FLOAT(0.0), dz),
    )


@wp.kernel
def calc_conditioned_cfl_dt_x_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_min_out: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    nx = water_depth.shape[1]
    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_CLOSED:
        return

    dz = WP_FLOAT(0.0)
    h_left = WP_FLOAT(0.0)
    h_right = WP_FLOAT(0.0)
    if btype == BOUNDARY_INTERNAL:
        dz = bed_slope_qx[j, i]
        h_left = water_depth[j, i - 1]
        h_right = water_depth[j, i]
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        h_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
        h_right = water_depth[j, i]
    elif btype == BOUNDARY_MASKED_EAST_STAGE:
        h_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
        h_left = water_depth[j, i - 1]
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        h_right = water_depth[j, i]
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        h_left = water_depth[j, i - 1]
    elif i == 0:
        h_right = water_depth[j, 0]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            h_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, 0])
    else:
        h_left = water_depth[j, nx - 1]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            h_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, nx])

    hf = calc_conditioned_face_h(h_left, h_right, dz)
    if hf > dry_depth:
        celerity = wp.sqrt(gravity * hf)
        if celerity > WP_FLOAT(0.0):
            wp.atomic_min(dt_min_out, 0, cfl_number * dx / celerity)


@wp.kernel
def calc_conditioned_cfl_dt_y_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_min_out: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    ny = water_depth.shape[0]
    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_CLOSED:
        return

    dz = WP_FLOAT(0.0)
    h_up = WP_FLOAT(0.0)
    h_down = WP_FLOAT(0.0)
    if btype == BOUNDARY_INTERNAL:
        dz = bed_slope_qy[j, i]
        h_up = water_depth[j - 1, i]
        h_down = water_depth[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        h_up = water_depth[j - 1, i]
        h_down = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[j, i])
    elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
        h_down = water_depth[j, i]
        h_up = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[j, i])
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        h_down = water_depth[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        h_up = water_depth[j - 1, i]
    elif j == 0:
        h_down = water_depth[0, i]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            h_up = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[0, i])
    else:
        h_up = water_depth[ny - 1, i]
        if btype == BOUNDARY_SPECIFIED_STAGE:
            h_down = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[ny, i])

    hf = calc_conditioned_face_h(h_up, h_down, dz)
    if hf > dry_depth:
        celerity = wp.sqrt(gravity * hf)
        if celerity > WP_FLOAT(0.0):
            wp.atomic_min(dt_min_out, 0, cfl_number * dy / celerity)

@wp.func
def _boundary_vector_qx(qy: wp.array2d(dtype=WP_FLOAT), j: int, i: int, btype: int) -> WP_FLOAT:
    """Centred interior reconstruction, one-sided constant exterior extension.

    Closed normal fluxes are already zero in the staggered state. At an active
    wet/dry interface their zeros enter the same bounded arithmetic stencil;
    no division by depth or artificial transverse velocity is introduced.
    """
    nx = qy.shape[1]
    if i == 0 or btype == BOUNDARY_MASKED_WEST_STAGE or btype == BOUNDARY_MASKED_WEST_FREE:
        return WP_FLOAT(0.5) * (qy[j, i] + qy[j + 1, i])
    if i == nx or btype == BOUNDARY_MASKED_EAST_FREE or btype == BOUNDARY_MASKED_EAST_STAGE:
        return WP_FLOAT(0.5) * (qy[j, i - 1] + qy[j + 1, i - 1])
    return WP_FLOAT(0.25) * (qy[j, i - 1] + qy[j, i] + qy[j + 1, i - 1] + qy[j + 1, i])


@wp.func
def _boundary_vector_qy(qx: wp.array2d(dtype=WP_FLOAT), j: int, i: int, btype: int) -> WP_FLOAT:
    ny = qx.shape[0]
    if j == 0 or btype == BOUNDARY_MASKED_SOUTH_FREE or btype == BOUNDARY_MASKED_SOUTH_STAGE:
        return WP_FLOAT(0.5) * (qx[j, i] + qx[j, i + 1])
    if j == ny or btype == BOUNDARY_MASKED_NORTH_STAGE or btype == BOUNDARY_MASKED_NORTH_FREE:
        return WP_FLOAT(0.5) * (qx[j - 1, i] + qx[j - 1, i + 1])
    return WP_FLOAT(0.25) * (qx[j - 1, i] + qx[j - 1, i + 1] + qx[j, i] + qx[j, i + 1])


@wp.func
def _update_qx_face(
    i: int,
    j: int,
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
) -> WP_FLOAT:
    # Shared body of update_qx_kernel and the qx branch of update_momentum_kernel.
    # Math is intentionally identical to the historical single-kernel form so the
    # fused and unfused paths produce bit-equal results.
    ny = qx_old.shape[0]
    nx = qx_old.shape[1] - 1

    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_CLOSED:
        return WP_FLOAT(0.0)

    if btype == BOUNDARY_FREE_OUTFLOW and free_outflow_mode == FREE_OUTFLOW_MODE_LISFLOOD_ACC:
        edge_column = nx - 1
        inward_column = nx - 2
        direction = WP_FLOAT(1.0)
        if i == 0:
            edge_column = 0
            inward_column = 1
            direction = WP_FLOAT(-1.0)
        hflow = water_depth[j, edge_column]
        if hflow <= dry_depth:
            return WP_FLOAT(0.0)
        # A non-zero FREE value is a prescribed slope (LISFLOOD BC_Val); the
        # legacy zero value is local slope, while NaN is the validation-only
        # marker for an explicitly prescribed zero slope.
        prescribed_slope = boundary_stage_qx[j, i]
        has_prescribed_slope = 0
        if wp.isnan(prescribed_slope):
            slope_magnitude = WP_FLOAT(0.0)
            has_prescribed_slope = 1
        elif wp.abs(prescribed_slope) > WP_FLOAT(0.0):
            slope_magnitude = WP_FLOAT(wp.abs(prescribed_slope))
            has_prescribed_slope = 1
        else:
            surface_drop = (bed_elevation[j, edge_column] + BED_FLOAT(hflow)) - (
                bed_elevation[j, inward_column] + BED_FLOAT(water_depth[j, inward_column])
            )
            slope_magnitude = WP_FLOAT(wp.abs(surface_drop / BED_FLOAT(dx)))
        q_old = qx_old[j, i]
        nf = manning_n[j, edge_column]
        friction = wp.abs(q_old)
        if use_two_dimensional_friction == 2:
            transverse = _boundary_vector_qx(qy_old, j, i, btype)
            friction = wp.sqrt(q_old * q_old + transverse * transverse)
        denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction / wp.pow(
            wp.max(hflow, depth_epsilon), WP_FLOAT(7.0 / 3.0)
        )
        return direction * (wp.abs(q_old) + wp.abs(gravity * dt * hflow * WP_FLOAT(slope_magnitude))) / denominator

    z_L = BED_FLOAT(0.0)
    z_R = BED_FLOAT(0.0)
    h_L = WP_FLOAT(0.0)
    h_R = WP_FLOAT(0.0)
    eta_L = BED_FLOAT(0.0)
    eta_R = BED_FLOAT(0.0)
    nf = WP_FLOAT(0.0)

    if btype == BOUNDARY_INTERNAL:
        z_L = bed_elevation[j, i-1]
        z_R = bed_elevation[j, i]
        h_L = water_depth[j, i-1]
        h_R = water_depth[j, i]
        eta_L = z_L + BED_FLOAT(h_L)
        eta_R = z_R + BED_FLOAT(h_R)
        if manning_face_mode == 1:
            nf = WP_FLOAT(0.5) * (manning_n[j, i-1] + manning_n[j, i])
        elif manning_face_mode == 2:
            nf = wp.sqrt(WP_FLOAT(0.5) * (manning_n[j, i-1] * manning_n[j, i-1] + manning_n[j, i] * manning_n[j, i]))
        else:
            nf = wp.max(manning_n[j, i-1], manning_n[j, i])
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        z_R = bed_elevation[j, i]
        z_L = z_R
        h_R = water_depth[j, i]
        eta_R = z_R + BED_FLOAT(h_R)
        eta_L = wp.max(boundary_stage_qx[j, i], z_L)
        h_L = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_L - z_L))
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_EAST_STAGE:
        z_L = bed_elevation[j, i - 1]
        z_R = z_L
        h_L = water_depth[j, i - 1]
        eta_L = z_L + BED_FLOAT(h_L)
        eta_R = wp.max(boundary_stage_qx[j, i], z_R)
        h_R = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_R - z_R))
        nf = manning_n[j, i - 1]
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        # Ghost cell on the inactive side: ghost bed = active bed, ghost
        # depth = 0 (free drainage over a sill at bed level).
        z_R = bed_elevation[j, i]
        z_L = z_R
        h_R = water_depth[j, i]
        eta_R = z_R + BED_FLOAT(h_R)
        h_L = WP_FLOAT(0.0)
        eta_L = z_L
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        z_L = bed_elevation[j, i - 1]
        z_R = z_L
        h_L = water_depth[j, i - 1]
        eta_L = z_L + BED_FLOAT(h_L)
        h_R = WP_FLOAT(0.0)
        eta_R = z_R
        nf = manning_n[j, i - 1]
    elif btype == BOUNDARY_FREE_OUTFLOW or btype == BOUNDARY_SPECIFIED_STAGE:
        if i == 0:
            z_R = bed_elevation[j, 0]
            h_R = water_depth[j, 0]
            eta_R = z_R + BED_FLOAT(h_R)
            z_L = z_R
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_L = WP_FLOAT(0.0)
                eta_L = z_L + BED_FLOAT(h_L)
            else:
                eta_L = wp.max(boundary_stage_qx[j, 0], z_L)
                h_L = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_L - z_L))
            nf = manning_n[j, 0]
        else:
            z_L = bed_elevation[j, nx-1]
            h_L = water_depth[j, nx-1]
            eta_L = z_L + BED_FLOAT(h_L)
            z_R = z_L
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_R = WP_FLOAT(0.0)
                eta_R = z_R + BED_FLOAT(h_R)
            else:
                eta_R = wp.max(boundary_stage_qx[j, nx], z_R)
                h_R = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_R - z_R))
            nf = manning_n[j, nx-1]

    hf = calc_face_h(eta_L, eta_R, z_L, z_R)

    if hf < dry_depth:
        return WP_FLOAT(0.0)

    hf_hydraulic = WP_FLOAT(hf)

    if USE_DIRECT_GRADIENT == 1 and btype == BOUNDARY_INTERNAL:
        surface_drop = (z_R - z_L) + (BED_FLOAT(h_R) - BED_FLOAT(h_L))
        Sf = WP_FLOAT(surface_drop / BED_FLOAT(dx))
    else:
        Sf = WP_FLOAT((eta_R - eta_L) / BED_FLOAT(dx))
    q_old = qx_old[j, i]
    friction_discharge = wp.abs(q_old)
    if use_two_dimensional_friction == 2:
        transverse = _boundary_vector_qx(qy_old, j, i, btype)
        friction_discharge = wp.sqrt(q_old * q_old + transverse * transverse)
    if use_two_dimensional_friction == 1 and i > 0 and i < nx:
        transverse_discharge = WP_FLOAT(0.25) * (
            qy_old[j, i - 1] + qy_old[j, i] + qy_old[j + 1, i - 1] + qy_old[j + 1, i]
        )
        friction_discharge = wp.sqrt(q_old * q_old + transverse_discharge * transverse_discharge)

    denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction_discharge / wp.pow(
        wp.max(hf_hydraulic, depth_epsilon), WP_FLOAT(7.0/3.0)
    )
    numerator = q_old - gravity * hf_hydraulic * dt * Sf
    q_new = numerator / denominator

    if btype == BOUNDARY_FREE_OUTFLOW:
        if i == 0:
            q_new = wp.min(q_new, WP_FLOAT(0.0))
        else:
            q_new = wp.max(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        # Interior faces need an orientation-based clamp: outward is
        # westward (q < 0); nodata can never be a source.
        q_new = wp.min(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        q_new = wp.max(q_new, WP_FLOAT(0.0))

    return q_new


@wp.kernel
def update_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
):
    j, i = wp.tid()
    qx[j, i] = _update_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_elevation, manning_n,
        boundary_type_qx, boundary_stage_qx,
        dx, dt, gravity, dry_depth, depth_epsilon,
        manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
    )

@wp.func
def _update_qy_face(
    i: int,
    j: int,
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
) -> WP_FLOAT:
    # Shared body of update_qy_kernel and the qy branch of update_momentum_kernel.
    ny = qy_old.shape[0] - 1
    nx = qy_old.shape[1]

    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_CLOSED:
        return WP_FLOAT(0.0)

    if btype == BOUNDARY_FREE_OUTFLOW and free_outflow_mode == FREE_OUTFLOW_MODE_LISFLOOD_ACC:
        edge_row = ny - 1
        inward_row = ny - 2
        direction = WP_FLOAT(1.0)
        if j == 0:
            edge_row = 0
            inward_row = 1
            direction = WP_FLOAT(-1.0)
        hflow = water_depth[edge_row, i]
        if hflow <= dry_depth:
            return WP_FLOAT(0.0)
        # A non-zero FREE value is a prescribed slope (LISFLOOD BC_Val); the
        # legacy zero value is local slope, while NaN is the validation-only
        # marker for an explicitly prescribed zero slope.
        prescribed_slope = boundary_stage_qy[j, i]
        has_prescribed_slope = 0
        if wp.isnan(prescribed_slope):
            slope_magnitude = WP_FLOAT(0.0)
            has_prescribed_slope = 1
        elif wp.abs(prescribed_slope) > WP_FLOAT(0.0):
            slope_magnitude = WP_FLOAT(wp.abs(prescribed_slope))
            has_prescribed_slope = 1
        else:
            surface_drop = (bed_elevation[edge_row, i] + BED_FLOAT(hflow)) - (
                bed_elevation[inward_row, i] + BED_FLOAT(water_depth[inward_row, i])
            )
            slope_magnitude = WP_FLOAT(wp.abs(surface_drop / BED_FLOAT(dy)))
        q_old = qy_old[j, i]
        nf = manning_n[edge_row, i]
        friction = wp.abs(q_old)
        if use_two_dimensional_friction == 2:
            transverse = _boundary_vector_qy(qx_old, j, i, btype)
            friction = wp.sqrt(q_old * q_old + transverse * transverse)
        denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction / wp.pow(
            wp.max(hflow, depth_epsilon), WP_FLOAT(7.0 / 3.0)
        )
        return direction * (wp.abs(q_old) + wp.abs(gravity * dt * hflow * WP_FLOAT(slope_magnitude))) / denominator

    z_U = BED_FLOAT(0.0)
    z_D = BED_FLOAT(0.0)
    h_U = WP_FLOAT(0.0)
    h_D = WP_FLOAT(0.0)
    eta_U = BED_FLOAT(0.0)
    eta_D = BED_FLOAT(0.0)
    nf = WP_FLOAT(0.0)

    if btype == BOUNDARY_INTERNAL:
        z_U = bed_elevation[j-1, i]
        z_D = bed_elevation[j, i]
        h_U = water_depth[j-1, i]
        h_D = water_depth[j, i]
        eta_U = z_U + BED_FLOAT(h_U)
        eta_D = z_D + BED_FLOAT(h_D)
        if manning_face_mode == 1:
            nf = WP_FLOAT(0.5) * (manning_n[j-1, i] + manning_n[j, i])
        elif manning_face_mode == 2:
            nf = wp.sqrt(WP_FLOAT(0.5) * (manning_n[j-1, i] * manning_n[j-1, i] + manning_n[j, i] * manning_n[j, i]))
        else:
            nf = wp.max(manning_n[j-1, i], manning_n[j, i])
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        z_U = bed_elevation[j - 1, i]
        z_D = z_U
        h_U = water_depth[j - 1, i]
        eta_U = z_U + BED_FLOAT(h_U)
        eta_D = wp.max(boundary_stage_qy[j, i], z_D)
        h_D = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_D - z_D))
        nf = manning_n[j - 1, i]
    elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
        z_D = bed_elevation[j, i]
        z_U = z_D
        h_D = water_depth[j, i]
        eta_D = z_D + BED_FLOAT(h_D)
        eta_U = wp.max(boundary_stage_qy[j, i], z_U)
        h_U = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_U - z_U))
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        # Ghost cell on the inactive side: ghost bed = active bed, ghost
        # depth = 0 (free drainage over a sill at bed level).
        z_D = bed_elevation[j, i]
        z_U = z_D
        h_D = water_depth[j, i]
        eta_D = z_D + BED_FLOAT(h_D)
        h_U = WP_FLOAT(0.0)
        eta_U = z_U
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        z_U = bed_elevation[j - 1, i]
        z_D = z_U
        h_U = water_depth[j - 1, i]
        eta_U = z_U + BED_FLOAT(h_U)
        h_D = WP_FLOAT(0.0)
        eta_D = z_D
        nf = manning_n[j - 1, i]
    elif btype == BOUNDARY_FREE_OUTFLOW or btype == BOUNDARY_SPECIFIED_STAGE:
        if j == 0:
            z_D = bed_elevation[0, i]
            h_D = water_depth[0, i]
            eta_D = z_D + BED_FLOAT(h_D)
            z_U = z_D
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_U = WP_FLOAT(0.0)
                eta_U = z_U + BED_FLOAT(h_U)
            else:
                eta_U = wp.max(boundary_stage_qy[0, i], z_U)
                h_U = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_U - z_U))
            nf = manning_n[0, i]
        else:
            z_U = bed_elevation[ny-1, i]
            h_U = water_depth[ny-1, i]
            eta_U = z_U + BED_FLOAT(h_U)
            z_D = z_U
            if btype == BOUNDARY_FREE_OUTFLOW:
                h_D = WP_FLOAT(0.0)
                eta_D = z_D + BED_FLOAT(h_D)
            else:
                eta_D = wp.max(boundary_stage_qy[ny, i], z_D)
                h_D = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_D - z_D))
            nf = manning_n[ny-1, i]

    hf = calc_face_h(eta_U, eta_D, z_U, z_D)

    if hf < dry_depth:
        return WP_FLOAT(0.0)

    hf_hydraulic = WP_FLOAT(hf)

    if USE_DIRECT_GRADIENT == 1 and btype == BOUNDARY_INTERNAL:
        surface_drop = (z_D - z_U) + (BED_FLOAT(h_D) - BED_FLOAT(h_U))
        Sf = WP_FLOAT(surface_drop / BED_FLOAT(dy))
    else:
        Sf = WP_FLOAT((eta_D - eta_U) / BED_FLOAT(dy))
    q_old = qy_old[j, i]
    friction_discharge = wp.abs(q_old)
    if use_two_dimensional_friction == 2:
        transverse = _boundary_vector_qy(qx_old, j, i, btype)
        friction_discharge = wp.sqrt(q_old * q_old + transverse * transverse)
    if use_two_dimensional_friction == 1 and j > 0 and j < ny:
        transverse_discharge = WP_FLOAT(0.25) * (
            qx_old[j - 1, i] + qx_old[j - 1, i + 1] + qx_old[j, i] + qx_old[j, i + 1]
        )
        friction_discharge = wp.sqrt(q_old * q_old + transverse_discharge * transverse_discharge)

    denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction_discharge / wp.pow(
        wp.max(hf_hydraulic, depth_epsilon), WP_FLOAT(7.0/3.0)
    )
    numerator = q_old - gravity * hf_hydraulic * dt * Sf
    q_new = numerator / denominator

    if btype == BOUNDARY_FREE_OUTFLOW:
        if j == 0:
            q_new = wp.min(q_new, WP_FLOAT(0.0))
        else:
            q_new = wp.max(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        # Orientation-based clamp: outward is southward (q < 0); nodata
        # can never be a source.
        q_new = wp.min(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        q_new = wp.max(q_new, WP_FLOAT(0.0))

    return q_new


@wp.kernel
def update_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
):
    j, i = wp.tid()
    qy[j, i] = _update_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_elevation, manning_n,
        boundary_type_qy, boundary_stage_qy,
        dy, dt, gravity, dry_depth, depth_epsilon,
        manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def update_momentum_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
):
    """Fused qx + qy momentum update over the union face grid (ny+1, nx+1).

    Each thread (j, i) updates the qx face at [j, i] when j < ny and the qy face
    at [j, i] when i < nx. qx reads qy_old and qy reads qx_old, so the two updates
    are independent within a thread and may be computed together without hazards.
    """
    j, i = wp.tid()
    if j < qx.shape[0]:
        qx[j, i] = _update_qx_face(
            i, j, qx_old, qy_old, water_depth, bed_elevation, manning_n,
            boundary_type_qx, boundary_stage_qx,
            dx, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
        )
    if i < qy.shape[1]:
        qy[j, i] = _update_qy_face(
            i, j, qy_old, qx_old, water_depth, bed_elevation, manning_n,
            boundary_type_qy, boundary_stage_qy,
            dy, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
        )


@wp.func
def _diagnostic_free_override(
    q_current: WP_FLOAT,
    q_old: WP_FLOAT,
    edge_depth: WP_FLOAT,
    edge_bed: BED_FLOAT,
    inward_depth: WP_FLOAT,
    inward_bed: BED_FLOAT,
    manning_n: WP_FLOAT,
    prescribed_slope: STAGE_FLOAT,
    spacing: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    direction: WP_FLOAT,
    arithmetic_mode: int,
) -> WP_FLOAT:
    """Override one local-slope FREE face for an opt-in arithmetic oracle."""
    if edge_depth <= dry_depth:
        return WP_FLOAT(0.0)
    if (
        arithmetic_mode == FREE_OUTFLOW_MODE_LISFLOOD_FLOAT_OPERATOR
        and wp.float32(edge_depth) <= wp.float32(dry_depth)
    ):
        return WP_FLOAT(0.0)
    if wp.isnan(prescribed_slope) or wp.abs(prescribed_slope) > STAGE_FLOAT(0.0):
        return q_current
    if arithmetic_mode == FREE_OUTFLOW_MODE_LISFLOOD_FLOAT_OPERATOR:
        return _lisflood_float_free_unit_discharge(
            q_old, edge_depth, edge_bed, inward_depth, inward_bed,
            manning_n, spacing, dt, gravity, direction,
        )
    if arithmetic_mode == FREE_OUTFLOW_MODE_LISFLOOD_FP64_OPERATOR:
        return _lisflood_fp64_free_unit_discharge(
            q_old, edge_depth, edge_bed, inward_depth, inward_bed,
            manning_n, spacing, dt, gravity, direction,
        )
    slope_magnitude = _lisflood_float_surface_slope_magnitude(
        edge_bed, edge_depth, inward_bed, inward_depth, spacing,
    )
    denominator = WP_FLOAT(1.0) + gravity * manning_n * manning_n * dt * wp.abs(q_old) / wp.pow(
        wp.max(edge_depth, depth_epsilon), WP_FLOAT(7.0 / 3.0)
    )
    return direction * (
        wp.abs(q_old) + wp.abs(gravity * dt * edge_depth * slope_magnitude)
    ) / denominator


@wp.kernel
def override_free_outflow_arithmetic_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    arithmetic_mode: int,
):
    """Apply expensive float FREE arithmetic only in diagnostic runs."""
    j, i = wp.tid()
    ny = water_depth.shape[0]
    nx = water_depth.shape[1]
    if j < ny and boundary_type_qx[j, i] == BOUNDARY_FREE_OUTFLOW:
        edge_column = nx - 1
        inward_column = nx - 2
        direction = WP_FLOAT(1.0)
        if i == 0:
            edge_column = 0
            inward_column = 1
            direction = WP_FLOAT(-1.0)
        qx[j, i] = _diagnostic_free_override(
            qx[j, i], qx_old[j, i], water_depth[j, edge_column],
            bed_elevation[j, edge_column], water_depth[j, inward_column],
            bed_elevation[j, inward_column], manning_n[j, edge_column],
            boundary_stage_qx[j, i], dx, dt, gravity, dry_depth,
            depth_epsilon, direction, arithmetic_mode,
        )
    if i < nx and boundary_type_qy[j, i] == BOUNDARY_FREE_OUTFLOW:
        edge_row = ny - 1
        inward_row = ny - 2
        direction = WP_FLOAT(1.0)
        if j == 0:
            edge_row = 0
            inward_row = 1
            direction = WP_FLOAT(-1.0)
        qy[j, i] = _diagnostic_free_override(
            qy[j, i], qy_old[j, i], water_depth[edge_row, i],
            bed_elevation[edge_row, i], water_depth[inward_row, i],
            bed_elevation[inward_row, i], manning_n[edge_row, i],
            boundary_stage_qy[j, i], dy, dt, gravity, dry_depth,
            depth_epsilon, direction, arithmetic_mode,
        )


@wp.func
def _update_conditioned_qx_face(
    i: int,
    j: int,
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qx: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
) -> WP_FLOAT:
    """Conditioned momentum, with dry stage ghosts at the local bed datum."""
    nx = qx_old.shape[1] - 1
    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_CLOSED:
        return WP_FLOAT(0.0)

    if btype == BOUNDARY_FREE_OUTFLOW and free_outflow_mode == FREE_OUTFLOW_MODE_LISFLOOD_ACC:
        edge_column = nx - 1
        inward_column = nx - 2
        direction = WP_FLOAT(1.0)
        if i == 0:
            edge_column = 0
            inward_column = 1
            direction = WP_FLOAT(-1.0)
        hflow = water_depth[j, edge_column]
        if hflow <= dry_depth:
            return WP_FLOAT(0.0)
        prescribed_slope = boundary_free_slope_qx[j, i]
        slope_magnitude = wp.abs(prescribed_slope)
        if wp.isnan(prescribed_slope):
            slope_magnitude = WP_FLOAT(0.0)
        elif slope_magnitude == WP_FLOAT(0.0):
            surface_drop = bed_slope_qx[j, i] + hflow - water_depth[j, inward_column]
            slope_magnitude = wp.abs(surface_drop / dx)
        q_old = qx_old[j, i]
        nf = manning_n[j, edge_column]
        friction = wp.abs(q_old)
        if use_two_dimensional_friction == 2:
            transverse = _boundary_vector_qx(qy_old, j, i, btype)
            friction = wp.sqrt(q_old * q_old + transverse * transverse)
        denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction / wp.pow(
            wp.max(hflow, depth_epsilon), WP_FLOAT(7.0 / 3.0)
        )
        return direction * (wp.abs(q_old) + wp.abs(gravity * dt * hflow * slope_magnitude)) / denominator

    dz = WP_FLOAT(0.0)
    h_left = WP_FLOAT(0.0)
    h_right = WP_FLOAT(0.0)
    surface_left = WP_FLOAT(0.0)
    surface_right = WP_FLOAT(0.0)
    nf = WP_FLOAT(0.0)
    if btype == BOUNDARY_INTERNAL:
        dz = bed_slope_qx[j, i]
        h_left = water_depth[j, i - 1]
        h_right = water_depth[j, i]
        surface_left = h_left
        surface_right = dz + h_right
        if manning_face_mode == 1:
            nf = WP_FLOAT(0.5) * (manning_n[j, i - 1] + manning_n[j, i])
        elif manning_face_mode == 2:
            nf = wp.sqrt(WP_FLOAT(0.5) * (manning_n[j, i-1] * manning_n[j, i-1] + manning_n[j, i] * manning_n[j, i]))
        else:
            nf = wp.max(manning_n[j, i - 1], manning_n[j, i])
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        h_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
        h_right = water_depth[j, i]
        surface_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
        surface_right = h_right
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_EAST_STAGE:
        h_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
        h_left = water_depth[j, i - 1]
        surface_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
        surface_left = h_left
        nf = manning_n[j, i - 1]
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        # Ghost on the inactive side: zero depth at the active bed, so
        # the ghost surface is exactly the local datum (dz = 0).
        h_right = water_depth[j, i]
        surface_right = h_right
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        h_left = water_depth[j, i - 1]
        surface_left = h_left
        nf = manning_n[j, i - 1]
    elif i == 0:
        h_right = water_depth[j, 0]
        surface_right = h_right
        if btype == BOUNDARY_SPECIFIED_STAGE:
            surface_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, 0])
            h_left = wp.max(WP_FLOAT(0.0), surface_left)
        nf = manning_n[j, 0]
    else:
        h_left = water_depth[j, nx - 1]
        surface_left = h_left
        if btype == BOUNDARY_SPECIFIED_STAGE:
            surface_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, nx])
            h_right = wp.max(WP_FLOAT(0.0), surface_right)
        nf = manning_n[j, nx - 1]

    hf = calc_conditioned_face_h(h_left, h_right, dz)
    if hf < dry_depth:
        return WP_FLOAT(0.0)
    sf = (surface_right - surface_left) / dx
    q_old = qx_old[j, i]
    friction_discharge = wp.abs(q_old)
    if use_two_dimensional_friction == 2:
        transverse = _boundary_vector_qx(qy_old, j, i, btype)
        friction_discharge = wp.sqrt(q_old * q_old + transverse * transverse)
    if use_two_dimensional_friction == 1 and i > 0 and i < nx:
        transverse_discharge = WP_FLOAT(0.25) * (
            qy_old[j, i - 1] + qy_old[j, i] + qy_old[j + 1, i - 1] + qy_old[j + 1, i]
        )
        friction_discharge = wp.sqrt(q_old * q_old + transverse_discharge * transverse_discharge)
    denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction_discharge / wp.pow(
        wp.max(hf, depth_epsilon), WP_FLOAT(7.0 / 3.0)
    )
    q_new = (q_old - gravity * hf * dt * sf) / denominator
    if btype == BOUNDARY_FREE_OUTFLOW:
        if i == 0:
            q_new = wp.min(q_new, WP_FLOAT(0.0))
        else:
            q_new = wp.max(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        q_new = wp.min(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        q_new = wp.max(q_new, WP_FLOAT(0.0))
    return q_new


@wp.func
def _update_conditioned_qy_face(
    i: int,
    j: int,
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qy: wp.array2d(dtype=WP_FLOAT),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
) -> WP_FLOAT:
    """Conditioned momentum, with dry stage ghosts at the local bed datum."""
    ny = qy_old.shape[0] - 1
    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_CLOSED:
        return WP_FLOAT(0.0)

    if btype == BOUNDARY_FREE_OUTFLOW and free_outflow_mode == FREE_OUTFLOW_MODE_LISFLOOD_ACC:
        edge_row = ny - 1
        inward_row = ny - 2
        direction = WP_FLOAT(1.0)
        if j == 0:
            edge_row = 0
            inward_row = 1
            direction = WP_FLOAT(-1.0)
        hflow = water_depth[edge_row, i]
        if hflow <= dry_depth:
            return WP_FLOAT(0.0)
        prescribed_slope = boundary_free_slope_qy[j, i]
        slope_magnitude = wp.abs(prescribed_slope)
        if wp.isnan(prescribed_slope):
            slope_magnitude = WP_FLOAT(0.0)
        elif slope_magnitude == WP_FLOAT(0.0):
            surface_drop = bed_slope_qy[j, i] + hflow - water_depth[inward_row, i]
            slope_magnitude = wp.abs(surface_drop / dy)
        q_old = qy_old[j, i]
        nf = manning_n[edge_row, i]
        friction = wp.abs(q_old)
        if use_two_dimensional_friction == 2:
            transverse = _boundary_vector_qy(qx_old, j, i, btype)
            friction = wp.sqrt(q_old * q_old + transverse * transverse)
        denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction / wp.pow(
            wp.max(hflow, depth_epsilon), WP_FLOAT(7.0 / 3.0)
        )
        return direction * (wp.abs(q_old) + wp.abs(gravity * dt * hflow * slope_magnitude)) / denominator

    dz = WP_FLOAT(0.0)
    h_up = WP_FLOAT(0.0)
    h_down = WP_FLOAT(0.0)
    surface_up = WP_FLOAT(0.0)
    surface_down = WP_FLOAT(0.0)
    nf = WP_FLOAT(0.0)
    if btype == BOUNDARY_INTERNAL:
        dz = bed_slope_qy[j, i]
        h_up = water_depth[j - 1, i]
        h_down = water_depth[j, i]
        surface_up = h_up
        surface_down = dz + h_down
        if manning_face_mode == 1:
            nf = WP_FLOAT(0.5) * (manning_n[j - 1, i] + manning_n[j, i])
        elif manning_face_mode == 2:
            nf = wp.sqrt(WP_FLOAT(0.5) * (manning_n[j-1, i] * manning_n[j-1, i] + manning_n[j, i] * manning_n[j, i]))
        else:
            nf = wp.max(manning_n[j - 1, i], manning_n[j, i])
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        h_up = water_depth[j - 1, i]
        surface_up = h_up
        surface_down = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[j, i])
        h_down = wp.max(WP_FLOAT(0.0), surface_down)
        nf = manning_n[j - 1, i]
    elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
        h_down = water_depth[j, i]
        surface_down = h_down
        surface_up = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[j, i])
        h_up = wp.max(WP_FLOAT(0.0), surface_up)
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        # Ghost on the inactive side: zero depth at the active bed, so
        # the ghost surface is exactly the local datum (dz = 0).
        h_down = water_depth[j, i]
        surface_down = h_down
        nf = manning_n[j, i]
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        h_up = water_depth[j - 1, i]
        surface_up = h_up
        nf = manning_n[j - 1, i]
    elif j == 0:
        h_down = water_depth[0, i]
        surface_down = h_down
        if btype == BOUNDARY_SPECIFIED_STAGE:
            surface_up = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[0, i])
            h_up = wp.max(WP_FLOAT(0.0), surface_up)
        nf = manning_n[0, i]
    else:
        h_up = water_depth[ny - 1, i]
        surface_up = h_up
        if btype == BOUNDARY_SPECIFIED_STAGE:
            surface_down = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[ny, i])
            h_down = wp.max(WP_FLOAT(0.0), surface_down)
        nf = manning_n[ny - 1, i]

    hf = calc_conditioned_face_h(h_up, h_down, dz)
    if hf < dry_depth:
        return WP_FLOAT(0.0)
    sf = (surface_down - surface_up) / dy
    q_old = qy_old[j, i]
    friction_discharge = wp.abs(q_old)
    if use_two_dimensional_friction == 2:
        transverse = _boundary_vector_qy(qx_old, j, i, btype)
        friction_discharge = wp.sqrt(q_old * q_old + transverse * transverse)
    if use_two_dimensional_friction == 1 and j > 0 and j < ny:
        transverse_discharge = WP_FLOAT(0.25) * (
            qx_old[j - 1, i] + qx_old[j - 1, i + 1] + qx_old[j, i] + qx_old[j, i + 1]
        )
        friction_discharge = wp.sqrt(q_old * q_old + transverse_discharge * transverse_discharge)
    denominator = WP_FLOAT(1.0) + gravity * nf * nf * dt * friction_discharge / wp.pow(
        wp.max(hf, depth_epsilon), WP_FLOAT(7.0 / 3.0)
    )
    q_new = (q_old - gravity * hf * dt * sf) / denominator
    if btype == BOUNDARY_FREE_OUTFLOW:
        if j == 0:
            q_new = wp.min(q_new, WP_FLOAT(0.0))
        else:
            q_new = wp.max(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        q_new = wp.min(q_new, WP_FLOAT(0.0))
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        q_new = wp.max(q_new, WP_FLOAT(0.0))
    return q_new


@wp.kernel
def update_conditioned_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qx: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    j, i = wp.tid()
    qx[j, i] = _update_conditioned_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_slope_qx, manning_n,
        boundary_type_qx, boundary_stage_depth_qx, boundary_free_slope_qx,
        dx, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
        use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def update_conditioned_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qy: wp.array2d(dtype=WP_FLOAT),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    j, i = wp.tid()
    qy[j, i] = _update_conditioned_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_slope_qy, manning_n,
        boundary_type_qy, boundary_stage_depth_qy, boundary_free_slope_qy,
        dy, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
        use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def update_conditioned_momentum_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qy: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    j, i = wp.tid()
    if j < qx.shape[0]:
        qx[j, i] = _update_conditioned_qx_face(
            i, j, qx_old, qy_old, water_depth, bed_slope_qx, manning_n,
            boundary_type_qx, boundary_stage_depth_qx, boundary_free_slope_qx,
            dx, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
            use_two_dimensional_friction, free_outflow_mode,
        )
    if i < qy.shape[1]:
        qy[j, i] = _update_conditioned_qy_face(
            i, j, qy_old, qx_old, water_depth, bed_slope_qy, manning_n,
            boundary_type_qy, boundary_stage_depth_qy, boundary_free_slope_qy,
            dy, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
            use_two_dimensional_friction, free_outflow_mode,
        )

@wp.kernel
def calc_outflow_limiter_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    free_outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    epsilon: WP_FLOAT,
    free_outflow_limiter_mode: int,
    free_outflow_diagnostics: wp.array(dtype=wp.float64),
    collect_free_outflow_diagnostics: int,
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        qx_east = qx[j, i+1]
        qx_west = qx[j, i]
        qy_south = qy[j+1, i]
        qy_north = qy[j, i]
        _write_outflow_limiter_cell(
            j, i, water_depth, qx_east, qx_west, qy_south, qy_north,
            outflow_limiter, free_outflow_limiter, boundary_type_qx,
            boundary_type_qy, dx, dy, dt, epsilon, free_outflow_limiter_mode,
            free_outflow_diagnostics, collect_free_outflow_diagnostics,
        )
    else:
        outflow_limiter[j, i] = WP_FLOAT(1.0)
        free_outflow_limiter[j, i] = WP_FLOAT(1.0)

@wp.kernel
def apply_limiter_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    free_outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32)
):
    j, i = wp.tid()
    q = qx[j, i]
    btype = boundary_type_qx[j, i]
    nx = outflow_limiter.shape[1]

    if q != WP_FLOAT(0.0) and btype != BOUNDARY_CLOSED:
        # qx > 0 means flowing west to east (from i-1 to i)
        # qx < 0 means flowing east to west (from i to i-1)

        # Donor cell limits flow
        if q > WP_FLOAT(0.0):
            # i-1 is the donor. if i==0, the external boundary is the donor, which we don't limit.
            if i > 0:
                limiter = outflow_limiter[j, i-1]
                if btype == BOUNDARY_FREE_OUTFLOW:
                    limiter = free_outflow_limiter[j, i-1]
                qx[j, i] = q * limiter
        elif q < WP_FLOAT(0.0):
            # i is the donor. if i==nx, the external boundary is the donor.
            if i < nx:
                limiter = outflow_limiter[j, i]
                if btype == BOUNDARY_FREE_OUTFLOW:
                    limiter = free_outflow_limiter[j, i]
                qx[j, i] = q * limiter

@wp.kernel
def apply_limiter_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    free_outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32)
):
    j, i = wp.tid()
    q = qy[j, i]
    btype = boundary_type_qy[j, i]
    ny = outflow_limiter.shape[0]

    if q != WP_FLOAT(0.0) and btype != BOUNDARY_CLOSED:
        # qy > 0 means flowing south to north (from j-1 to j)
        if q > WP_FLOAT(0.0):
            if j > 0:
                limiter = outflow_limiter[j-1, i]
                if btype == BOUNDARY_FREE_OUTFLOW:
                    limiter = free_outflow_limiter[j-1, i]
                qy[j, i] = q * limiter
        elif q < WP_FLOAT(0.0):
            if j < ny:
                limiter = outflow_limiter[j, i]
                if btype == BOUNDARY_FREE_OUTFLOW:
                    limiter = free_outflow_limiter[j, i]
                qy[j, i] = q * limiter


@wp.kernel
def apply_outflow_limiter_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    epsilon: WP_FLOAT,
    free_outflow_limiter_mode: int,
    free_outflow_diagnostics: wp.array(dtype=wp.float64),
    collect_free_outflow_diagnostics: int,
):
    """Fused cell-centric outflow limiter: compute alpha and apply it in one pass.

    Replaces calc_outflow_limiter_kernel + apply_limiter_qx_kernel +
    apply_limiter_qy_kernel and drops the per-cell ``outflow_limiter`` temporary.

    A face contributes to a cell's ``q_out`` only when that cell is the donor
    (flow leaves the cell), and a face is written only by its donor cell. So the
    faces a cell reads into ``q_out`` are exactly the ones it will write, and it
    reads them before writing; inflow faces are read but discarded. This makes the
    single-pass compute-then-apply race-free and bit-equal to the two-phase form.
    """
    j, i = wp.tid()
    _apply_outflow_limiter_cell(
        j, i, water_depth, qx, qy, active_mask, boundary_type_qx,
        boundary_type_qy, dx, dy, dt, epsilon, free_outflow_limiter_mode,
        free_outflow_diagnostics, collect_free_outflow_diagnostics,
    )


@wp.func
def _apply_outflow_limiter_cell(
    j: int,
    i: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    epsilon: WP_FLOAT,
    free_outflow_limiter_mode: int,
    free_outflow_diagnostics: wp.array(dtype=wp.float64),
    collect_free_outflow_diagnostics: int,
):
    """Shared body of apply_outflow_limiter_kernel and the device-loop variant."""
    if active_mask[j, i] != 1:
        return

    qx_east = qx[j, i + 1]
    qx_west = qx[j, i]
    qy_south = qy[j + 1, i]
    qy_north = qy[j, i]

    alpha_other, alpha_free = _outflow_limiter_alphas(
        j, i, water_depth, qx_east, qx_west, qy_south, qy_north,
        boundary_type_qx, boundary_type_qy, dx, dy, dt, epsilon,
        free_outflow_limiter_mode,
    )
    _accumulate_free_outflow_limiter_diagnostics(
        j, i, qx_east, qx_west, qy_south, qy_north, boundary_type_qx,
        boundary_type_qy, dx, dy, dt, alpha_free, free_outflow_diagnostics,
        collect_free_outflow_diagnostics,
    )

    if qx_east > WP_FLOAT(0.0):
        limiter = alpha_other
        if boundary_type_qx[j, i + 1] == BOUNDARY_FREE_OUTFLOW:
            limiter = alpha_free
        qx[j, i + 1] = qx_east * limiter
    if qx_west < WP_FLOAT(0.0):
        limiter = alpha_other
        if boundary_type_qx[j, i] == BOUNDARY_FREE_OUTFLOW:
            limiter = alpha_free
        qx[j, i] = qx_west * limiter
    if qy_south > WP_FLOAT(0.0):
        limiter = alpha_other
        if boundary_type_qy[j + 1, i] == BOUNDARY_FREE_OUTFLOW:
            limiter = alpha_free
        qy[j + 1, i] = qy_south * limiter
    if qy_north < WP_FLOAT(0.0):
        limiter = alpha_other
        if boundary_type_qy[j, i] == BOUNDARY_FREE_OUTFLOW:
            limiter = alpha_free
        qy[j, i] = qy_north * limiter


@wp.func
def _outflow_limiter_alphas(
    j: int,
    i: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx_east: WP_FLOAT,
    qx_west: WP_FLOAT,
    qy_south: WP_FLOAT,
    qy_north: WP_FLOAT,
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    epsilon: WP_FLOAT,
    free_outflow_limiter_mode: int,
) -> tuple[WP_FLOAT, WP_FLOAT]:
    """Return limiter factors for ordinary and outward FREE face fluxes.

    Mode 0 is the historic donor-cell limiter. Mode 1 reserves available cell
    volume for outward FREE faces before limiting all other donor outflows.
    Mode 2 is an intentionally diagnostic-only FREE bypass; continuity still
    detects any negative depth it causes.
    """
    q_free_out = WP_FLOAT(0.0)
    q_other_out = WP_FLOAT(0.0)
    if qx_east > WP_FLOAT(0.0):
        if boundary_type_qx[j, i + 1] == BOUNDARY_FREE_OUTFLOW:
            q_free_out = q_free_out + qx_east * dy
        else:
            q_other_out = q_other_out + qx_east * dy
    if qx_west < WP_FLOAT(0.0):
        if boundary_type_qx[j, i] == BOUNDARY_FREE_OUTFLOW:
            q_free_out = q_free_out - qx_west * dy
        else:
            q_other_out = q_other_out - qx_west * dy
    if qy_south > WP_FLOAT(0.0):
        if boundary_type_qy[j + 1, i] == BOUNDARY_FREE_OUTFLOW:
            q_free_out = q_free_out + qy_south * dx
        else:
            q_other_out = q_other_out + qy_south * dx
    if qy_north < WP_FLOAT(0.0):
        if boundary_type_qy[j, i] == BOUNDARY_FREE_OUTFLOW:
            q_free_out = q_free_out - qy_north * dx
        else:
            q_other_out = q_other_out - qy_north * dx

    V_avail = water_depth[j, i] * dx * dy
    q_total = q_free_out + q_other_out
    alpha_total = WP_FLOAT(1.0)
    if q_total > WP_FLOAT(0.0):
        alpha_total = wp.min(WP_FLOAT(1.0), V_avail / (dt * q_total + epsilon))
    alpha_other = alpha_total
    alpha_free = alpha_total
    if free_outflow_limiter_mode == 1:
        if q_free_out > WP_FLOAT(0.0):
            alpha_free = wp.min(WP_FLOAT(1.0), V_avail / (dt * q_free_out + epsilon))
        V_remaining = wp.max(WP_FLOAT(0.0), V_avail - dt * q_free_out * alpha_free)
        alpha_other = WP_FLOAT(1.0)
        if q_other_out > WP_FLOAT(0.0):
            alpha_other = wp.min(WP_FLOAT(1.0), V_remaining / (dt * q_other_out + epsilon))
    elif free_outflow_limiter_mode == 2:
        alpha_free = WP_FLOAT(1.0)
    return alpha_other, alpha_free


@wp.func
def _write_outflow_limiter_cell(
    j: int,
    i: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx_east: WP_FLOAT,
    qx_west: WP_FLOAT,
    qy_south: WP_FLOAT,
    qy_north: WP_FLOAT,
    outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    free_outflow_limiter: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    epsilon: WP_FLOAT,
    free_outflow_limiter_mode: int,
    free_outflow_diagnostics: wp.array(dtype=wp.float64),
    collect_free_outflow_diagnostics: int,
):
    alpha_other, alpha_free = _outflow_limiter_alphas(
        j, i, water_depth, qx_east, qx_west, qy_south, qy_north,
        boundary_type_qx, boundary_type_qy, dx, dy, dt, epsilon,
        free_outflow_limiter_mode,
    )
    outflow_limiter[j, i] = alpha_other
    free_outflow_limiter[j, i] = alpha_free
    _accumulate_free_outflow_limiter_diagnostics(
        j, i, qx_east, qx_west, qy_south, qy_north, boundary_type_qx,
        boundary_type_qy, dx, dy, dt, alpha_free, free_outflow_diagnostics,
        collect_free_outflow_diagnostics,
    )


@wp.func
def _accumulate_free_outflow_limiter_diagnostics(
    j: int,
    i: int,
    qx_east: WP_FLOAT,
    qx_west: WP_FLOAT,
    qy_south: WP_FLOAT,
    qy_north: WP_FLOAT,
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    alpha_free: WP_FLOAT,
    free_outflow_diagnostics: wp.array(dtype=wp.float64),
    collect_free_outflow_diagnostics: int,
):
    """Accumulate sparse FREE-face diagnostics without altering hydraulics."""
    if collect_free_outflow_diagnostics == 0:
        return
    q_free_out = WP_FLOAT(0.0)
    if qx_east > WP_FLOAT(0.0) and boundary_type_qx[j, i + 1] == BOUNDARY_FREE_OUTFLOW:
        q_free_out = q_free_out + qx_east * dy
    if qx_west < WP_FLOAT(0.0) and boundary_type_qx[j, i] == BOUNDARY_FREE_OUTFLOW:
        q_free_out = q_free_out - qx_west * dy
    if qy_south > WP_FLOAT(0.0) and boundary_type_qy[j + 1, i] == BOUNDARY_FREE_OUTFLOW:
        q_free_out = q_free_out + qy_south * dx
    if qy_north < WP_FLOAT(0.0) and boundary_type_qy[j, i] == BOUNDARY_FREE_OUTFLOW:
        q_free_out = q_free_out - qy_north * dx
    if q_free_out > WP_FLOAT(0.0):
        raw_volume = wp.float64(q_free_out) * wp.float64(dt)
        wp.atomic_add(free_outflow_diagnostics, 0, raw_volume)
        wp.atomic_add(free_outflow_diagnostics, 1, raw_volume * wp.float64(alpha_free))
        wp.atomic_add(free_outflow_diagnostics, 2, wp.float64(1.0))
        wp.atomic_add(free_outflow_diagnostics, 3, wp.float64(alpha_free))
        if alpha_free < WP_FLOAT(1.0):
            wp.atomic_add(free_outflow_diagnostics, 4, wp.float64(1.0))
        wp.atomic_min(free_outflow_diagnostics, 5, wp.float64(alpha_free))


# Compact diagnostic layout for an all-FREE-face snapshot.  It is deliberately
# separate from the hydraulic update and is launched only by opt-in validation
# tooling at device-loop sync points.
FREE_TRACE_COUNT = 0
FREE_TRACE_QOUT = 1
FREE_TRACE_EDGE_MIN = 2
FREE_TRACE_EDGE_MAX = 3
FREE_TRACE_EDGE_SUM = 4
FREE_TRACE_INWARD_MIN = 5
FREE_TRACE_INWARD_MAX = 6
FREE_TRACE_INWARD_SUM = 7
FREE_TRACE_DH_MIN = 8
FREE_TRACE_DH_MAX = 9
FREE_TRACE_DH_SUM = 10
FREE_TRACE_ABS_SLOPE_SUM = 11
FREE_TRACE_ABS_SLOPE_MAX = 12
FREE_TRACE_ABS_QOLD_SUM = 13
FREE_TRACE_ABS_QNEW_SUM = 14
FREE_TRACE_GRAVITY_INCREMENT_SUM = 15
FREE_TRACE_FRICTION_DENOMINATOR_SUM = 16
FREE_TRACE_SURFACE_VOLUME = 17
FREE_TRACE_SIZE = 18


@wp.kernel
def accumulate_free_outlet_trace_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx_new: wp.array2d(dtype=WP_FLOAT),
    qy_new: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    free_outflow_mode: int,
    stats: wp.array(dtype=wp.float64),
):
    """Reduce outlet state and FREE momentum terms into a tiny trace array."""
    j, i = wp.tid()
    ny = water_depth.shape[0]
    nx = water_depth.shape[1]
    if active_mask[j, i] != 1:
        return

    wp.atomic_add(
        stats, FREE_TRACE_SURFACE_VOLUME,
        wp.float64(water_depth[j, i]) * wp.float64(dx) * wp.float64(dy),
    )

    # Each possible free face is owned by its adjacent active cell, avoiding
    # double counting while supporting arbitrary perimeter configurations.
    is_free = 0
    edge_depth = WP_FLOAT(0.0)
    inward_depth = WP_FLOAT(0.0)
    edge_bed = BED_FLOAT(0.0)
    inward_bed = BED_FLOAT(0.0)
    q_old = WP_FLOAT(0.0)
    q_new = WP_FLOAT(0.0)
    width = WP_FLOAT(0.0)
    prescribed_slope = WP_FLOAT(0.0)
    direction = WP_FLOAT(0.0)
    spacing = WP_FLOAT(0.0)

    if i == nx - 1 and boundary_type_qx[j, nx] == BOUNDARY_FREE_OUTFLOW:
        is_free = 1
        edge_depth = water_depth[j, nx - 1]
        inward_depth = water_depth[j, nx - 2]
        edge_bed = bed_elevation[j, nx - 1]
        inward_bed = bed_elevation[j, nx - 2]
        q_old = qx_old[j, nx]
        q_new = qx_new[j, nx]
        width = dy
        prescribed_slope = WP_FLOAT(boundary_stage_qx[j, nx])
        direction = WP_FLOAT(1.0)
        spacing = dx
    elif i == 0 and boundary_type_qx[j, 0] == BOUNDARY_FREE_OUTFLOW:
        is_free = 1
        edge_depth = water_depth[j, 0]
        inward_depth = water_depth[j, 1]
        edge_bed = bed_elevation[j, 0]
        inward_bed = bed_elevation[j, 1]
        q_old = qx_old[j, 0]
        q_new = qx_new[j, 0]
        width = dy
        prescribed_slope = WP_FLOAT(boundary_stage_qx[j, 0])
        direction = WP_FLOAT(-1.0)
        spacing = dx
    elif j == ny - 1 and boundary_type_qy[ny, i] == BOUNDARY_FREE_OUTFLOW:
        is_free = 1
        edge_depth = water_depth[ny - 1, i]
        inward_depth = water_depth[ny - 2, i]
        edge_bed = bed_elevation[ny - 1, i]
        inward_bed = bed_elevation[ny - 2, i]
        q_old = qy_old[ny, i]
        q_new = qy_new[ny, i]
        width = dx
        prescribed_slope = WP_FLOAT(boundary_stage_qy[ny, i])
        direction = WP_FLOAT(1.0)
        spacing = dy
    elif j == 0 and boundary_type_qy[0, i] == BOUNDARY_FREE_OUTFLOW:
        is_free = 1
        edge_depth = water_depth[0, i]
        inward_depth = water_depth[1, i]
        edge_bed = bed_elevation[0, i]
        inward_bed = bed_elevation[1, i]
        q_old = qy_old[0, i]
        q_new = qy_new[0, i]
        width = dx
        prescribed_slope = WP_FLOAT(boundary_stage_qy[0, i])
        direction = WP_FLOAT(-1.0)
        spacing = dy

    if is_free == 0:
        return

    dh = WP_FLOAT((edge_bed + BED_FLOAT(edge_depth)) - (inward_bed + BED_FLOAT(inward_depth)))
    slope = -dh / spacing
    has_prescribed_slope = 0
    if wp.isnan(prescribed_slope):
        slope = WP_FLOAT(0.0)
        has_prescribed_slope = 1
    elif wp.abs(prescribed_slope) > WP_FLOAT(0.0):
        slope = prescribed_slope
        has_prescribed_slope = 1
    elif USE_CONDITIONED_MIXED_KERNELS == 1:
        # Match the mixed hot kernel: retain the FP64 DEM difference before
        # combining it with FP32 depths, rather than forming absolute WSEs.
        dh = WP_FLOAT(edge_bed - inward_bed) + edge_depth - inward_depth
        slope = -dh / spacing
    elif free_outflow_mode >= FREE_OUTFLOW_MODE_LISFLOOD_FLOAT_SURFACE:
        dh = _lisflood_float_surface_dh(
            edge_bed, edge_depth, inward_bed, inward_depth, spacing,
        )
        slope = WP_FLOAT(wp.float32(-wp.float32(dh) / wp.float32(spacing)))
    n_value = manning_n[j, i]
    gravity_increment = gravity * edge_depth * slope
    denominator = WP_FLOAT(1.0) + gravity * dt * edge_depth * n_value * n_value * wp.abs(q_old) / wp.pow(
        wp.max(edge_depth, depth_epsilon), WP_FLOAT(10.0 / 3.0)
    )
    if (
        has_prescribed_slope == 0
        and free_outflow_mode == FREE_OUTFLOW_MODE_LISFLOOD_FLOAT_OPERATOR
    ):
        g = wp.float32(gravity)
        tstep = wp.float32(dt)
        hflow = wp.float32(edge_depth)
        fn = wp.float32(n_value)
        q0 = wp.float32(q_old)
        sf = wp.float32(slope)
        gravity_increment_float = wp.float32(g * tstep)
        gravity_increment_float = wp.float32(gravity_increment_float * hflow)
        gravity_increment = WP_FLOAT(wp.float32(gravity_increment_float * sf))
        h_power = wp.pow(hflow, wp.float32(10.0) / wp.float32(3.0))
        friction = wp.float32(g * tstep)
        friction = wp.float32(friction * hflow)
        friction = wp.float32(friction * fn)
        friction = wp.float32(friction * fn)
        friction = wp.float32(friction * wp.abs(q0))
        denominator = WP_FLOAT(wp.float32(1.0 + friction / h_power))

    wp.atomic_add(stats, FREE_TRACE_COUNT, wp.float64(1.0))
    wp.atomic_add(stats, FREE_TRACE_QOUT, wp.float64(direction * q_new * width))
    wp.atomic_min(stats, FREE_TRACE_EDGE_MIN, wp.float64(edge_depth))
    wp.atomic_max(stats, FREE_TRACE_EDGE_MAX, wp.float64(edge_depth))
    wp.atomic_add(stats, FREE_TRACE_EDGE_SUM, wp.float64(edge_depth))
    wp.atomic_min(stats, FREE_TRACE_INWARD_MIN, wp.float64(inward_depth))
    wp.atomic_max(stats, FREE_TRACE_INWARD_MAX, wp.float64(inward_depth))
    wp.atomic_add(stats, FREE_TRACE_INWARD_SUM, wp.float64(inward_depth))
    wp.atomic_min(stats, FREE_TRACE_DH_MIN, wp.float64(dh))
    wp.atomic_max(stats, FREE_TRACE_DH_MAX, wp.float64(dh))
    wp.atomic_add(stats, FREE_TRACE_DH_SUM, wp.float64(dh))
    wp.atomic_add(stats, FREE_TRACE_ABS_SLOPE_SUM, wp.float64(wp.abs(slope)))
    wp.atomic_max(stats, FREE_TRACE_ABS_SLOPE_MAX, wp.float64(wp.abs(slope)))
    wp.atomic_add(stats, FREE_TRACE_ABS_QOLD_SUM, wp.float64(wp.abs(q_old)))
    wp.atomic_add(stats, FREE_TRACE_ABS_QNEW_SUM, wp.float64(wp.abs(q_new)))
    wp.atomic_add(stats, FREE_TRACE_GRAVITY_INCREMENT_SUM, wp.float64(wp.abs(gravity_increment)))
    wp.atomic_add(stats, FREE_TRACE_FRICTION_DENOMINATOR_SUM, wp.float64(denominator))


@wp.func
def _accumulate_boundary_flux_edges(
    j: int,
    i: int,
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
):
    """Fold open-boundary flux volumes into the on-device mass totals.

    Called from the continuity kernels (one thread per cell): thread (j, i)
    handles the west qx face when i == 0, the east qx face when i == nx - 1,
    the north qy face when j == 0, the south qy face when j == ny - 1,
    masked west-stage/west-FREE faces at qx[j, i], a masked east-FREE face
    at qx[j, i + 1], a masked south-FREE face at qy[j, i], and masked
    north-stage/north-FREE faces at qy[j + 1, i] when present. Index
    layout: cumulative_totals[3] = inflow, [4] = outflow. Runs after the
    limiter, so qx/qy are final; matches the historical standalone accumulator
    kernels (only the atomic summation order differs).
    """
    ny = qy.shape[0] - 1
    nx = qx.shape[1] - 1

    # Both faces are checked independently so single-cell-wide domains
    # (nx == 1 or ny == 1) still account for their two opposing edges.
    if i == 0:
        boundary_type = boundary_type_qx[j, 0]
        if boundary_type != BOUNDARY_CLOSED and boundary_type != BOUNDARY_INTERNAL:
            q = qx[j, 0]
            volume = wp.float64(wp.abs(q)) * wp.float64(dy) * wp.float64(dt)
            # West edge: q > 0 flows into the domain.
            if q > WP_FLOAT(0.0):
                wp.atomic_add(cumulative_totals, 3, volume)
            else:
                wp.atomic_add(cumulative_totals, 4, volume)
    if i == nx - 1:
        boundary_type = boundary_type_qx[j, nx]
        if boundary_type != BOUNDARY_CLOSED and boundary_type != BOUNDARY_INTERNAL:
            q = qx[j, nx]
            volume = wp.float64(wp.abs(q)) * wp.float64(dy) * wp.float64(dt)
            # East edge: q < 0 flows into the domain.
            if q < WP_FLOAT(0.0):
                wp.atomic_add(cumulative_totals, 3, volume)
            else:
                wp.atomic_add(cumulative_totals, 4, volume)

    # A masked west-stage face qx[j, i] has this active cell on
    # its i side. Negative q therefore leaves the polygon.
    if i > 0 and boundary_type_qx[j, i] == BOUNDARY_MASKED_WEST_STAGE:
        q = qx[j, i]
        volume = wp.float64(wp.abs(q)) * wp.float64(dy) * wp.float64(dt)
        if q > WP_FLOAT(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)

    # A masked west-FREE face qx[j, i] has this active cell on its i
    # side; negative q leaves the domain (inflow is clamped to zero).
    if i > 0 and boundary_type_qx[j, i] == BOUNDARY_MASKED_WEST_FREE:
        q = qx[j, i]
        volume = wp.float64(wp.abs(q)) * wp.float64(dy) * wp.float64(dt)
        if q > WP_FLOAT(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)

    # A masked east-FREE face qx[j, i + 1] has this active cell on its
    # i-1 side; positive q leaves the domain (inflow is clamped to zero).
    if i < nx - 1 and (boundary_type_qx[j, i + 1] == BOUNDARY_MASKED_EAST_FREE or boundary_type_qx[j, i + 1] == BOUNDARY_MASKED_EAST_STAGE):
        q = qx[j, i + 1]
        volume = wp.float64(wp.abs(q)) * wp.float64(dy) * wp.float64(dt)
        if q < WP_FLOAT(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)

    if j == 0:
        boundary_type = boundary_type_qy[0, i]
        if boundary_type != BOUNDARY_CLOSED and boundary_type != BOUNDARY_INTERNAL:
            q = qy[0, i]
            volume = wp.float64(wp.abs(q)) * wp.float64(dx) * wp.float64(dt)
            # North edge: q > 0 flows into the domain.
            if q > WP_FLOAT(0.0):
                wp.atomic_add(cumulative_totals, 3, volume)
            else:
                wp.atomic_add(cumulative_totals, 4, volume)
    if j == ny - 1:
        boundary_type = boundary_type_qy[ny, i]
        if boundary_type != BOUNDARY_CLOSED and boundary_type != BOUNDARY_INTERNAL:
            q = qy[ny, i]
            volume = wp.float64(wp.abs(q)) * wp.float64(dx) * wp.float64(dt)
            # South edge: q < 0 flows into the domain.
            if q < WP_FLOAT(0.0):
                wp.atomic_add(cumulative_totals, 3, volume)
            else:
                wp.atomic_add(cumulative_totals, 4, volume)

    # A masked north-stage face qy[j+1, i] has this active cell on
    # its j-1 side. Positive q therefore leaves the polygon.
    if j < ny - 1 and boundary_type_qy[j + 1, i] == BOUNDARY_MASKED_NORTH_STAGE:
        q = qy[j + 1, i]
        volume = wp.float64(wp.abs(q)) * wp.float64(dx) * wp.float64(dt)
        if q < WP_FLOAT(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)

    # A masked south-FREE face qy[j, i] has this active cell on its j
    # side; negative q leaves the domain (inflow is clamped to zero).
    if j > 0 and (boundary_type_qy[j, i] == BOUNDARY_MASKED_SOUTH_FREE or boundary_type_qy[j, i] == BOUNDARY_MASKED_SOUTH_STAGE):
        q = qy[j, i]
        volume = wp.float64(wp.abs(q)) * wp.float64(dx) * wp.float64(dt)
        if q > WP_FLOAT(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)

    # A masked north-FREE face qy[j + 1, i] has this active cell on its
    # j-1 side; positive q leaves the domain (inflow is clamped to zero).
    if j < ny - 1 and boundary_type_qy[j + 1, i] == BOUNDARY_MASKED_NORTH_FREE:
        q = qy[j + 1, i]
        volume = wp.float64(wp.abs(q)) * wp.float64(dx) * wp.float64(dt)
        if q < WP_FLOAT(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)


@wp.func
def _fold_source_mass_sums(
    j: int,
    i: int,
    rain_sum: wp.array(dtype=wp.float64),
    infil_sum: wp.array(dtype=wp.float64),
    other_sum: wp.array(dtype=wp.float64),
    cell_area: wp.float64,
    cumulative_totals: wp.array(dtype=wp.float64),
):
    """Single-thread fold of device-reduced source sums into the accumulator.

    Executed by thread (0, 0) of the continuity kernels on accounting-flush
    steps (the CUB tree-sums are enqueued earlier in the same stream). The
    read-modify-writes are uncontended: no other thread touches elements 0-2.
    """
    if j == 0 and i == 0:
        cumulative_totals[0] = cumulative_totals[0] + wp.float64(rain_sum[0]) * cell_area
        cumulative_totals[1] = cumulative_totals[1] + wp.float64(infil_sum[0]) * cell_area
        cumulative_totals[2] = cumulative_totals[2] + wp.float64(other_sum[0]) * cell_area


@wp.kernel
def continuity_update_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    cumulative_totals: wp.array(dtype=wp.float64),
    rain_sum: wp.array(dtype=wp.float64),
    infil_sum: wp.array(dtype=wp.float64),
    other_sum: wp.array(dtype=wp.float64),
    cell_area: wp.float64,
    fold_source_mass: int,
):
    j, i = wp.tid()
    _accumulate_boundary_flux_edges(
        j, i, qx, qy, boundary_type_qx, boundary_type_qy, dx, dy, dt, cumulative_totals
    )
    if fold_source_mass == 1:
        _fold_source_mass_sums(
            j, i, rain_sum, infil_sum, other_sum, cell_area, cumulative_totals
        )
    if active_mask[j, i] == 1:
        dqx = (qx[j, i+1] - qx[j, i]) / dx
        dqy = (qy[j+1, i] - qy[j, i]) / dy

        h_new = water_depth[j, i] - dt * (dqx + dqy)
        water_depth[j, i] = h_new
    else:
        water_depth[j, i] = WP_FLOAT(0.0)


@wp.kernel
def continuity_update_and_check_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    neg_depth_tol: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    reduce_max_depth: int,
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    cumulative_totals: wp.array(dtype=wp.float64),
    rain_sum: wp.array(dtype=wp.float64),
    infil_sum: wp.array(dtype=wp.float64),
    other_sum: wp.array(dtype=wp.float64),
    cell_area: wp.float64,
    fold_source_mass: int,
):
    """Fused continuity update + negative-depth check + (optional) depth reduction.

    ``control[0]`` receives the maximum post-continuity active depth when
    ``reduce_max_depth == 1`` (ACC host-stepped timestep path); ``control[1]``
    receives a binary negative-depth error flag via atomic_max for both profiles.

    Note: the max-depth reduction uses a per-cell atomic_max rather than a separate
    reduction pass. Unlike atomic_add, atomic_max is "sticky" (a CAS from a smaller
    value fails immediately without retry), so it does not serialize; and the value
    is already in registers here, avoiding a second global-memory pass. A separate
    chunked reduction was measured to be slower (it re-reads N cells).
    """
    j, i = wp.tid()
    _accumulate_boundary_flux_edges(
        j, i, qx, qy, boundary_type_qx, boundary_type_qy, dx, dy, dt, cumulative_totals
    )
    if fold_source_mass == 1:
        _fold_source_mass_sums(
            j, i, rain_sum, infil_sum, other_sum, cell_area, cumulative_totals
        )
    if active_mask[j, i] != 1:
        water_depth[j, i] = WP_FLOAT(0.0)
        return

    dqx = (qx[j, i + 1] - qx[j, i]) / dx
    dqy = (qy[j + 1, i] - qy[j, i]) / dy
    h_new = water_depth[j, i] - dt * (dqx + dqy)

    if h_new < WP_FLOAT(0.0):
        if h_new >= -neg_depth_tol:
            h_new = WP_FLOAT(0.0)
        else:
            wp.atomic_max(control, 1, wp.float64(1.0))

    water_depth[j, i] = h_new
    if reduce_max_depth == 1:
        wp.atomic_max(control, 0, wp.float64(h_new))


@wp.kernel
def apply_sources_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_capacity: wp.array2d(dtype=WP_FLOAT),
    infiltration_f0: wp.array2d(dtype=WP_FLOAT),
    infiltration_fd: wp.array2d(dtype=WP_FLOAT),
    infiltration_k: wp.array2d(dtype=WP_FLOAT),
    green_ampt_ks: wp.array2d(dtype=WP_FLOAT),
    green_ampt_psi: wp.array2d(dtype=WP_FLOAT),
    green_ampt_delta_theta: wp.array2d(dtype=WP_FLOAT),
    cumulative_infiltration: wp.array2d(dtype=WP_FLOAT),
    infiltration_method: int,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    active_mask: wp.array2d(dtype=wp.int32),
    dt: WP_FLOAT,
    source_mode: int,
    infiltration_wet_depth: WP_FLOAT,
    applied_rain: wp.array2d(dtype=wp.float64),
    applied_other: wp.array2d(dtype=wp.float64),
    applied_infil: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
    accumulate_accounting: int,
    accumulate_output: int,
    additional_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_exclusion_mask: wp.array2d(dtype=wp.int32),
    infiltration_cap_active: int,
    infiltration_cap_remaining_m: wp.array2d(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    # Sources lead momentum on this path, so no committed fluxes exist yet:
    # pass inert dummies with the cap disabled (never dereferenced).
    _apply_sources_cell(
        j, i, water_depth, rainfall_rate[j, i], other_source_rate[j, i],
        additional_source_rate[j, i],
        infiltration_capacity[j, i], infiltration_f0, infiltration_fd,
        infiltration_k, green_ampt_ks, green_ampt_psi,
        green_ampt_delta_theta, cumulative_infiltration, infiltration_method,
        subgrid_infiltration_enabled, subgrid_cell_z,
        subgrid_cell_storage_breakpoints, subgrid_cell_count,
        active_mask, dt, source_mode,
        infiltration_wet_depth, water_depth, water_depth, 0,
        infiltration_exclusion_mask[j, i], infiltration_cap_active,
        infiltration_cap_remaining_m,
        applied_rain, applied_other, applied_infil,
        dx, dy, cumulative_totals, accumulate_accounting, accumulate_output,
    )


@wp.kernel
def apply_sources_committed_flux_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_capacity: wp.array2d(dtype=WP_FLOAT),
    infiltration_f0: wp.array2d(dtype=WP_FLOAT),
    infiltration_fd: wp.array2d(dtype=WP_FLOAT),
    infiltration_k: wp.array2d(dtype=WP_FLOAT),
    green_ampt_ks: wp.array2d(dtype=WP_FLOAT),
    green_ampt_psi: wp.array2d(dtype=WP_FLOAT),
    green_ampt_delta_theta: wp.array2d(dtype=WP_FLOAT),
    cumulative_infiltration: wp.array2d(dtype=WP_FLOAT),
    infiltration_method: int,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    active_mask: wp.array2d(dtype=wp.int32),
    dt: WP_FLOAT,
    source_mode: int,
    infiltration_wet_depth: WP_FLOAT,
    committed_qx: wp.array2d(dtype=WP_FLOAT),
    committed_qy: wp.array2d(dtype=WP_FLOAT),
    applied_rain: wp.array2d(dtype=wp.float64),
    applied_other: wp.array2d(dtype=wp.float64),
    applied_infil: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
    accumulate_accounting: int,
    accumulate_output: int,
    additional_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_exclusion_mask: wp.array2d(dtype=wp.int32),
    infiltration_cap_active: int,
    infiltration_cap_remaining_m: wp.array2d(dtype=WP_FLOAT),
):
    """apply_sources_kernel for sources-after-momentum profiles.

    ``committed_qx``/``committed_qy`` are the post-limiter accepted face
    fluxes; the infiltration sink is capped at the depth remaining after
    those committed outfluxes (see _apply_sources_cell).
    """
    j, i = wp.tid()
    _apply_sources_cell(
        j, i, water_depth, rainfall_rate[j, i], other_source_rate[j, i],
        additional_source_rate[j, i],
        infiltration_capacity[j, i], infiltration_f0, infiltration_fd,
        infiltration_k, green_ampt_ks, green_ampt_psi,
        green_ampt_delta_theta, cumulative_infiltration, infiltration_method,
        subgrid_infiltration_enabled, subgrid_cell_z,
        subgrid_cell_storage_breakpoints, subgrid_cell_count,
        active_mask, dt, source_mode,
        infiltration_wet_depth, committed_qx, committed_qy, 1,
        infiltration_exclusion_mask[j, i], infiltration_cap_active,
        infiltration_cap_remaining_m,
        applied_rain, applied_other, applied_infil,
        dx, dy, cumulative_totals, accumulate_accounting, accumulate_output,
    )


@wp.func
def _green_ampt_potential_depth(
    cumulative_infiltration: WP_FLOAT,
    ks: WP_FLOAT,
    suction: WP_FLOAT,
    delta_theta: WP_FLOAT,
    dt: WP_FLOAT,
) -> WP_FLOAT:
    """Finite-step ponded Green-Ampt increment using safeguarded Newton."""
    if ks <= WP_FLOAT(0.0) or dt <= WP_FLOAT(0.0):
        return WP_FLOAT(0.0)
    target = ks * dt
    a = suction * delta_theta
    if a <= WP_FLOAT(0.0):
        return target
    f0 = wp.max(cumulative_infiltration, WP_FLOAT(0.0))
    lower = WP_FLOAT(0.0)
    upper = target + WP_FLOAT(2.0) * wp.sqrt(a * target) + WP_FLOAT(2.0) * a
    x = wp.min(upper, target + wp.sqrt(WP_FLOAT(2.0) * a * target))
    # Evaluate r-log(1+r) by its series near zero to avoid cancellation.
    # Check convergence before tightening the bracket: an exact Newton root
    # is otherwise rejected as a bracket endpoint on the next iteration.
    for _ in range(64):
        ratio = x / (f0 + a)
        remainder = ratio - wp.log(WP_FLOAT(1.0) + ratio)
        if ratio < WP_FLOAT(0.001):
            remainder = ratio * ratio * (WP_FLOAT(0.5) + ratio * (WP_FLOAT(-1.0 / 3.0) + ratio * (WP_FLOAT(0.25) + ratio * (WP_FLOAT(-0.2) + ratio / WP_FLOAT(6.0)))))
        residual = x * (f0 / (f0 + a)) + a * remainder - target
        derivative = (f0 + x) / (f0 + a + x)
        update = residual / wp.max(derivative, WP_FLOAT(1.0e-30))
        tolerance = WP_FLOAT(2.0e-14) * wp.max(x, target)
        if wp.abs(residual) <= WP_FLOAT(2.0e-14) * target or wp.abs(update) <= tolerance:
            return wp.max(WP_FLOAT(0.0), x)
        if residual > WP_FLOAT(0.0):
            upper = x
        else:
            lower = x
        candidate = x - update
        if not wp.isfinite(candidate) or candidate <= lower or candidate >= upper:
            candidate = lower + WP_FLOAT(0.5) * (upper - lower)
        if wp.abs(candidate - x) <= tolerance:
            return wp.max(WP_FLOAT(0.0), candidate)
        x = candidate
    return wp.max(WP_FLOAT(0.0), wp.min(upper, wp.max(lower, x)))


@wp.func
def _green_ampt_dry_source_depth(
    cumulative_infiltration: WP_FLOAT,
    ks: WP_FLOAT,
    suction: WP_FLOAT,
    delta_theta: WP_FLOAT,
    source_rate: WP_FLOAT,
    dt: WP_FLOAT,
) -> WP_FLOAT:
    """Rain-limited infiltration followed by ponding within one step (m).

    For source rate r > Ks, ponding begins at Fp = Ks*A/(r-Ks), where
    A = psi_f*delta_theta. No initial surface storage is assumed here.
    """
    if source_rate <= WP_FLOAT(0.0) or dt <= WP_FLOAT(0.0) or ks <= WP_FLOAT(0.0):
        return WP_FLOAT(0.0)
    supply = source_rate * dt
    if source_rate <= ks:
        return supply
    a = suction * delta_theta
    if a <= WP_FLOAT(0.0):
        return wp.min(supply, ks * dt)
    f0 = wp.max(cumulative_infiltration, WP_FLOAT(0.0))
    ponding_f = ks * a / (source_rate - ks)
    if f0 < ponding_f:
        pre_time = wp.min(dt, (ponding_f - f0) / source_rate)
        pre_depth = source_rate * pre_time
        if pre_time >= dt:
            return pre_depth
        ponded_depth = _green_ampt_potential_depth(
            f0 + pre_depth, ks, suction, delta_theta, dt - pre_time
        )
        return wp.min(supply, pre_depth + ponded_depth)
    return wp.min(
        supply,
        _green_ampt_potential_depth(f0, ks, suction, delta_theta, dt),
    )


@wp.func
def _rain_first_infiltration_from_fraction(
    r_depth: WP_FLOAT, ponded_available: WP_FLOAT, capacity_depth: WP_FLOAT,
    wet_fraction: WP_FLOAT,
) -> WP_FLOAT:
    infiltration_from_rain=wp.min(r_depth,capacity_depth)
    remaining_capacity=wp.max(WP_FLOAT(0.0),capacity_depth-infiltration_from_rain)
    infiltration_from_ponding=wp.min(ponded_available,wet_fraction*remaining_capacity)
    return infiltration_from_rain+infiltration_from_ponding


@wp.func
def _rain_first_infiltration_depth(
    r_depth: WP_FLOAT,
    wet_fraction_storage: WP_FLOAT,
    ponded_available: WP_FLOAT,
    capacity_depth: WP_FLOAT,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    j: int,
    i: int,
    subgrid_cell_count: int,
) -> WP_FLOAT:
    """Shared rain-first infiltration split (constant and exponential).

    Rainfall can infiltrate over the full plan area even before the resolved
    subgrid becomes wet. Only capacity drawn from pre-existing ponded storage
    is scaled by the hydraulic wet fraction derived from
    ``wet_fraction_storage`` (which must exclude this interval's rainfall),
    and the ponded draw is capped by ``ponded_available`` (which may include
    other surface sources and committed-outflow adjustments).

    Capacity is supplied by the caller: the constant method passes its fixed
    ``rate * dt`` and the exponential method its ``fc(F) * dt``. Only the
    exponential caller updates cumulative infiltration state.
    """
    wet_fraction = WP_FLOAT(1.0)
    if subgrid_infiltration_enabled == 1:
        wet_fraction = subgrid_wet_fraction_from_storage(
            wet_fraction_storage,
            subgrid_cell_z,
            subgrid_cell_storage_breakpoints,
            j,
            i,
            subgrid_cell_count,
        )
    return _rain_first_infiltration_from_fraction(
        r_depth, ponded_available, capacity_depth, wet_fraction,
    )


@wp.func
def _apply_source_state(
    h: WP_FLOAT, raw_rain_rate: WP_FLOAT, raw_other_rate: WP_FLOAT, raw_gw_rate: WP_FLOAT,
    raw_infil_capacity: WP_FLOAT, infiltration_f0_value: WP_FLOAT,
    infiltration_fd_value: WP_FLOAT, infiltration_k_value: WP_FLOAT,
    green_ampt_ks_value: WP_FLOAT, green_ampt_psi_value: WP_FLOAT,
    green_ampt_delta_theta_value: WP_FLOAT, cumulative_value: WP_FLOAT,
    infiltration_method: int, subgrid_infiltration_enabled: int,
    wet_fraction: WP_FLOAT, dt: WP_FLOAT, source_mode: int,
    infiltration_wet_depth: WP_FLOAT, committed_outflow_depth: WP_FLOAT,
    cap_infil_committed_flux: int, infiltration_excluded: int,
    infiltration_cap_active: int, remaining_value: WP_FLOAT,
):
    """Shared production source physics for a conservative storage control volume.

    h, source depths and soil uptake are per unit of that state's physical plan
    area. The caller supplies its own hypsometric wet fraction and committed
    outward depth. No state may supply another state's infiltration demand.
    Returns h, cumulative soil uptake, remaining capacity and accepted depths.
    """
    updated_cumulative=cumulative_value
    updated_remaining=remaining_value
    r_rate = wp.max(WP_FLOAT(0.0), raw_rain_rate)
    o_rate = wp.max(WP_FLOAT(0.0), raw_other_rate)
    g_rate = wp.max(WP_FLOAT(0.0), raw_gw_rate)
    i_cap = wp.max(WP_FLOAT(0.0), raw_infil_capacity)
    if infiltration_method == 2:
        i_cap = WP_FLOAT(0.0)
    cumulative = WP_FLOAT(0.0)
    if infiltration_method == 1:
        f0 = infiltration_f0_value
        fd = infiltration_fd_value
        k = infiltration_k_value
        fd_safe = wp.max(fd, WP_FLOAT(1.0e-30))
        cumulative = wp.max(cumulative_value, WP_FLOAT(0.0))
        i_cap = k + (f0 - k) * wp.exp(-cumulative / fd_safe)
        i_cap = wp.max(WP_FLOAT(0.0), i_cap)
    elif infiltration_method == 3:
        cumulative = wp.max(cumulative_value, WP_FLOAT(0.0))
        i_cap = wp.max(WP_FLOAT(0.0), green_ampt_ks_value)

    r_depth = r_rate * dt
    o_depth = o_rate * dt
    g_depth = g_rate * dt

    i_max_depth = WP_FLOAT(0.0)
    if infiltration_method == 1 or infiltration_method == 3 or (
        infiltration_method == 0 and subgrid_infiltration_enabled == 1
    ):
        # Shared rain-first application for exponential/Green-Ampt (all
        # modes) and constant (subgrid mode); each method supplies a
        # finite-step potential depth before the shared water-availability cap.
        # Sources-after-momentum profiles have already committed the
        # donor-limited outflow. Derive the ponding wet fraction from the
        # remaining storage, but never include this source interval's rain:
        # rainfall is intentionally allowed to infiltrate over the full
        # plan area.
        capacity_depth = i_cap * dt
        if infiltration_method == 3:
            capacity_depth = _green_ampt_potential_depth(
                cumulative, green_ampt_ks_value, green_ampt_psi_value,
                green_ampt_delta_theta_value, dt,
            )
        wet_fraction_storage = h
        if cap_infil_committed_flux == 1:
            wet_fraction_storage = wp.max(
                WP_FLOAT(0.0), h - committed_outflow_depth
            )
        if infiltration_method == 3 and wet_fraction_storage <= WP_FLOAT(0.0):
            capacity_depth = _green_ampt_dry_source_depth(
                cumulative, green_ampt_ks_value, green_ampt_psi_value,
                green_ampt_delta_theta_value, r_rate + o_rate + g_rate, dt,
            )
        ponded_available = h + o_depth + g_depth
        if cap_infil_committed_flux == 1:
            ponded_available = wp.max(
                WP_FLOAT(0.0), ponded_available - committed_outflow_depth
            )
        i_max_depth = _rain_first_infiltration_from_fraction(
            r_depth, ponded_available, capacity_depth, wet_fraction,
        )
    elif source_mode == 1:
        if h > infiltration_wet_depth:
            i_max_depth = wp.min(i_cap * dt, h)
        if cap_infil_committed_flux == 1:
            i_max_depth = wp.min(
                i_max_depth,
                wp.max(WP_FLOAT(0.0), h - committed_outflow_depth),
            )
    else:
        h_after_sources = h + r_depth + o_depth + g_depth
        i_max_depth = wp.min(i_cap * dt, h_after_sources)
        if cap_infil_committed_flux == 1:
            i_max_depth = wp.min(
                i_max_depth,
                wp.max(WP_FLOAT(0.0), h_after_sources - committed_outflow_depth),
            )

    # Finite-soil-storage constraints, applied after method physics and
    # before any state update: the exclusion mask zeroes the sink, and
    # the persistent storage cap draws down by the applied sink so the
    # cumulative infiltration accepted over a soil-cap interval cannot
    # exceed the limit set for that interval.
    if infiltration_excluded == 1:
        i_max_depth = WP_FLOAT(0.0)
    elif infiltration_cap_active == 1:
        remaining = wp.max(remaining_value, WP_FLOAT(0.0))
        i_max_depth = wp.min(i_max_depth, remaining)
        updated_remaining = wp.max(
            WP_FLOAT(0.0), remaining_value - i_max_depth
        )

    if infiltration_method == 1 or infiltration_method == 3:
        updated_cumulative = cumulative + i_max_depth

    if infiltration_method == 1 or infiltration_method == 3 or (
        infiltration_method == 0 and subgrid_infiltration_enabled == 1
    ):
        updated_depth = h + r_depth + o_depth + g_depth - i_max_depth
    elif source_mode == 1:
        updated_depth = h - i_max_depth + r_depth + o_depth + g_depth
    else:
        updated_depth = h + r_depth + o_depth + g_depth - i_max_depth

    return updated_depth, updated_cumulative, updated_remaining, r_depth, o_depth, g_depth, i_max_depth


@wp.func
def _apply_sources_cell(
    j: int,
    i: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    raw_rain_rate: WP_FLOAT,
    raw_other_rate: WP_FLOAT,
    raw_gw_rate: WP_FLOAT,
    raw_infil_capacity: WP_FLOAT,
    infiltration_f0: wp.array2d(dtype=WP_FLOAT),
    infiltration_fd: wp.array2d(dtype=WP_FLOAT),
    infiltration_k: wp.array2d(dtype=WP_FLOAT),
    green_ampt_ks: wp.array2d(dtype=WP_FLOAT),
    green_ampt_psi: wp.array2d(dtype=WP_FLOAT),
    green_ampt_delta_theta: wp.array2d(dtype=WP_FLOAT),
    cumulative_infiltration: wp.array2d(dtype=WP_FLOAT),
    infiltration_method: int,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    active_mask: wp.array2d(dtype=wp.int32),
    dt: WP_FLOAT,
    source_mode: int,
    infiltration_wet_depth: WP_FLOAT,
    committed_qx: wp.array2d(dtype=WP_FLOAT),
    committed_qy: wp.array2d(dtype=WP_FLOAT),
    cap_infil_committed_flux: int,
    infiltration_excluded: int,
    infiltration_cap_active: int,
    infiltration_cap_remaining_m: wp.array2d(dtype=WP_FLOAT),
    applied_rain: wp.array2d(dtype=wp.float64),
    applied_other: wp.array2d(dtype=wp.float64),
    applied_infil: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
    accumulate_accounting: int,
    accumulate_output: int,
):
    """Shared body of apply_sources_kernel and the device-loop variant.

    ``raw_gw_rate`` is the external additional surface-source source rate [m/s].
    It behaves like the other external source: it ponds on the surface, is
    available to infiltration from ponded storage, and contributes to the
    dry-cell Green-Ampt supply rate. ``infiltration_excluded`` disables the
    infiltration sink entirely for this cell (external saturation mask) and
    ``infiltration_cap_active``/``infiltration_cap_remaining_m`` implement a
    persistent per-cell interval storage cap that the applied sink draws
    down; both are supplied by an external surface input driver and never
    modify the configured soil properties.
    """
    if active_mask[j, i] == 1:
        h = water_depth[j, i]
        committed_outflow_depth = WP_FLOAT(0.0)
        if cap_infil_committed_flux == 1:
            # Profiles applying sources after momentum run the donor-cell
            # outflow limiter BEFORE this kernel: the limiter budgeted the
            # committed (limited) fluxes against pre-source depth, so the
            # infiltration sink must not exceed the depth remaining after
            # those outfluxes.  Inflows are ignored (conservative bound,
            # matching LISFLOOD FPInfiltration's cap at the post-update
            # depth from below).  Without this cap a fully draining cell
            # overshoots by up to i_cap * dt and trips the negative-depth
            # invariant.
            committed_outflow_depth = dt * (
                (
                    wp.max(WP_FLOAT(0.0), committed_qx[j, i + 1])
                    + wp.max(WP_FLOAT(0.0), -committed_qx[j, i])
                )
                / dx
                + (
                    wp.max(WP_FLOAT(0.0), committed_qy[j + 1, i])
                    + wp.max(WP_FLOAT(0.0), -committed_qy[j, i])
                )
                / dy
            )
        wet_fraction=WP_FLOAT(1.0)
        if subgrid_infiltration_enabled == 1 and (infiltration_method == 0 or infiltration_method == 1 or infiltration_method == 3):
            wet_storage=h
            if cap_infil_committed_flux == 1:
                wet_storage=wp.max(WP_FLOAT(0.0),h-committed_outflow_depth)
            wet_fraction=subgrid_wet_fraction_from_storage(wet_storage,subgrid_cell_z,
                subgrid_cell_storage_breakpoints,j,i,subgrid_cell_count)
        updated_depth, updated_cumulative, updated_remaining, r_depth, o_depth, g_depth, i_max_depth = _apply_source_state(
            h,raw_rain_rate,raw_other_rate,raw_gw_rate,raw_infil_capacity,
            infiltration_f0[j,i],infiltration_fd[j,i],infiltration_k[j,i],green_ampt_ks[j,i],
            green_ampt_psi[j,i],green_ampt_delta_theta[j,i],cumulative_infiltration[j,i],
            infiltration_method,subgrid_infiltration_enabled,wet_fraction,dt,source_mode,
            infiltration_wet_depth,committed_outflow_depth,cap_infil_committed_flux,
            infiltration_excluded,infiltration_cap_active,infiltration_cap_remaining_m[j,i])
        water_depth[j,i]=updated_depth
        if infiltration_method == 1 or infiltration_method == 3:
            cumulative_infiltration[j,i]=updated_cumulative
        if infiltration_excluded != 1 and infiltration_cap_active == 1:
            infiltration_cap_remaining_m[j,i]=updated_remaining

        # Deferred mass accounting (accounting_interval > 1): the kernel
        # accumulates the per-step applied depths in place into persistent
        # pending_* output fields instead of overwriting them. The host then
        # tree-sums the pending fields once per accounting flush, replacing
        # three CUB reductions per step with three per flush while still
        # accounting for every step exactly.
        if accumulate_output == 1:
            applied_rain[j, i] = applied_rain[j, i] + wp.float64(r_depth)
            applied_other[j, i] = applied_other[j, i] + wp.float64(o_depth)
            applied_infil[j, i] = applied_infil[j, i] + wp.float64(i_max_depth)
        else:
            applied_rain[j, i] = wp.float64(r_depth)
            applied_other[j, i] = wp.float64(o_depth)
            applied_infil[j, i] = wp.float64(i_max_depth)

        # On-device source-volume accounting. When accumulate_accounting == 1 the
        # per-cell volumes are accumulated here with float64 atomics (used on small
        # grids where launch overhead would exceed the atomic cost). On large grids
        # the host instead reduces the applied_* arrays once per step
        # (see LocalInertialSolver._reduce_mass_accounting) to avoid ~N global
        # atomics contending on three scalars.
        cell_area = wp.float64(dx) * wp.float64(dy)
        if accumulate_accounting == 1:
            wp.atomic_add(cumulative_totals, 0, wp.float64(r_depth) * cell_area)
            wp.atomic_add(cumulative_totals, 1, wp.float64(i_max_depth) * cell_area)
            wp.atomic_add(cumulative_totals, 2, wp.float64(o_depth) * cell_area)
        # The additional_source source has no tree-reduction path; it always uses a
        # direct atomic (slot 5) so its accounting stays exact in every mode.
        wp.atomic_add(cumulative_totals, 5, wp.float64(g_depth) * cell_area)
    else:
        if accumulate_output == 0:
            applied_rain[j, i] = wp.float64(0.0)
            applied_other[j, i] = wp.float64(0.0)
            applied_infil[j, i] = wp.float64(0.0)


@wp.kernel
def accumulate_mass_kernel(
    rain_sum: wp.array(dtype=wp.float64),
    infil_sum: wp.array(dtype=wp.float64),
    other_sum: wp.array(dtype=wp.float64),
    cell_area: wp.float64,
    cumulative_totals: wp.array(dtype=wp.float64),
):
    """Fold three device-reduced per-step source sums into the accumulator.

    Single-thread (dim=1), so the read-modify-writes are uncontended. Pairs with
    ``wp.utils.array_sum`` reductions of the per-cell ``applied_*`` arrays to
    replace ~3*N global float64 atomics per step with three native tree sums.
    """
    cumulative_totals[0] = cumulative_totals[0] + wp.float64(rain_sum[0]) * cell_area
    cumulative_totals[1] = cumulative_totals[1] + wp.float64(infil_sum[0]) * cell_area
    cumulative_totals[2] = cumulative_totals[2] + wp.float64(other_sum[0]) * cell_area


@wp.kernel
def check_negative_depth_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    neg_depth_tol: WP_FLOAT,
    error_flag: wp.array(dtype=wp.int32)
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        h = water_depth[j, i]
        if h < WP_FLOAT(0.0):
            if h >= -neg_depth_tol:
                water_depth[j, i] = WP_FLOAT(0.0)
            else:
                wp.atomic_add(error_flag, 0, 1)


@wp.kernel
def check_negative_depth_and_max_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    neg_depth_tol: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
):
    """Validate depth and reduce the next ACC timestep's maximum depth.

    ``control[0]`` is the active maximum depth and ``control[1]`` is a binary
    negative-depth error flag. This runs after continuity, so its maximum is
    exactly the field used by the next host-stepped ACC timestep selection.
    """
    j, i = wp.tid()
    if active_mask[j, i] != 1:
        return

    h = water_depth[j, i]
    if h < WP_FLOAT(0.0):
        if h >= -neg_depth_tol:
            water_depth[j, i] = WP_FLOAT(0.0)
            h = WP_FLOAT(0.0)
        else:
            wp.atomic_max(control, 1, wp.float64(1.0))

    wp.atomic_max(control, 0, wp.float64(h))

@wp.kernel
def sum_array_kernel(
    arr: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    total_vol: wp.array(dtype=wp.float64)
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        wp.atomic_add(
            total_vol,
            0,
            wp.float64(arr[j, i]) * wp.float64(dx) * wp.float64(dy),
        )
# ---------------------------------------------------------------------------
# Device-side simulation loop (opt-in; see LocalInertialSolver.run_device_loop)
#
# The loop advances whole windows of steps without host synchronisation. All
# per-step scalars live in a float64 control array so kernels read the current
# timestep and forcing directly instead of receiving launch scalars. Every
# formula, clamp, and evaluation order below mirrors the host-stepped path
# exactly so trajectories and dt sequences are bit-identical.
#
# Control array layout (indices):
DL_TIME = 0       # simulation clock (advanced by the continuity kernel)
DL_DT = 1         # timestep for the current step
DL_ERROR = 2      # 0 ok / 1 significant negative depth / 2 dt below minimum_dt
DL_MAX_DEPTH = 3  # max active depth from the previous continuity (ACC dt)
DL_RAIN_RATE = 4  # current uniform rainfall rate sampled from the hyetograph
DL_MAX_DT = 5     # max_dt cap for the current step (default-profile predict)
DL_STAGE_QX = 6   # current uniform specified-stage value for qx faces
DL_STAGE_QY = 7   # current uniform specified-stage value for qy faces
DL_COUNTER = 8    # number of accepted steps in this device-loop run
DL_DONE = 9       # 1 once the clock reached the end time (extra steps no-op)
DL_INTERP_SCRATCH = 10  # rounding barrier slot for the finalize kernel
DL_GROWTH_REFERENCE = 11  # previous unconstrained adaptive dt for controlled growth
DL_LAST_ACCEPTED_DT = 12  # preserved when the terminal no-op clears DL_DT
DL_CHECKPOINT_INDEX = 13  # next device-resident checkpoint trace row
DL_RECORD_CHECKPOINT = 14  # current step lands exactly on that checkpoint
DL_DT_MIN = 15            # minimum accepted dt in this device-loop call
DL_DT_MAX = 16            # maximum accepted dt in this device-loop call
DL_DT_SUM = 17            # sum of accepted dt in this device-loop call
DL_SIZE = 18


@wp.func
def _device_piecewise_linear_interp(
    times: wp.array(dtype=wp.float64),
    values: wp.array(dtype=wp.float64),
    t: wp.float64,
    scratch: wp.array(dtype=wp.float64),
) -> wp.float64:
    """Match numpy.interp exactly (clamped ends, slope*(t - t0) + v0 form).

    The segment is the one whose left node is the largest node time <= t, so
    exact-node hits return the node value with no slope arithmetic, mirroring
    numpy's binary_search behaviour.
    """
    n = times.shape[0]
    if t <= times[0]:
        return values[0]
    if t >= times[n - 1]:
        return values[n - 1]
    idx = int(0)
    while idx < n - 2 and times[idx + 1] <= t:
        idx += 1
    slope = (values[idx + 1] - values[idx]) / (times[idx + 1] - times[idx])
    # Route the product through memory so the compiler cannot contract
    # mul+add into an FMA; np.interp rounds the multiply and add separately.
    scratch[DL_INTERP_SCRATCH] = slope * (t - times[idx])
    return scratch[DL_INTERP_SCRATCH] + values[idx]


@wp.func
def _device_next_breakpoint(times: wp.array(dtype=wp.float64), t: wp.float64) -> wp.float64:
    """Match Hyetograph.next_breakpoint_after (searchsorted side='right')."""
    n = times.shape[0]
    idx = int(0)
    while idx < n - 1 and times[idx] <= t:
        idx += 1
    return times[idx]


@wp.func
def _device_hard_dt_cap(
    control: wp.array(dtype=wp.float64),
    end_time: wp.float64,
    maximum_dt: wp.float64,
    hyeto_times: wp.array(dtype=wp.float64),
    use_hyetograph: int,
    storm_duration: wp.float64,
    checkpoint_times: wp.array(dtype=wp.float64),
    checkpoint_count: int,
) -> wp.float64:
    """Return the hard end/forcing/maximum timestep cap."""
    t = control[DL_TIME]
    max_dt = wp.min(maximum_dt, end_time - t)
    if use_hyetograph == 1:
        if t < storm_duration:
            breakpoint_time = _device_next_breakpoint(hyeto_times, t)
            max_dt = wp.min(max_dt, breakpoint_time - t)
    checkpoint_index = int(control[DL_CHECKPOINT_INDEX])
    if checkpoint_index < checkpoint_count:
        max_dt = wp.min(max_dt, checkpoint_times[checkpoint_index] - t)
    return max_dt


@wp.func
def _device_max_dt_cap(
    control: wp.array(dtype=wp.float64),
    end_time: wp.float64,
    maximum_dt: wp.float64,
    adaptive_dt_growth_factor: wp.float64,
    hyeto_times: wp.array(dtype=wp.float64),
    use_hyetograph: int,
    storm_duration: wp.float64,
    checkpoint_times: wp.array(dtype=wp.float64),
    checkpoint_count: int,
) -> wp.float64:
    """Return the hard cap additionally limited by controlled adaptive growth."""
    max_dt = _device_hard_dt_cap(
        control, end_time, maximum_dt, hyeto_times, use_hyetograph,
        storm_duration, checkpoint_times, checkpoint_count
    )
    previous_dt = control[DL_GROWTH_REFERENCE]
    if adaptive_dt_growth_factor >= wp.float64(1.0) and previous_dt > wp.float64(0.0):
        max_dt = wp.min(max_dt, previous_dt * adaptive_dt_growth_factor)
    return max_dt


@wp.kernel
def device_loop_timestep_cap_kernel(
    control: wp.array(dtype=wp.float64),
    dt_min_out: wp.array(dtype=WP_FLOAT),
    end_time: wp.float64,
    maximum_dt: wp.float64,
    adaptive_dt_growth_factor: wp.float64,
    hyeto_times: wp.array(dtype=wp.float64),
    use_hyetograph: int,
    storm_duration: wp.float64,
    checkpoint_times: wp.array(dtype=wp.float64),
    checkpoint_count: int,
):
    """Store the hard/growth-limited dt cap and reset the CFL minimum."""
    if control[DL_DONE] != 0.0:
        return
    control[DL_MAX_DT] = _device_max_dt_cap(
        control, end_time, maximum_dt, adaptive_dt_growth_factor,
        hyeto_times, use_hyetograph, storm_duration,
        checkpoint_times, checkpoint_count
    )
    dt_min_out[0] = WP_FLOAT(maximum_dt)


@wp.kernel
def device_loop_prepare_forcing_kernel(
    control: wp.array(dtype=wp.float64),
    hyeto_times: wp.array(dtype=wp.float64), hyeto_rates: wp.array(dtype=wp.float64),
    use_hyetograph: int, storm_duration: wp.float64,
    stage_times_qx: wp.array(dtype=wp.float64), stage_values_qx: wp.array(dtype=wp.float64), use_stage_qx: int,
    stage_times_qy: wp.array(dtype=wp.float64), stage_values_qy: wp.array(dtype=wp.float64), use_stage_qy: int,
):
    """Sample current stages and a conservative rain bound before CFL."""
    if control[DL_DONE] != wp.float64(0.0):
        return
    t=control[DL_TIME]
    cap=control[DL_MAX_DT]
    if use_stage_qx == 1:
        if t < stage_times_qx[stage_times_qx.shape[0]-1]:
            cap=wp.min(cap,_device_next_breakpoint(stage_times_qx,t)-t)
        control[DL_STAGE_QX]=_device_piecewise_linear_interp(stage_times_qx,stage_values_qx,t,control)
    if use_stage_qy == 1:
        if t < stage_times_qy[stage_times_qy.shape[0]-1]:
            cap=wp.min(cap,_device_next_breakpoint(stage_times_qy,t)-t)
        control[DL_STAGE_QY]=_device_piecewise_linear_interp(stage_times_qy,stage_values_qy,t,control)
    control[DL_MAX_DT]=cap
    if use_hyetograph == 1:
        rate=wp.float64(0.0)
        if t < storm_duration:
            rate=wp.max(_device_piecewise_linear_interp(hyeto_times,hyeto_rates,t,control),
                _device_piecewise_linear_interp(hyeto_times,hyeto_rates,t+cap,control))
        control[DL_RAIN_RATE]=rate


@wp.kernel
def device_loop_predict_depth_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    rainfall_spatial_multiplier: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    additional_source_rate: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    control: wp.array(dtype=wp.float64),
    use_uniform_rate: int,
    predicted_depth: wp.array2d(dtype=WP_FLOAT)
):
    """predict_depth_for_cfl_kernel with dt/rate read from the control array."""
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        rain_value = rainfall_rate[j, i]
        if use_uniform_rate == 1:
            rain_value = WP_FLOAT(control[DL_RAIN_RATE]) * rainfall_spatial_multiplier[j, i]
        source_rate = wp.max(WP_FLOAT(0.0), rain_value) + wp.max(WP_FLOAT(0.0), other_source_rate[j, i]) + wp.max(WP_FLOAT(0.0), additional_source_rate[j, i])
        predicted_depth[j, i] = water_depth[j, i] + source_rate * WP_FLOAT(control[DL_MAX_DT])
    else:
        predicted_depth[j, i] = WP_FLOAT(0.0)


@wp.kernel
def device_loop_max_depth_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    control: wp.array(dtype=wp.float64),
):
    """First-step ACC max-depth reduction (mirrors max_active_depth_kernel)."""
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        wp.atomic_max(control, DL_MAX_DEPTH, wp.float64(water_depth[j, i]))


@wp.kernel
def device_loop_max_depth_partial_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    total_threads: int,
    control: wp.array(dtype=wp.float64),
    partials: wp.array(dtype=wp.float64),
):
    """Per-thread partial max of the post-continuity depths (no atomics).

    Each of ``total_threads`` threads reduces a coalesced strided slice in a
    register and stores it.  Reads ``water_depth`` (the just-accepted
    post-continuity depths), so it must be launched after the continuity
    kernel of the same step.
    """
    t = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    nx = water_depth.shape[1]
    cell_count = water_depth.shape[0] * nx
    local_max = wp.float64(0.0)
    index = t
    while index < cell_count:
        r = index // nx
        c = index - r * nx
        if active_mask[r, c] == 1:
            local_max = wp.max(local_max, wp.float64(water_depth[r, c]))
        index += total_threads
    partials[t] = local_max


@wp.kernel
def device_loop_max_depth_finalize_kernel(
    partials: wp.array(dtype=wp.float64),
    total_threads: int,
    control: wp.array(dtype=wp.float64),
):
    """Fold the partial maxima into control[DL_MAX_DEPTH].

    Replaces the former per-cell global atomic_max in the continuity kernels,
    whose CAS retry loop on one address dominated the device-loop step cost.
    Each thread strides over a slice of the partials and issues one
    atomic_max, so contention stays bounded at ``total_threads`` CAS
    operations.  Float64 max is order-exact, so the result is bit-identical
    to the per-cell atomic reduction.
    """
    t = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    count = partials.shape[0]
    local_max = wp.float64(0.0)
    index = t
    while index < count:
        local_max = wp.max(local_max, partials[index])
        index += total_threads
    wp.atomic_max(control, DL_MAX_DEPTH, local_max)


@wp.func
def _device_timestep_finalize_from_cfl(
    control: wp.array(dtype=wp.float64),
    cfl_dt: wp.float64,
    end_time: wp.float64,
    minimum_dt: wp.float64,
    maximum_dt: wp.float64,
    adaptive_dt_growth_factor: wp.float64,
    use_acc_timestep: int,
    cfl_number: WP_FLOAT,
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    hyeto_times: wp.array(dtype=wp.float64),
    hyeto_rates: wp.array(dtype=wp.float64),
    use_hyetograph: int,
    storm_duration: wp.float64,
    sample_at_half_step: int,
    stage_times_qx: wp.array(dtype=wp.float64),
    stage_values_qx: wp.array(dtype=wp.float64),
    use_stage_qx: int,
    stage_times_qy: wp.array(dtype=wp.float64),
    stage_values_qy: wp.array(dtype=wp.float64),
    use_stage_qy: int,
    checkpoint_times: wp.array(dtype=wp.float64),
    checkpoint_count: int,
):
    """Select dt on-device with the identical host formula and clamps.

    ACC: dt = max_dt when max depth <= dry_depth, else
         min(max_dt, cfl*dx/sqrt(gravity*max_depth)) using the max depth
         reduced by the previous step's continuity kernel.
    Default: dt = min(max_dt, min(maximum_dt, CFL face minimum)) where the CFL
         minimum was reduced over the source-predicted depth field. ``max_dt``
         includes the optional controlled-growth cap from the previous adaptive step.
    Also samples the rainfall rate (step start, or t + dt/2 when sources lead
    momentum) and any specified-stage series at the step start time.
    """
    if control[DL_RECORD_CHECKPOINT] != wp.float64(0.0):
        control[DL_CHECKPOINT_INDEX] = control[DL_CHECKPOINT_INDEX] + wp.float64(1.0)
        control[DL_RECORD_CHECKPOINT] = wp.float64(0.0)
    if control[DL_DONE] != 0.0:
        control[DL_DT] = wp.float64(0.0)
        return
    t = control[DL_TIME]
    if t >= end_time:
        control[DL_DONE] = wp.float64(1.0)
        control[DL_DT] = wp.float64(0.0)
        return
    remaining = end_time - t
    checkpoint_index = int(control[DL_CHECKPOINT_INDEX])
    if checkpoint_index < checkpoint_count:
        remaining = wp.min(remaining, checkpoint_times[checkpoint_index] - t)
    hard_max_dt = _device_hard_dt_cap(
        control, end_time, maximum_dt, hyeto_times, use_hyetograph,
        storm_duration, checkpoint_times, checkpoint_count
    )
    max_dt = _device_max_dt_cap(
        control, end_time, maximum_dt, adaptive_dt_growth_factor,
        hyeto_times, use_hyetograph, storm_duration,
        checkpoint_times, checkpoint_count
    )

    if use_stage_qx == 1 and t < stage_times_qx[stage_times_qx.shape[0]-1]:
        stage_cap=_device_next_breakpoint(stage_times_qx,t)-t
        hard_max_dt=wp.min(hard_max_dt,stage_cap)
        max_dt=wp.min(max_dt,stage_cap)
    if use_stage_qy == 1 and t < stage_times_qy[stage_times_qy.shape[0]-1]:
        stage_cap=_device_next_breakpoint(stage_times_qy,t)-t
        hard_max_dt=wp.min(hard_max_dt,stage_cap)
        max_dt=wp.min(max_dt,stage_cap)

    dt = max_dt
    # Stable timestep without the hard end/breakpoint cap (growth cap only).
    uncapped_dt = maximum_dt
    if adaptive_dt_growth_factor >= wp.float64(1.0) and control[DL_GROWTH_REFERENCE] > wp.float64(0.0):
        uncapped_dt = wp.min(
            uncapped_dt,
            control[DL_GROWTH_REFERENCE] * adaptive_dt_growth_factor,
        )
    if use_acc_timestep == 1:
        maximum_depth_m = control[DL_MAX_DEPTH]
        if maximum_depth_m > wp.float64(dry_depth):
            acc_dt = wp.float64(cfl_number) * wp.float64(dx) / wp.sqrt(wp.float64(gravity) * maximum_depth_m)
            dt = wp.min(max_dt, acc_dt)
            uncapped_dt = wp.min(uncapped_dt, acc_dt)
    else:
        dt = wp.min(max_dt, wp.min(maximum_dt, cfl_dt))
        uncapped_dt = wp.min(uncapped_dt, cfl_dt)

    # The host raises when the stable timestep collapses below minimum_dt
    # unless the remaining interval is smaller than minimum_dt.  The error
    # test therefore uses the timestep *without* the hard end/breakpoint cap:
    # a sub-minimum step that merely closes a small gap to a hyetograph
    # breakpoint or the run end is permitted, while a genuine CFL/ACC
    # collapse is flagged and the host raises at the next sync.
    if uncapped_dt < minimum_dt and remaining >= minimum_dt:
        control[DL_ERROR] = wp.float64(2.0)
    if adaptive_dt_growth_factor >= wp.float64(1.0):
        boundary_limited = hard_max_dt < maximum_dt and dt >= hard_max_dt - wp.float64(1.0e-12)
        if not boundary_limited or control[DL_GROWTH_REFERENCE] <= wp.float64(0.0):
            control[DL_GROWTH_REFERENCE] = dt
    control[DL_DT] = dt
    if checkpoint_index < checkpoint_count:
        if t + dt >= checkpoint_times[checkpoint_index]:
            control[DL_RECORD_CHECKPOINT] = wp.float64(1.0)
    # Consumed for this step; the next continuity reduction starts from zero
    # (mirrors the host's per-step lisflood_step_control.zero_()).
    control[DL_MAX_DEPTH] = wp.float64(0.0)

    if use_hyetograph == 1:
        t_sample = t
        if sample_at_half_step == 1:
            control[DL_INTERP_SCRATCH] = wp.float64(0.5) * dt
            t_sample = t_sample + control[DL_INTERP_SCRATCH]
        # Hyetograph semantics: rainfall is zero outside the storm.  Sampling
        # must not clamp onto the final node — that would hold the terminal
        # rate forever once t passes the storm end.
        if t_sample >= storm_duration:
            control[DL_RAIN_RATE] = wp.float64(0.0)
        else:
            control[DL_RAIN_RATE] = _device_piecewise_linear_interp(hyeto_times, hyeto_rates, t_sample, control)
    if use_stage_qx == 1:
        control[DL_STAGE_QX] = _device_piecewise_linear_interp(stage_times_qx, stage_values_qx, t, control)
    if use_stage_qy == 1:
        control[DL_STAGE_QY] = _device_piecewise_linear_interp(stage_times_qy, stage_values_qy, t, control)



@wp.kernel
def device_loop_timestep_finalize_kernel(
    control: wp.array(dtype=wp.float64),
    dt_min: wp.array(dtype=WP_FLOAT),
    end_time: wp.float64,
    minimum_dt: wp.float64,
    maximum_dt: wp.float64,
    adaptive_dt_growth_factor: wp.float64,
    use_acc_timestep: int,
    cfl_number: WP_FLOAT,
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    hyeto_times: wp.array(dtype=wp.float64),
    hyeto_rates: wp.array(dtype=wp.float64),
    use_hyetograph: int,
    storm_duration: wp.float64,
    sample_at_half_step: int,
    stage_times_qx: wp.array(dtype=wp.float64),
    stage_values_qx: wp.array(dtype=wp.float64),
    use_stage_qx: int,
    stage_times_qy: wp.array(dtype=wp.float64),
    stage_values_qy: wp.array(dtype=wp.float64),
    use_stage_qy: int,
    checkpoint_times: wp.array(dtype=wp.float64),
    checkpoint_count: int,
):
    """Production controller, shared with component hydraulics."""
    _device_timestep_finalize_from_cfl(
        control, wp.float64(dt_min[0]), end_time, minimum_dt, maximum_dt, adaptive_dt_growth_factor, use_acc_timestep, cfl_number, dx, gravity, dry_depth, hyeto_times, hyeto_rates, use_hyetograph, storm_duration, sample_at_half_step, stage_times_qx, stage_values_qx, use_stage_qx, stage_times_qy, stage_values_qy, use_stage_qy, checkpoint_times, checkpoint_count
    )

@wp.kernel
def device_loop_stage_fill_kernel(
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    control: wp.array(dtype=wp.float64),
    use_stage_qx: int,
    use_stage_qy: int,
):
    """Write the sampled stage values to all SPECIFIED_STAGE faces."""
    j, i = wp.tid()
    if use_stage_qx == 1 and j < boundary_stage_qx.shape[0]:
        if boundary_type_qx[j, i] == BOUNDARY_SPECIFIED_STAGE or boundary_type_qx[j, i] == BOUNDARY_MASKED_WEST_STAGE or boundary_type_qx[j, i] == BOUNDARY_MASKED_EAST_STAGE:
            boundary_stage_qx[j, i] = STAGE_FLOAT(control[DL_STAGE_QX])
    if use_stage_qy == 1 and i < boundary_stage_qy.shape[1]:
        if boundary_type_qy[j, i] == BOUNDARY_SPECIFIED_STAGE or boundary_type_qy[j, i] == BOUNDARY_MASKED_NORTH_STAGE or boundary_type_qy[j, i] == BOUNDARY_MASKED_SOUTH_STAGE:
            boundary_stage_qy[j, i] = STAGE_FLOAT(control[DL_STAGE_QY])


@wp.kernel
def device_loop_conditioned_stage_fill_kernel(
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_qx: wp.array2d(dtype=STAGE_FLOAT),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_stage_reference_qx: wp.array2d(dtype=STAGE_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_qy: wp.array2d(dtype=STAGE_FLOAT),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_stage_reference_qy: wp.array2d(dtype=STAGE_FLOAT),
    control: wp.array(dtype=wp.float64),
    use_stage_qx: int,
    use_stage_qy: int,
):
    """Write datum-relative FP32 stage depths for conditioned mixed kernels."""
    j, i = wp.tid()
    if use_stage_qx == 1 and j < boundary_stage_depth_qx.shape[0]:
        if boundary_type_qx[j, i] == BOUNDARY_SPECIFIED_STAGE or boundary_type_qx[j, i] == BOUNDARY_MASKED_WEST_STAGE or boundary_type_qx[j, i] == BOUNDARY_MASKED_EAST_STAGE:
            stage = STAGE_FLOAT(control[DL_STAGE_QX])
            boundary_stage_qx[j, i] = stage
            boundary_stage_depth_qx[j, i] = WP_FLOAT(stage - boundary_stage_reference_qx[j, i])
    if use_stage_qy == 1 and i < boundary_stage_depth_qy.shape[1]:
        if boundary_type_qy[j, i] == BOUNDARY_SPECIFIED_STAGE or boundary_type_qy[j, i] == BOUNDARY_MASKED_NORTH_STAGE or boundary_type_qy[j, i] == BOUNDARY_MASKED_SOUTH_STAGE:
            stage = STAGE_FLOAT(control[DL_STAGE_QY])
            boundary_stage_qy[j, i] = stage
            boundary_stage_depth_qy[j, i] = WP_FLOAT(stage - boundary_stage_reference_qy[j, i])


@wp.kernel
def device_loop_apply_sources_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    rainfall_spatial_multiplier: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_capacity: wp.array2d(dtype=WP_FLOAT),
    infiltration_f0: wp.array2d(dtype=WP_FLOAT),
    infiltration_fd: wp.array2d(dtype=WP_FLOAT),
    infiltration_k: wp.array2d(dtype=WP_FLOAT),
    green_ampt_ks: wp.array2d(dtype=WP_FLOAT),
    green_ampt_psi: wp.array2d(dtype=WP_FLOAT),
    green_ampt_delta_theta: wp.array2d(dtype=WP_FLOAT),
    cumulative_infiltration: wp.array2d(dtype=WP_FLOAT),
    infiltration_method: int,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    active_mask: wp.array2d(dtype=wp.int32),
    control: wp.array(dtype=wp.float64),
    use_uniform_rate: int,
    source_mode: int,
    infiltration_wet_depth: WP_FLOAT,
    committed_qx: wp.array2d(dtype=WP_FLOAT),
    committed_qy: wp.array2d(dtype=WP_FLOAT),
    cap_infil_committed_flux: int,
    applied_rain: wp.array2d(dtype=wp.float64),
    applied_other: wp.array2d(dtype=wp.float64),
    applied_infil: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
    accumulate_accounting: int,
    accumulate_output: int,
    additional_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_exclusion_mask: wp.array2d(dtype=wp.int32),
    infiltration_cap_active: int,
    infiltration_cap_remaining_m: wp.array2d(dtype=WP_FLOAT),
):
    """apply_sources_kernel with dt (and optionally a uniform rate) on-device."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    rain_value = rainfall_rate[j, i]
    if use_uniform_rate == 1:
        rain_value = WP_FLOAT(control[DL_RAIN_RATE]) * rainfall_spatial_multiplier[j, i]
    _apply_sources_cell(
        j, i, water_depth, rain_value, other_source_rate[j, i],
        additional_source_rate[j, i],
        infiltration_capacity[j, i], infiltration_f0, infiltration_fd,
        infiltration_k, green_ampt_ks, green_ampt_psi,
        green_ampt_delta_theta, cumulative_infiltration, infiltration_method,
        subgrid_infiltration_enabled, subgrid_cell_z,
        subgrid_cell_storage_breakpoints, subgrid_cell_count,
        active_mask, WP_FLOAT(control[DL_DT]),
        source_mode, infiltration_wet_depth, committed_qx, committed_qy,
        cap_infil_committed_flux, infiltration_exclusion_mask[j, i],
        infiltration_cap_active, infiltration_cap_remaining_m,
        applied_rain, applied_other, applied_infil,
        dx, dy, cumulative_totals, accumulate_accounting,
        accumulate_output,
    )


@wp.kernel
def device_loop_update_momentum_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
):
    """Fused momentum update with dt read from the control array."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    dt = WP_FLOAT(control[DL_DT])
    if j < qx.shape[0]:
        qx[j, i] = _update_qx_face(
            i, j, qx_old, qy_old, water_depth, bed_elevation, manning_n,
            boundary_type_qx, boundary_stage_qx,
            dx, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
        )
    if i < qy.shape[1]:
        qy[j, i] = _update_qy_face(
            i, j, qy_old, qx_old, water_depth, bed_elevation, manning_n,
            boundary_type_qy, boundary_stage_qy,
            dy, dt, gravity, dry_depth, depth_epsilon,
            manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
        )


@wp.kernel
def device_loop_override_free_outflow_arithmetic_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    arithmetic_mode: int,
):
    """Device-loop counterpart of the opt-in FREE arithmetic override."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    ny = water_depth.shape[0]
    nx = water_depth.shape[1]
    dt = WP_FLOAT(control[DL_DT])
    if j < ny and boundary_type_qx[j, i] == BOUNDARY_FREE_OUTFLOW:
        edge_column = nx - 1
        inward_column = nx - 2
        direction = WP_FLOAT(1.0)
        if i == 0:
            edge_column = 0
            inward_column = 1
            direction = WP_FLOAT(-1.0)
        qx[j, i] = _diagnostic_free_override(
            qx[j, i], qx_old[j, i], water_depth[j, edge_column],
            bed_elevation[j, edge_column], water_depth[j, inward_column],
            bed_elevation[j, inward_column], manning_n[j, edge_column],
            boundary_stage_qx[j, i], dx, dt, gravity, dry_depth,
            depth_epsilon, direction, arithmetic_mode,
        )
    if i < nx and boundary_type_qy[j, i] == BOUNDARY_FREE_OUTFLOW:
        edge_row = ny - 1
        inward_row = ny - 2
        direction = WP_FLOAT(1.0)
        if j == 0:
            edge_row = 0
            inward_row = 1
            direction = WP_FLOAT(-1.0)
        qy[j, i] = _diagnostic_free_override(
            qy[j, i], qy_old[j, i], water_depth[edge_row, i],
            bed_elevation[edge_row, i], water_depth[inward_row, i],
            bed_elevation[inward_row, i], manning_n[edge_row, i],
            boundary_stage_qy[j, i], dy, dt, gravity, dry_depth,
            depth_epsilon, direction, arithmetic_mode,
        )


@wp.kernel
def device_loop_update_conditioned_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qx: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    """Conditioned mixed x-momentum update with device-resident timestep.

    This intentionally remains separate from the y update.  Combining both
    directional face calculations produces enough live FP32 values to reduce
    occupancy on consumer GPUs; qx and qy still read only the immutable old
    ping-pong buffers, so the two launches are numerically independent.
    """
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    qx[j, i] = _update_conditioned_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_slope_qx, manning_n,
        boundary_type_qx, boundary_stage_depth_qx, boundary_free_slope_qx,
        dx, WP_FLOAT(control[DL_DT]), gravity, dry_depth, depth_epsilon,
        manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def device_loop_update_conditioned_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qy: wp.array2d(dtype=WP_FLOAT),
    dy: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    """Conditioned mixed y-momentum update with device-resident timestep."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    qy[j, i] = _update_conditioned_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_slope_qy, manning_n,
        boundary_type_qy, boundary_stage_depth_qy, boundary_free_slope_qy,
        dy, WP_FLOAT(control[DL_DT]), gravity, dry_depth, depth_epsilon,
        manning_face_mode, use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def device_loop_apply_outflow_limiter_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    epsilon: WP_FLOAT,
    free_outflow_limiter_mode: int,
    free_outflow_diagnostics: wp.array(dtype=wp.float64),
    collect_free_outflow_diagnostics: int,
):
    """Fused outflow limiter with dt read from the control array."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    _apply_outflow_limiter_cell(
        j, i, water_depth, qx, qy, active_mask, boundary_type_qx,
        boundary_type_qy, dx, dy, WP_FLOAT(control[DL_DT]), epsilon,
        free_outflow_limiter_mode, free_outflow_diagnostics,
        collect_free_outflow_diagnostics,
    )


@wp.func
def _device_commit_clock(
    control: wp.array(dtype=wp.float64), dt_history: wp.array(dtype=wp.float64),
    history_capacity: int, record_history: int,
):
    step_index = int(control[DL_COUNTER])
    if record_history == 1 and step_index < history_capacity:
        dt_history[step_index] = control[DL_DT]
    control[DL_LAST_ACCEPTED_DT] = control[DL_DT]
    control[DL_DT_MIN] = wp.min(control[DL_DT_MIN], control[DL_DT])
    control[DL_DT_MAX] = wp.max(control[DL_DT_MAX], control[DL_DT])
    control[DL_DT_SUM] = control[DL_DT_SUM] + control[DL_DT]
    control[DL_COUNTER] = control[DL_COUNTER] + wp.float64(1.0)
    control[DL_TIME] = control[DL_TIME] + control[DL_DT]


@wp.kernel
def device_loop_continuity_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    neg_depth_tol: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    cumulative_totals: wp.array(dtype=wp.float64),
    dt_history: wp.array(dtype=wp.float64),
    history_capacity: int,
    record_history: int,
):
    """Legacy device-loop continuity without checkpoint tracing.

    The ACC max-depth reduction is performed by the separately launched
    device_loop_max_depth_partial/finalize kernels (contention-free; the
    float64 max result is order-exact, hence bit-identical).
    """
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    dt = WP_FLOAT(control[DL_DT])
    _accumulate_boundary_flux_edges(
        j, i, qx, qy, boundary_type_qx, boundary_type_qy, dx, dy, dt, cumulative_totals
    )
    if active_mask[j, i] != 1:
        water_depth[j, i] = WP_FLOAT(0.0)
    else:
        dqx = (qx[j, i + 1] - qx[j, i]) / dx
        dqy = (qy[j + 1, i] - qy[j, i]) / dy
        h_new = water_depth[j, i] - dt * (dqx + dqy)
        if h_new < WP_FLOAT(0.0):
            if h_new >= -neg_depth_tol:
                h_new = WP_FLOAT(0.0)
            else:
                wp.atomic_max(control, DL_ERROR, wp.float64(1.0))
        water_depth[j, i] = h_new
    if j == 0 and i == 0:
        if control[DL_DONE] == 0.0:
            _device_commit_clock(control,dt_history,history_capacity,record_history)


@wp.kernel
def device_loop_continuity_checkpoint_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    neg_depth_tol: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    cumulative_totals: wp.array(dtype=wp.float64),
    dt_history: wp.array(dtype=wp.float64),
    history_capacity: int,
    record_history: int,
    checkpoint_outlet_rows: wp.array(dtype=wp.int32),
    checkpoint_field_indices: wp.array(dtype=wp.int32),
    checkpoint_times_out: wp.array(dtype=wp.float64),
    checkpoint_outlet_out: wp.array2d(dtype=wp.float64),
    checkpoint_volume_out: wp.array(dtype=wp.float64),
    checkpoint_depth_out: wp.array3d(dtype=WP_FLOAT),
    checkpoint_count: int,
):
    """Device-loop continuity: update + check + accounting + clock advance.

    Mirrors continuity_update_and_check_kernel (same update, tolerance clamp,
    boundary-flux accounting) with the error flag in control[DL_ERROR].  The
    ACC max-depth reduction is performed by the separately launched
    device_loop_max_depth_partial/finalize kernels (contention-free; the
    float64 max result is order-exact, hence bit-identical).  Thread (0, 0)
    then appends dt to the optional history, increments the step counter, and
    advances the simulation clock — the same order as host step() (state
    update first, then ``simulation_time += dt``).
    """
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    dt = WP_FLOAT(control[DL_DT])
    _accumulate_boundary_flux_edges(
        j, i, qx, qy, boundary_type_qx, boundary_type_qy, dx, dy, dt, cumulative_totals
    )
    h_new = WP_FLOAT(0.0)
    if active_mask[j, i] != 1:
        water_depth[j, i] = WP_FLOAT(0.0)
    else:
        dqx = (qx[j, i + 1] - qx[j, i]) / dx
        dqy = (qy[j + 1, i] - qy[j, i]) / dy
        h_new = water_depth[j, i] - dt * (dqx + dqy)

        if h_new < WP_FLOAT(0.0):
            if h_new >= -neg_depth_tol:
                h_new = WP_FLOAT(0.0)
            else:
                wp.atomic_max(control, DL_ERROR, wp.float64(1.0))

        water_depth[j, i] = h_new

    checkpoint_index = int(control[DL_CHECKPOINT_INDEX])
    if control[DL_RECORD_CHECKPOINT] != wp.float64(0.0) and checkpoint_index < checkpoint_count:
        if active_mask[j, i] == 1:
            wp.atomic_add(
                checkpoint_volume_out,
                checkpoint_index,
                wp.float64(h_new) * wp.float64(dx) * wp.float64(dy),
            )
        if i == water_depth.shape[1] - 1 and checkpoint_outlet_rows[j] == 1:
            checkpoint_outlet_out[checkpoint_index, j] = wp.float64(
                wp.max(qx[j, water_depth.shape[1]], WP_FLOAT(0.0))
            )
        field_index = checkpoint_field_indices[checkpoint_index]
        if field_index >= 0:
            checkpoint_depth_out[field_index, j, i] = h_new

    if j == 0 and i == 0:
        if control[DL_DONE] == 0.0:
            _device_commit_clock(control,dt_history,history_capacity,record_history)
            if control[DL_RECORD_CHECKPOINT] != wp.float64(0.0):
                checkpoint_times_out[checkpoint_index] = control[DL_TIME]


# ---------------------------------------------------------------------------
# Spatially adaptive local timestepping (LTS)
# ---------------------------------------------------------------------------
#
# The LTS path groups cells and faces into power-of-two timestep levels.  Face
# fluxes are integrated into a conservative per-cell volume register.  A cell
# consumes that register only when its own local interval ends, so a fine cell
# may subcycle against a frozen coarse neighbour without losing water across
# the shared face.

# Small device-resident schedule control. Integer-valued entries are stored in
# float64 so one tiny snapshot can carry both scalar timestep state and level
# counts back to the host when compact launch dimensions are required.
LTS_CTRL_BASE_DT = 0
LTS_CTRL_MACRO_DT = 1
LTS_CTRL_MACRO_LEVEL = 2
LTS_CTRL_MACRO_TICKS = 3
LTS_CTRL_ERROR = 4
LTS_CTRL_SIZE = 5

# Accuracy-controlled classification statistics (int64 array, run-accumulated;
# classifications can exceed 2**31 on large grids over long runs).
# The scalar block is followed by a fixed log-spaced histogram of the relative
# effective-depth change indicator so the host can derive quantiles from one
# tiny read instead of a full-grid copy.
LTS_STATS_ACTIVE = 0           # active-cell classifications
LTS_STATS_CFL_LIMITED = 1      # cells whose target level came from the CFL limit
LTS_STATS_ACCURACY_LIMITED = 2 # cells tightened by the temporal accuracy limit
LTS_STATS_WET_GUARD_LIMITED = 3   # cells clamped by the wet/source guard
LTS_STATS_FRONT_GUARD_LIMITED = 4 # dry cells clamped by the wetting-front guard
LTS_STATS_PROMOTIONS = 5       # accepted one-level promotions
LTS_STATS_DEMOTIONS = 6        # immediate demotions
LTS_STATS_PROMOTION_HELD = 7   # promotions withheld by the hysteresis margin
LTS_STATS_DRAINAGE_LIMITED = 8 # unforced draining cells clamped for the fringe
LTS_STATS_SCALARS = 9
LTS_STATS_HIST_BINS = 64
LTS_STATS_SIZE = LTS_STATS_SCALARS + LTS_STATS_HIST_BINS
LTS_HIST_LOG_MIN = -16.0        # histogram lower edge: log10(relative_change)
LTS_HIST_BINS_PER_DEX = 3.2     # 64 bins over the log10 range [-16, 4)


@wp.func
def _lts_finalize_schedule_from_dt(
    stable_dt: wp.float64,
    cap_dt: wp.float64,
    minimum_dt: wp.float64,
    max_levels: int,
    control: wp.array(dtype=wp.float64),
):
    base_dt = wp.min(cap_dt, stable_dt)
    error = wp.float64(0.0)
    if base_dt <= wp.float64(0.0):
        error = wp.float64(1.0)
    elif base_dt < minimum_dt and cap_dt >= minimum_dt:
        error = wp.float64(2.0)

    macro_level = int(0)
    if error == wp.float64(0.0):
        trial_dt = base_dt * wp.float64(2.0)
        while macro_level < max_levels and trial_dt <= cap_dt * wp.float64(1.0 + 1.0e-12):
            macro_level += 1
            trial_dt = trial_dt * wp.float64(2.0)

    macro_ticks = 1 << macro_level
    macro_dt = base_dt * wp.float64(macro_ticks)
    control[LTS_CTRL_BASE_DT] = base_dt
    control[LTS_CTRL_MACRO_DT] = macro_dt
    control[LTS_CTRL_MACRO_LEVEL] = wp.float64(macro_level)
    control[LTS_CTRL_MACRO_TICKS] = wp.float64(macro_ticks)
    control[LTS_CTRL_ERROR] = error


@wp.kernel
def lts_finalize_schedule_control_kernel(
    dt_min: wp.array(dtype=WP_FLOAT),cap_dt: wp.float64,minimum_dt: wp.float64,
    max_levels: int,control: wp.array(dtype=wp.float64)):
    """Finalize the LTS base timestep and macro hierarchy on-device."""
    if wp.tid()==0:
        _lts_finalize_schedule_from_dt(wp.float64(dt_min[0]),cap_dt,minimum_dt,max_levels,control)


@wp.kernel
def lts_predict_depth_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    rainfall_spatial_multiplier: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    additional_source_rate: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dt_cap: WP_FLOAT,
    uniform_rain_rate: WP_FLOAT,
    use_uniform_rain: int,
    predicted_depth: wp.array2d(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        rain = rainfall_rate[j, i]
        if use_uniform_rain == 1:
            rain = uniform_rain_rate * rainfall_spatial_multiplier[j, i]
        source_rate = wp.max(WP_FLOAT(0.0), rain) + wp.max(
            WP_FLOAT(0.0), other_source_rate[j, i]
        ) + wp.max(WP_FLOAT(0.0), additional_source_rate[j, i])
        predicted_depth[j, i] = water_depth[j, i] + source_rate * dt_cap
    else:
        predicted_depth[j, i] = WP_FLOAT(0.0)


@wp.kernel
def lts_conditioned_face_dt_x_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_cap: WP_FLOAT,
    face_dt: wp.array2d(dtype=WP_FLOAT),
    global_dt_min: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    nx = water_depth.shape[1]
    btype = boundary_type_qx[j, i]
    dt = dt_cap
    if btype != BOUNDARY_CLOSED:
        dz = WP_FLOAT(0.0)
        h_left = WP_FLOAT(0.0)
        h_right = WP_FLOAT(0.0)
        if btype == BOUNDARY_INTERNAL:
            dz = bed_slope_qx[j, i]
            h_left = water_depth[j, i - 1]
            h_right = water_depth[j, i]
        elif btype == BOUNDARY_MASKED_WEST_STAGE:
            h_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
            h_right = water_depth[j, i]
        elif btype == BOUNDARY_MASKED_EAST_STAGE:
            h_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, i])
            h_left = water_depth[j, i - 1]
        elif btype == BOUNDARY_MASKED_WEST_FREE:
            h_right = water_depth[j, i]
        elif btype == BOUNDARY_MASKED_EAST_FREE:
            h_left = water_depth[j, i - 1]
        elif i == 0:
            h_right = water_depth[j, 0]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                h_left = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, 0])
        else:
            h_left = water_depth[j, nx - 1]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                h_right = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qx[j, nx])
        hf = calc_conditioned_face_h(h_left, h_right, dz)
        if hf > dry_depth:
            celerity = wp.sqrt(gravity * hf)
            if celerity > WP_FLOAT(0.0):
                dt = wp.min(dt_cap, cfl_number * dx / celerity)
                wp.atomic_min(global_dt_min, 0, dt)
    face_dt[j, i] = dt


@wp.kernel
def lts_conditioned_face_dt_y_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    cfl_number: WP_FLOAT,
    dt_cap: WP_FLOAT,
    face_dt: wp.array2d(dtype=WP_FLOAT),
    global_dt_min: wp.array(dtype=WP_FLOAT),
):
    j, i = wp.tid()
    ny = water_depth.shape[0]
    btype = boundary_type_qy[j, i]
    dt = dt_cap
    if btype != BOUNDARY_CLOSED:
        dz = WP_FLOAT(0.0)
        h_up = WP_FLOAT(0.0)
        h_down = WP_FLOAT(0.0)
        if btype == BOUNDARY_INTERNAL:
            dz = bed_slope_qy[j, i]
            h_up = water_depth[j - 1, i]
            h_down = water_depth[j, i]
        elif btype == BOUNDARY_MASKED_NORTH_STAGE:
            h_up = water_depth[j - 1, i]
            h_down = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[j, i])
        elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
            h_down = water_depth[j, i]
            h_up = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[j, i])
        elif btype == BOUNDARY_MASKED_SOUTH_FREE:
            h_down = water_depth[j, i]
        elif btype == BOUNDARY_MASKED_NORTH_FREE:
            h_up = water_depth[j - 1, i]
        elif j == 0:
            h_down = water_depth[0, i]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                h_up = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[0, i])
        else:
            h_up = water_depth[ny - 1, i]
            if btype == BOUNDARY_SPECIFIED_STAGE:
                h_down = wp.max(WP_FLOAT(0.0), boundary_stage_depth_qy[ny, i])
        hf = calc_conditioned_face_h(h_up, h_down, dz)
        if hf > dry_depth:
            celerity = wp.sqrt(gravity * hf)
            if celerity > WP_FLOAT(0.0):
                dt = wp.min(dt_cap, cfl_number * dy / celerity)
                wp.atomic_min(global_dt_min, 0, dt)
    face_dt[j, i] = dt


@wp.kernel
def lts_face_dt_x_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    nx = water_depth.shape[1]
    btype = boundary_type_qx[j, i]
    dt = dt_cap
    if btype != BOUNDARY_CLOSED:
        z_left = BED_FLOAT(0.0)
        z_right = BED_FLOAT(0.0)
        h_left = WP_FLOAT(0.0)
        h_right = WP_FLOAT(0.0)
        eta_left = BED_FLOAT(0.0)
        eta_right = BED_FLOAT(0.0)
        if btype == BOUNDARY_INTERNAL:
            z_left = bed_elevation[j, i - 1]
            z_right = bed_elevation[j, i]
            h_left = water_depth[j, i - 1]
            h_right = water_depth[j, i]
            eta_left = z_left + BED_FLOAT(h_left)
            eta_right = z_right + BED_FLOAT(h_right)
        elif btype == BOUNDARY_MASKED_WEST_STAGE:
            z_right = bed_elevation[j, i]
            z_left = z_right
            h_right = water_depth[j, i]
            eta_right = z_right + BED_FLOAT(h_right)
            eta_left = wp.max(boundary_stage_qx[j, i], z_left)
            h_left = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_left - z_left))
        elif btype == BOUNDARY_MASKED_EAST_STAGE:
            z_right = bed_elevation[j, i - 1]
            z_left = z_right
            h_left = water_depth[j, i - 1]
            eta_right = z_right + BED_FLOAT(h_left)
            eta_left = wp.max(boundary_stage_qx[j, i], z_left)
            h_right = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_left - z_left))
        elif btype == BOUNDARY_MASKED_WEST_FREE:
            z_right = bed_elevation[j, i]
            z_left = z_right
            h_right = water_depth[j, i]
            eta_right = z_right + BED_FLOAT(h_right)
            eta_left = z_left
        elif btype == BOUNDARY_MASKED_EAST_FREE:
            z_left = bed_elevation[j, i - 1]
            z_right = z_left
            h_left = water_depth[j, i - 1]
            eta_left = z_left + BED_FLOAT(h_left)
            eta_right = z_right
        elif i == 0:
            z_right = bed_elevation[j, 0]
            z_left = z_right
            h_right = water_depth[j, 0]
            eta_right = z_right + BED_FLOAT(h_right)
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_left = wp.max(boundary_stage_qx[j, 0], z_left)
                h_left = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_left - z_left))
            else:
                eta_left = z_left
        else:
            z_left = bed_elevation[j, nx - 1]
            z_right = z_left
            h_left = water_depth[j, nx - 1]
            eta_left = z_left + BED_FLOAT(h_left)
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_right = wp.max(boundary_stage_qx[j, nx], z_right)
                h_right = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_right - z_right))
            else:
                eta_right = z_right
        hf = calc_face_h(eta_left, eta_right, z_left, z_right)
        if hf > BED_FLOAT(dry_depth):
            celerity = wp.sqrt(BED_FLOAT(gravity) * hf)
            if celerity > BED_FLOAT(0.0):
                dt = wp.min(dt_cap, WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dx) / celerity))
                wp.atomic_min(global_dt_min, 0, dt)
    face_dt[j, i] = dt


@wp.kernel
def lts_face_dt_y_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    ny = water_depth.shape[0]
    btype = boundary_type_qy[j, i]
    dt = dt_cap
    if btype != BOUNDARY_CLOSED:
        z_up = BED_FLOAT(0.0)
        z_down = BED_FLOAT(0.0)
        h_up = WP_FLOAT(0.0)
        h_down = WP_FLOAT(0.0)
        eta_up = BED_FLOAT(0.0)
        eta_down = BED_FLOAT(0.0)
        if btype == BOUNDARY_INTERNAL:
            z_up = bed_elevation[j - 1, i]
            z_down = bed_elevation[j, i]
            h_up = water_depth[j - 1, i]
            h_down = water_depth[j, i]
            eta_up = z_up + BED_FLOAT(h_up)
            eta_down = z_down + BED_FLOAT(h_down)
        elif btype == BOUNDARY_MASKED_NORTH_STAGE:
            z_up = bed_elevation[j - 1, i]
            z_down = z_up
            h_up = water_depth[j - 1, i]
            eta_up = z_up + BED_FLOAT(h_up)
            eta_down = wp.max(boundary_stage_qy[j, i], z_down)
            h_down = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_down - z_down))
        elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
            z_up = bed_elevation[j, i]
            z_down = z_up
            h_down = water_depth[j, i]
            eta_up = z_up + BED_FLOAT(h_down)
            eta_down = wp.max(boundary_stage_qy[j, i], z_down)
            h_up = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_down - z_down))
        elif btype == BOUNDARY_MASKED_SOUTH_FREE:
            z_down = bed_elevation[j, i]
            z_up = z_down
            h_down = water_depth[j, i]
            eta_down = z_down + BED_FLOAT(h_down)
            eta_up = z_up
        elif btype == BOUNDARY_MASKED_NORTH_FREE:
            z_up = bed_elevation[j - 1, i]
            z_down = z_up
            h_up = water_depth[j - 1, i]
            eta_up = z_up + BED_FLOAT(h_up)
            eta_down = z_down
        elif j == 0:
            z_down = bed_elevation[0, i]
            z_up = z_down
            h_down = water_depth[0, i]
            eta_down = z_down + BED_FLOAT(h_down)
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_up = wp.max(boundary_stage_qy[0, i], z_up)
                h_up = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_up - z_up))
            else:
                eta_up = z_up
        else:
            z_up = bed_elevation[ny - 1, i]
            z_down = z_up
            h_up = water_depth[ny - 1, i]
            eta_up = z_up + BED_FLOAT(h_up)
            if btype == BOUNDARY_SPECIFIED_STAGE:
                eta_down = wp.max(boundary_stage_qy[ny, i], z_down)
                h_down = WP_FLOAT(wp.max(BED_FLOAT(0.0), eta_down - z_down))
            else:
                eta_down = z_down
        hf = calc_face_h(eta_up, eta_down, z_up, z_down)
        if hf > BED_FLOAT(dry_depth):
            celerity = wp.sqrt(BED_FLOAT(gravity) * hf)
            if celerity > BED_FLOAT(0.0):
                dt = wp.min(dt_cap, WP_FLOAT(BED_FLOAT(cfl_number) * BED_FLOAT(dy) / celerity))
                wp.atomic_min(global_dt_min, 0, dt)
    face_dt[j, i] = dt


@wp.kernel
def lts_cell_level_kernel(
    qx_dt: wp.array2d(dtype=WP_FLOAT),
    qy_dt: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    base_dt: WP_FLOAT,
    max_level: int,
    levels: wp.array2d(dtype=wp.int32),
):
    j, i = wp.tid()
    if active_mask[j, i] != 1:
        levels[j, i] = -1
        return
    local_dt = wp.min(
        wp.min(qx_dt[j, i], qx_dt[j, i + 1]),
        wp.min(qy_dt[j, i], qy_dt[j + 1, i]),
    )
    level = int(0)
    trial_dt = base_dt * WP_FLOAT(2.0)
    while level < max_level and trial_dt <= local_dt:
        level = level + 1
        trial_dt = trial_dt * WP_FLOAT(2.0)
    levels[j, i] = level


@wp.kernel
def lts_classify_cell_levels_kernel(
    qx_dt: wp.array2d(dtype=WP_FLOAT),
    qy_dt: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    predicted_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    previous_levels: wp.array2d(dtype=wp.int32),
    previous_effective_depth: wp.array2d(dtype=WP_FLOAT),
    control: wp.array(dtype=wp.float64),
    sample_interval: WP_FLOAT,
    accuracy_fraction: WP_FLOAT,
    depth_scale: WP_FLOAT,
    promotion_margin: WP_FLOAT,
    guard_depth: WP_FLOAT,
    front_max_level: int,
    drainage_max_level: int,
    cell_area: WP_FLOAT,
    levels: wp.array2d(dtype=wp.int32),
    stats: wp.array(dtype=wp.int64),
):
    """Assign accuracy-controlled power-of-two levels with hysteresis.

    The target level combines the face-CFL limit with a local temporal
    accuracy limit derived from the recent evolution of the effective cell
    storage ``water_depth + pending_flux_volume / cell_area`` — the state the
    LTS scheme actually advances.  With ``dh/dt`` estimated between successive
    schedule rebuilds, the characteristic change time is
    ``tau_h = max(h_eff, depth_scale) / |dh/dt|`` and the accuracy limit is
    ``accuracy_fraction * tau_h``; a non-positive ``sample_interval`` (first
    classification after a reset) disables the accuracy limit.

    Demotion to a finer level is immediate.  Promotion is limited to one level
    per rebuild and additionally requires the combined constraint to permit
    the promoted level's timestep with ``promotion_margin`` headroom, which
    damps level chatter (the default margin of 2 exactly matches the
    fresh-pick quantisation requirement, so only the one-level-per-rebuild cap
    then remains).  Dry cells adjacent to wet or source-predicted-wet
    neighbours are clamped to ``front_max_level`` (negative disables) so a
    moving wetting front cannot be held on a coarse timestep.
    ``guard_depth`` may exceed the profile dry depth so sub-threshold but
    hydraulically active cells near the wet/dry fringe are protected as well.
    Cells holding water above ``guard_depth`` without meaningful positive
    source forcing (``predicted_depth - water_depth`` small) are draining:
    their evolution is purely momentum-driven, and validation showed held
    coarse fluxes then mis-time the recession fringe, so they are clamped to
    ``drainage_max_level`` (negative disables).  Inactive cells carry level -1
    and their history is cleared so stale levels cannot persist.
    """
    j, i = wp.tid()
    if active_mask[j, i] != 1:
        levels[j, i] = -1
        previous_levels[j, i] = -1
        previous_effective_depth[j, i] = WP_FLOAT(0.0)
        return

    base_dt = WP_FLOAT(control[LTS_CTRL_BASE_DT])
    max_level = int(control[LTS_CTRL_MACRO_LEVEL])

    h_eff = water_depth[j, i] + WP_FLOAT(pending_flux_volume[j, i]) / cell_area
    h_eff = wp.max(h_eff, WP_FLOAT(0.0))

    cfl_dt = wp.min(
        wp.min(qx_dt[j, i], qx_dt[j, i + 1]),
        wp.min(qy_dt[j, i], qy_dt[j + 1, i]),
    )

    accuracy_dt = WP_FLOAT(3.0e38)
    if sample_interval > WP_FLOAT(0.0):
        dh = h_eff - previous_effective_depth[j, i]
        dh_rate = wp.abs(dh) / sample_interval
        relative_change = wp.abs(dh) / wp.max(h_eff, depth_scale)
        log_change = wp.log(wp.max(relative_change, WP_FLOAT(1.0e-16))) / WP_FLOAT(
            2.302585092994046
        )
        bin_index = int(
            (log_change - WP_FLOAT(LTS_HIST_LOG_MIN)) * WP_FLOAT(LTS_HIST_BINS_PER_DEX)
        )
        bin_index = wp.min(wp.max(bin_index, 0), LTS_STATS_HIST_BINS - 1)
        wp.atomic_add(stats, LTS_STATS_SCALARS + bin_index, wp.int64(1))
        if dh_rate > WP_FLOAT(1.0e-30):
            accuracy_dt = accuracy_fraction * wp.max(h_eff, depth_scale) / dh_rate

    cfl_level = int(0)
    trial_dt = base_dt * WP_FLOAT(2.0)
    while cfl_level < max_level and trial_dt <= cfl_dt:
        cfl_level += 1
        trial_dt = trial_dt * WP_FLOAT(2.0)

    accuracy_level = int(0)
    trial_dt = base_dt * WP_FLOAT(2.0)
    while accuracy_level < max_level and trial_dt <= accuracy_dt:
        accuracy_level += 1
        trial_dt = trial_dt * WP_FLOAT(2.0)

    target_level = wp.min(cfl_level, accuracy_level)
    if target_level < cfl_level:
        wp.atomic_add(stats, LTS_STATS_ACCURACY_LIMITED, wp.int64(1))
    else:
        wp.atomic_add(stats, LTS_STATS_CFL_LIMITED, wp.int64(1))

    # Drainage guard: unforced cells holding water evolve purely by their own
    # momentum, so coarse held fluxes alias the drainage wave.  Source-forced
    # cells are continuously driven and tolerate coarsening.  "Unforced" means
    # the predicted positive source addition over the whole macrostep is below
    # a tenth of the guard depth; "holding water" starts at a twentieth of it
    # so sub-threshold flowing cells are protected as well.
    if (
        drainage_max_level >= 0
        and target_level > drainage_max_level
        and h_eff > WP_FLOAT(5.0e-2) * guard_depth
    ):
        source_add = predicted_depth[j, i] - water_depth[j, i]
        if source_add <= WP_FLOAT(1.0e-1) * guard_depth:
            target_level = wp.min(target_level, drainage_max_level)
            wp.atomic_add(stats, LTS_STATS_DRAINAGE_LIMITED, wp.int64(1))

    # Wetting-front guard: a currently dry cell next to a wet or imminently
    # wet neighbour must stay on a prompt timestep so front arrival is
    # resolved rather than smeared over a coarse interval.
    if (
        front_max_level >= 0
        and h_eff <= guard_depth
        and predicted_depth[j, i] <= guard_depth
        and target_level > front_max_level
    ):
        front_nearby = False
        if i > 0 and active_mask[j, i - 1] == 1:
            h_neighbour = water_depth[j, i - 1] + WP_FLOAT(
                pending_flux_volume[j, i - 1]
            ) / cell_area
            if (
                wp.max(h_neighbour, WP_FLOAT(0.0)) > guard_depth
                or predicted_depth[j, i - 1] > guard_depth
            ):
                front_nearby = True
        if i + 1 < water_depth.shape[1] and active_mask[j, i + 1] == 1:
            h_neighbour = water_depth[j, i + 1] + WP_FLOAT(
                pending_flux_volume[j, i + 1]
            ) / cell_area
            if (
                wp.max(h_neighbour, WP_FLOAT(0.0)) > guard_depth
                or predicted_depth[j, i + 1] > guard_depth
            ):
                front_nearby = True
        if j > 0 and active_mask[j - 1, i] == 1:
            h_neighbour = water_depth[j - 1, i] + WP_FLOAT(
                pending_flux_volume[j - 1, i]
            ) / cell_area
            if (
                wp.max(h_neighbour, WP_FLOAT(0.0)) > guard_depth
                or predicted_depth[j - 1, i] > guard_depth
            ):
                front_nearby = True
        if j + 1 < water_depth.shape[0] and active_mask[j + 1, i] == 1:
            h_neighbour = water_depth[j + 1, i] + WP_FLOAT(
                pending_flux_volume[j + 1, i]
            ) / cell_area
            if (
                wp.max(h_neighbour, WP_FLOAT(0.0)) > guard_depth
                or predicted_depth[j + 1, i] > guard_depth
            ):
                front_nearby = True
        if front_nearby:
            target_level = wp.min(target_level, front_max_level)
            wp.atomic_add(stats, LTS_STATS_FRONT_GUARD_LIMITED, wp.int64(1))

    # Hysteresis against the level this cell actually held at the previous
    # schedule rebuild: immediate demotion, gradual margin-checked promotion.
    previous_level = previous_levels[j, i]
    new_level = target_level
    if previous_level >= 0:
        if target_level < previous_level:
            wp.atomic_add(stats, LTS_STATS_DEMOTIONS, wp.int64(1))
        elif target_level > previous_level:
            promotion_dt = base_dt * WP_FLOAT(2.0)
            k = int(0)
            while k < previous_level:
                promotion_dt = promotion_dt * WP_FLOAT(2.0)
                k += 1
            # promotion_dt == base_dt * 2**(previous_level + 1): the timestep
            # the cell would actually use at the promoted level, demanded with
            # promotion_margin headroom.  The default margin of 2 exactly
            # matches the fresh-pick quantisation requirement, so promotion is
            # then limited only by the one-level-per-rebuild cap; larger
            # margins demand additional headroom.
            if wp.min(cfl_dt, accuracy_dt) >= promotion_margin * promotion_dt:
                new_level = wp.min(target_level, previous_level + 1)
                wp.atomic_add(stats, LTS_STATS_PROMOTIONS, wp.int64(1))
            else:
                new_level = previous_level
                wp.atomic_add(stats, LTS_STATS_PROMOTION_HELD, wp.int64(1))

    levels[j, i] = new_level
    previous_levels[j, i] = new_level
    previous_effective_depth[j, i] = h_eff
    wp.atomic_add(stats, LTS_STATS_ACTIVE, wp.int64(1))


@wp.kernel
def lts_wet_level_guard_kernel(
    levels: wp.array2d(dtype=wp.int32),
    predicted_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    wet_depth: WP_FLOAT,
    wet_max_level: int,
    stats: wp.array(dtype=wp.int64),
):
    """Prevent wet or imminently wet cells from taking very coarse hydraulic steps.

    Large dry domains can safely occupy high LTS levels, but allowing shallow
    active flow to do the same causes routing/amplitude drift on large grids.
    ``predicted_depth`` includes positive forcing over the current cap interval,
    so cells about to wet are guarded before the front arrives.  This is a
    secondary safety guard behind the accuracy-controlled classifier; the
    ``wet_depth`` threshold may be raised above the profile dry depth to also
    protect the sub-threshold flowing fringe.
    """
    j, i = wp.tid()
    if active_mask[j, i] != 1:
        return
    if predicted_depth[j, i] > wet_depth and levels[j, i] > wet_max_level:
        levels[j, i] = wet_max_level
        wp.atomic_add(stats, LTS_STATS_WET_GUARD_LIMITED, wp.int64(1))


@wp.kernel
def lts_smooth_cell_levels_kernel(
    levels_in: wp.array2d(dtype=wp.int32),
    active_mask: wp.array2d(dtype=wp.int32),
    levels_out: wp.array2d(dtype=wp.int32),
):
    """Enforce a one-level maximum jump between active neighbours."""
    j, i = wp.tid()
    ny = levels_in.shape[0]
    nx = levels_in.shape[1]
    if active_mask[j, i] != 1:
        levels_out[j, i] = -1
        return
    level = levels_in[j, i]
    if i > 0 and active_mask[j, i - 1] == 1:
        level = wp.min(level, levels_in[j, i - 1] + 1)
    if i + 1 < nx and active_mask[j, i + 1] == 1:
        level = wp.min(level, levels_in[j, i + 1] + 1)
    if j > 0 and active_mask[j - 1, i] == 1:
        level = wp.min(level, levels_in[j - 1, i] + 1)
    if j + 1 < ny and active_mask[j + 1, i] == 1:
        level = wp.min(level, levels_in[j + 1, i] + 1)
    levels_out[j, i] = level


@wp.kernel
def lts_face_levels_x_kernel(
    cell_levels: wp.array2d(dtype=wp.int32),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    face_levels: wp.array2d(dtype=wp.int32),
):
    j, i = wp.tid()
    nx = cell_levels.shape[1]
    btype = boundary_type_qx[j, i]
    level = -1
    if btype != BOUNDARY_CLOSED:
        if btype == BOUNDARY_INTERNAL:
            left_level = cell_levels[j, i - 1]
            right_level = cell_levels[j, i]
            if left_level >= 0 and right_level >= 0:
                level = wp.min(left_level, right_level)
        elif btype == BOUNDARY_MASKED_WEST_STAGE:
            level = cell_levels[j, i]
        elif btype == BOUNDARY_MASKED_EAST_STAGE:
            level = cell_levels[j, i - 1]
        elif btype == BOUNDARY_MASKED_WEST_FREE:
            level = cell_levels[j, i]
        elif btype == BOUNDARY_MASKED_EAST_FREE:
            level = cell_levels[j, i - 1]
        elif i == 0:
            level = cell_levels[j, 0]
        elif i == nx:
            level = cell_levels[j, nx - 1]
    face_levels[j, i] = level


@wp.kernel
def lts_face_levels_y_kernel(
    cell_levels: wp.array2d(dtype=wp.int32),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    face_levels: wp.array2d(dtype=wp.int32),
):
    j, i = wp.tid()
    ny = cell_levels.shape[0]
    btype = boundary_type_qy[j, i]
    level = -1
    if btype != BOUNDARY_CLOSED:
        if btype == BOUNDARY_INTERNAL:
            up_level = cell_levels[j - 1, i]
            down_level = cell_levels[j, i]
            if up_level >= 0 and down_level >= 0:
                level = wp.min(up_level, down_level)
        elif btype == BOUNDARY_MASKED_NORTH_STAGE:
            level = cell_levels[j - 1, i]
        elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
            level = cell_levels[j, i]
        elif btype == BOUNDARY_MASKED_SOUTH_FREE:
            level = cell_levels[j, i]
        elif btype == BOUNDARY_MASKED_NORTH_FREE:
            level = cell_levels[j - 1, i]
        elif j == 0:
            level = cell_levels[0, i]
        elif j == ny:
            level = cell_levels[ny - 1, i]
    face_levels[j, i] = level


@wp.kernel
def lts_limiter_level_kernel(
    qx_levels: wp.array2d(dtype=wp.int32),
    qy_levels: wp.array2d(dtype=wp.int32),
    active_mask: wp.array2d(dtype=wp.int32),
    limiter_levels: wp.array2d(dtype=wp.int32),
):
    j, i = wp.tid()
    if active_mask[j, i] != 1:
        limiter_levels[j, i] = -1
        return
    level = int(2147483647)
    west = qx_levels[j, i]
    east = qx_levels[j, i + 1]
    north = qy_levels[j, i]
    south = qy_levels[j + 1, i]
    if west >= 0:
        level = wp.min(level, west)
    if east >= 0:
        level = wp.min(level, east)
    if north >= 0:
        level = wp.min(level, north)
    if south >= 0:
        level = wp.min(level, south)
    if level == 2147483647:
        level = -1
    limiter_levels[j, i] = level


@wp.kernel
def lts_count_cell_levels_kernel(
    levels: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    count_offset: int,
):
    j, i = wp.tid()
    level = levels[j, i]
    if level >= 0:
        wp.atomic_add(counts, count_offset + level, 1)


@wp.kernel
def lts_count_face_levels_kernel(
    levels: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    count_offset: int,
):
    j, i = wp.tid()
    level = levels[j, i]
    if level >= 0:
        wp.atomic_add(counts, count_offset + level, 1)


@wp.kernel
def lts_prepare_level_offsets_kernel(
    counts: wp.array(dtype=wp.int32),
    capacity: int,
    cell_write: wp.array(dtype=wp.int32),
    limiter_write: wp.array(dtype=wp.int32),
    qx_write: wp.array(dtype=wp.int32),
    qy_write: wp.array(dtype=wp.int32),
):
    """Build four short level-prefix tables entirely on-device."""
    if wp.tid() != 0:
        return

    running_cell = int(0)
    running_limiter = int(0)
    running_qx = int(0)
    running_qy = int(0)
    level = int(0)
    while level < capacity:
        cell_write[level] = running_cell
        limiter_write[level] = running_limiter
        qx_write[level] = running_qx
        qy_write[level] = running_qy
        running_cell += counts[level]
        running_limiter += counts[capacity + level]
        running_qx += counts[2 * capacity + level]
        running_qy += counts[3 * capacity + level]
        level += 1


@wp.kernel
def lts_schedule_snapshot_kernel(
    control: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    error_flag: wp.array(dtype=wp.int32),
    snapshot: wp.array(dtype=wp.float64),
):
    """Pack schedule scalars, prior-step error state, and counts for one host read."""
    tid = wp.tid()
    if tid < LTS_CTRL_SIZE:
        snapshot[tid] = control[tid]
    elif tid == LTS_CTRL_SIZE:
        snapshot[tid] = wp.float64(error_flag[0])
    else:
        idx = tid - (LTS_CTRL_SIZE + 1)
        if idx < counts.shape[0]:
            snapshot[tid] = wp.float64(counts[idx])


@wp.kernel
def lts_compact_cell_levels_kernel(
    levels: wp.array2d(dtype=wp.int32),
    write_positions: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
):
    j, i = wp.tid()
    level = levels[j, i]
    if level >= 0:
        position = wp.atomic_add(write_positions, level, 1)
        indices[position] = j * levels.shape[1] + i


@wp.kernel
def lts_compact_face_levels_kernel(
    levels: wp.array2d(dtype=wp.int32),
    write_positions: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
):
    j, i = wp.tid()
    level = levels[j, i]
    if level >= 0:
        position = wp.atomic_add(write_positions, level, 1)
        indices[position] = j * levels.shape[1] + i


@wp.kernel
def lts_apply_sources_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    effective_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    rainfall_spatial_multiplier: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_capacity: wp.array2d(dtype=WP_FLOAT),
    infiltration_f0: wp.array2d(dtype=WP_FLOAT),
    infiltration_fd: wp.array2d(dtype=WP_FLOAT),
    infiltration_k: wp.array2d(dtype=WP_FLOAT),
    green_ampt_ks: wp.array2d(dtype=WP_FLOAT),
    green_ampt_psi: wp.array2d(dtype=WP_FLOAT),
    green_ampt_delta_theta: wp.array2d(dtype=WP_FLOAT),
    cumulative_infiltration: wp.array2d(dtype=WP_FLOAT),
    infiltration_method: int,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    active_mask: wp.array2d(dtype=wp.int32),
    dt: WP_FLOAT,
    uniform_rain_rate: WP_FLOAT,
    use_uniform_rain: int,
    source_mode: int,
    infiltration_wet_depth: WP_FLOAT,
    applied_rain: wp.array2d(dtype=wp.float64),
    applied_other: wp.array2d(dtype=wp.float64),
    applied_infil: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    negative_depth_tolerance: WP_FLOAT,
    error_flag: wp.array(dtype=wp.int32),
):
    """Advance sources on the base clock using the conservative pending storage.

    Between local continuity synchronisations, ``water_depth`` is the last
    committed cell state and ``pending_flux_volume`` contains every hydraulic
    and source volume change that has occurred since that commit. Source
    physics therefore uses the effective current storage rather than
    ``water_depth`` alone.

    The source increment is written back to the pending register, not directly
    to ``water_depth``. This prevents a coarse cell from spending the same
    storage once through coarse hydraulic outflow and again through repeated
    base-clock infiltration before continuity consumes the register.
    """
    tid = wp.tid()
    flat = indices[start + tid]
    nx = water_depth.shape[1]
    j = flat // nx
    i = flat - j * nx

    if active_mask[j, i] != 1:
        effective_depth[j, i] = WP_FLOAT(0.0)
        return

    cell_area = wp.float64(dx) * wp.float64(dy)
    committed_volume = wp.float64(water_depth[j, i]) * cell_area
    current_volume = committed_volume + pending_flux_volume[j, i]
    tolerance_volume = wp.float64(negative_depth_tolerance) * cell_area

    if current_volume < wp.float64(0.0):
        if current_volume < -tolerance_volume:
            wp.atomic_add(error_flag, 0, 1)
        current_volume = wp.float64(0.0)
        pending_flux_volume[j, i] = -committed_volume

    h = WP_FLOAT(current_volume / cell_area)

    rain = rainfall_rate[j, i]
    if use_uniform_rain == 1:
        rain = uniform_rain_rate * rainfall_spatial_multiplier[j, i]

    r_rate = wp.max(WP_FLOAT(0.0), rain)
    o_rate = wp.max(WP_FLOAT(0.0), other_source_rate[j, i])
    i_cap = wp.max(WP_FLOAT(0.0), infiltration_capacity[j, i])
    if infiltration_method == 2:
        i_cap = WP_FLOAT(0.0)
    cumulative = WP_FLOAT(0.0)
    if infiltration_method == 1:
        f0 = infiltration_f0[j, i]
        fd = infiltration_fd[j, i]
        k = infiltration_k[j, i]
        fd_safe = wp.max(fd, WP_FLOAT(1.0e-30))
        cumulative = wp.max(cumulative_infiltration[j, i], WP_FLOAT(0.0))
        i_cap = wp.max(WP_FLOAT(0.0), k + (f0 - k) * wp.exp(-cumulative / fd_safe))
    elif infiltration_method == 3:
        cumulative = wp.max(cumulative_infiltration[j, i], WP_FLOAT(0.0))
        i_cap = wp.max(WP_FLOAT(0.0), green_ampt_ks[j, i])

    r_depth = r_rate * dt
    o_depth = o_rate * dt
    i_depth = WP_FLOAT(0.0)

    if infiltration_method == 1 or infiltration_method == 3 or (
        infiltration_method == 0 and subgrid_infiltration_enabled == 1
    ):
        # Exponential and Green-Ampt use cumulative capacities; constant
        # infiltration joins this path in subgrid mode. The shared application
        # preserves the existing source ordering and available-water limit.
        capacity_depth = i_cap * dt
        if infiltration_method == 3:
            capacity_depth = _green_ampt_potential_depth(
                cumulative,
                green_ampt_ks[j, i],
                green_ampt_psi[j, i],
                green_ampt_delta_theta[j, i],
                dt,
            )
            if h <= WP_FLOAT(0.0):
                capacity_depth = _green_ampt_dry_source_depth(
                    cumulative, green_ampt_ks[j, i], green_ampt_psi[j, i],
                    green_ampt_delta_theta[j, i], r_rate + o_rate, dt,
                )
        i_depth = _rain_first_infiltration_depth(
            r_depth,
            h,
            h + o_depth,
            capacity_depth,
            subgrid_infiltration_enabled,
            subgrid_cell_z,
            subgrid_cell_storage_breakpoints,
            j,
            i,
            subgrid_cell_count,
        )
        h_new = h + r_depth + o_depth - i_depth
    elif source_mode == 1:
        if h > infiltration_wet_depth:
            i_depth = wp.min(i_cap * dt, h)
        h_new = h - i_depth + r_depth + o_depth
    else:
        h_after_sources = h + r_depth + o_depth
        i_depth = wp.min(i_cap * dt, h_after_sources)
        h_new = h_after_sources - i_depth

    delta_volume = wp.float64(h_new - h) * cell_area
    pending_flux_volume[j, i] = pending_flux_volume[j, i] + delta_volume
    effective_depth[j, i] = wp.max(WP_FLOAT(0.0), h_new)

    if infiltration_method == 1 or infiltration_method == 3:
        cumulative_infiltration[j, i] = cumulative + i_depth

    applied_rain[j, i] = applied_rain[j, i] + wp.float64(r_depth)
    applied_other[j, i] = applied_other[j, i] + wp.float64(o_depth)
    applied_infil[j, i] = applied_infil[j, i] + wp.float64(i_depth)


@wp.kernel
def lts_update_conditioned_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qx: wp.array2d(dtype=WP_FLOAT),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qx_old.shape[1]
    j = flat // width
    i = flat - j * width
    qx_next[j, i] = _update_conditioned_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_slope_qx, manning_n,
        boundary_type_qx, boundary_stage_depth_qx, boundary_free_slope_qx,
        dx, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
        use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def lts_update_conditioned_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qy: wp.array2d(dtype=WP_FLOAT),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qy_old.shape[1]
    j = flat // width
    i = flat - j * width
    qy_next[j, i] = _update_conditioned_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_slope_qy, manning_n,
        boundary_type_qy, boundary_stage_depth_qy, boundary_free_slope_qy,
        dy, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
        use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def lts_update_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qx_old.shape[1]
    j = flat // width
    i = flat - j * width
    qx_next[j, i] = _update_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_elevation, manning_n,
        boundary_type_qx, boundary_stage_qx, dx, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction,
        free_outflow_mode,
    )


@wp.kernel
def lts_update_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qy_old.shape[1]
    j = flat // width
    i = flat - j * width
    qy_next[j, i] = _update_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_elevation, manning_n,
        boundary_type_qy, boundary_stage_qy, dy, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction,
        free_outflow_mode,
    )


@wp.func
def _lts_face_is_scheduled(level: int, tick: int) -> int:
    if level < 0:
        return 0
    period = 1 << level
    if tick % period == 0:
        return 1
    return 0


@wp.kernel
def lts_limit_outflow_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qx_levels: wp.array2d(dtype=wp.int32),
    qy_levels: wp.array2d(dtype=wp.int32),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    base_dt: WP_FLOAT,
    tick: int,
    epsilon: WP_FLOAT,
):
    """Limit scheduled outflow against the cell's current effective storage."""
    tid = wp.tid()
    flat = indices[start + tid]
    nx = water_depth.shape[1]
    j = flat // nx
    i = flat - j * nx
    if active_mask[j, i] != 1:
        return

    west_level = qx_levels[j, i]
    east_level = qx_levels[j, i + 1]
    north_level = qy_levels[j, i]
    south_level = qy_levels[j + 1, i]

    qx_west = qx_next[j, i]
    qx_east = qx_next[j, i + 1]
    qy_north = qy_next[j, i]
    qy_south = qy_next[j + 1, i]

    requested_out = wp.float64(0.0)
    if _lts_face_is_scheduled(east_level, tick) == 1 and qx_east > WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(qx_east) * wp.float64(dy) * wp.float64(base_dt) * wp.float64(1 << east_level)
    if _lts_face_is_scheduled(west_level, tick) == 1 and qx_west < WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(-qx_west) * wp.float64(dy) * wp.float64(base_dt) * wp.float64(1 << west_level)
    if _lts_face_is_scheduled(south_level, tick) == 1 and qy_south > WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(qy_south) * wp.float64(dx) * wp.float64(base_dt) * wp.float64(1 << south_level)
    if _lts_face_is_scheduled(north_level, tick) == 1 and qy_north < WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(-qy_north) * wp.float64(dx) * wp.float64(base_dt) * wp.float64(1 << north_level)

    cell_area = wp.float64(dx) * wp.float64(dy)
    available = wp.float64(water_depth[j, i]) * cell_area + pending_flux_volume[j, i]
    available = wp.max(wp.float64(0.0), available)

    alpha = WP_FLOAT(1.0)
    if requested_out > wp.float64(0.0):
        alpha = WP_FLOAT(wp.min(wp.float64(1.0), available / (requested_out + wp.float64(epsilon))))

    if alpha < WP_FLOAT(1.0):
        if _lts_face_is_scheduled(east_level, tick) == 1 and qx_east > WP_FLOAT(0.0):
            qx_next[j, i + 1] = qx_east * alpha
        if _lts_face_is_scheduled(west_level, tick) == 1 and qx_west < WP_FLOAT(0.0):
            qx_next[j, i] = qx_west * alpha
        if _lts_face_is_scheduled(south_level, tick) == 1 and qy_south > WP_FLOAT(0.0):
            qy_next[j + 1, i] = qy_south * alpha
        if _lts_face_is_scheduled(north_level, tick) == 1 and qy_north < WP_FLOAT(0.0):
            qy_next[j, i] = qy_north * alpha


@wp.kernel
def lts_commit_qx_volume_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qx: wp.array2d(dtype=WP_FLOAT),
    qx_next: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    dy: WP_FLOAT,
    dt: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qx.shape[1]
    nx = width - 1
    j = flat // width
    i = flat - j * width
    q = qx_next[j, i]
    volume = wp.float64(q) * wp.float64(dy) * wp.float64(dt)
    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_INTERNAL:
        wp.atomic_add(pending_flux_volume, j, i - 1, -volume)
        wp.atomic_add(pending_flux_volume, j, i, volume)
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        wp.atomic_add(pending_flux_volume, j, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        wp.atomic_add(pending_flux_volume, j, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif btype == BOUNDARY_MASKED_EAST_FREE or btype == BOUNDARY_MASKED_EAST_STAGE:
        wp.atomic_add(pending_flux_volume, j, i - 1, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    elif i == 0:
        wp.atomic_add(pending_flux_volume, j, 0, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif i == nx:
        wp.atomic_add(pending_flux_volume, j, nx - 1, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    qx[j, i] = q


@wp.kernel
def lts_commit_qy_volume_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    qy: wp.array2d(dtype=WP_FLOAT),
    qy_next: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    cumulative_totals: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    flat = indices[start + tid]
    width = qy.shape[1]
    ny = qy.shape[0] - 1
    j = flat // width
    i = flat - j * width
    q = qy_next[j, i]
    volume = wp.float64(q) * wp.float64(dx) * wp.float64(dt)
    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_INTERNAL:
        wp.atomic_add(pending_flux_volume, j - 1, i, -volume)
        wp.atomic_add(pending_flux_volume, j, i, volume)
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        wp.atomic_add(pending_flux_volume, j - 1, i, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    elif btype == BOUNDARY_MASKED_SOUTH_FREE or btype == BOUNDARY_MASKED_SOUTH_STAGE:
        wp.atomic_add(pending_flux_volume, j, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        wp.atomic_add(pending_flux_volume, j - 1, i, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    elif j == 0:
        wp.atomic_add(pending_flux_volume, 0, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif j == ny:
        wp.atomic_add(pending_flux_volume, ny - 1, i, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    qy[j, i] = q


@wp.kernel
def lts_continuity_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    start: int,
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    negative_depth_tolerance: WP_FLOAT,
    error_flag: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    flat = indices[start + tid]
    nx = water_depth.shape[1]
    j = flat // nx
    i = flat - j * nx
    if active_mask[j, i] != 1:
        pending_flux_volume[j, i] = wp.float64(0.0)
        water_depth[j, i] = WP_FLOAT(0.0)
        return
    volume = pending_flux_volume[j, i]
    h_new = water_depth[j, i] + WP_FLOAT(
        volume / (wp.float64(dx) * wp.float64(dy))
    )
    if h_new < WP_FLOAT(0.0):
        if h_new >= -negative_depth_tolerance:
            h_new = WP_FLOAT(0.0)
        else:
            wp.atomic_add(error_flag, 0, 1)
    water_depth[j, i] = h_new
    pending_flux_volume[j, i] = wp.float64(0.0)

# ---------------------------------------------------------------------------
# Device-graph LTS subcycling
# ---------------------------------------------------------------------------
# Momentum is evaluated only on each face's local power-of-two cadence, but
# the held discharge transports water on every base tick. This is important:
# committing q * local_dt at the start of a coarse interval is conservative in
# total volume but advances future water too early and produces a scale-
# dependent phase/amplitude error over long routing paths.


@wp.kernel
def lts_graph_init_kernel(
    tick: wp.array(dtype=wp.int32),
    condition: wp.array(dtype=wp.int32),
    forcing: wp.array(dtype=wp.float64),
    macro_ticks: int,
    rain_start: wp.float64,
    rain_slope: wp.float64,
):
    if wp.tid() == 0:
        tick[0] = 0
        condition[0] = macro_ticks
        forcing[0] = rain_start
        forcing[1] = rain_slope


@wp.kernel
def lts_graph_apply_sources_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    effective_depth: wp.array2d(dtype=WP_FLOAT),
    rainfall_rate: wp.array2d(dtype=WP_FLOAT),
    rainfall_spatial_multiplier: wp.array2d(dtype=WP_FLOAT),
    other_source_rate: wp.array2d(dtype=WP_FLOAT),
    infiltration_capacity: wp.array2d(dtype=WP_FLOAT),
    infiltration_f0: wp.array2d(dtype=WP_FLOAT),
    infiltration_fd: wp.array2d(dtype=WP_FLOAT),
    infiltration_k: wp.array2d(dtype=WP_FLOAT),
    green_ampt_ks: wp.array2d(dtype=WP_FLOAT),
    green_ampt_psi: wp.array2d(dtype=WP_FLOAT),
    green_ampt_delta_theta: wp.array2d(dtype=WP_FLOAT),
    cumulative_infiltration: wp.array2d(dtype=WP_FLOAT),
    infiltration_method: int,
    subgrid_infiltration_enabled: int,
    subgrid_cell_z: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    subgrid_cell_count: int,
    active_mask: wp.array2d(dtype=wp.int32),
    schedule_control: wp.array(dtype=wp.float64),
    tick: wp.array(dtype=wp.int32),
    forcing: wp.array(dtype=wp.float64),
    use_uniform_rain: int,
    source_mode: int,
    infiltration_wet_depth: WP_FLOAT,
    applied_rain: wp.array2d(dtype=wp.float64),
    applied_other: wp.array2d(dtype=wp.float64),
    applied_infil: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    negative_depth_tolerance: WP_FLOAT,
    error_flag: wp.array(dtype=wp.int32),
):
    """Base-clock source update used inside the CUDA graph LTS loop.

    A schedule macrostep is capped at rainfall-hyetograph breakpoints, so the
    uniform forcing within one macrostep is one linear segment. ``forcing``
    stores the segment start rate and slope; midpoint sampling therefore gives
    the exact mean rate over each base interval.
    """
    tid = wp.tid()
    flat = indices[tid]
    nx = water_depth.shape[1]
    j = flat // nx
    i = flat - j * nx

    if active_mask[j, i] != 1:
        effective_depth[j, i] = WP_FLOAT(0.0)
        return

    cell_area = wp.float64(dx) * wp.float64(dy)
    committed_volume = wp.float64(water_depth[j, i]) * cell_area
    current_volume = committed_volume + pending_flux_volume[j, i]
    tolerance_volume = wp.float64(negative_depth_tolerance) * cell_area

    if current_volume < wp.float64(0.0):
        if current_volume < -tolerance_volume:
            wp.atomic_add(error_flag, 0, 1)
        current_volume = wp.float64(0.0)
        pending_flux_volume[j, i] = -committed_volume

    h = WP_FLOAT(current_volume / cell_area)
    dt = WP_FLOAT(schedule_control[LTS_CTRL_BASE_DT])

    rain = rainfall_rate[j, i]
    if use_uniform_rain == 1:
        tick_value = wp.float64(tick[0])
        elapsed_mid = (tick_value + wp.float64(0.5)) * wp.float64(dt)
        rain = WP_FLOAT(forcing[0] + forcing[1] * elapsed_mid) * rainfall_spatial_multiplier[j, i]

    r_rate = wp.max(WP_FLOAT(0.0), rain)
    o_rate = wp.max(WP_FLOAT(0.0), other_source_rate[j, i])
    i_cap = wp.max(WP_FLOAT(0.0), infiltration_capacity[j, i])
    if infiltration_method == 2:
        i_cap = WP_FLOAT(0.0)
    cumulative = WP_FLOAT(0.0)
    if infiltration_method == 1:
        f0 = infiltration_f0[j, i]
        fd = infiltration_fd[j, i]
        k = infiltration_k[j, i]
        fd_safe = wp.max(fd, WP_FLOAT(1.0e-30))
        cumulative = wp.max(cumulative_infiltration[j, i], WP_FLOAT(0.0))
        i_cap = wp.max(WP_FLOAT(0.0), k + (f0 - k) * wp.exp(-cumulative / fd_safe))
    elif infiltration_method == 3:
        cumulative = wp.max(cumulative_infiltration[j, i], WP_FLOAT(0.0))
        i_cap = wp.max(WP_FLOAT(0.0), green_ampt_ks[j, i])

    r_depth = r_rate * dt
    o_depth = o_rate * dt
    i_depth = WP_FLOAT(0.0)

    if infiltration_method == 1 or infiltration_method == 3 or (
        infiltration_method == 0 and subgrid_infiltration_enabled == 1
    ):
        # Exponential and Green-Ampt use cumulative capacities; constant
        # infiltration joins this path in subgrid mode. The shared application
        # preserves the existing source ordering and available-water limit.
        capacity_depth = i_cap * dt
        if infiltration_method == 3:
            capacity_depth = _green_ampt_potential_depth(
                cumulative,
                green_ampt_ks[j, i],
                green_ampt_psi[j, i],
                green_ampt_delta_theta[j, i],
                dt,
            )
            if h <= WP_FLOAT(0.0):
                capacity_depth = _green_ampt_dry_source_depth(
                    cumulative, green_ampt_ks[j, i], green_ampt_psi[j, i],
                    green_ampt_delta_theta[j, i], r_rate + o_rate, dt,
                )
        i_depth = _rain_first_infiltration_depth(
            r_depth,
            h,
            h + o_depth,
            capacity_depth,
            subgrid_infiltration_enabled,
            subgrid_cell_z,
            subgrid_cell_storage_breakpoints,
            j,
            i,
            subgrid_cell_count,
        )
        h_new = h + r_depth + o_depth - i_depth
    elif source_mode == 1:
        if h > infiltration_wet_depth:
            i_depth = wp.min(i_cap * dt, h)
        h_new = h - i_depth + r_depth + o_depth
    else:
        h_after_sources = h + r_depth + o_depth
        i_depth = wp.min(i_cap * dt, h_after_sources)
        h_new = h_after_sources - i_depth

    delta_volume = wp.float64(h_new - h) * cell_area
    pending_flux_volume[j, i] = pending_flux_volume[j, i] + delta_volume
    effective_depth[j, i] = wp.max(WP_FLOAT(0.0), h_new)

    if infiltration_method == 1 or infiltration_method == 3:
        cumulative_infiltration[j, i] = cumulative + i_depth

    applied_rain[j, i] = applied_rain[j, i] + wp.float64(r_depth)
    applied_other[j, i] = applied_other[j, i] + wp.float64(o_depth)
    applied_infil[j, i] = applied_infil[j, i] + wp.float64(i_depth)


@wp.kernel
def lts_graph_update_conditioned_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qx: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qx: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qx: wp.array2d(dtype=WP_FLOAT),
    face_levels: wp.array2d(dtype=wp.int32),
    schedule_control: wp.array(dtype=wp.float64),
    tick: wp.array(dtype=wp.int32),
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
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
    qx_next[j, i] = _update_conditioned_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_slope_qx, manning_n,
        boundary_type_qx, boundary_stage_depth_qx, boundary_free_slope_qx,
        dx, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
        use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def lts_graph_update_conditioned_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_slope_qy: wp.array2d(dtype=WP_FLOAT),
    manning_n: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    boundary_stage_depth_qy: wp.array2d(dtype=WP_FLOAT),
    boundary_free_slope_qy: wp.array2d(dtype=WP_FLOAT),
    face_levels: wp.array2d(dtype=wp.int32),
    schedule_control: wp.array(dtype=wp.float64),
    tick: wp.array(dtype=wp.int32),
    dy: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    depth_epsilon: WP_FLOAT,
    manning_face_mode: int,
    use_two_dimensional_friction: int,
    free_outflow_mode: int,
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
    qy_next[j, i] = _update_conditioned_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_slope_qy, manning_n,
        boundary_type_qy, boundary_stage_depth_qy, boundary_free_slope_qy,
        dy, dt, gravity, dry_depth, depth_epsilon, manning_face_mode,
        use_two_dimensional_friction, free_outflow_mode,
    )


@wp.kernel
def lts_graph_update_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qx_next: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
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
    qx_next[j, i] = _update_qx_face(
        i, j, qx_old, qy_old, water_depth, bed_elevation, manning_n,
        boundary_type_qx, boundary_stage_qx, dx, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction,
        free_outflow_mode,
    )


@wp.kernel
def lts_graph_update_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qy_next: wp.array2d(dtype=WP_FLOAT),
    qy_old: wp.array2d(dtype=WP_FLOAT),
    qx_old: wp.array2d(dtype=WP_FLOAT),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    bed_elevation: wp.array2d(dtype=BED_FLOAT),
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
    free_outflow_mode: int,
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
    qy_next[j, i] = _update_qy_face(
        i, j, qy_old, qx_old, water_depth, bed_elevation, manning_n,
        boundary_type_qy, boundary_stage_qy, dy, dt, gravity, dry_depth,
        depth_epsilon, manning_face_mode, use_two_dimensional_friction,
        free_outflow_mode,
    )


@wp.kernel
def lts_graph_accept_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qx: wp.array2d(dtype=WP_FLOAT),
    qx_next: wp.array2d(dtype=WP_FLOAT),
    face_levels: wp.array2d(dtype=wp.int32),
    tick: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    flat = indices[tid]
    width = qx.shape[1]
    j = flat // width
    i = flat - j * width
    if _lts_face_is_scheduled(face_levels[j, i], tick[0]) == 1:
        qx[j, i] = qx_next[j, i]


@wp.kernel
def lts_graph_accept_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qy: wp.array2d(dtype=WP_FLOAT),
    qy_next: wp.array2d(dtype=WP_FLOAT),
    face_levels: wp.array2d(dtype=wp.int32),
    tick: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    flat = indices[tid]
    width = qy.shape[1]
    j = flat // width
    i = flat - j * width
    if _lts_face_is_scheduled(face_levels[j, i], tick[0]) == 1:
        qy[j, i] = qy_next[j, i]


@wp.kernel
def lts_limit_held_outflow_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    schedule_control: wp.array(dtype=wp.float64),
    epsilon: WP_FLOAT,
):
    """Limit the held discharge volume for one base-clock transport tick."""
    tid = wp.tid()
    flat = indices[tid]
    nx = water_depth.shape[1]
    j = flat // nx
    i = flat - j * nx
    if active_mask[j, i] != 1:
        return

    qx_west = qx[j, i]
    qx_east = qx[j, i + 1]
    qy_north = qy[j, i]
    qy_south = qy[j + 1, i]
    dt = wp.float64(schedule_control[LTS_CTRL_BASE_DT])

    requested_out = wp.float64(0.0)
    if qx_east > WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(qx_east) * wp.float64(dy) * dt
    if qx_west < WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(-qx_west) * wp.float64(dy) * dt
    if qy_south > WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(qy_south) * wp.float64(dx) * dt
    if qy_north < WP_FLOAT(0.0):
        requested_out = requested_out + wp.float64(-qy_north) * wp.float64(dx) * dt

    cell_area = wp.float64(dx) * wp.float64(dy)
    available = wp.float64(water_depth[j, i]) * cell_area + pending_flux_volume[j, i]
    available = wp.max(wp.float64(0.0), available)

    alpha = WP_FLOAT(1.0)
    if requested_out > wp.float64(0.0):
        alpha = WP_FLOAT(
            wp.min(wp.float64(1.0), available / (requested_out + wp.float64(epsilon)))
        )

    if alpha < WP_FLOAT(1.0):
        if qx_east > WP_FLOAT(0.0):
            qx[j, i + 1] = qx_east * alpha
        if qx_west < WP_FLOAT(0.0):
            qx[j, i] = qx_west * alpha
        if qy_south > WP_FLOAT(0.0):
            qy[j + 1, i] = qy_south * alpha
        if qy_north < WP_FLOAT(0.0):
            qy[j, i] = qy_north * alpha


@wp.kernel
def lts_transport_qx_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qx: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qx: wp.array2d(dtype=wp.int32),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    dy: WP_FLOAT,
    schedule_control: wp.array(dtype=wp.float64),
    cumulative_totals: wp.array(dtype=wp.float64),
):
    """Transport held x-face discharge over one base interval."""
    tid = wp.tid()
    flat = indices[tid]
    width = qx.shape[1]
    nx = width - 1
    j = flat // width
    i = flat - j * width
    volume = wp.float64(qx[j, i]) * wp.float64(dy) * schedule_control[LTS_CTRL_BASE_DT]
    btype = boundary_type_qx[j, i]
    if btype == BOUNDARY_INTERNAL:
        wp.atomic_add(pending_flux_volume, j, i - 1, -volume)
        wp.atomic_add(pending_flux_volume, j, i, volume)
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        wp.atomic_add(pending_flux_volume, j, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        wp.atomic_add(pending_flux_volume, j, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif btype == BOUNDARY_MASKED_EAST_FREE or btype == BOUNDARY_MASKED_EAST_STAGE:
        wp.atomic_add(pending_flux_volume, j, i - 1, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    elif i == 0:
        wp.atomic_add(pending_flux_volume, j, 0, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif i == nx:
        wp.atomic_add(pending_flux_volume, j, nx - 1, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)


@wp.kernel
def lts_transport_qy_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    qy: wp.array2d(dtype=WP_FLOAT),
    boundary_type_qy: wp.array2d(dtype=wp.int32),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    dx: WP_FLOAT,
    schedule_control: wp.array(dtype=wp.float64),
    cumulative_totals: wp.array(dtype=wp.float64),
):
    """Transport held y-face discharge over one base interval."""
    tid = wp.tid()
    flat = indices[tid]
    width = qy.shape[1]
    ny = qy.shape[0] - 1
    j = flat // width
    i = flat - j * width
    volume = wp.float64(qy[j, i]) * wp.float64(dx) * schedule_control[LTS_CTRL_BASE_DT]
    btype = boundary_type_qy[j, i]
    if btype == BOUNDARY_INTERNAL:
        wp.atomic_add(pending_flux_volume, j - 1, i, -volume)
        wp.atomic_add(pending_flux_volume, j, i, volume)
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        wp.atomic_add(pending_flux_volume, j - 1, i, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    elif btype == BOUNDARY_MASKED_SOUTH_FREE or btype == BOUNDARY_MASKED_SOUTH_STAGE:
        wp.atomic_add(pending_flux_volume, j, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        wp.atomic_add(pending_flux_volume, j - 1, i, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)
    elif j == 0:
        wp.atomic_add(pending_flux_volume, 0, i, volume)
        if volume >= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, volume)
        else:
            wp.atomic_add(cumulative_totals, 4, -volume)
    elif j == ny:
        wp.atomic_add(pending_flux_volume, ny - 1, i, -volume)
        if volume <= wp.float64(0.0):
            wp.atomic_add(cumulative_totals, 3, -volume)
        else:
            wp.atomic_add(cumulative_totals, 4, volume)


@wp.kernel
def lts_graph_continuity_indexed_kernel(
    indices: wp.array(dtype=wp.int32),
    water_depth: wp.array2d(dtype=WP_FLOAT),
    pending_flux_volume: wp.array2d(dtype=wp.float64),
    cell_levels: wp.array2d(dtype=wp.int32),
    active_mask: wp.array2d(dtype=wp.int32),
    tick: wp.array(dtype=wp.int32),
    dx: WP_FLOAT,
    dy: WP_FLOAT,
    negative_depth_tolerance: WP_FLOAT,
    error_flag: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    flat = indices[tid]
    nx = water_depth.shape[1]
    j = flat // nx
    i = flat - j * nx
    if active_mask[j, i] != 1:
        pending_flux_volume[j, i] = wp.float64(0.0)
        water_depth[j, i] = WP_FLOAT(0.0)
        return

    level = cell_levels[j, i]
    period = 1 << level
    end_tick = tick[0] + 1
    if end_tick % period != 0:
        return

    volume = pending_flux_volume[j, i]
    h_new = water_depth[j, i] + WP_FLOAT(
        volume / (wp.float64(dx) * wp.float64(dy))
    )
    if h_new < WP_FLOAT(0.0):
        if h_new >= -negative_depth_tolerance:
            h_new = WP_FLOAT(0.0)
        else:
            wp.atomic_add(error_flag, 0, 1)
    water_depth[j, i] = h_new
    pending_flux_volume[j, i] = wp.float64(0.0)


@wp.kernel
def lts_graph_advance_tick_kernel(
    tick: wp.array(dtype=wp.int32),
    condition: wp.array(dtype=wp.int32),
):
    if wp.tid() == 0:
        tick[0] = tick[0] + 1
        condition[0] = condition[0] - 1


@wp.kernel
def accumulate_face_volume_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    cumulative_net_volume: wp.array2d(dtype=wp.float64),
    cumulative_absolute_volume: wp.array2d(dtype=wp.float64),
    peak_rows: wp.array(dtype=wp.float64),
    face_width: WP_FLOAT,
    dt: WP_FLOAT,
    track_absolute: int,
):
    """Accumulate accepted x-face transport for an opt-in diagnostic."""
    j, i = wp.tid()
    volume = wp.float64(qx[j, i]) * wp.float64(face_width) * wp.float64(dt)
    cumulative_net_volume[j, i] = cumulative_net_volume[j, i] + volume
    if track_absolute == 1:
        cumulative_absolute_volume[j, i] = cumulative_absolute_volume[j, i] + wp.abs(volume)
    wp.atomic_max(peak_rows, j, wp.abs(wp.float64(qx[j, i])))


@wp.kernel
def accumulate_face_volume_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    cumulative_net_volume: wp.array2d(dtype=wp.float64),
    cumulative_absolute_volume: wp.array2d(dtype=wp.float64),
    peak_rows: wp.array(dtype=wp.float64),
    face_width: WP_FLOAT,
    dt: WP_FLOAT,
    track_absolute: int,
):
    """Accumulate accepted y-face transport for an opt-in diagnostic."""
    j, i = wp.tid()
    volume = wp.float64(qy[j, i]) * wp.float64(face_width) * wp.float64(dt)
    cumulative_net_volume[j, i] = cumulative_net_volume[j, i] + volume
    if track_absolute == 1:
        cumulative_absolute_volume[j, i] = cumulative_absolute_volume[j, i] + wp.abs(volume)
    wp.atomic_max(peak_rows, j, wp.abs(wp.float64(qy[j, i])))


@wp.kernel
def device_loop_accumulate_face_volume_qx_kernel(
    qx: wp.array2d(dtype=WP_FLOAT),
    cumulative_net_volume: wp.array2d(dtype=wp.float64),
    cumulative_absolute_volume: wp.array2d(dtype=wp.float64),
    peak_rows: wp.array(dtype=wp.float64),
    face_width: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    track_absolute: int,
):
    """Accumulate accepted x-face transport using the device-loop timestep."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    volume = wp.float64(qx[j, i]) * wp.float64(face_width) * wp.float64(control[DL_DT])
    cumulative_net_volume[j, i] = cumulative_net_volume[j, i] + volume
    if track_absolute == 1:
        cumulative_absolute_volume[j, i] = cumulative_absolute_volume[j, i] + wp.abs(volume)
    wp.atomic_max(peak_rows, j, wp.abs(wp.float64(qx[j, i])))


@wp.kernel
def device_loop_accumulate_face_volume_qy_kernel(
    qy: wp.array2d(dtype=WP_FLOAT),
    cumulative_net_volume: wp.array2d(dtype=wp.float64),
    cumulative_absolute_volume: wp.array2d(dtype=wp.float64),
    peak_rows: wp.array(dtype=wp.float64),
    face_width: WP_FLOAT,
    control: wp.array(dtype=wp.float64),
    track_absolute: int,
):
    """Accumulate accepted y-face transport using the device-loop timestep."""
    j, i = wp.tid()
    if control[DL_DONE] != 0.0:
        return
    volume = wp.float64(qy[j, i]) * wp.float64(face_width) * wp.float64(control[DL_DT])
    cumulative_net_volume[j, i] = cumulative_net_volume[j, i] + volume
    if track_absolute == 1:
        cumulative_absolute_volume[j, i] = cumulative_absolute_volume[j, i] + wp.abs(volume)
    wp.atomic_max(peak_rows, j, wp.abs(wp.float64(qy[j, i])))


@wp.kernel
def reduce_hydraulic_diagnostic_state_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),
    qy: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    cell_area: wp.float64,
    peak_qx_rows: wp.array(dtype=wp.float64),
    peak_qy_rows: wp.array(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
):
    """Reduce current activity/storage and copy the running diagnostic peak.

    ``output`` is ``[max_abs_q, active_storage_m3, running_peak_q]``. It is
    intentionally a three-scalar device array so recession checks never copy
    hydraulic rasters to the host.
    """
    j, i = wp.tid()
    if j < water_depth.shape[0] and i < water_depth.shape[1]:
        if active_mask[j, i] == 1:
            wp.atomic_add(output, 1, wp.float64(water_depth[j, i]) * cell_area)
    if j < qx.shape[0] and i < qx.shape[1]:
        wp.atomic_max(output, 0, wp.abs(wp.float64(qx[j, i])))
    if j < qy.shape[0] and i < qy.shape[1]:
        wp.atomic_max(output, 0, wp.abs(wp.float64(qy[j, i])))
    if i == 0:
        if j < peak_qx_rows.shape[0]:
            wp.atomic_max(output, 2, peak_qx_rows[j])
        if j < peak_qy_rows.shape[0]:
            wp.atomic_max(output, 2, peak_qy_rows[j])

@wp.kernel
def accumulate_max_active_depth_kernel(
    water_depth: wp.array2d(dtype=WP_FLOAT),
    active_mask: wp.array2d(dtype=wp.int32),
    maximum_depth: wp.array2d(dtype=WP_FLOAT),
):
    """Track per-cell maximum storage depth without host transfers."""
    j, i = wp.tid()
    if active_mask[j, i] == 1:
        maximum_depth[j, i] = wp.max(maximum_depth[j, i], water_depth[j, i])


@wp.kernel
def device_loop_graph_condition_kernel(
    control: wp.array(dtype=wp.float64),
    condition: wp.array(dtype=wp.int32),
    end_time: wp.float64,
    progress_budget: wp.array(dtype=wp.float64),
):
    """Pause for optional reporting without changing hydraulic time boundaries."""
    condition[0] = int(control[DL_DONE] == wp.float64(0.0)
        and control[DL_ERROR] == wp.float64(0.0) and control[DL_TIME] < end_time
        and (progress_budget[0] <= wp.float64(0.0)
            or control[DL_COUNTER] < progress_budget[0]))
