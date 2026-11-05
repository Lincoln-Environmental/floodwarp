"""LISFLOOD-FP FREE boundary arithmetic oracle, strictly for validation.

Re-expresses the FREE1 acceleration boundary calculation from LISFLOOD-FP
8.3 source/boundary.cpp, lines 98-125, with explicit floating-point ordering.
LISFLOOD-FP 8.3 is distributed under GNU GPLv3:
https://github.com/ihaseeburrehman/LISFLOOD-FP-8.3

This independent NumPy test oracle is not imported by production FloodWarp.
It is not a FloodWarp hydraulic operator. The float_surface_only option
is a historical hybrid sensitivity control, not exact LISFLOOD-FP arithmetic.
"""

import numpy as np


def free_boundary_unit_discharge(
    *,
    q_old: float,
    edge_depth: float,
    edge_bed: float,
    inward_depth: float,
    inward_bed: float,
    manning_n: float,
    spacing: float,
    dt: float,
    gravity: float = 9.80665,
    direction: float = 1.0,
    arithmetic: str = "float32",
) -> np.floating:
    """Compute a reference FREE1 unit discharge.

    :param q_old: Previous signed discharge per unit width (m2/s).
    :param edge_depth: Depth at the outlet cell (m), positive.
    :param edge_bed: Outlet bed elevation (m).
    :param inward_depth: Depth at the next inward cell (m).
    :param inward_bed: Inward cell bed elevation (m).
    :param manning_n: Manning roughness.
    :param spacing: Cell spacing normal to outlet (m), positive.
    :param dt: Timestep (s).
    :param gravity: Gravitational acceleration (m/s2).
    :param direction: Outward orientation, +1 or -1.
    :param arithmetic: float32, float64 or float_surface_only (hybrid).
    """
    if arithmetic not in ("float32", "float64", "float_surface_only"):
        raise ValueError("unknown reference arithmetic mode")
    if edge_depth <= 0.0 or spacing <= 0.0:
        raise ValueError("reference equation requires positive depth and spacing")
    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")

    if arithmetic == "float64":
        z0 = np.float64(edge_bed)
        h0 = np.float64(edge_depth)
        z1 = np.float64(inward_bed)
        h1 = np.float64(inward_depth)
        dh = np.float64(z0 + h0)
        dh = np.float64(dh - z1)
        dh = np.float64(dh - h1)
        sf = np.float64(-dh / np.float64(spacing))
    else:
        z0 = np.float32(edge_bed)
        h0 = np.float32(edge_depth)
        z1 = np.float32(inward_bed)
        h1 = np.float32(inward_depth)
        dh = np.float32(z0 + h0)
        dh = np.float32(dh - z1)
        dh = np.float32(dh - h1)
        sf = np.float32(-dh / np.float32(spacing))

    if arithmetic == "float32":
        q0 = np.float32(q_old)
        fn = np.float32(manning_n)
        g = np.float32(gravity)
        timestep = np.float32(dt)
        gravity_increment = np.float32(g * timestep)
        gravity_increment = np.float32(gravity_increment * h0)
        gravity_increment = np.float32(gravity_increment * sf)
        numerator = np.float32(np.abs(q0) + np.abs(gravity_increment))
        h_power = np.float32(h0 ** np.float32(np.float32(10.0) / np.float32(3.0)))
        friction = np.float32(g * timestep)
        friction = np.float32(friction * h0)
        friction = np.float32(friction * fn)
        friction = np.float32(friction * fn)
        friction = np.float32(friction * np.abs(q0))
        denominator = np.float32(1.0 + friction / h_power)
        return np.float32(np.float32(direction) * np.float32(numerator / denominator))

    depth = np.float64(edge_depth)
    old = np.float64(q_old)
    g = np.float64(gravity)
    timestep = np.float64(dt)
    rough = np.float64(manning_n)
    slope = np.float64(sf)
    numerator = np.abs(old) + np.abs(g * timestep * depth * slope)
    if arithmetic == "float_surface_only":
        # Historical hybrid sensitivity option (not the native FREE1 operator).
        friction_power = np.float64(7.0) / np.float64(3.0)
    else:
        friction_power = np.float64(10.0) / np.float64(3.0)
    denominator = np.float64(1.0) + (
        g * timestep * depth * rough * rough * np.abs(old)
        / np.power(depth, friction_power)
    )
    return np.float64(np.float64(direction) * numerator / denominator)
