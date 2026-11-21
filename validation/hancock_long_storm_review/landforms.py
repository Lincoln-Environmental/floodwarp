"""Deterministic continuous DEMs. Coordinates and bed elevations are metres."""

import numpy as np

from floodwarp.inputs import terrain_from_array, boundary_faces
from floodwarp.model import create_model


def physical_fields(*, landform, spec, seed, length, width, dx, terrain_dx):
    ny, nx = round(width / dx), round(length / dx)
    y, x = np.indices((ny, nx), dtype=float)
    x, y = (x + 0.5) * dx, (y + 0.5) * dx
    # Piecewise-constant representation experiment: sample a common parent DEM.
    xb = (np.floor(x / terrain_dx) + 0.5) * terrain_dx
    yb = (np.floor(y / terrain_dx) + 0.5) * terrain_dx
    u, v = xb / length, yb / width
    phase = np.random.default_rng(seed=seed).uniform(low=-0.3, high=0.3)
    centre = 0.47 + 0.10 * np.sin(2 * np.pi * u + phase)
    grade, incision, initial = spec
    base = grade * (length - xb)
    valley = np.exp(-(((v - centre) / 0.13) ** 2))
    if landform == "mixed":
        base = 0.001 * (length - xb) + grade * np.maximum(0.48 * length - xb, 0.0)
    z = base - incision * valley
    if landform == "plain":
        z -= 0.035 * np.exp(-(((u - 0.65) / 0.12) ** 2) - ((v - 0.65) / 0.18) ** 2)
    elif landform == "floodplain":
        # Overtoppable embankment interrupted by a low saddle.
        z += (
            0.10
            * np.exp(-(((u - 0.7) / 0.045) ** 2))
            * (1.0 - 0.93 * np.exp(-(((v - 0.55) / 0.11) ** 2)))
        )
    elif landform == "coastal":
        z -= 0.055 * np.exp(-(((u - 0.75) / 0.19) ** 2) - ((v - 0.35) / 0.19) ** 2)
    elif landform == "rolling":
        z += 0.17 * np.sin(3 * np.pi * u) * np.cos(2 * np.pi * v + phase)
    elif landform == "fan":
        z += 0.10 * np.cos(3 * np.pi * (v - 0.5) / (u + 0.5)) * np.sin(np.pi * u)
        z -= 0.07 * np.exp(-(((v - 0.45 - 0.3 * u) / 0.07) ** 2))
    elif landform in ("hill", "incised", "mixed"):
        z += (0.014 if landform == "hill" else 0.032) * width * (v - centre) ** 2
        z += (
            0.15
            * np.exp(-(((u - 0.61) / 0.05) ** 2))
            * (1.0 - np.exp(-(((v - centre) / 0.045) ** 2)))
        )
    h = np.zeros_like(z)
    if initial == "channel":
        h = 0.035 * np.maximum(0.0, (valley - 0.65) / 0.35)
    elif initial == "pond":
        h = 0.025 * np.maximum(
            0.0, 1.0 - ((u - 0.67) / 0.15) ** 2 - ((v - 0.43) / 0.18) ** 2
        )
    return (
        z,
        h,
        # Use the common 0.1 mm momentum threshold for every formulation.
        np.where(h > 0.0001, 0.12 * h, 0.0)
        if initial == "channel"
        else np.zeros_like(h),
        x / length,
        y / width,
    )


def storm_model(
    *,
    config,
    landform,
    storm,
    method,
    mode,
    dx,
    terrain_dx,
    dimensions,
    duration,
    device,
    directional=False,
):
    z, h, hu, u, v = physical_fields(
        landform=landform,
        spec=config["landforms"][landform],
        seed=config["seed"],
        length=dimensions[0],
        width=dimensions[1],
        dx=dx,
        terrain_dx=terrain_dx,
    )
    terrain = terrain_from_array(
        elevation=z, dx=dx, dy=dx, row_order="south_to_north", vertical_datum="local"
    )
    bx, by = boundary_faces(terrain=terrain)
    bx[:, -1] = 3 if landform == "coastal" else 2
    times = np.r_[np.asarray(config["hours"]) * 3600.0, duration]
    rates = (
        np.r_[config["storms_mm_h"]["spatial" if storm == "moving" else storm], 0.0]
        / 3.6e6
    )
    # Identical dimensionless rain footprint and soil capacity for all formulations.
    multiplier = np.ones_like(z)
    if storm == "spatial":
        multiplier = np.maximum(
            0.0, 1.0 - ((u - 0.30) / 0.23) ** 2 - ((v - 0.70) / 0.20) ** 2
        )
    infil = ((6.0 + 22.0 * u + 6.0 * v) if storm == "soil" else 2.0) / 3.6e6
    options = (
        dict(execution_mode=mode)
        if method != "li"
        else dict(
            track_cumulative_face_volumes=True,
            track_absolute_face_volumes=False,
            track_max_water_depth=True,
        )
    )
    if directional:
        options["cfl_policy"] = "directional"
    if method == "fv2":
        options["limiter"] = "adaptive"
    m = create_model(
        terrain=terrain,
        solver_method=method,
        boundary_qx=bx,
        boundary_qy=by,
        stage_qx=0.015 if landform == "coastal" else 0.0,
        stage_datum="local",
        initial_depth=h,
        manning_n=0.035,
        rainfall_series=(times, rates),
        infiltration=infil,
        device=device,
        **options,
    )
    m.solver.set_rainfall_spatial_multiplier(multiplier=multiplier)
    if method == "li":
        qx = np.zeros(bx.shape)
        qx[:, 1:-1] = 0.5 * (hu[:, 1:] + hu[:, :-1])
        qx[:, -1] = hu[:, -1]
        m.solver.set_face_fluxes(qx=qx, qy=np.zeros(by.shape))
    else:
        m.solver.set_momentum(hu=hu, hv=np.zeros_like(hu))
    return m, float(np.trapezoid(rates, times) * multiplier.sum() * dx**2)
