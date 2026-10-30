"""Masked one-way FREE outflow faces on interior active/inactive mask edges."""

import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import (
    BOUNDARY_MASKED_EAST_FREE,
    BOUNDARY_MASKED_NORTH_FREE,
    BOUNDARY_MASKED_SOUTH_FREE,
    BOUNDARY_MASKED_WEST_FREE,
    get_boundary_name,
)
from floodwarp.config import PRECISION_MODE, _float_env
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_solver import LocalInertialSolver


def _devices():
    devices = ["cpu"]
    if wp.is_device_available("cuda:0"):
        devices.append("cuda:0")
    return devices


def _mass_tolerance() -> float:
    return 1.0e-12 if _float_env == "float64" else 2.0e-6


# Each orientation: which strip of the 4x4 grid is inactive (nodata), which
# interior face array entry carries the masked FREE code, where water ponds
# initially, and the sign of outward flux through the masked face.
#   qx face i lies between cells i-1 and i; qy face j lies between rows j-1
#   and j.  Outward signs: west/south faces drain at q < 0, east/north at
#   q > 0.
ORIENTATIONS = ["west", "east", "south", "north"]


def _face_code(orientation: str) -> int:
    return {
        "west": BOUNDARY_MASKED_WEST_FREE,
        "east": BOUNDARY_MASKED_EAST_FREE,
        "south": BOUNDARY_MASKED_SOUTH_FREE,
        "north": BOUNDARY_MASKED_NORTH_FREE,
    }[orientation]


def _outward_sign(orientation: str) -> float:
    return -1.0 if orientation in ("west", "south") else 1.0


def _build_orientation_solver(
    orientation: str,
    *,
    device: str,
    use_fused_kernels: bool = True,
    track_cumulative_face_volumes: bool = False,
    depth: float = 0.4,
    mode: str = "coarse",
) -> LocalInertialSolver:
    """4x4 grid with one exterior nodata strip and a masked FREE face line."""
    ny = nx = 4
    active_mask = np.ones((ny, nx), dtype=np.int32)
    depth_field = np.zeros((ny, nx), dtype=np.float64)
    if orientation == "west":
        active_mask[:, 0] = 0
        depth_field[:, 1] = depth
    elif orientation == "east":
        active_mask[:, -1] = 0
        depth_field[:, nx - 2] = depth
    elif orientation == "south":
        active_mask[0, :] = 0
        depth_field[1, :] = depth
    else:
        active_mask[-1, :] = 0
        depth_field[ny - 2, :] = depth
    native_bed = None
    if mode == "traditional":
        native_bed = np.repeat(
            np.repeat(np.where(active_mask == 1, 0.0, 50.0), 2, axis=0),
            2, axis=1,
        )
    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((ny, nx), dtype=np.float64),
        manning_n=np.full((ny, nx), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device=device,
        maximum_dt=0.05,
        hydraulic_geometry_mode=mode,
        subgrid_bed_elevation=native_bed,
        subgrid_resolution_m=2.5 if native_bed is not None else None,
        use_fused_kernels=use_fused_kernels,
        track_cumulative_face_volumes=track_cumulative_face_volumes,
    )
    if orientation in ("west", "east"):
        boundary = solver.boundary_type_qx.numpy()
        boundary[:, 1 if orientation == "west" else nx - 1] = _face_code(orientation)
        solver.set_boundary_type_qx(boundary_type=boundary)
    else:
        boundary = solver.boundary_type_qy.numpy()
        boundary[1 if orientation == "south" else ny - 1, :] = _face_code(orientation)
        solver.set_boundary_type_qy(boundary_type=boundary)
    solver.set_water_depth(depth=depth_field)
    return solver


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("execution", ("step", "lts"))
def test_subgrid_masked_free_drains_and_conserves(
    orientation: str,
    device: str,
    execution: str,
) -> None:
    solver = _build_orientation_solver(
        orientation=orientation,
        device=device,
        mode="traditional",
    )
    initial_volume = solver.initial_volume
    face_z = (
        solver._subgrid_geometry.face_z_x
        if orientation in ("west", "east")
        else solver._subgrid_geometry.face_z_y
    )
    masked_index = 1 if orientation in ("west", "south") else 3
    if orientation in ("west", "east"):
        np.testing.assert_array_equal(face_z[:, masked_index, :], 0.0)
    else:
        np.testing.assert_array_equal(face_z[masked_index, :, :], 0.0)

    if execution == "step":
        _advance(solver=solver, steps=5, dt=0.05)
    else:
        solver.run_spatially_adaptive_loop(
            duration=0.25,
            max_levels=2,
            wet_max_level=1,
        )

    flux = _masked_face_flux(solver=solver, orientation=orientation)
    accounting = solver.cumulative_accounting()
    final_volume = float(solver.get_water_depth().sum()) * float(solver.dx) * float(solver.dy)
    assert np.all(_outward_sign(orientation) * flux >= 0.0)
    assert np.any(_outward_sign(orientation) * flux > 0.0)
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
    assert np.min(solver.get_water_depth()) >= -1.0e-12
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=_mass_tolerance()
    )


def _masked_face_flux(solver: LocalInertialSolver, orientation: str) -> np.ndarray:
    if orientation == "west":
        return solver.get_qx()[:, 1]
    if orientation == "east":
        return solver.get_qx()[:, solver.nx - 1]
    if orientation == "south":
        return solver.get_qy()[1, :]
    return solver.get_qy()[solver.ny - 1, :]


def _advance(solver: LocalInertialSolver, *, steps: int, dt: float) -> None:
    for _ in range(steps):
        solver.step(remaining_time=dt, fixed_dt=dt)


def test_get_boundary_name_covers_masked_free_codes() -> None:
    assert get_boundary_name(BOUNDARY_MASKED_WEST_FREE) == "MASKED_WEST_FREE"
    assert get_boundary_name(BOUNDARY_MASKED_EAST_FREE) == "MASKED_EAST_FREE"
    assert get_boundary_name(BOUNDARY_MASKED_SOUTH_FREE) == "MASKED_SOUTH_FREE"
    assert get_boundary_name(BOUNDARY_MASKED_NORTH_FREE) == "MASKED_NORTH_FREE"


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_masked_free_drains_outward_only(orientation: str, device: str) -> None:
    solver = _build_orientation_solver(orientation, device=device)
    sign = _outward_sign(orientation)
    initial_volume = solver.initial_volume

    for _ in range(5):
        solver.step(remaining_time=0.05, fixed_dt=0.05)
        flux = _masked_face_flux(solver, orientation)
        # One-way clamp: the flux can never carry the inward sign.
        assert np.all(sign * flux >= 0.0)

    depth = solver.get_water_depth()
    accounting = solver.cumulative_accounting()
    diagnostics = compute_diagnostics(solver)
    final_volume = float(depth.sum()) * float(solver.dx) * float(solver.dy)

    assert np.all(sign * _masked_face_flux(solver, orientation) > 0.0)
    assert np.all(depth[solver.active_mask.numpy() == 0] == 0.0)
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=_mass_tolerance()
    )
    assert abs(diagnostics["mass_balance_residual"]) < _mass_tolerance()


@pytest.mark.parametrize("device", _devices())
def test_masked_free_zero_depth_produces_no_flow(device: str) -> None:
    solver = _build_orientation_solver("north", device=device, depth=0.0)
    _advance(solver, steps=3, dt=0.05)

    assert np.all(_masked_face_flux(solver, "north") == 0.0)
    accounting = solver.cumulative_accounting()
    assert accounting["boundary_outflow"] == 0.0
    assert accounting["boundary_inflow"] == 0.0


@pytest.mark.parametrize("device", _devices())
def test_masked_free_face_volumes_close_with_accounting(device: str) -> None:
    solver = _build_orientation_solver(
        "north", device=device, track_cumulative_face_volumes=True
    )
    initial_volume = solver.initial_volume
    _advance(solver, steps=5, dt=0.05)

    volumes = solver.get_cumulative_face_volumes()
    masked_face = (
        solver.boundary_type_qy.numpy() == BOUNDARY_MASKED_NORTH_FREE
    )
    outward_face_volume = float(
        volumes["cumulative_abs_qy_volume_m3"][masked_face].sum()
    )
    accounting = solver.cumulative_accounting()
    final_volume = (
        float(solver.get_water_depth().sum()) * float(solver.dx) * float(solver.dy)
    )

    assert outward_face_volume > 0.0
    assert outward_face_volume == pytest.approx(
        accounting["boundary_outflow"], rel=1.0e-9
    )
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], rel=1.0e-9
    )


@pytest.mark.parametrize("device", _devices())
def test_inactive_island_without_masked_codes_stays_closed(device: str) -> None:
    ny = nx = 5
    active_mask = np.ones((ny, nx), dtype=np.int32)
    active_mask[2, 2] = 0
    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((ny, nx), dtype=np.float64),
        manning_n=np.full((ny, nx), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device=device,
        maximum_dt=0.05,
    )
    depth_field = np.full((ny, nx), 0.3, dtype=np.float64)
    depth_field[2, 2] = 0.0
    solver.set_water_depth(depth=depth_field)

    _advance(solver, steps=5, dt=0.05)

    qx = solver.get_qx()
    qy = solver.get_qy()
    assert solver.get_water_depth()[2, 2] == 0.0
    # Every face touching the island is CLOSED and must carry no flux.
    assert qx[2, 2] == 0.0
    assert qx[2, 3] == 0.0
    assert qy[2, 2] == 0.0
    assert qy[3, 2] == 0.0
    accounting = solver.cumulative_accounting()
    assert accounting["boundary_outflow"] == 0.0
    assert accounting["boundary_inflow"] == 0.0


def _base_solver_for_validation(
    active_mask: np.ndarray, device: str = "cpu", **kwargs
) -> LocalInertialSolver:
    ny, nx = active_mask.shape
    return LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((ny, nx), dtype=np.float64),
        manning_n=np.full((ny, nx), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device=device,
        **kwargs,
    )


def test_masked_free_rejects_outer_qx_edges() -> None:
    solver = _base_solver_for_validation(np.ones((2, 2), dtype=np.int32))
    boundary = solver.boundary_type_qx.numpy()
    boundary[:, 0] = BOUNDARY_MASKED_WEST_FREE
    with pytest.raises(ValueError, match="only valid on interior qx faces"):
        solver.set_boundary_type_qx(boundary_type=boundary)


def test_masked_free_rejects_outer_qy_edges() -> None:
    solver = _base_solver_for_validation(np.ones((2, 2), dtype=np.int32))
    boundary = solver.boundary_type_qy.numpy()
    boundary[0, :] = BOUNDARY_MASKED_SOUTH_FREE
    with pytest.raises(ValueError, match="only valid on interior qy faces"):
        solver.set_boundary_type_qy(boundary_type=boundary)


def test_masked_free_rejects_wrong_axis() -> None:
    solver = _base_solver_for_validation(np.ones((2, 2), dtype=np.int32))
    boundary_qx = solver.boundary_type_qx.numpy()
    boundary_qx[:, 1] = BOUNDARY_MASKED_NORTH_FREE
    with pytest.raises(ValueError, match="only valid on qy faces"):
        solver.set_boundary_type_qx(boundary_type=boundary_qx)
    boundary_qy = solver.boundary_type_qy.numpy()
    boundary_qy[1, :] = BOUNDARY_MASKED_WEST_FREE
    with pytest.raises(ValueError, match="only valid on qx faces"):
        solver.set_boundary_type_qy(boundary_type=boundary_qy)


def test_masked_west_free_rejects_wrong_active_side() -> None:
    solver = _base_solver_for_validation(np.array([[1, 0]], dtype=np.int32))
    boundary = solver.boundary_type_qx.numpy()
    boundary[0, 1] = BOUNDARY_MASKED_WEST_FREE
    with pytest.raises(ValueError, match="inactive i-1 cell"):
        solver.set_boundary_type_qx(boundary_type=boundary)


def test_masked_east_free_rejects_wrong_active_side() -> None:
    solver = _base_solver_for_validation(np.array([[0, 1]], dtype=np.int32))
    boundary = solver.boundary_type_qx.numpy()
    boundary[0, 1] = BOUNDARY_MASKED_EAST_FREE
    with pytest.raises(ValueError, match="active i-1 cell"):
        solver.set_boundary_type_qx(boundary_type=boundary)


def test_masked_south_free_rejects_wrong_active_side() -> None:
    solver = _base_solver_for_validation(np.array([[1], [0]], dtype=np.int32))
    boundary = solver.boundary_type_qy.numpy()
    boundary[1, 0] = BOUNDARY_MASKED_SOUTH_FREE
    with pytest.raises(ValueError, match="inactive j-1 cell"):
        solver.set_boundary_type_qy(boundary_type=boundary)


def test_masked_north_free_rejects_wrong_active_side() -> None:
    solver = _base_solver_for_validation(np.array([[0], [1]], dtype=np.int32))
    boundary = solver.boundary_type_qy.numpy()
    boundary[1, 0] = BOUNDARY_MASKED_NORTH_FREE
    with pytest.raises(ValueError, match="active j-1 cell"):
        solver.set_boundary_type_qy(boundary_type=boundary)


def test_masked_free_supported_in_subgrid_mode() -> None:
    active_mask = np.array([[0, 1], [0, 1]], dtype=np.int32)
    bed = np.zeros((2, 2), dtype=np.float64)
    factor = 5
    native_bed = np.repeat(np.repeat(bed, factor, axis=0), factor, axis=1)
    solver = _base_solver_for_validation(
        active_mask,
        hydraulic_geometry_mode="traditional",
        subgrid_bed_elevation=native_bed,
        subgrid_resolution_m=5.0 / factor,
    )
    boundary = solver.boundary_type_qx.numpy()
    boundary[:, 1] = BOUNDARY_MASKED_WEST_FREE
    solver.set_boundary_type_qx(boundary_type=boundary)
    depth = np.zeros((2, 2), dtype=np.float64)
    depth[:, 1] = 0.4
    solver.set_water_depth(depth=depth)
    solver.step(remaining_time=0.05, fixed_dt=0.05)
    assert np.all(solver.get_qx()[:, 1] < 0.0)
    assert solver.cumulative_accounting()["boundary_outflow"] > 0.0


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_masked_free_fused_and_unfused_are_bit_exact(
    orientation: str, device: str
) -> None:
    fused = _build_orientation_solver(
        orientation, device=device, use_fused_kernels=True
    )
    unfused = _build_orientation_solver(
        orientation, device=device, use_fused_kernels=False
    )
    _advance(fused, steps=5, dt=0.05)
    _advance(unfused, steps=5, dt=0.05)

    np.testing.assert_array_equal(fused.get_water_depth(), unfused.get_water_depth())
    np.testing.assert_array_equal(fused.get_qx(), unfused.get_qx())
    np.testing.assert_array_equal(fused.get_qy(), unfused.get_qy())
    assert fused.cumulative_accounting() == unfused.cumulative_accounting()


@pytest.mark.parametrize("device", _devices())
def test_masked_free_participates_in_cfl_restriction(device: str) -> None:
    active_mask = np.array([[0, 1]], dtype=np.int32)
    solver = _base_solver_for_validation(active_mask, device=device)
    boundary = solver.boundary_type_qx.numpy()
    boundary[0, 1] = BOUNDARY_MASKED_WEST_FREE
    solver.set_boundary_type_qx(boundary_type=boundary)
    solver.set_water_depth(depth=np.array([[0.0, 10.0]], dtype=np.float64))

    closed = _base_solver_for_validation(active_mask, device=device)
    closed.set_water_depth(depth=np.array([[0.0, 10.0]], dtype=np.float64))

    dt_free = solver.calculate_stable_timestep()
    dt_closed = closed.calculate_stable_timestep()
    assert dt_free < dt_closed
    # The masked FREE face restricts like a wet face at the active depth:
    # cfl * dx / sqrt(g * h) ~ 0.2 s here, far below the 60 s cap.
    assert dt_free < 1.0


@pytest.mark.parametrize("device", _devices())
def test_masked_free_runs_in_spatial_lts(device: str) -> None:
    solver = _build_orientation_solver("north", device=device)
    initial_volume = solver.initial_volume
    stats = solver.run_spatially_adaptive_loop(
        duration=0.2,
        max_levels=2,
        wet_max_level=0,
    )

    final_volume = (
        float(solver.get_water_depth().sum()) * float(solver.dx) * float(solver.dy)
    )
    accounting = solver.cumulative_accounting()
    assert stats["macro_steps"] > 0
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=_mass_tolerance()
    )


@pytest.mark.parametrize("device", _devices())
def test_channel_drains_through_masked_free_coast(device: str) -> None:
    """A 1-cell-wide sloped channel exits into an exterior nodata strip."""
    ny, nx = 6, 4
    active_mask = np.zeros((ny, nx), dtype=np.int32)
    active_mask[0:5, 1] = 1
    bed = np.zeros((ny, nx), dtype=np.float64)
    bed[0:5, 1] = 0.2 * np.arange(5, 0, -1, dtype=np.float64)
    solver = LocalInertialSolver(
        ny=ny,
        nx=nx,
        dx=5.0,
        dy=5.0,
        bed_elevation=bed,
        manning_n=np.full((ny, nx), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device=device,
    )
    boundary = solver.boundary_type_qy.numpy()
    boundary[5, 1] = BOUNDARY_MASKED_NORTH_FREE
    solver.set_boundary_type_qy(boundary_type=boundary)
    depth_field = np.zeros((ny, nx), dtype=np.float64)
    depth_field[0:5, 1] = 0.3
    solver.set_water_depth(depth=depth_field)
    initial_volume = solver.initial_volume

    _advance(solver, steps=200, dt=0.1)

    depth = solver.get_water_depth()
    accounting = solver.cumulative_accounting()
    final_volume = float(depth.sum()) * float(solver.dx) * float(solver.dy)

    assert np.all(depth[active_mask == 0] == 0.0)
    assert np.all(depth[0:5, 1] < 0.3)
    assert final_volume < 0.5 * initial_volume
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=_mass_tolerance()
    )


@pytest.mark.skipif(
    PRECISION_MODE != "mixed",
    reason="exercises the conditioned mixed-precision masked FREE branch",
)
def test_masked_free_conditioned_mixed_precision_drains() -> None:
    solver = _build_orientation_solver("west", device="cpu")
    _advance(solver, steps=5, dt=0.05)

    flux = _masked_face_flux(solver, "west")
    accounting = solver.cumulative_accounting()
    assert np.all(flux <= 0.0)
    assert np.any(flux < 0.0)
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
