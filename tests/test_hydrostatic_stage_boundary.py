"""Dry coastal lips use FV1's hydrostatic exterior head, not pressure in air."""

import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import (
    BOUNDARY_MASKED_NORTH_STAGE,
    BOUNDARY_MASKED_WEST_STAGE,
    BOUNDARY_SPECIFIED_STAGE,
)
from floodwarp.config import PRECISION_MODE
from floodwarp.local_inertial_reference import step_local_inertial_numpy
from floodwarp.local_inertial_solver import LocalInertialSolver


def stage_case(*, side, device, bed, stage, depth, mode="coarse", fused=True,minimum_dt=.001):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    shape = (1, 1)
    if side == "masked_west":
        shape = (1, 2)
    elif side == "masked_north":
        shape = (2, 1)
    active = np.ones(shape, dtype=np.int32)
    if side == "masked_west":
        active[0, 0] = 0
    elif side == "masked_north":
        active[1, 0] = 0
    terrain = np.where(active, bed, 1.0e6)
    options = {}
    if mode == "traditional":
        options = dict(subgrid_bed_elevation=terrain.repeat(2, axis=0).repeat(2, axis=1),
                       subgrid_resolution_m=1.0)
    solver = LocalInertialSolver(
        ny=shape[0], nx=shape[1], dx=2.0, dy=2.0,
        bed_elevation=terrain, manning_n=np.full(shape, 0.035),
        active_mask=active, device=device, maximum_dt=0.02,
        hydraulic_geometry_mode=mode, use_fused_kernels=fused,minimum_dt=minimum_dt, **options,
    )
    axis = "x" if side in ("west", "east", "masked_west") else "y"
    index = (0, 0)
    code = BOUNDARY_SPECIFIED_STAGE
    outward = -1.0
    if side in ("east", "masked_west"):
        index = (0, 1)
    elif side in ("north", "masked_north"):
        index = (1, 0)
    if side in ("east", "north", "masked_north"):
        outward = 1.0
    if side == "masked_west":
        code = BOUNDARY_MASKED_WEST_STAGE
    elif side == "masked_north":
        code = BOUNDARY_MASKED_NORTH_STAGE
    if axis == "x":
        types = solver.boundary_type_qx.numpy()
        values = solver.boundary_stage_qx.numpy()
        types[index], values[index] = code, stage
        solver.set_boundary_type_qx(boundary_type=types)
        solver.set_boundary_stage_qx(stage=values)
    else:
        types = solver.boundary_type_qy.numpy()
        values = solver.boundary_stage_qy.numpy()
        types[index], values[index] = code, stage
        solver.set_boundary_type_qy(boundary_type=types)
        solver.set_boundary_stage_qy(stage=values)
    solver.set_water_depth(depth=depth * active.astype(np.float64))
    return solver, axis, index, outward


def port_discharge(*, solver, axis, index, outward):
    values = solver.get_qx() if axis == "x" else solver.get_qy()
    return outward * float(values[index])


def assert_conservation(*, solver):
    totals = solver.cumulative_accounting()
    storage = float(solver.get_water_depth().sum()) * float(solver.dx) * float(solver.dy)
    residual = solver.initial_volume + totals["rain"] + totals["boundary_inflow"]
    residual -= storage + totals["infiltration"] + totals["boundary_outflow"]
    tolerance = 1.0e-12 if PRECISION_MODE == "fp64" else 2.0e-6
    assert abs(residual) < tolerance
    assert np.all(solver.get_water_depth() >= 0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("side", ["west", "east", "south", "north", "masked_west", "masked_north"])
def test_dry_lip_uses_water_depth_not_fall_to_sea(side, fused, device):
    solver, axis, index, outward = stage_case(
        side=side, device=device, bed=1.0, stage=0.0, depth=0.1, fused=fused,
    )
    dt = solver.step(remaining_time=0.02, fixed_dt=0.02)
    # Starting from rest, local inertia gives dq/dt = g h^2 / dx at a dry lip.
    # The previous raw sea-level gradient instead used h * (1 + h).
    expected = float(solver.gravity) * 0.1**2 * dt / 2.0
    tolerance = 1.0e-11 if PRECISION_MODE == "fp64" else 5.0e-6
    assert port_discharge(solver=solver, axis=axis, index=index, outward=outward) == pytest.approx(expected, rel=tolerance)
    assert solver.cumulative_accounting()["boundary_inflow"] == 0.0
    assert_conservation(solver=solver)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("side", ["west", "east", "south", "north", "masked_west", "masked_north"])
@pytest.mark.parametrize("bed,depth,expected_sign", [(1.0, 0.0, 0), (-1.0, 0.0, -1), (-1.0, 1.0, 0)])
def test_dry_land_sea_inflow_and_lake_equilibrium(side, device, bed, depth, expected_sign):
    solver, axis, index, outward = stage_case(
        side=side, device=device, bed=bed, stage=0.0, depth=depth,
    )
    solver.step(remaining_time=0.02, fixed_dt=0.02)
    discharge = port_discharge(solver=solver, axis=axis, index=index, outward=outward)
    if expected_sign == 0:
        assert discharge == 0.0
    else:
        assert discharge < 0.0
        assert solver.cumulative_accounting()["boundary_inflow"] > 0.0
    assert_conservation(solver=solver)


@pytest.mark.parametrize("mode", ["coarse", "traditional"])
def test_dry_lip_is_independent_of_sea_level_below_sill(mode):
    outputs = []
    for stage in (1.0, 0.0, -10.0):
        solver, axis, index, outward = stage_case(
            side="east", device="cpu", bed=1.0, stage=stage, depth=0.1, mode=mode,
        )
        solver.step(remaining_time=0.02, fixed_dt=0.02)
        outputs.append(port_discharge(solver=solver, axis=axis, index=index, outward=outward))
        assert_conservation(solver=solver)
    np.testing.assert_array_equal(actual=outputs, desired=np.repeat(outputs[0], len(outputs)))


def test_numpy_reference_agrees_at_above_sea_outlet():
    solver, unused_axis, unused_index, unused_outward = stage_case(
        side="east", device="cpu", bed=1.0, stage=0.0, depth=0.1,
    )
    zero = np.zeros((1, 1))
    expected = step_local_inertial_numpy(
        water_depth=solver.get_water_depth(), bed_elevation=solver.bed_elevation.numpy(),
        manning_n=solver.manning_n.numpy(), active_mask=solver.active_mask.numpy(),
        qx=solver.get_qx(), qy=solver.get_qy(), rainfall_rate=zero,
        other_source_rate=zero, infiltration_capacity=zero,
        boundary_type_qx=solver.boundary_type_qx.numpy(), boundary_stage_qx=solver.boundary_stage_qx.numpy(),
        boundary_type_qy=solver.boundary_type_qy.numpy(), boundary_stage_qy=solver.boundary_stage_qy.numpy(),
        dx=2.0, dy=2.0, dt=0.02,
    )
    solver.step(remaining_time=0.02, fixed_dt=0.02)
    tolerance = 1.0e-12 if PRECISION_MODE == "fp64" else 2.0e-6
    np.testing.assert_allclose(actual=solver.get_water_depth(), desired=expected["water_depth"], atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(actual=solver.get_qx(), desired=expected["qx"], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sgs_pressure_preserves_each_native_coastal_sill(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    solver = LocalInertialSolver(
        ny=1, nx=1, dx=2.0, dy=2.0, bed_elevation=np.array([[2.0]]),
        manning_n=np.array([[0.035]]), active_mask=np.ones((1, 1), dtype=np.int32),
        hydraulic_geometry_mode="traditional", subgrid_resolution_m=1.0,
        subgrid_bed_elevation=np.array([[1.0, 1.0], [3.0, 3.0]]), device=device,
    )
    boundary = solver.boundary_type_qx.numpy()
    boundary[0, 1] = BOUNDARY_SPECIFIED_STAGE
    solver.set_boundary_type_qx(boundary_type=boundary)
    solver.set_water_depth(depth=np.array([[2.0]]))  # Horizontal stage 4 m.
    dt = solver.step(remaining_time=0.01, fixed_dt=0.01)
    # Two equal-width native lips: pressure moments (4-1)^2 and (4-3)^2.
    expected = float(solver.gravity) * dt * 5.0 / 2.0
    tolerance = 1.0e-11 if PRECISION_MODE == "fp64" else 5.0e-6
    assert solver.get_qx()[0, 1] == pytest.approx(expected, rel=tolerance)
    assert_conservation(solver=solver)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_dry_lip_finite_donor_with_retained_outward_momentum(device):
    solver, unused_axis, unused_index, unused_outward = stage_case(
        side="east", device=device, bed=1.0, stage=0.0, depth=0.1,
    )
    wp.copy(solver.manning_n, wp.array(np.array([[1.0e-8]]), dtype=solver.manning_n.dtype, device=device))
    solver.set_face_fluxes(qx=np.array([[0.0, 100.0]]), qy=np.zeros((2, 1)))
    solver.step(remaining_time=0.02, fixed_dt=0.02)
    assert solver.cumulative_accounting()["boundary_outflow"] <= solver.initial_volume * (1.0 + 1.0e-6)
    assert_conservation(solver=solver)


def test_resident_gpu_tidal_transition_agrees_with_cpu_steps():
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    states = []
    stage_series = (np.array([0.0, 0.08, 0.16]), np.array([0.0, 1.3, 0.0]))
    from tests.test_device_loop import run_host_stepped
    for device in ("cpu", "cuda:0"):
        solver, unused_axis, unused_index, unused_outward = stage_case(
            side="west", device=device, bed=1.0, stage=0.0, depth=0.0,
            # The above-sill pulse rises 0.3 m in 18.5 ms. Resolving its
            # head variation needs sub-ms steps; production defaults stay put.
            minimum_dt=.0001,
        )
        solver.set_uniform_rainfall_rate(rainfall_rate=0.001)
        if device == "cpu":
            timesteps=run_host_stepped(solver=solver,duration=.16,hyetograph=None,stage_series=stage_series)
        else:
            result = solver.run_device_loop(duration=0.16, stage_series_qx=stage_series,
                                           use_cuda_graph=True, dt_history_capacity=10000)
            timesteps = result["dt_history"]
        states.append((solver.get_water_depth(), solver.get_qx(), timesteps))
        assert_conservation(solver=solver)
    tolerance = 1.0e-12 if PRECISION_MODE == "fp64" else 2.0e-6
    for cpu, gpu in zip(states[0], states[1]):
        np.testing.assert_allclose(actual=gpu, desired=cpu, atol=tolerance, rtol=tolerance)


def test_dry_outlet_timestep_refinement():
    storage = []
    for dt in (0.02, 0.01, 0.005, 0.00125):
        solver, unused_axis, unused_index, unused_outward = stage_case(
            side="east", device="cpu", bed=1.0, stage=0.0, depth=0.1,
        )
        for unused_step in range(round(0.5 / dt)):
            solver.step(fixed_dt=dt)
        storage.append(float(solver.get_water_depth().sum()))
        assert_conservation(solver=solver)
    assert abs(storage[2] - storage[3]) < abs(storage[1] - storage[3])
    assert abs(storage[1] - storage[3]) < abs(storage[0] - storage[3])
