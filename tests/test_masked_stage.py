"""Masked specified-stage boundaries on irregular north-facing polygon edges."""

import numpy as np
import pytest

from floodwarp.boundary_conditions import (
    BOUNDARY_MASKED_NORTH_STAGE,
    BOUNDARY_MASKED_WEST_STAGE,
)
from floodwarp.config import _float_env
from floodwarp.diagnostics import compute_diagnostics
from floodwarp.local_inertial_solver import LocalInertialSolver


def build_masked_stage_solver(*, use_fused_kernels: bool) -> LocalInertialSolver:
    """Construct a tiny irregular domain with two north-facing stage faces."""
    active_mask = np.array(
        [
            [1, 1],
            [1, 0],
            [0, 0],
        ],
        dtype=np.int32,
    )
    solver = LocalInertialSolver(
        ny=3,
        nx=2,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((3, 2), dtype=np.float64),
        manning_n=np.full((3, 2), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device="cpu",
        maximum_dt=0.05,
        use_fused_kernels=use_fused_kernels,
    )
    boundary_type_qy = solver.boundary_type_qy.numpy()
    boundary_type_qy[1, 1] = BOUNDARY_MASKED_NORTH_STAGE
    boundary_type_qy[2, 0] = BOUNDARY_MASKED_NORTH_STAGE
    solver.set_boundary_type_qy(boundary_type=boundary_type_qy)
    boundary_stage_qy = solver.boundary_stage_qy.numpy()
    boundary_stage_qy[1, 1] = 0.0
    boundary_stage_qy[2, 0] = 0.0
    solver.set_boundary_stage_qy(stage=boundary_stage_qy)
    solver.set_water_depth(
        depth=np.array(
            [
                [0.4, 0.4],
                [0.4, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    return solver


def build_masked_west_stage_solver(*, use_fused_kernels: bool) -> LocalInertialSolver:
    """Construct a tiny domain with one west-facing masked stage face."""
    active_mask = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=np.int32,
    )
    solver = LocalInertialSolver(
        ny=2,
        nx=2,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((2, 2), dtype=np.float64),
        manning_n=np.full((2, 2), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device="cpu",
        maximum_dt=0.05,
        use_fused_kernels=use_fused_kernels,
    )
    boundary_type_qx = solver.boundary_type_qx.numpy()
    boundary_type_qx[0, 1] = BOUNDARY_MASKED_WEST_STAGE
    solver.set_boundary_type_qx(boundary_type=boundary_type_qx)
    boundary_stage_qx = solver.boundary_stage_qx.numpy()
    boundary_stage_qx[0, 1] = 0.0
    solver.set_boundary_stage_qx(stage=boundary_stage_qx)
    solver.set_water_depth(
        depth=np.array(
            [
                [0.0, 0.4],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    return solver


def advance_fixed_steps(*, solver: LocalInertialSolver, steps: int, dt: float) -> None:
    """Advance a solver by a fixed number of equal steps."""
    for _ in range(steps):
        solver.step(remaining_time=dt, fixed_dt=dt)


def test_masked_north_stage_drains_and_accounts_conservatively() -> None:
    solver = build_masked_stage_solver(use_fused_kernels=True)
    initial_volume = solver.initial_volume
    advance_fixed_steps(solver=solver, steps=4, dt=0.05)

    depth = solver.get_water_depth()
    qy = solver.get_qy()
    accounting = solver.cumulative_accounting()
    diagnostics = compute_diagnostics(solver)
    final_volume = float(depth.sum()) * float(solver.dx) * float(solver.dy)

    assert qy[1, 1] > 0.0
    assert qy[2, 0] > 0.0
    assert np.all(depth[solver.active_mask.numpy() == 0] == 0.0)
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
    mass_tolerance = 1.0e-12 if _float_env == "float64" else 2.0e-6
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=mass_tolerance
    )
    assert abs(diagnostics["mass_balance_residual"]) < mass_tolerance


def test_masked_north_stage_allows_sea_level_inflow_below_zero_datum() -> None:
    bed = np.array([[-1.0], [-1.0]], dtype=np.float64)
    active_mask = np.array([[1], [0]], dtype=np.int32)
    solver = LocalInertialSolver(
        ny=2,
        nx=1,
        dx=5.0,
        dy=5.0,
        bed_elevation=bed,
        manning_n=np.full((2, 1), 0.04, dtype=np.float64),
        active_mask=active_mask,
        device="cpu",
    )
    boundary_type_qy = solver.boundary_type_qy.numpy()
    boundary_type_qy[1, 0] = BOUNDARY_MASKED_NORTH_STAGE
    solver.set_boundary_type_qy(boundary_type=boundary_type_qy)
    boundary_stage_qy = solver.boundary_stage_qy.numpy()
    boundary_stage_qy[1, 0] = 0.0
    solver.set_boundary_stage_qy(stage=boundary_stage_qy)
    solver.set_water_depth(depth=np.zeros((2, 1), dtype=np.float64))

    solver.step(remaining_time=0.05, fixed_dt=0.05)

    assert solver.get_qy()[1, 0] < 0.0
    assert solver.get_water_depth()[0, 0] > 0.0
    accounting = solver.cumulative_accounting()
    assert accounting["boundary_inflow"] > 0.0
    assert accounting["boundary_outflow"] == 0.0
    mass_tolerance = 1.0e-12 if _float_env == "float64" else 2.0e-6
    assert (
        abs(compute_diagnostics(solver)["mass_balance_residual"])
        < mass_tolerance
    )


def test_masked_north_stage_fused_and_unfused_are_bit_exact() -> None:
    fused = build_masked_stage_solver(use_fused_kernels=True)
    unfused = build_masked_stage_solver(use_fused_kernels=False)
    advance_fixed_steps(solver=fused, steps=4, dt=0.05)
    advance_fixed_steps(solver=unfused, steps=4, dt=0.05)

    np.testing.assert_array_equal(fused.get_water_depth(), unfused.get_water_depth())
    np.testing.assert_array_equal(fused.get_qx(), unfused.get_qx())
    np.testing.assert_array_equal(fused.get_qy(), unfused.get_qy())
    assert fused.cumulative_accounting() == unfused.cumulative_accounting()


def test_masked_north_stage_runs_in_spatial_lts() -> None:
    solver = build_masked_stage_solver(use_fused_kernels=True)
    initial_volume = solver.initial_volume
    stats = solver.run_spatially_adaptive_loop(
        duration=0.2,
        max_levels=2,
        wet_max_level=0,
    )

    final_volume = float(solver.get_water_depth().sum()) * float(solver.dx) * float(solver.dy)
    accounting = solver.cumulative_accounting()
    assert stats["macro_steps"] > 0
    assert accounting["boundary_outflow"] > 0.0
    mass_tolerance = 1.0e-12 if _float_env == "float64" else 2.0e-6
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=mass_tolerance
    )


def test_masked_north_stage_rejects_wrong_active_side() -> None:
    solver = LocalInertialSolver(
        ny=2,
        nx=1,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((2, 1), dtype=np.float64),
        manning_n=np.full((2, 1), 0.04, dtype=np.float64),
        active_mask=np.array([[0], [1]], dtype=np.int32),
        device="cpu",
    )
    boundary_type_qy = solver.boundary_type_qy.numpy()
    boundary_type_qy[1, 0] = BOUNDARY_MASKED_NORTH_STAGE
    with pytest.raises(ValueError, match="active j-1 cell"):
        solver.set_boundary_type_qy(boundary_type=boundary_type_qy)


def test_masked_west_stage_drains_and_accounts_conservatively() -> None:
    solver = build_masked_west_stage_solver(use_fused_kernels=True)
    initial_volume = solver.initial_volume
    advance_fixed_steps(solver=solver, steps=4, dt=0.05)

    depth = solver.get_water_depth()
    accounting = solver.cumulative_accounting()
    final_volume = float(depth.sum()) * float(solver.dx) * float(solver.dy)

    assert solver.get_qx()[0, 1] < 0.0
    assert accounting["boundary_outflow"] > 0.0
    assert accounting["boundary_inflow"] == 0.0
    mass_tolerance = 1.0e-12 if _float_env == "float64" else 2.0e-6
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=mass_tolerance
    )
    assert abs(compute_diagnostics(solver)["mass_balance_residual"]) < mass_tolerance


def test_masked_west_stage_fused_and_unfused_are_bit_exact() -> None:
    fused = build_masked_west_stage_solver(use_fused_kernels=True)
    unfused = build_masked_west_stage_solver(use_fused_kernels=False)
    advance_fixed_steps(solver=fused, steps=4, dt=0.05)
    advance_fixed_steps(solver=unfused, steps=4, dt=0.05)

    np.testing.assert_array_equal(fused.get_water_depth(), unfused.get_water_depth())
    np.testing.assert_array_equal(fused.get_qx(), unfused.get_qx())
    np.testing.assert_array_equal(fused.get_qy(), unfused.get_qy())
    assert fused.cumulative_accounting() == unfused.cumulative_accounting()


def test_masked_west_stage_runs_in_spatial_lts() -> None:
    solver = build_masked_west_stage_solver(use_fused_kernels=True)
    initial_volume = solver.initial_volume
    stats = solver.run_spatially_adaptive_loop(
        duration=0.2,
        max_levels=2,
        wet_max_level=0,
    )

    final_volume = float(solver.get_water_depth().sum()) * float(solver.dx) * float(solver.dy)
    accounting = solver.cumulative_accounting()
    assert stats["macro_steps"] > 0
    assert accounting["boundary_outflow"] > 0.0
    mass_tolerance = 1.0e-12 if _float_env == "float64" else 2.0e-6
    assert initial_volume - final_volume == pytest.approx(
        accounting["boundary_outflow"], abs=mass_tolerance
    )


def test_masked_west_stage_rejects_wrong_active_side() -> None:
    solver = LocalInertialSolver(
        ny=1,
        nx=2,
        dx=5.0,
        dy=5.0,
        bed_elevation=np.zeros((1, 2), dtype=np.float64),
        manning_n=np.full((1, 2), 0.04, dtype=np.float64),
        active_mask=np.array([[1, 0]], dtype=np.int32),
        device="cpu",
    )
    boundary_type_qx = solver.boundary_type_qx.numpy()
    boundary_type_qx[0, 1] = BOUNDARY_MASKED_WEST_STAGE
    with pytest.raises(ValueError, match="inactive i-1 cell"):
        solver.set_boundary_type_qx(boundary_type=boundary_type_qx)
