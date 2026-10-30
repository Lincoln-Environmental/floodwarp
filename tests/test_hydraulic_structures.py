"""Analytical conservation tests for compact internal structure transfers."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver


def culvert(**changes):
    record = {
        "id": "culvert_test", "type": "culvert",
        "inlet_row": 0, "inlet_col": 0, "outlet_row": 0, "outlet_col": 1,
        "shape": "box", "width_m": 1.0, "height_m": 1.0,
        "length_m": 1.0, "inlet_invert_m": 0.0, "outlet_invert_m": 0.0,
        "manning_n": 0.02, "entrance_loss_coefficient": 0.5,
        "exit_loss_coefficient": 1.0, "one_way": False,
    }
    record.update(changes)
    return record


def weir(**changes):
    record = {
        "id": "weir_test", "type": "weir",
        "inlet_row": 0, "inlet_col": 0, "outlet_row": 0, "outlet_col": 1,
        "crest_elevation_m": 0.5, "crest_length_m": 1.0,
        "discharge_coefficient": 0.62, "weir_type": "sharp_crested",
        "one_way": False,
    }
    record.update(changes)
    return record


def make_solver(records, *, depth, dx=10.0, device="cpu"):
    solver = LocalInertialSolver(
        ny=1, nx=np.asarray(depth).shape[1], dx=dx, dy=dx,
        bed_elevation=np.zeros_like(depth, dtype=np.float64),
        manning_n=np.full_like(depth, 0.035, dtype=np.float64),
        active_mask=np.ones_like(depth, dtype=np.int32),
        device=device, structures=tuple(records),
    )
    solver.set_water_depth(np.asarray(depth, dtype=np.float64))
    return solver


def transfer(solver, dt=1.0):
    before = float(solver.get_water_depth().sum() * float(solver.dx) ** 2)
    solver._apply_structure_transfers(dt=dt)
    after = float(solver.get_water_depth().sum() * float(solver.dx) ** 2)
    return before, after


def test_culvert_forward_reverse_cap_dry_and_one_way():
    forward = make_solver([culvert(maximum_discharge_m3s=0.2)], depth=np.array([[1.0, 0.0]]))
    before, after = transfer(forward, dt=10.0)
    assert forward.get_structure_diagnostics()[0]["instantaneous_q_m3s"] == pytest.approx(0.2)
    assert after == pytest.approx(before)
    assert forward.get_water_depth()[0, 1] == pytest.approx(0.02)
    assert forward.get_structure_mass_balance_diagnostics()["internal_transfer_residual_m3"] == 0.0

    reverse = make_solver([culvert(maximum_discharge_m3s=0.2)], depth=np.array([[0.0, 1.0]]))
    transfer(reverse, dt=10.0)
    assert reverse.get_structure_diagnostics()[0]["flow_direction"] == "B_to_A"
    assert reverse.get_water_depth()[0, 0] == pytest.approx(0.02)

    dry = make_solver([culvert()], depth=np.zeros((1, 2)))
    transfer(dry)
    assert dry.get_structure_diagnostics()[0]["instantaneous_q_m3s"] == 0.0

    one_way = make_solver([culvert(one_way=True)], depth=np.array([[0.0, 1.0]]))
    transfer(one_way)
    assert one_way.get_structure_diagnostics()[0]["instantaneous_q_m3s"] == 0.0


def test_donor_limiter_and_shared_donor_conserve_without_negative_depth():
    first = culvert(id="c1", outlet_col=1, maximum_discharge_m3s=100.0)
    second = culvert(id="c2", outlet_col=2, maximum_discharge_m3s=100.0)
    solver = make_solver([first, second], depth=np.array([[0.01, 0.0, 0.0]]), dx=1.0)
    before, after = transfer(solver, dt=100.0)
    result = solver.get_water_depth()
    assert np.all(result >= 0.0)
    assert float(result.sum()) == pytest.approx(0.01, abs=1e-12)
    assert after == pytest.approx(before, abs=1e-12)
    diagnostics = solver.get_structure_diagnostics()
    assert all(item["limiter_activations"] == 1 for item in diagnostics)
    assert diagnostics[0]["net_transferred_volume_m3"] == pytest.approx(
        diagnostics[1]["net_transferred_volume_m3"]
    )


def test_weir_below_crest_free_flow_tailwater_and_reverse():
    closed = make_solver([weir()], depth=np.array([[0.4, 0.0]]))
    transfer(closed)
    assert closed.get_structure_diagnostics()[0]["instantaneous_q_m3s"] == 0.0

    open_weir = make_solver([weir()], depth=np.array([[1.5, 0.0]]))
    transfer(open_weir)
    q = open_weir.get_structure_diagnostics()[0]["instantaneous_q_m3s"]
    expected = 0.62 * (2.0 / 3.0) * np.sqrt(2.0 * 9.80665) * (1.0 ** 1.5)
    assert q == pytest.approx(expected)

    submerged = make_solver([weir()], depth=np.array([[1.5, 0.8]]))
    transfer(submerged)
    assert submerged.get_structure_diagnostics()[0]["instantaneous_q_m3s"] < q

    reverse = make_solver([weir()], depth=np.array([[0.0, 1.5]]))
    transfer(reverse)
    assert reverse.get_structure_diagnostics()[0]["flow_direction"] == "B_to_A"


def test_bridge_opening_is_internal_and_surface_overtopping_remains_separate():
    record = {
        "id": "bridge_test", "type": "bridge",
        "inlet_row": 0, "inlet_col": 0, "outlet_row": 0, "outlet_col": 2,
        "opening_width_m": 1.0, "opening_height_m": 1.0,
        "opening_invert_m": 0.0, "deck_elevation_m": 0.5,
        "length_m": 2.0, "manning_n": 0.03,
        "entrance_loss_coefficient": 0.5, "exit_loss_coefficient": 1.0,
    }
    bed = np.array([[0.0, 10.0, 0.0]])
    low_stage = make_solver([record], depth=np.array([[1.0, 0.0, 0.0]]))
    low_stage.bed_elevation = wp.array(bed, dtype=low_stage.bed_elevation.dtype, device=low_stage.device)
    low_stage.step(fixed_dt=0.001)
    assert low_stage.get_structure_diagnostics()[0]["instantaneous_q_m3s"] > 0.0
    assert low_stage.get_qx()[0, 1] == pytest.approx(0.0)  # below deck: no surface overtopping

    solver = make_solver([record], depth=np.array([[11.0, 0.0, 0.0]]))
    solver.bed_elevation = wp.array(bed, dtype=solver.bed_elevation.dtype, device=solver.device)
    initial_volume = float(solver.get_water_depth().sum() * 100.0)
    solver.step(fixed_dt=0.001)
    final_volume = float(solver.get_water_depth().sum() * 100.0)
    assert solver.get_structure_diagnostics()[0]["instantaneous_q_m3s"] > 0.0
    assert solver.get_qx()[0, 1] > 0.0  # road overtopping remains ordinary surface flow
    assert final_volume == pytest.approx(initial_volume, abs=1.0e-8)
    # The elevated road/deck terrain remains in the 2D domain.
    np.testing.assert_array_equal(solver.bed_elevation.numpy(), bed)


def test_structure_step_is_rejected_by_execution_paths_that_cannot_run_it():
    solver = make_solver([culvert()], depth=np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="host-stepped"):
        solver.run_device_loop(duration=1.0)
    with pytest.raises(ValueError, match="host-stepped"):
        solver.run_spatially_adaptive_loop(duration=1.0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device unavailable")
def test_structure_transfer_cpu_cuda_parity():
    initial = np.array([[0.75, 0.05]], dtype=np.float64)
    cpu = make_solver([culvert(maximum_discharge_m3s=0.1)], depth=initial.copy(), device="cpu")
    gpu = make_solver([culvert(maximum_discharge_m3s=0.1)], depth=initial.copy(), device="cuda:0")
    transfer(cpu, dt=2.0)
    transfer(gpu, dt=2.0)
    np.testing.assert_allclose(gpu.get_water_depth(), cpu.get_water_depth(), rtol=1e-12, atol=1e-12)
    assert gpu.get_structure_diagnostics()[0]["net_transferred_volume_m3"] == pytest.approx(
        cpu.get_structure_diagnostics()[0]["net_transferred_volume_m3"], rel=1e-12, abs=1e-12
    )













def test_empty_structure_path_keeps_legacy_explicit_state_bit_identical():
    plain = LocalInertialSolver(1, 2, 2.0, 2.0, np.zeros((1, 2)), np.full((1, 2), 0.035), np.ones((1, 2), np.int32))
    explicit_empty = LocalInertialSolver(1, 2, 2.0, 2.0, np.zeros((1, 2)), np.full((1, 2), 0.035), np.ones((1, 2), np.int32), structures=())
    initial = np.array([[0.2, 0.1]])
    plain.set_water_depth(initial.copy())
    explicit_empty.set_water_depth(initial.copy())
    plain.step(fixed_dt=0.01)
    explicit_empty.step(fixed_dt=0.01)
    np.testing.assert_array_equal(plain.get_water_depth(), explicit_empty.get_water_depth())
    np.testing.assert_array_equal(plain.get_qx(), explicit_empty.get_qx())
    np.testing.assert_array_equal(plain.get_qy(), explicit_empty.get_qy())


def test_subgrid_structure_heads_use_hypsometric_stage_not_coarse_bed_plus_depth():
    bed = np.full((1, 2), 5.0, dtype=np.float64)
    solver = LocalInertialSolver(
        ny=1, nx=2, dx=2.0, dy=2.0,
        bed_elevation=bed,
        manning_n=np.full((1, 2), 0.035),
        active_mask=np.ones((1, 2), dtype=np.int32),
        hydraulic_geometry_mode="traditional",
        subgrid_bed_elevation=np.zeros((2, 4), dtype=np.float64),
        subgrid_resolution_m=1.0,
        structures=(culvert(maximum_discharge_m3s=100.0),),
    )
    solver.set_water_depth(np.array([[0.1, 0.0]]))
    solver._apply_structure_transfers(dt=0.01)
    diagnostic = solver.get_structure_diagnostics()[0]
    assert diagnostic["maximum_head_difference_m"] == pytest.approx(0.1, abs=1e-10)
    assert diagnostic["instantaneous_q_m3s"] > 0.0


def test_structure_diagnostics_reset_with_hydraulic_state():
    solver = make_solver([culvert(maximum_discharge_m3s=0.1)], depth=np.array([[1.0, 0.0]]))
    transfer(solver, dt=1.0)
    assert solver.get_structure_diagnostics()[0]["net_transferred_volume_m3"] > 0.0
    solver.set_water_depth(np.array([[0.5, 0.0]]))
    diagnostic = solver.get_structure_diagnostics()[0]
    assert diagnostic["instantaneous_q_m3s"] == 0.0
    assert diagnostic["net_transferred_volume_m3"] == 0.0
    assert diagnostic["maximum_abs_q_m3s"] == 0.0


def test_ponds_separated_by_terrain_ridge_exchange_through_structure_links():
    records = [
        culvert(id="culvert_ponds", outlet_col=2, maximum_discharge_m3s=0.05),
        {
            "id": "weir_ponds", "type": "weir",
            "inlet_row": 0, "inlet_col": 0, "outlet_row": 0, "outlet_col": 2,
            "crest_elevation_m": 0.2, "crest_length_m": 1.0,
            "discharge_coefficient": 0.62, "weir_type": "broad_crested",
        },
    ]
    solver = make_solver(records, depth=np.array([[1.0, 0.0, 0.0]]), dx=1.0)
    solver.set_bed_elevation(np.array([[0.0, 100.0, 0.0]]))
    initial_volume = float(solver.get_water_depth().sum())
    solver.step(fixed_dt=0.01)
    depth = solver.get_water_depth()
    assert solver.get_qx()[0, 1] == pytest.approx(0.0)
    assert solver.get_qx()[0, 2] == pytest.approx(0.0)
    assert depth[0, 0] < 1.0
    assert depth[0, 2] > 0.0
    assert float(depth.sum()) == pytest.approx(initial_volume, abs=1.0e-10)
    assert all(item["instantaneous_q_m3s"] > 0 for item in solver.get_structure_diagnostics())
