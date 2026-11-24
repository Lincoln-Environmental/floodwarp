import numpy as np
import pytest
from floodwarp.solver_advisor.model import load_model, recommend

from research.floodwarp_methods_paper.fidelity import (
    native_advisor,
    outlet_results,
    repeat_costs,
    timing_repeats,
)


def test_reference_pair_needs_two_matched_events_on_independent_terrains():
    rows = [
        dict(
            objective="peak",
            terrain_id=terrain,
            dx=dx,
            case=case,
            chosen="fv1",
            feasible=True,
        )
        for terrain, dx, case in (
            ("a", 5, "storm-a"),
            ("a", 10, "storm-a"),
            ("b", 5, "storm-b1"),
            ("b", 10, "storm-b2"),
        )
    ]
    assert native_advisor.validated_pairs(rows=rows, minimum=2) == {}
    rows[-1]["case"] = "storm-b1"
    assert native_advisor.validated_pairs(rows=rows, minimum=2) == {"peak": [[5, 10]]}


def test_repeat_summary_keeps_independent_completed_measurements(tmp_path):
    event = dict(
        case="wet",
        method="fv1",
        execution_mode="sparse",
        dx_m=5,
        factor=1.0,
        interval_s=60,
        end_h=16.0,
        native_sha="dem",
        solver_source="solver",
        hardware="GPU",
        steps=30,
        net_outlet_m3=100.0,
        infiltration_m3=3.0,
        storage_m3=0.0,
        outlet_peak_m3s=1.0,
    )
    rows = [
        dict(
            event,
            horizon=cohort,
            source_id=str(i),
            timing_repeat=i,
            hydraulic_s=t,
            end_to_end_s=t + 1,
        )
        for i, (cohort, t) in enumerate(
            (
                ("complete-native-metre", 8.0),
                ("complete-native-repeat-1", 6.0),
                ("complete-native-repeat-2", 7.0),
            )
        )
    ]
    repeats = repeat_costs.collect(rows=rows, target=tmp_path)
    assert repeats["0"]["hydraulic_median_s"] == 7
    assert repeats["0"]["end_to_end_median_s"] == 8
    assert repeats["0"]["has_three_timings"]
    rows[1]["net_outlet_m3"] -= 1
    with pytest.raises(ValueError, match="net_outlet_m3"):
        repeat_costs.collect(rows=rows, target=tmp_path)


def test_marginal_policy_winners_are_repeated_together():
    assert timing_repeats.close_options(
        points={"dense": 10.0, "tiles": 10.1, "auto": 12.0}, relative_gap=0.05
    ) == {"dense", "tiles"}
    assert not timing_repeats.close_options(
        points={"dense": 9.0, "tiles": 10.0}, relative_gap=0.05
    )


@pytest.mark.parametrize(
    "objective", ("cumulative_infiltration", "net_exported_volume")
)
def test_historical_advisor_abstains_on_new_objectives(objective):
    model = load_model()
    decision = recommend(
        model=model, features=model["training"][0]["features"], objective=objective
    )
    assert decision["status"] in ("insufficient_evidence", "refine_terrain")
    assert decision["method"] is None


def test_export_difference_closes_inside_and_outside_rain_footprint():
    base = dict(
        method="hancock_adaptive",
        run_id="h",
        dx_m=5,
        extension_m=0,
        initial_storage_m3=10.0,
        footprint_rain_m3=50.0,
        infiltration_inside_rain_m3=9.0,
        infiltration_outside_rain_m3=13.0,
        footprint_infiltration_m3=22.0,
        footprint_storage_m3=2.0,
        gauge_m3=36.0,
    )
    peer = dict(
        base,
        method="fv1",
        run_id="f",
        infiltration_inside_rain_m3=10.0,
        infiltration_outside_rain_m3=8.0,
        footprint_infiltration_m3=18.0,
        footprint_storage_m3=3.0,
        gauge_m3=39.0,
    )
    (delta,) = outlet_results.water_differences(summaries=[base, peer])
    assert delta["first_method"] == "fv1"
    assert delta["gauge_m3"] == 3
    assert delta["unexplained_export_m3"] == 0


def test_spatial_qualification_requires_independent_time_control():
    coarse, fine = np.ones((2, 2)), np.ones((4, 4))
    peers, states = [], {}
    for ident, method, dx, cfl, steps in (
        ("coarse", "adaptive", 10, 1.0, 20),
        ("fine", "adaptive", 5, 1.0, 40),
        ("cross", "hancock_adaptive", 5, 1.0, 40),
        ("time", "adaptive", 5, 0.5, 40),
    ):
        peers.append(
            dict(
                run_id=ident,
                method=method,
                dx_m=dx,
                cfl_factor=cfl,
                steps=steps,
                extension_m=0,
            )
        )
        field = coarse if dx == 10 else fine
        states[ident, dx] = (field, field)
    options = dict(
        peers=peers,
        states=states,
        hour=8.0,
        thresholds=(0.05, 0.1),
        settings=dict(
            depth_floor_m=0.01,
            depth_relative=0.05,
            maximum_depth_m=0.03,
            extent_error=0.05,
        ),
    )
    rows = outlet_results.spatial_qualification(**options)
    assert not any(row["qualified"] for row in rows)
    peers[-1]["steps"] = 80
    rows = outlet_results.spatial_qualification(**options)
    assert all(row["qualified"] == (row["run_id"] == "coarse") for row in rows)


@pytest.mark.parametrize(
    "method,mode", (("hancock", "dense"), ("hancock", "tiles"), ("fv2", "dense"))
)
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_sloping_lake_with_stage_and_opposite_free_face(method, mode, device):
    import warp as wp
    from floodwarp.inputs import boundary_faces, terrain_from_array
    from floodwarp.model import advance, create_model

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    dx = 5.0
    x = (np.arange(32) + 0.5) * dx
    y = np.arange(4)[:, None] * dx
    bed = -0.002 * x + 0.0001 * y
    level = 0.7
    terrain = terrain_from_array(
        elevation=bed, dx=dx, dy=dx, row_order="south_to_north", vertical_datum="local"
    )
    bx, by = boundary_faces(terrain=terrain)
    bx[:, 0] = 3
    bx[:, -1] = 2
    model = create_model(
        terrain=terrain,
        solver_method=method,
        execution_mode=mode,
        device=device,
        manning_n=0.035,
        limiter="adaptive",
        explicit_maximum_dt=0.1,
        initial_depth=level - bed,
        boundary_qx=bx,
        boundary_qy=by,
        stage_qx=np.full_like(bx, level, dtype=float),
        stage_datum="local",
    )
    advance(model=model, end_time=5.0)
    solver = model.solver
    np.testing.assert_allclose(
        solver.get_water_depth(), level - bed, atol=3.0e-14, rtol=0
    )
    np.testing.assert_allclose(
        solver.get_conservative_state()[..., 1:], 0.0, atol=3.0e-14, rtol=0
    )
    np.testing.assert_allclose(
        solver.get_cumulative_face_volumes()["qx"][:, -1], 0.0, atol=1.0e-12, rtol=0
    )
