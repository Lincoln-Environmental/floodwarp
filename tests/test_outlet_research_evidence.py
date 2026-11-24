import numpy as np
import pytest
from floodwarp.solver_advisor.model import load_model, recommend

from research.floodwarp_methods_paper.fidelity import (
    native_advisor,
    outlet_observables,
    outlet_results,
    repeat_costs,
    timing_repeats,
)


def test_revision_comparison_requires_the_same_exact_reference_and_reports_timestep_parity_separately():
    from research.floodwarp_methods_paper.analytical_revision import field_comparison

    original = {name: np.array([0.0, 1.0]) for name in
                ("x_m", "t_s", "exact_cell_averages", "exact_gauges", "model_centreline", "model_gauges", "accepted_dt_s")}
    revised = dict(original, accepted_dt_s=np.array([0.5, 0.5]))
    result = field_comparison(before=original, after=revised)
    assert result == dict(model_centreline_equal=True, model_gauges_equal=True, accepted_dt_s_equal=False)
    with pytest.raises(AssertionError, match="Reference differs"):
        field_comparison(before=original, after=dict(revised, exact_gauges=np.array([0.0, 1.01])))

def test_fixed_reach_report_rejects_displaced_section(tmp_path):
    import json
    from research.floodwarp_methods_paper import metrics, runner
    from research.floodwarp_methods_paper.fidelity.repaired_outlet_report import reach_comparison

    settings = dict(times_s=[0, 60], section_x_m=200, section_y_bounds_m=[10, 70])
    (tmp_path / "free_reach_settings.json").write_text(json.dumps(settings))
    (tmp_path / "mean_storms").mkdir()
    terrain = {"native_spacing": 1}
    (tmp_path / "mean_storms/terrain.json").write_text(json.dumps(terrain))
    contract = dict(settings=settings, solver="test-repair", physical_sha=runner.digest(terrain), device="cpu")
    folder = tmp_path / "free_reach" / runner.digest(contract)[:24]
    folder.mkdir(parents=True)
    (folder / "inputs.json").write_text(json.dumps(contract))
    sample = dict(start_s=0, end_s=60, section_x_m=200, south_m=10, north_m=70, grid_m=5,
                  extension_m=100, section_m3s=0.7, residual_m3=0)
    options = dict(directory=tmp_path, destination=tmp_path,
                   revisions={"repair": {"source_sha256": {"long-storms": "test-repair"}}})
    metrics.csv_write(file=folder/"comparison.csv", rows=[sample])
    reach_comparison(**options)
    assert metrics.csv_read(file=tmp_path/"incised_reach.csv")[0]["physical_sha"] == runner.digest(terrain)
    metrics.csv_write(file=folder/"comparison.csv", rows=[dict(sample, section_x_m=300)])
    with pytest.raises(ValueError, match="physical sections"):
        reach_comparison(**options)

def test_missing_incised_checkpoint_restoration_checks_fields_and_interval_transport(tmp_path, monkeypatch):
    import json
    from floodwarp.inputs import terrain_from_array
    from floodwarp.model import advance, create_model, save_checkpoint
    from research.floodwarp_methods_paper.fidelity import outlet_states

    base = tmp_path/"primary"
    base.mkdir()
    cache = tmp_path/"checkpoints"
    identity = dict(dx=5, cfl_factor=1, extension_m=0, settings={}, stopping={}, method="hancock")
    window = dict(start_s=0, end_s=1, frames=[dict(t_s=1, gauge=0.0, hydraulic_s=.9)])

    def build(*, path):
        terrain = terrain_from_array(elevation=np.zeros((3, 3)), dx=5., dy=5.,
                                     row_order="south_to_north", vertical_datum="local")
        model = create_model(terrain=terrain, solver_method="hancock", device="cpu", initial_depth=.1)
        advance(model=model, end_time=1.)
        save_checkpoint(model=model, path=path)

    def replay(*, run_dir, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        build(path=cache/run_dir.name/"1.npz")
        (run_dir/"1.json").write_text(json.dumps(dict(window, frames=[dict(window["frames"][0], hydraulic_s=3.0)])))

    build(path=tmp_path/"original.npz")
    proof = outlet_states.state_receipt(path=tmp_path/"original.npz", ident=base.name, end=1)
    (base/"state_1.json").write_text(json.dumps(proof))
    (base/"1.json").write_text(json.dumps(window))
    monkeypatch.setattr(outlet_states.outlet_storms, "trajectory", replay)
    options = dict(directory=tmp_path, base=base, identity=identity, missing=[1], cache=cache, control={})
    outlet_states.restore(**options)
    assert outlet_states.state_receipt(path=cache/"primary/1.npz", ident=base.name, end=1) == proof
    (base/"1.json").write_text(json.dumps(dict(window, frames=[dict(window["frames"][0], gauge=.01)])))
    with pytest.raises(ValueError, match="accepted hydraulic measurements"):
        outlet_states.restore(**options)


def test_spatial_cost_ends_at_the_snapshot():
    frames = [dict(t_s=t, steps=n, hydraulic_s=s) for t, n, s in
              ((60, 5, 0.25), (120, 4, 0.15), (180, 8, 0.76))]
    assert outlet_observables.window_cost(frames=frames, end_s=120) == (0.4, 9)
    with pytest.raises(ValueError, match="endpoint"):
        outlet_observables.window_cost(frames=frames, end_s=150)


def test_storm_timing_table_requires_distinct_matches_on_the_same_numerical_build():
    from research.floodwarp_methods_paper.fidelity.repaired_outlet_report import timing_table

    row = dict(primary="a", end_s="57600", gpu="board", cpu_model="cpu", warp="warp", python="python",
               precision="FP64", solver_sha="source", dx_m="10", method="hancock", extension_m="0",
               policy="tiles", steps="100", interval_s="60", hydraulic_s="2", end_to_end_s="")
    peers = [dict(row, id="a"), dict(row, id="b", interval_s="60.0", hydraulic_s="2.5", end_to_end_s="5"),
             dict(row, id="c", hydraulic_s="2.2", end_to_end_s="4")]
    rendered = "\n".join(timing_table(records=peers))
    assert "2.200" in rendered and "4.00–5.00 (2)" in rendered
    for replacement in (dict(peers[2], id="b"), dict(peers[2], solver_sha="other")):
        with pytest.raises(ValueError, match="three matched"):
            timing_table(records=[*peers[:2], replacement])

def test_extent_display_aggregates_fine_cell_areas_before_drawing(tmp_path):
    a = np.array([[.08, .0], [.0, .0]])
    b = np.zeros_like(a)
    outlet_results.write_contrast_map(
        destination=tmp_path, first=dict(run_id="fv1", footprint_width_m=10), second=dict(run_id="hancock"),
        dx=5, bin_m=10, hour=8, footprint=np.array([[True, True], [False, False]]), soil_delta=a-b,
        disagreements={5: ((a > .05) & (b <= .05), (b > .05) & (a <= .05))},
    )
    with np.load(tmp_path/"hour_eight_contrasts.npz") as fields:
        assert fields["fv1_only_5cm"].item() == .25 and fields["hancock_only_5cm"].item() == 0
        assert fields["rain_fraction"].item() == .5
        assert fields["infiltration_difference_m"].item() == .02


def test_simultaneous_extent_distinguishes_moving_wave_from_cumulative_footprint():
    frames = {t: dict(area_above_5cm_m2=area, ever_above_5cm_m2=envelope,
                      area_above_10cm_m2=0, ever_above_10cm_m2=0)
              for t, area, envelope in ((0, 0, 0), (900, 100, 100), (1800, 100, 200), (2700, 25, 225))}
    args = dict(case={"run_id": "fixture"}, end=2700, cadence=900, thresholds_cm=(5, 10))
    samples, (wet, dry) = outlet_results.extent_history(frames=frames, **args)
    assert len(samples) == 4 and (wet["largest_sampled_area_m2"], wet["area_ever_flooded_m2"]) == (100, 225)
    assert wet["first_peak_h"] == .25 and dry["first_peak_h"] is None
    frames[2700]["ever_above_5cm_m2"] = 125
    with pytest.raises(ValueError, match="envelope"):
        outlet_results.extent_history(frames=frames, **args)


def test_recession_export_difference_includes_storage_at_rain_cessation():
    def account(*, storage, inside, outside, gauge):
        return dict(footprint_storage_m3=storage, footprint_rain_m3=100, gauge=gauge,
                    footprint_infiltration_m3=inside+outside, infiltration_inside_rain_m3=inside,
                    infiltration_outside_rain_m3=outside)

    cases = []
    for method, start, stop in (
        ("fv1", account(storage=40, inside=5, outside=10, gauge=45), account(storage=5, inside=7, outside=13, gauge=75)),
        ("hancock", account(storage=25, inside=5, outside=18, gauge=52), account(storage=0, inside=7, outside=26, gauge=67)),
    ):
        cases.append(outlet_results.phase_account(case=dict(run_id=method, method=method, dx_m=5, extension_m=0),
                                                   start=start, stop=stop))
    row, = outlet_results.water_differences(summaries=cases)
    assert row["initial_storage_m3"] == 15 and row["gauge_m3"] == 15
    assert row["unexplained_export_m3"] == 0 and row["infiltration_outside_rain_m3"] == -5
def test_repeat_selection_covers_cross_grid_decisions_on_the_same_snapshot():
    from research.floodwarp_methods_paper.fidelity.outlet_timing import marginal_decisions

    common = dict(case="fixed gauge", representation="no buffer", reference_id="fine",
                  station="domain", objective="depth_rmse_m", cap="0", interval_s="60", end_h=8,
                  eligible="True", reference_sensitivity="0.002")
    coarse = dict(common, method="hancock_adaptive", dx="10", runtime_s="102", absolute_error="0.004",
                  experiment_id="coarse")
    fine = dict(common, method="fv1", dx="5", runtime_s="100", absolute_error="0.008",
                experiment_id="fine-candidate", interval_s="60.0")
    later = dict(coarse, end_h=16, experiment_id="other-endpoint")
    unresolved = dict(coarse, method="adaptive", eligible="False", experiment_id="unqualified")
    result = marginal_decisions(frontier=[coarse, fine, later, unresolved],
                                tolerances={"depth_rmse_m": [0.01]}, fraction=0.05)
    assert result == {"coarse", "fine-candidate"}


def test_transfer_inventory_retains_contract_for_selected_run(tmp_path):
    from research.floodwarp_methods_paper.fidelity.orton_transfer import inventory

    physical = dict(directory=tmp_path, identity={"dem": "fixed"},
                    settings={"methods": ["fv1", "hancock", "rk2"], "grids_m": [20, 10, 5]},
                    native_shape=(1123, 843), case="orton-*")
    all_runs = inventory(**physical, method=None, spacing=None)
    entry, = inventory(**physical, method="hancock", spacing=10)
    assert len(all_runs) == 9 and entry in all_runs
    assert (entry["ny"], entry["nx"]) == (562, 422)
    folder = tmp_path / "orton_transfer" / entry["id"]
    folder.mkdir(parents=True)
    (folder / "hour_4.json").touch()
    resumed, = inventory(**physical, method="hancock", spacing=10)
    assert resumed["id"] == entry["id"] and resumed["archived_windows"] == 1
    assert not resumed["complete"]


def test_transfer_figure_reads_the_observed_times_and_sections(tmp_path):
    import json
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.figures.generate import orton_transfer

    folder = tmp_path / "fidelity/orton_transfer"
    sample = folder / "fixture"
    sample.mkdir(parents=True)
    (folder.parent / "orton_transfer_settings.json").write_text(json.dumps(dict(gauges_m=[20], checkpoint_h=1)))
    (sample / "inputs.json").write_text(json.dumps(dict(gauges_above_southern_origin_m=[20])))
    record = dict(id="fixture", method="fv1", dx_m=5, duration_h=1, interval_s=1800, peak_m3s=3)
    metrics.csv_write(file=folder / "transfer.csv", rows=[record])
    frames = [dict(t_s=t, net_outlet_m3s=q, gauge_m3s=[q / 2]) for t, q in ((1800, 3), (3600, 1))]
    (sample / "hour_1.json").write_text(json.dumps(dict(frames=frames)))
    out = tmp_path / "figures"
    orton_transfer(study=tmp_path, out=out)
    assert all((out / f"20_orton_transfer.{ext}").is_file() for ext in ("svg", "png", "pdf"))
    metrics.csv_write(file=folder / "transfer.csv", rows=[dict(record, peak_m3s=4)])
    with pytest.raises(ValueError, match="peak"):
        orton_transfer(study=tmp_path, out=out)


def test_historical_transfer_uses_identical_windows_and_checks_accepted_volume(tmp_path):
    import json
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import orton_controls

    release = tmp_path / "release"
    folder = tmp_path / "transfer"
    sample = folder / "run"
    sample.mkdir(parents=True)
    archive = release / "outputs/orton_bradley_hydrographs_72h.csv"
    archive.parent.mkdir(parents=True)
    measurements = []
    for start, end, flow, volume in ((0, 1800, 1, 1800), (1800, 3600, 2, 5400), (3600, 5400, 4, 12600)):
        row = dict(interval_start_s=start, interval_end_s=end)
        for method in ("floodwarp_fv1", "lisflood_fv1", "floodwarp_li"):
            row[method + "_mean_net_outlet_m3_s"] = flow
            row[method + "_cumulative_net_export_m3"] = volume
        measurements.append(row)
    metrics.csv_write(file=archive, rows=measurements)
    entry = dict(id="run", method="hancock", dx_m=5, duration_h=1, interval_s=1800,
                 boundary_inflow=100, boundary_outflow=5500)
    frames = [dict(t_s=1800, net_outlet_m3s=2), dict(t_s=3600, net_outlet_m3s=1)]
    (sample / "hour_1.json").write_text(json.dumps(dict(frames=frames)))
    options = dict(release=release, destination=folder, settings=dict(cadence_s=1800, checkpoint_h=1))
    orton_controls.compare(**options, rows=[entry])
    contrasts = metrics.csv_read(file=folder / "historical_transfer.csv")
    assert len(contrasts) == 3
    assert all(float(row["net_export_difference_m3"]) == 0 and float(row["peak_end_difference_s"]) == -1800
               and float(row["hydrograph_rms_difference_m3s"]) == 1 for row in contrasts)
    with pytest.raises(AssertionError):
        orton_controls.compare(**options, rows=[dict(entry, boundary_inflow=0)])
    metrics.csv_write(file=archive, rows=[measurements[0], measurements[2]])
    with pytest.raises(AssertionError):
        orton_controls.compare(**options, rows=[entry])


def test_transfer_preserves_failed_and_resumed_wall_timings(tmp_path):
    import json
    from research.floodwarp_methods_paper.fidelity.orton_transfer import measured_attempt

    options = dict(folder=tmp_path, ident="fixture", machine={"gpu": "fixture"})
    with pytest.raises(ValueError, match="stop"):
        with measured_attempt(**options):
            (tmp_path / "hour_4.json").touch()
            raise ValueError("stop")
    with measured_attempt(**options):
        (tmp_path / "complete.json").write_text("{}")
    failed, = tmp_path.glob("attempt_*/failed.json")
    completed, = tmp_path.glob("attempt_*/completed.json")
    assert failed.parent != completed.parent
    assert "ValueError: stop" in json.loads(failed.read_text())["traceback"]
    assert json.loads(completed.read_text())["uninterrupted_from_start"] is False


def test_publication_inventory_requires_incised_decisions_and_measured_orton_products(tmp_path):
    from research.floodwarp_methods_paper.analysis import verify_publication_inventory

    for group in ("outlet_results/events.csv", "orton_transfer/transfer.csv"):
        path = tmp_path / "fidelity" / group
        path.parent.mkdir(parents=True)
        path.touch()
    generated = set()
    for category, numbers, formats in (("tables", [*range(1, 14), *range(16, 21)], ("md", "csv", "tex")),
                                       ("figures", range(1, 21), ("svg", "pdf", "png"))):
        folder = tmp_path / category
        folder.mkdir()
        for number in numbers:
            name = f"{number:02d}_product"
            for extension in formats:
                (folder / f"{name}.{extension}").touch()
            if category == "tables":
                generated.add(name)
    assert verify_publication_inventory(study=tmp_path, rendered=generated) == (18, 20)
    (tmp_path/"figures/20_product.svg").unlink()
    with pytest.raises(ValueError, match="Incomplete svg"):
        verify_publication_inventory(study=tmp_path, rendered=generated)


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


def test_lost_connection_cannot_supply_development_hydraulic_error():
    row = dict(features=dict(critical_connection_lost=True), errors={"peak_discharge": 0.1},
               qualified={"peak_discharge": True}, runtime_s=5.0)
    training, = native_advisor.admissible_training(rows=[row])
    assert training["runtime_s"] == 5.0 and training["errors"]["peak_discharge"] is None
    assert training["qualified"] == {"peak_discharge": False}
    assert row["errors"] == {"peak_discharge": 0.1}


def test_joint_oracle_excludes_unqualified_numeric_error():
    from research.floodwarp_methods_paper.fidelity.paired_advisor import evaluate_pairs

    records = [dict(case="event", terrain_id="terrain", native_sha="inputs", end_h=16,
                    method="fv1", run_id=str(dx), features=dict(resolution=dx),
                    interval_s=cadence, errors={"depth": 0.001}, qualified={"depth": dx == 5},
                    runtime_s=cost) for dx, cadence, cost in ((10, "60", 10), (5, "60.0", 20))]
    trials = [dict(case="event", objective="depth", rule="oracle", dx=dx, tolerance=.01,
                   status="oracle", terrain_admissible=True) for dx in (5, 10)]
    rows = evaluate_pairs(model={}, rows=records, trials=trials, spacings=[5, 10])
    assert len(rows) == 2 and all(r["optimal_dx"] == r["chosen_dx"] == 5 for r in rows)


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


def test_remeasure_an_accuracy_improvement_just_outside_an_fv1_budget():
    points = [
        dict(method=m, dx=dx, runtime_s=s, eligible=eligible, absolute_error=error)
        for m, dx, s, eligible, error in (
            ("fv1", 5, 100, "False", ""), ("hancock_dense", 10, 90, "True", 2),
            ("adaptive", 10, 104, "True", 0.5), ("euler_mc", 10, 99, "True", 3),
        )
    ]
    assert timing_repeats.marginal_budget_options(rows=points, relative_gap=0.05) == {
        ("fv1", 5), ("adaptive", 10)
    }


def test_budget_keeps_close_errors_unresolved_at_reference_sensitivity(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import budgets

    rows = [dict(method=m, experiment_id=m, dx=10, runtime_s=t, eligible=True,
                 absolute_error=e, reference_sensitivity=0.4, reference_id="fine",
                 objective="peak_bias_pct", execution_mode="dense", end_h=16, interval_s=60)
            for m, t, e in (("fv1", 10, 1.0), ("hancock_dense", 9, 1.5), ("adaptive", 8, 3.0))]
    budgets.feasible_cost(rows=rows, target=tmp_path, tolerances=[5], spacings=[10])
    selected, = metrics.csv_read(file=tmp_path / "equal_cost_fidelity.csv")
    assert selected["selected_id"] == "fv1"
    assert selected["possible_best_ids"].split() == ["fv1", "hancock_dense"]
    assert selected["ranking_status"] == "overlapping error ranges"


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
def test_internal_mask_edge_transmits_only_at_explicit_free_contact(method, mode, device):
    import warp as wp
    from floodwarp.inputs import boundary_faces, terrain_from_array
    from floodwarp.model import advance, create_model

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    dx, grade, n, depth = 10., .07329, .035, .03
    active = np.zeros((5, 36), dtype=bool)
    active[1, :32] = active[3, :32] = True
    speed = depth**(2/3) * np.sqrt(grade) / n
    bed = np.broadcast_to(-grade * (np.arange(36) + .5) * dx, active.shape)
    terrain = terrain_from_array(elevation=bed, active=active, dx=dx, dy=dx,
                                 row_order="south_to_north", vertical_datum="local")
    bx, by = boundary_faces(terrain=terrain)
    bx[1, 32] = 2
    model = create_model(terrain=terrain, solver_method=method, limiter="adaptive", execution_mode=mode,
                         device=device, initial_depth=np.where(active, depth, 0.), manning_n=n,
                         boundary_qx=bx, boundary_qy=by, explicit_maximum_dt=.05)
    s = model.solver
    s.set_momentum(hu=active * (depth * speed), hv=np.zeros_like(bed))
    advance(model=model, end_time=5.)
    start_volume = s.get_cumulative_face_volumes()["qx"][1, 32]
    advance(model=model, end_time=10.)
    volume = s.get_cumulative_face_volumes()
    assert (volume["qx"][1, 32] - start_volume) / 5 == pytest.approx(dx * depth * speed, rel=.01)
    assert volume["qx"][3, 32] == 0
    np.testing.assert_array_equal(volume["qy"], np.zeros_like(volume["qy"]))
    assert not np.any(s.get_water_depth()[~active])
    ledger = s.cumulative_accounting()
    assert ledger["boundary_outflow"] == pytest.approx(volume["qx"][1, 32], abs=1e-10)
    assert abs(ledger["residual"]) < 1e-9


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


@pytest.mark.parametrize("method,mode", (("hancock", "dense"), ("hancock", "tiles"), ("fv2", "dense")))
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_moving_subcritical_reach_responds_to_tailwater(method, mode, device):
    import warp as wp
    from floodwarp.inputs import boundary_faces, terrain_from_array
    from floodwarp.model import advance, create_model

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    shape = (4, 32)
    depth, speed, dx, duration = 0.75, 0.2, 1.0, 2.0
    terrain = terrain_from_array(elevation=np.zeros(shape), dx=dx, dy=dx,
                                 row_order="south_to_north", vertical_datum="local")
    bx, by = boundary_faces(terrain=terrain)
    bx[:, [0, -1]] = 3
    # Both lateral faces are FREE with zero normal flow in the exact uniform state.
    by[[0, -1], :] = 2
    roughness, gravity, step = 0.035, 9.81, 0.05
    drag = gravity * roughness**2 / depth**(4/3)
    section = depth * shape[0] * dx
    # Uniform flat-bed flow obeys du/dt = -drag*u². With equal stages the
    # accepted face transport is a midpoint quadrature of its velocity.
    exact = section * np.log1p(drag * speed * duration) / drag
    midpoint_bound = section * duration * step**2 * (2 * drag**2 * speed**3) / 24
    for increase in (0.0, 0.05):
        stages = np.full(bx.shape, depth)
        stages[:, -1] += increase
        model = create_model(
            terrain=terrain, solver_method=method, execution_mode=mode, device=device,
            manning_n=roughness, gravity=gravity, limiter="adaptive", explicit_maximum_dt=step,
            initial_depth=depth, boundary_qx=bx, boundary_qy=by, stage_qx=stages,
            stage_datum="local", stage_boundary_momentum="velocity_extrapolation",
        )
        model.solver.set_momentum(hu=np.full(shape, depth * speed), hv=np.zeros(shape))
        advance(model=model, end_time=duration)
        flux = model.solver.get_cumulative_face_volumes()["qx"].sum(axis=0)
        water = model.solver.get_water_depth()
        if increase == 0:
            np.testing.assert_allclose(flux, exact, atol=midpoint_bound + 2e-13, rtol=0)
            np.testing.assert_allclose(water, depth, atol=2e-14, rtol=0)
        else:
            assert flux[-1] < exact and flux[-2] < exact
            assert water[:, -2].min() > depth
        assert abs(model.solver.cumulative_accounting()["residual"]) < 1e-11
