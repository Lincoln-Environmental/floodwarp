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


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_local_inertia_recession_records_accepted_model_time(tmp_path, monkeypatch, device):
    import json
    import warp as wp
    from floodwarp.inputs import terrain_from_array
    from floodwarp.model import create_model, advance, model_time
    from research.floodwarp_methods_paper.fidelity import continuation, recession

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    terrain = terrain_from_array(elevation=np.zeros((6, 6)), dx=5., dy=5., row_order="south_to_north",
                                 vertical_datum="local")
    model = create_model(terrain=terrain, solver_method="li", device=device, initial_depth=.02,
                         track_cumulative_face_volumes=True)
    advance(model=model, end_time=1.)
    (tmp_path / "spatial_settings.json").write_text(json.dumps(dict(interval_s=1, support_m={})))
    frames = []

    def observe(**kwargs):
        frames.extend(recession.advance_frames(model=model, spec=dict(method="li", interval=1, coastal_stage=False),
                                              start=1, end=3, options={}, device=device))

    monkeypatch.setattr(recession, "run", observe)
    continuation.extend(directory=tmp_path, settings=dict(device=device), cohorts=("native-metre",))
    assert [r["t_s"] for r in frames] == [2, 3] and model_time(model=model) == 3
    assert all(r["wet_fraction"] == 1 and r["section_m3s"] == 0 and r["steps"] > 0 for r in frames)
    assert all((r["live_pool_mib"] is None) == (device == "cpu") for r in frames)


def test_native_archive_resume_checks_entire_contract_and_spatial_fields(tmp_path):
    import json
    from research.floodwarp_methods_paper import runner
    from research.floodwarp_methods_paper.fidelity.native_resume import previous_prefix

    old = dict(spec={"case": "plain"}, solver="s", sources={"record.py": "old"}, hardware=["board"])
    ident = runner.digest(old)[:24]
    output = tmp_path / "cases"
    folder = output / ident
    attempt = folder / "attempt_1"
    attempt.mkdir(parents=True)
    path = folder / "result.json"
    path.write_text(json.dumps(dict(inputs=old, data={"steps": 10})))
    field = folder / "spatial.npz"
    np.savez(field, depth=[0.01])
    (attempt / "complete.json").write_text(json.dumps(dict(result_sha256=runner.checksum(path),
                                                            spatial_sha256=runner.checksum(field))))
    (tmp_path / "native_prefix_compatibility.json").write_text(json.dumps(dict(
        case="plain", prefix_ids=[ident], sources={"record.py": {"original": "old", "repaired": "new"}})))
    forecast = dict(old, sources={"record.py": "new"})
    options = dict(directory=tmp_path, output=output)
    assert previous_prefix(**options, definition=forecast)["id"] == ident
    assert previous_prefix(**options, definition=dict(forecast, solver="changed")) is None
    assert previous_prefix(**options, definition=dict(forecast, hardware=["other"])) is None
    np.savez(field, depth=[0.02])
    with pytest.raises(ValueError, match="fields"):
        previous_prefix(**options, definition=forecast)

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


def test_advisor_schedule_keeps_half_cfl_reference_receipts_separate_from_candidates():
    from research.floodwarp_methods_paper.fidelity.advisor_design import measured_schedule

    base = dict(case="same", dx_m="1", method="adaptive", factor="1", solver_source="current",
                horizon="complete-native-metre", end_h="28", source_id="normal")
    half = dict(base, factor=".5", source_id="half")
    measured = measured_schedule(windows=[
        base, half, dict(half, horizon="original_window", end_h=8, source_id="prefix"),
        dict(half, solver_source="archived"), dict(base, timing_repeat="1", source_id="repeat"),
    ], numerical_source="current")
    assert measured == {
        ("same", 1, "adaptive", 1, "complete-native-metre"): ("normal", 28.),
        ("same", 1, "adaptive", .5, "complete-native-metre"): ("half", 28.),
        ("same", 1, "adaptive", .5, "original_window"): ("prefix", 8.),
    }


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
    (folder / "complete.json").touch()
    orphan, = inventory(**physical, method="hancock", spacing=10)
    assert not orphan["complete"]


def test_native_frontier_includes_water_accounts_and_depth_at_matched_cadence(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity.native_publication import measured_fronts

    folder = tmp_path / "native_results"
    folder.mkdir()
    output = tmp_path / "figures"
    output.mkdir()
    run = dict(cohort="complete", case="storm", representation="native", reference_id="same",
               interval_s="60", end_h=16, station="outlet", eligible=True, nondominated=True,
               runtime_s=1, absolute_error=2, formulation="FV1", execution_mode="sparse", dx=5)
    metrics.csv_write(file=folder / "hydrograph_pareto.csv", rows=[
        dict(run, objective="net_export_bias_pct"),
        dict(run, interval_s="60.0", objective="infiltration_bias_pct", eligible=False, nondominated=None),
    ])
    metrics.csv_write(file=folder / "recession_depth_pareto.csv", rows=[
        dict(run, station="domain", interval_s="60.0", objective="depth_rmse_m", absolute_error=.001),
    ])
    counts = measured_fronts(directory=tmp_path, output=output)
    assert {r["metric"]: r["eligible"] for r in counts} == {
        "net_export_bias_pct": 1, "infiltration_bias_pct": 0, "depth_rmse_m": 1,
    }
    assert len(list(output.glob("*_pareto.pdf"))) == 1


def test_automatic_schedule_choice_cost_is_distinct_from_run_variability():
    from research.floodwarp_methods_paper.fidelity.native_publication import selection_costs

    event = dict(cohort="complete-native-metre", case="wet", dx_m=10, end_h="28", dense_id="d",
                 tiled_id="t", auto_id="a")
    auto = dict(horizon=event["cohort"], source_id="a", resolved_execution_mode="dense", method="hancock_auto")
    samples = [dict(horizon="completed recession", run_ids=ids, independent_timings=3,
                    hydraulic_median_s=cost, hydraulic_min_s=cost-.1, hydraulic_max_s=cost+.1)
               for ids, cost in (("a a1 a2", 11.9), ("d d1 d2", 12.), ("t t1 t2", 10.))]
    result, = selection_costs(events=[event], windows=[auto], timing=samples)
    assert result["fastest_explicit"] == "tiles" and result["chosen"] == "dense"
    assert result["chosen_over_fastest"] == 1.2
    assert result["auto_over_chosen"] == pytest.approx(11.9 / 12.)
    assert result["tiles_run_ids"] == "t t1 t2"


def test_recession_without_rk2_retains_its_real_endpoint_and_no_reference(tmp_path):
    from pathlib import Path
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import final_depth, native_publication

    study = Path(__file__).resolve().parents[1] / "research/floodwarp_methods_paper/fidelity"
    target = tmp_path / "native_results"
    target.mkdir()
    forecasts = [dict(id="measured-fv1", factor=1, d=dict(hydraulic_s=138., steps=74331),
                      s=dict(method="fv1", dx=10, interval=300, hours=28))]
    final_depth.process(groups={("complete-native-metre", "floodplain", "native"): forecasts},
                        directory=study, target=target, settings={"thresholds_m": [.05, .1]})
    rows = metrics.csv_read(file=target / "recession_depth_pareto.csv")
    assert rows and all(r["end_h"] == "28" and r["reference_id"] == ""
                        and r["eligible"] == "False" for r in rows)
    metrics.csv_write(file=target / "hydrograph_pareto.csv", rows=[])
    counts = native_publication.measured_fronts(directory=tmp_path, output=tmp_path)
    assert counts and all(r["eligible"] == 0 and r["end_h"] == 28 for r in counts)


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


def test_progress_report_pairs_the_same_gauge_window_and_timestep_policy(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity.progress import repaired_incised

    folder = tmp_path / "outlet_results"
    folder.mkdir()
    base = dict(dx_m=10, method="adaptive", cfl_factor=1, end_h=16, extension_m=0,
                minute_peak_m3s=42, gauge_m3=100, outer_boundary_m3=50)
    rows = [base, dict(base, extension_m=400, minute_peak_m3s=41, gauge_m3=99),
            dict(base, extension_m=400, end_h=20, minute_peak_m3s=80),
            dict(base, extension_m=400, minute_peak_m3s=70, cfl_factor=0.5)]
    metrics.csv_write(file=folder / "events.csv", rows=rows)
    report = repaired_incised(directory=tmp_path)
    assert [line for line in report if line.startswith("| 10 ")] == [
        "| 10 | FV2 RK2 | 16 | 400 | 42.000 | 41.000 | 100 | 99 |"
    ]


def test_transfer_geometry_keeps_gauges_fixed_across_represented_outlet_changes(tmp_path):
    from research.floodwarp_methods_paper import runner
    from research.floodwarp_methods_paper.fidelity.orton_controls import matched_geometry

    rows = []
    for method, dx, width in (("fv1", 5, 1060), ("fv1", 10, 1110), ("hancock", 10, 1110)):
        identity = dict(method=method, dx=dx, dem="native", inputs="forcing", spatial_reference={"south_m": 0})
        run_id = runner.digest(identity)[:24]
        rows.append(dict(id=run_id, method=method, dx_m=str(dx)))
        manifest = dict(identity=identity, geometry=dict(stage_width_m=width),
                        gauges_above_southern_origin_m=[2000, 4000])
        path = tmp_path / run_id / "inputs.json"
        runner.write_json(path=path, data=manifest)
    assert matched_geometry(directory=tmp_path, rows=rows)[10]["stage_width_m"] == 1110
    runner.write_json(path=path, data=dict(manifest, gauges_above_southern_origin_m=[2005, 4000]))
    with pytest.raises(ValueError, match="gauge positions"):
        matched_geometry(directory=tmp_path, rows=rows)
    runner.write_json(path=path, data=dict(manifest, geometry=dict(stage_width_m=1100)))
    with pytest.raises(ValueError, match="different represented outlet geometry"):
        matched_geometry(directory=tmp_path, rows=rows)


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


def test_transfer_resume_detects_changed_internal_gauge_observation(tmp_path):
    import json
    from research.floodwarp_methods_paper import metrics, runner
    from research.floodwarp_methods_paper.fidelity import orton_transfer

    identity = dict(dx=10, method="rk2", settings={"checkpoint_h": 4, "cadence_s": 60})
    ident = runner.digest(identity)[:24]
    folder = tmp_path / ident
    runner.write_json(path=folder / "inputs.json", data={"identity": identity, "forcing": {"duration_h": 4}})
    window = folder / "hour_4.json"
    runner.write_json(path=window, data={"gauge_m3s": [5., 3.]})
    fields = orton_transfer.observation_checksums(folder=folder, hours=4, checkpoint_h=4)
    entry = dict(id=ident, dx_m=10, method="rk2", interval_s=60, duration_h=4, observation_sha256=runner.digest(fields))
    runner.write_json(path=folder / "complete.json", data=entry)
    with pytest.raises(ValueError, match="no completed execution receipt"):
        orton_transfer.read_complete(folder=folder, identity=identity)
    with orton_transfer.measured_attempt(folder=folder, machine={"gpu": "reference"}, ident=ident):
        pass
    receipt, = folder.glob("attempt_*/completed.json")
    assert json.loads(receipt.read_text())["restored_unconfirmed_summary"] is True
    assert orton_transfer.read_complete(folder=folder, identity=identity) == entry
    metrics.csv_write(file=tmp_path / "transfer.csv", rows=[entry])
    assert orton_transfer.verify_index(directory=tmp_path, required_ids={ident}) == 1
    with pytest.raises(ValueError, match="Missing completed transfer forecasts"):
        orton_transfer.verify_index(directory=tmp_path, required_ids={ident, "missing"})
    runner.write_json(path=folder / "complete.json", data=dict(entry, duration_h=2))
    with pytest.raises(ValueError, match="cover the imposed storm"):
        orton_transfer.read_complete(folder=folder, identity=identity)
    runner.write_json(path=folder / "complete.json", data=entry)
    metrics.csv_write(file=tmp_path / "transfer.csv", rows=[dict(entry, dx_m=20)])
    with pytest.raises(ValueError, match="index differs"):
        orton_transfer.verify_index(directory=tmp_path)
    runner.write_json(path=window, data={"gauge_m3s": [5., 2.]})
    with pytest.raises(ValueError, match="Changed transfer observations"):
        orton_transfer.read_complete(folder=folder, identity=identity)


def test_section_storage_uses_aligned_gauges_and_checks_the_domain_ledger():
    from research.floodwarp_methods_paper.fidelity.native_products import section_storage

    data = np.array([[0., .4], [.1, .5]])
    measured = section_storage(depths=data, support_m=10, section_x_m=10, storage_m3=100.)
    assert measured == {"upstream_storage_m3": 10., "downstream_storage_m3": 90.}
    unsplit = section_storage(depths=data, support_m=10, section_x_m=5, storage_m3=100.)
    assert all(amount is None for amount in unsplit.values())
    with pytest.raises(ValueError, match="endpoint water account"):
        section_storage(depths=data, support_m=10, section_x_m=10, storage_m3=99.)


def test_model_grid_extent_counts_subcell_wet_area_before_aggregation():
    from research.floodwarp_methods_paper.fidelity.native_products import threshold_areas

    data = {"wet_fraction_5cm": np.array([[.25, .75]]), "wet_fraction_10cm": np.array([[0., .25]])}
    assert threshold_areas(archive=data, support_m=10, dx_m=5) == {
        "model_grid_wet_area_5cm_m2": 100, "model_grid_wet_area_10cm_m2": 25}
    with pytest.raises(ValueError, match="cell area"):
        threshold_areas(archive=dict(data, wet_fraction_5cm=np.array([[.1]])), support_m=10, dx_m=5)


def test_cpu_cuda_comparison_uses_each_accepted_transport_interval_and_actual_rain_window():
    from research.floodwarp_methods_paper.fidelity import device_controls

    cpu = [dict(t_s=2, steps=4, section_m3s=2., outlet_m3s=1.), dict(t_s=5, steps=5, section_m3s=3., outlet_m3s=2.)]
    gpu = [dict(cpu[0], outlet_m3s=2.), dict(cpu[1], section_m3s=1., steps=7)]
    measured = device_controls.interval_differences(cpu=cpu, gpu=gpu)
    assert measured["intervals_with_different_step_count"] == 1 and measured["outlet_net_difference_m3"] == 2
    assert measured["section_net_difference_m3"] == -6 and measured["section_max_abs_difference_m3s"] == 2.
    short = dict(rain_times_s=[0, 2, 5], rain_rates_ms=[1, 0, 0], roughness=.03)
    long = dict(short, rain_times_s=[0, 2, 10])
    device_controls.same_forcing(cpu=long, gpu=short, end=5)
    with pytest.raises(ValueError, match="Different rainfall"):
        device_controls.same_forcing(cpu=long, gpu=short, end=6)
    with pytest.raises(ValueError, match="Different rainfall"):
        device_controls.same_forcing(cpu=dict(long, rain_rates_ms=[1, 1, 0]), gpu=dict(short, rain_rates_ms=[1, 1, 0]), end=5)


def test_cpu_recession_requires_all_intervals_from_the_physical_start(tmp_path):
    import json
    from research.floodwarp_methods_paper.fidelity.device_controls import cpu_recession

    scheme = {"limiter": "adaptive"}
    for start, end in ((0, 60), (60, 120)):
        block = dict(start_s=start, end_s=end, frames=[dict(t_s=end)], budget=dict(numerical=scheme))
        (tmp_path / f"{end}.json").write_text(json.dumps(block))
    frames, final, files = cpu_recession(folder=tmp_path, end=120, numerical=scheme)
    assert [r["t_s"] for r in frames] == [60, 120] and final["end_s"] == 120 and len(files) == 2
    (tmp_path / "60.json").unlink()
    with pytest.raises(ValueError, match="observation windows"):
        cpu_recession(folder=tmp_path, end=120, numerical=scheme)


def test_budget_reprices_on_admissible_terrain_and_preserves_hydraulic_winner(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import budgets

    point = dict(case="plain", cohort="complete-native-metre", reference_id="fine", station="outlet",
                 end_h=20, interval_s=60, representation="native", cap=0, objective="peak_bias_pct",
                 eligible=True, reference_sensitivity=1., execution_mode="dense", steps=10)
    rows = [dict(point, dx=10, method="fv1", experiment_id="cheap", absolute_error=1., runtime_s=2.),
            dict(point, dx=5, method="hancock_dense", experiment_id="fine", absolute_error=2., runtime_s=4.),
            dict(point, dx=5, method="fv1", experiment_id="unknown", absolute_error=1., runtime_s=3.)]
    states = {"cheap": dict(identity="coarse", features=dict(critical_connection_lost=True)),
              "fine": dict(identity="finer", features=dict(critical_connection_lost=False))}
    budgets.native_cost(rows=rows, target=tmp_path, tolerances=[5], spacings=[5, 10], name="fidelity", controls=states)
    original, = metrics.csv_read(file=tmp_path / "least_cost_fidelity.csv")
    screened, = metrics.csv_read(file=tmp_path / "least_cost_terrain_fidelity.csv")
    assert original["id"] == "cheap" and screened["id"] == "fine"
    assert (screened["terrain_requires_refinement"], screened["terrain_unassessed"]) == ("1", "1")
    assert screened["conservative_error"] == "3.0"
    equal = metrics.csv_read(file=tmp_path / "equal_cost_terrain_fidelity.csv")
    assert len(equal) == 2 and all(r["selected_id"] == "" for r in equal)
    audit = metrics.csv_read(file=tmp_path / "terrain_screen_fidelity.csv")
    assert [r["disposition"] for r in audit] == ["refine terrain", "admissible", "unassessed"]


def test_native_budget_terrain_sources_agree_between_gauges_and_spatial_fields():
    from pathlib import Path
    from research.floodwarp_methods_paper import metrics

    root = Path(__file__).resolve().parents[1] / "research/floodwarp_methods_paper/fidelity/native_results"
    gauges = {r["experiment_id"]: r for r in metrics.csv_read(file=root / "terrain_screen_fidelity.csv")}
    fields = metrics.csv_read(file=root / "terrain_screen_depth_extent.csv")
    assert fields
    for r in fields:
        wave = gauges[r["experiment_id"]]
        assert r["descriptor_id"] == wave["descriptor_id"]
        assert r["critical_connection_lost"] == wave["critical_connection_lost"]


def test_manuscript_budget_decisions_keep_depth_and_extent_thresholds_in_their_own_units(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.manuscript.generate import native_choice_table

    file = tmp_path / "cost.csv"
    base = dict(case="hill", end_h=12, interval_s=60, reference_id="ref", cohort="complete-native-metre",
                station="domain", representation="native", cap=0, dx=10, runtime_s=2., method="fv1")
    scopes = {("domain", "maximum_depth_rmse_m"): ("Maximum-depth map", 0.01),
              ("domain", "maximum_extent_error_5cm"): ("Maximum 5 cm extent", 0.05)}
    rows = [dict(base, objective="maximum_depth_rmse_m", tolerance=.01, assessable=6, id="h", method="hancock_dense"),
            dict(base, objective="maximum_extent_error_5cm", tolerance=.05, assessable=0, id=""),
            dict(base, objective="maximum_extent_error_5cm", tolerance=.1, assessable=6, id="f")]
    metrics.csv_write(file=file, rows=rows)
    table = native_choice_table(file=file, references={("hill", "ref")}, objectives=scopes)
    assert table[-1] == "| hill | 12 | 60 | Hancock dense 10 m (2 s) | Unresolved |"
    rows[1]["assessable"] = 6
    metrics.csv_write(file=file, rows=rows)
    assert native_choice_table(file=file, references={("hill", "ref")}, objectives=scopes)[-1].endswith("Outside tolerance |")


def test_manuscript_reports_single_measured_hancock_policy_and_checks_dual_policy_parity(tmp_path):
    from research.floodwarp_methods_paper import metrics, runner
    from research.floodwarp_methods_paper.manuscript.generate import native_operational_balances

    directory = tmp_path / "fidelity/native_results"
    runner.write_json(path=tmp_path / "fidelity/outlet_results/sources.json", data={"solver": "repaired"})
    base = dict(case="hill", native_sha="bed", dx_m=1, interval_s=60, end_h=12, hardware="test",
                cells=1000000, factor=1, solver_source="repaired", horizon="complete-native-metre",
                net_outlet_m3=1300, infiltration_m3=100, outlet_peak_m3s=10,
                initial_m3=0, rain_m3=1600, storage_m3=200,
                model_grid_wet_area_5cm_m2=20, model_grid_wet_area_10cm_m2=5)
    dense = dict(base, method="hancock_dense", source_id="h", execution_mode="dense")
    fv1 = dict(base, method="fv1", source_id="f", execution_mode="sparse")
    metrics.csv_write(file=directory / "measured_windows.csv", rows=[dense, fv1])
    metrics.csv_write(file=directory / "policy_cost_events.csv", rows=[])
    metrics.csv_write(file=directory / "timing_distribution.csv", rows=[
        dict(horizon="completed recession", hydraulic_median_s=t, run_ids=ident)
        for ident, t in (("h", 1.5), ("f", 1), ("t", 1.1))])
    report = native_operational_balances(study=tmp_path, minimum_cells=1000000)
    assert "| dense only measured | 1.500 |" in report
    assert "| 1,600 | 1.60 | 200 | 200 | 20 | 20 | 5 | 5 |" in report
    metrics.csv_write(file=directory / "measured_windows.csv", rows=[dict(dense, rain_m3=1500), fv1])
    with pytest.raises(ValueError, match="storm inputs disagree in rain_m3"):
        native_operational_balances(study=tmp_path, minimum_cells=1000000)
    tiled = dict(base, method="hancock_adaptive", source_id="t", execution_mode="tiles")
    metrics.csv_write(file=directory / "measured_windows.csv", rows=[dense, fv1, tiled])
    with pytest.raises(ValueError, match="lack hydraulic parity"):
        native_operational_balances(study=tmp_path, minimum_cells=1000000)
    metrics.csv_write(file=directory / "policy_cost_events.csv", rows=[
        dict(fv1_id="f", dense_id="h", tiled_id="t", cohort="complete-native-metre")])
    assert "| tiles | 1.100 |" in native_operational_balances(study=tmp_path, minimum_cells=1000000)


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


def test_small_reference_domain_cannot_authorise_a_large_domain_grid_change():
    from floodwarp.solver_advisor.model import choose_resolution

    rows = [dict(objective="peak", chosen="fv1", feasible=True, dx=dx,
                 terrain_id=t, case=t, cells=1_000_000 // dx**2)
            for t in ("a", "b") for dx in (5, 10)]
    model = dict(native_cross_grid_validation=native_advisor.validated_pairs(rows=rows, minimum=2),
                 native_cross_grid_areas_m2=native_advisor.validated_areas(rows=rows, minimum=2))
    options = [dict(status="recommended", resolution_m=dx, runtime_s=cost, warnings=[],
                    represented_area_m2=1_000_000, objective="peak") for dx, cost in ((5, 3), (10, 1))]
    assert choose_resolution(model=model, results=options) is options[1]
    for candidate in options:
        candidate["represented_area_m2"] = 100_000_000
    assert choose_resolution(model=model, results=options) is options[0]
    assert "domain-area range" in options[0]["warnings"][-1]


def test_lost_connection_cannot_supply_development_hydraulic_error():
    row = dict(features=dict(critical_connection_lost=True), errors={"peak_discharge": 0.1},
               qualified={"peak_discharge": True}, runtime_s=5.0)
    training, = native_advisor.admissible_training(rows=[row])
    assert training["runtime_s"] == 5.0 and training["errors"]["peak_discharge"] is None
    assert training["qualified"] == {"peak_discharge": False}
    assert row["errors"] == {"peak_discharge": 0.1}


def test_unassessed_native_control_cannot_supply_training_validation_or_public_advice():
    from floodwarp.solver_advisor.model import recommend, terrain_control_status

    f = dict(critical_connection_lost=None, resolution=5)
    row = dict(features=f, errors={"depth": .002}, qualified={"depth": True}, runtime_s=4, terrain_id="plain")
    filtered, = native_advisor.admissible_training(rows=[row])
    assert filtered["errors"]["depth"] is None and filtered["runtime_s"] == 4
    assert native_advisor.support(rows=[row], objectives=["depth"], minimum=1) == {"depth": []}
    model = dict(native_objective_spacings={"depth": [5]}, tolerances={"depth": .01}, revision="test", hardware="GPU")
    choice = recommend(model=model, features=f, objective="depth")
    assert choice["status"] == "insufficient_evidence" and choice["terrain_control_status"] == "unassessed"
    assert terrain_control_status(features=dict(f, maximum_barrier_lowering_m=.1, barrier_lowering_m=.2),
                                 objective="depth") == "refine terrain"


def test_joint_oracle_excludes_unqualified_numeric_error():
    from research.floodwarp_methods_paper.fidelity.paired_advisor import evaluate_pairs

    records = [dict(case="event", terrain_id="terrain", native_sha="inputs", end_h=16,
                    method="fv1", run_id=str(dx), features=dict(resolution=dx, critical_connection_lost=False),
                    interval_s=cadence, spatial_reference_id="rk2-1m", spatial_support_m=10,
                    errors={"depth": 0.001}, qualified={"depth": dx == 5},
                    runtime_s=cost) for dx, cadence, cost in ((10, "60", 10), (5, "60.0", 20))]
    trials = [dict(case="event", objective="depth", rule="oracle", dx=dx, tolerance=.01,
                   status="oracle", terrain_admissible=True) for dx in (5, 10)]
    rows = evaluate_pairs(model={}, rows=records, trials=trials, spacings=[5, 10])
    assert len(rows) == 2 and all(r["optimal_dx"] == r["chosen_dx"] == 5 for r in rows)
    assert {(r["reference_id"], r["spatial_support_m"]) for r in rows} == {("rk2-1m", 10)}
    for field, value in (("spatial_reference_id", "different-ref"), ("spatial_support_m", 5)):
        different = dict(records[0], qualified={"depth": True})
        different[field] = value
        with pytest.raises(ValueError, match="different numerical references or spatial supports"):
            evaluate_pairs(model={}, rows=[different, records[1]], trials=trials, spacings=[5, 10])


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


def test_repeat_driver_selects_only_requested_complete_case_policy_and_grid():
    choices = {("plain-a", 20): {("fv1", 5), ("hancock_dense", 10), ("hancock_dense", 5)},
               ("rolling-b", 40): {("hancock_dense", 5)}}
    assert timing_repeats.requested_options(selected=choices, case="plain-*", method="hancock_dense", dx=5) == {
        ("plain-a", 20): {("hancock_dense", 5)}}
    assert not timing_repeats.requested_options(selected=choices, case="*", method="li", dx=None)


def test_overlapping_repeat_subsets_count_each_independent_execution_once():
    from research.floodwarp_methods_paper.fidelity.native_products import merged_repeat_receipt

    hardware = dict(gpu="reference", cpu_model="cpu", warp="1.17", python="3.14", numpy="2", precision="FP64")
    base = dict(case="plain", representation="sampled", cohort="native-repeat-1", end_s=72000,
                extended=True, settings={"timing_repeat_end_s": 72000}, source="same", hardware=hardware)
    first = dict(base, sources=["b", "a"], series=["extended-b", "extended-a"])
    second = dict(base, sources=["c", "b"], series=["extended-c", "extended-b"])
    merged = merged_repeat_receipt(receipts=[first, second])
    assert merged["sources"] == ["a", "b", "c"]
    assert merged["series"] == ["extended-a", "extended-b", "extended-c"]
    with pytest.raises(ValueError, match="Different recession histories"):
        merged_repeat_receipt(receipts=[first, dict(second, series=["extended-c", "unmatched"])])
    with pytest.raises(ValueError, match="physical endpoint"):
        merged_repeat_receipt(receipts=[first, dict(second, end_s=57600)])
    with pytest.raises(ValueError, match="hardware"):
        merged_repeat_receipt(receipts=[first, dict(second, hardware=dict(hardware, gpu="other"))])


def test_partial_repeat_receipts_publish_on_the_original_common_horizon(tmp_path):
    from research.floodwarp_methods_paper import metrics, runner
    from research.floodwarp_methods_paper.fidelity.native_products import grouped

    hardware = dict(gpu="reference", cpu_model="cpu", warp="1.17", python="3.14", numpy="2", precision="FP64")
    settings = dict(rain_end_s=21600, window_s=7200, minimum_peak_m3s=1e-5,
                    minimum_volume_m3=0.01, dry_supply_fraction=1e-6)
    data = dict(frames=[dict(t_s=28800, section_m3s=0., outlet_m3s=0.)],
                rain_m3=100, initial_m3=0, final_storage_m3=0, infiltration_m3=100, residual_m3=0)
    records = {}
    for index, (method, parents) in enumerate((("fv1", ["a"]), ("adaptive", ["a", "b"]))):
        ident = parents[-1]
        records[ident] = dict(id=ident, s=dict(case="hill", method=method, hours=8, dx=5), d=data,
                              native_sha="terrain", hardware=hardware, solver_source="numerics")
        runner.write_json(path=tmp_path / "extended_runs" / f"group_{index}.json",
                          data=dict(case="hill", representation="sampled", cohort="native-repeat-1",
                                    end_s=28800, extended=False, settings=settings, source="driver",
                                    hardware=hardware, sources=parents, series=[]))
    target = tmp_path / "products"
    result = grouped(directory=tmp_path, records=records, expected={}, target=target)
    assert [r["id"] for r in result["complete-native-repeat-1", "hill", "native metre area means"]] == ["a", "b"]
    horizon, = metrics.csv_read(file=target / "recession_horizons.csv")
    assert int(horizon["configurations"]) == 2
    assert float(horizon["common_end_h"]) == 8
    assert len(metrics.csv_read(file=target / "recession_terminal.csv")) == 4


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
