import json
from pathlib import Path

import numpy as np
import pytest

from research.floodwarp_methods_paper.fidelity.hydrograph import evaluate


@pytest.fixture
def config():
    source = (
        Path(__file__).resolve().parents[1]
        / "research/floodwarp_methods_paper/fidelity/design.json"
    )
    return json.loads(source.read_text())


def compare(*, model, reference, cfg, edges=None):
    if edges is None:
        edges = np.arange(len(model) + 1)
    return evaluate(edges=edges, model=model, reference=reference, config=cfg)


def triangle(*, half_width, centre, bins=100):
    wave = np.maximum(0.0, 1 - np.abs(np.arange(bins) - centre) / half_width)
    return wave / wave.sum()


def test_progressive_equal_volume_broadening(config):
    ref = triangle(half_width=8, centre=40)
    last = 0
    for width in (12, 18, 24):
        row = compare(
            model=triangle(half_width=width, centre=40), reference=ref, cfg=config
        )
        assert row["volume_bias_pct"] == pytest.approx(0, abs=1e-12)
        assert row["peak_bias_pct"] == pytest.approx(100 * (8 / width - 1))
        assert row["centroid_delay_s"] == pytest.approx(0, abs=1e-12)
        assert row["width80_bias_pct"] > last
        assert row["shape_tv_pct"] > 0
        last = row["width80_bias_pct"]


def test_delay_and_scaling(config):
    ref = triangle(half_width=10, centre=35)
    row = compare(model=np.roll(ref, 7) * 1.4, reference=ref, cfg=config)
    for key in (
        "peak_delay_s",
        "t10_delay_s",
        "centroid_delay_s",
        "xcorr_lag_s",
        "wasserstein_s",
    ):
        assert row[key] == pytest.approx(7)
    assert row["width80_bias_pct"] == pytest.approx(0, abs=1e-11)
    assert row["volume_bias_pct"] == pytest.approx(40)
    assert row["aligned_tv_pct"] == pytest.approx(0, abs=1e-10)
    assert row["shape_tv_pct"] > 0


def test_native_gauges_resolve_internal_transport_and_dry_outlet():
    from research.floodwarp_methods_paper.fidelity.native_products import gauge_observation

    frames = [dict(t_s=t, section_m3s=q, outlet_m3s=0.0)
              for t, q in ((5, -2), (18, 4), (20, 1))]
    item = {"d": {"frames": frames}}
    section = gauge_observation(item=item, station="section")
    assert section["peak_m3s"] == 4
    assert section["unique_outward_peak_interval_end_s"] == 18
    assert section["net_m3"] == 44
    outlet = gauge_observation(item=item, station="outlet")
    assert outlet["net_m3"] == outlet["peak_m3s"] == 0
    assert outlet["unique_outward_peak_interval_end_s"] is None
    frames[-1]["section_m3s"] = 4
    assert gauge_observation(item=item, station="section")["unique_outward_peak_interval_end_s"] is None


def test_irregular_bins_and_combined(config):
    t = np.array([0, 1, 2, 4, 8, 12, 17, 21, 25])
    ref = np.array([0, 1, 3, 1, 0, 0, 0, 0.0])
    a = np.array([0, 0, 1, 2, 1, 0.5, 0, 0.0])
    row = compare(model=a, reference=ref, cfg=config, edges=t)
    assert row["volume_bias_pct"] == pytest.approx(
        100 * ((a @ np.diff(t)) / (ref @ np.diff(t)) - 1)
    )
    assert row["centroid_delay_s"] > 0 and row["peak_bias_pct"] < 0
    assert row["shape_tv_pct"] > row["aligned_tv_pct"] > 0
    assert "xcorr_lag_s" not in row


def test_multiple_negligible_truncated_and_reversed(config):
    a = triangle(half_width=6, centre=20) + triangle(half_width=6, centre=60)
    row = compare(model=a, reference=a, cfg=config)
    assert (
        row["model_modes"] == 2
        and "peak_delay_s" not in row
        and "fwhm_bias_pct" not in row
    )
    for scale in (0, 1e-10):
        assert "volume_bias_pct" not in compare(
            model=a, reference=a * scale, cfg=config
        )
    assert compare(model=a[:21], reference=a[:21], cfg=config)["reference_truncated"]
    b = a.copy()
    b[0] = -1
    assert "shape_tv_pct" not in compare(model=b, reference=a, cfg=config)
    with pytest.raises(ValueError):
        compare(model=a, reference=a, cfg=config, edges=np.ones(len(a) + 1))


def test_tail_padding_preserves_active_error(config):
    ref = triangle(half_width=11, centre=45)
    model = triangle(half_width=18, centre=45)
    short = compare(model=model, reference=ref, cfg=config)
    long = compare(
        model=np.pad(model, (0, 100)), reference=np.pad(ref, (0, 100)), cfg=config
    )
    assert short["active_nrmse_peak"] == pytest.approx(long["active_nrmse_peak"])
    assert long["nrmse_peak"] == pytest.approx(short["nrmse_peak"] / np.sqrt(2))
    assert long["integrated_abs_pct"] == pytest.approx(short["integrated_abs_pct"])


def test_cadence_limits_plateau_and_truncation(config):
    from research.floodwarp_methods_paper.fidelity.assessment import limits

    ref = np.r_[np.zeros(4), np.ones(6), np.zeros(5)]
    row = compare(model=np.roll(ref, 2), reference=ref, cfg=config)
    assert "peak_delay_s" not in row
    assert row["peak_delay_lower_s"] < 0 < row["peak_delay_upper_s"]
    censored = compare(model=ref[:9], reference=ref[:9], cfg=config)
    assert not limits(row=censored, key="shape_tv_pct")


def test_reference_scale_and_baseline(config):
    t = np.arange(101.0)
    ref = triangle(half_width=12, centre=55)
    model = np.roll(ref, 3)
    base = evaluate(edges=t, model=model, reference=ref, config=config)
    scaled = evaluate(
        edges=t,
        model=10 * model + 2,
        reference=10 * ref + 2,
        background=2,
        config=config,
    )
    for k in ("peak_bias_pct", "shape_tv_pct", "centroid_delay_pct", "t75_delay_s"):
        assert base[k] == pytest.approx(scaled[k], abs=1e-10)
    assert scaled["model_net_m3"] == pytest.approx(10 * base["model_net_m3"])


def test_fidelity_advisor_preserves_old_evidence_and_separates_objectives():
    from floodwarp.solver_advisor.fidelity import attach
    from floodwarp.solver_advisor.model import recommend

    model = {
        "training": [
            {"run_id": "x", "errors": {"depth": 0.01}, "qualified": {"depth": True}}
        ],
        "tolerances": {"depth": 0.05},
        "hardware": "test GPU",
        "revision": "test",
    }
    supplement = {
        "training": {"x": {"hydrograph_shape": 4.0}},
        "tolerances": {"hydrograph_shape": 10.0},
        "padding": {},
        "validation_support": {},
    }
    extended = attach(model=model, supplement=supplement)
    assert model["training"][0]["errors"] == {"depth": 0.01}
    result = recommend(
        model=extended, features={"resolution": 10}, objective="full_hydrograph"
    )
    assert result["objective"] == "hydrograph_shape"
    assert result["status"] == "insufficient_evidence"


def test_recession_requires_both_stations_and_late_volume():
    from research.floodwarp_methods_paper.fidelity.recession import settled

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(
        (
            root / "research/floodwarp_methods_paper/fidelity/recession_settings.json"
        ).read_text()
    )
    frames = [
        dict(t_s=t, outlet_m3s=0.0, section_m3s=0.0)
        for t in range(900, 12 * 3600 + 1, 900)
    ]
    assert not settled(frames=frames, storage=1.0, supply=100.0, config=cfg)[0]
    assert settled(frames=frames, storage=0.0, supply=100.0, config=cfg)[0]
    frames[8].update(outlet_m3s=10.0, section_m3s=10.0)
    frames[-2]["section_m3s"] = 2.0
    ok, checks = settled(frames=frames, storage=0.0, supply=100.0, config=cfg)
    assert checks["outlet"]["passed"] and not ok
    frames[-2]["section_m3s"] = 0.0
    for frame in frames[-8:]:
        frame["section_m3s"] = 0.4
    assert not settled(frames=frames, storage=0.0, supply=100.0, config=cfg)[0]


def test_recession_replay_checks_every_original_step():
    from research.floodwarp_methods_paper.fidelity.recession import verify_prefix

    row = dict(t_s=60, steps=10, section_m3s=0.02, outlet_m3s=0.01)
    verify_prefix(old={"frames": [row]}, new=[row.copy()])
    with pytest.raises(AssertionError):
        verify_prefix(old={"samples": [row]}, new=[dict(row, steps=9)])


def test_recession_endpoint_is_an_outcome_and_keeps_design_features():
    from floodwarp.solver_advisor.fidelity import attach

    model = dict(
        training=[
            dict(
                run_id="abc",
                features={"forecast_duration_h": 8.0},
                errors={},
                qualified={},
            )
        ],
        tolerances={},
        interval_padding={},
        scales={},
    )
    supplement = dict(
        training={"abc": {"integrated_runoff": 4.2}},
        forecasts={"abc": dict(runtime_s=12.5, end_h=24.0, evidence_id="extended")},
        tolerances={},
        padding={},
        validation_support={},
    )
    result = attach(model=model, supplement=supplement)
    assert result["training"][0]["runtime_s"] == 12.5
    assert result["training"][0]["features"]["forecast_duration_h"] == 8.0


def test_spatial_front_and_conserved_momentum(tmp_path):
    from research.floodwarp_methods_paper.fidelity.transient import summary

    spec = dict(
        case="synthetic",
        dx=5,
        dem=5,
        method="adaptive",
        cfl=0.4,
        interval=900,
        support=10,
    )
    reference = tmp_path / "reference"
    model = tmp_path / "candidate"
    reference.mkdir()
    model.mkdir()
    for time in (0, 900, 1800, 2700):
        for folder, arrives in ((model, 1800), (reference, 900)):
            h = np.array([[0.2 if time >= arrives else 0, 0.2]])
            np.savez(folder / f"{time}.npz", h=h, hu=h * 2, hv=h)
    rows, result, delay = summary(
        pair=(model, spec),
        reference=(reference, spec),
        edges=[0, 900, 1800, 2700],
        threshold=0.01,
    )
    assert delay[0, 0] == 900 and np.isnan(delay[0, 1])
    assert result["co_arrival_cells"] == 1
    assert rows[1]["momentum_vector_rmse_m2s"] == pytest.approx(np.sqrt(0.1))
    assert rows[-1]["iou"] == 1


def test_interval_receipt_preserves_shorter_horizon(tmp_path):
    from research.floodwarp_methods_paper.fidelity import receipts

    f = dict(t_s=900, steps=3, section_m3s=0.04, outlet_m3s=0.02)
    run = dict(id="abc", end_s=900, frames=[f])
    receipts.freeze(folder=tmp_path / "abc/horizons", data=run)
    (tmp_path / "abc/series.json").write_text(
        json.dumps(dict(run, end_s=1800, frames=[f, dict(f, t_s=1800)]))
    )
    assert receipts.read(root=tmp_path, ident="abc", end=900) == run
    assert receipts.read(root=tmp_path, ident="abc", end=1800)["end_s"] == 1800


def test_csv_archive_preserves_missing_and_numeric_types(tmp_path):
    import sqlite3
    from research.floodwarp_methods_paper.fidelity.catalogue import database

    (tmp_path / "observations.csv").write_text(
        "experiment_id,delay_s,qualified\n00001234,-3.5,True\n00005678,,False\n"
    )
    database(target=tmp_path)
    with sqlite3.connect(tmp_path / "fidelity.sqlite") as db:
        assert db.execute(
            "select * from observations order by experiment_id"
        ).fetchall() == [("00001234", -3.5, 1), ("00005678", None, 0)]
        assert db.execute("select rows from csv_sources").fetchone() == (2,)


def test_irregular_accepted_volume_brackets_each_quantile(config):
    t = np.array([0.0, 1.0, 4.0, 5.0, 9.0, 11.0, 12.0, 15.0, 18.0, 20.0, 25.0])
    ref = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mod = np.roll(ref, 3)
    row = compare(model=mod, reference=ref, edges=t, cfg=config)
    for name in ("t10", "t25", "t50", "t75", "t90", "centroid"):
        assert (
            row[name + "_delay_lower_s"]
            <= row[name + "_delay_s"]
            <= row[name + "_delay_upper_s"]
        )
        assert row[name + "_direction"] == "delayed"
    identical = compare(model=ref, reference=ref, edges=t, cfg=config)
    assert identical["centroid_direction"] == "unresolved_within_interval"


def test_absent_candidate_still_has_scalar_errors(config):
    from research.floodwarp_methods_paper.fidelity.assessment import limits

    ref = triangle(half_width=12, centre=40)
    row = compare(model=np.zeros_like(ref), reference=ref, cfg=config)
    assert row["volume_bias_pct"] == pytest.approx(-100)
    assert row["peak_bias_pct"] == -100
    assert limits(row=row, key="peak_bias_pct") and limits(
        row=row, key="volume_bias_pct"
    )
    assert "shape_tv_pct" not in row and row["active_nrmse_peak"] > 0


def test_equal_cost_decisions_keep_ties_and_unresolved_cases(tmp_path):
    from research.floodwarp_methods_paper.fidelity.budgets import feasible_cost
    from research.floodwarp_methods_paper.metrics import csv_read

    base = dict(
        cohort="pulse",
        case="front",
        reference_id="exact",
        station=10,
        objective="peak_bias_pct",
        representation="flat",
        cap=0.1,
        interval_s=1,
        dx=5,
        execution_mode="dense",
        eligible=True,
    )
    rows = [
        dict(
            base,
            method="fv1",
            absolute_error=25.0,
            runtime_s=1.0,
            experiment_id="budget",
        ),
        dict(
            base,
            method="hancock_dense",
            dx=10,
            absolute_error=9.0,
            runtime_s=0.8,
            experiment_id="h",
        ),
        dict(
            base,
            method="adaptive",
            dx=10,
            absolute_error=9.0,
            runtime_s=0.8,
            experiment_id="r",
        ),
        dict(
            base,
            method="fv1",
            dx=1,
            eligible=False,
            absolute_error=0.0,
            runtime_s=0.2,
            experiment_id="unresolved",
        ),
    ]
    feasible_cost(rows=rows, target=tmp_path, tolerances=(5, 10), spacings=(5,))
    tied = csv_read(tmp_path / "equal_cost_fidelity.csv")
    assert {r["selected_id"] for r in tied} == {"r", "h"}
    least = csv_read(tmp_path / "least_cost_fidelity.csv")
    assert [r["id"] for r in least if r["tolerance_pct"] == "5"] == [""]


def test_recession_cost_compares_matching_intervals_and_handles_unmeasured_work(
    tmp_path,
):
    from research.floodwarp_methods_paper.fidelity.phase_cost import collect
    from research.floodwarp_methods_paper.metrics import csv_read

    def forecast(*, method, runtime, depth_fraction):
        frames = [
            dict(t_s=t, steps=steps, hydraulic_s=wall)
            for t, steps, wall in zip((900, 1800), (10, 20), runtime, strict=True)
        ]
        spatial = [
            dict(t_s=t, wet_fraction=v)
            for t, v in zip((900, 1800), depth_fraction, strict=True)
        ]
        return dict(
            id=method,
            s=dict(dx=10, method=method),
            factor=1,
            d=dict(frames=frames, spatial=spatial, hydraulic_s=sum(runtime)),
        )

    fv = forecast(method="fv1", runtime=(2, 1), depth_fraction=(0.005, 0.50))
    tiled = forecast(
        method="hancock_adaptive", runtime=(4, 1), depth_fraction=(0.02, 0.9)
    )
    collect(
        groups={("complete-expanded", "mixed", "sampled"): [fv, tiled]},
        directory=tmp_path,
        target=tmp_path,
        cadence=900,
        edges=(0, 0.01, 0.05, 0.2, 0.5, 1),
        phase_edges=(6, 8, 12),
    )
    output = {
        (r["axis"], r["level"]): r
        for r in csv_read(tmp_path / "cost_by_phase.csv")
        if r["method"] == "hancock_adaptive"
    }
    assert float(output["phase", "rain"]["ratio"]) == pytest.approx(5 / 3)
    assert float(output["fv1_wet", "0–1%"]["ratio"]) == 2
    assert float(output["fv1_wet", "50–100%"]["ratio"]) == 1
    assert output["scheduled_work", "unmeasured"]["windows"] == "2"


def test_recession_pareto_keeps_reference_and_unresolved_extent_separate():
    from research.floodwarp_methods_paper.fidelity.statistics import frontiers

    common = dict(
        cohort="complete-expanded",
        case="hill",
        representation="sampled",
        station="domain",
        dx=5,
        method="fv1",
        qualified_extent=True,
    )
    first = dict(
        common, experiment_id="a", reference_id="coarse", runtime_s=1, extent=0.1
    )
    second = dict(
        common, experiment_id="b", reference_id="fine", runtime_s=2, extent=0.2
    )
    unknown = dict(
        common,
        experiment_id="c",
        reference_id="fine",
        runtime_s=0.1,
        extent=None,
        qualified_extent=False,
    )
    result = {
        r["experiment_id"]: r
        for r in frontiers(rows=[first, second, unknown], objectives=["extent"])
    }
    assert result["a"]["nondominated"] and result["b"]["nondominated"]
    assert result["c"]["nondominated"] is None and result["c"]["absolute_error"] is None


def test_recession_depth_extent_and_outlet_connectivity():
    from research.floodwarp_methods_paper.fidelity.final_depth import difference

    ref = np.full((1, 5), 0.06)
    h = ref.copy()
    h[0, 3] = 0
    row = difference(
        candidate=(h, h), reference=(ref, ref), support=1, thresholds=(0.01, 0.05, 0.10)
    )
    assert row["iou_5cm"] == 0.8
    assert row["component_difference_1cm"] == 1
    assert row["outlet_connected_area_difference_1cm_m2"] == -4
    assert row["iou_10cm"] is None


def test_long_storage_tail_integrates_partial_bins_and_reversing_transport():
    from research.floodwarp_methods_paper.fidelity.long_tail import late_export

    frames = [
        dict(t_s=t, outlet_m3s=q, section_m3s=-q)
        for t, q in ((5, 2.0), (8, 1.0), (12, -1.0))
    ]
    result = late_export(frames=frames, window=6, fraction=0.1)
    assert result["outlet"]["final_outward_m3"] == 13
    assert result["outlet"]["late_outward_m3"] == 2
    assert not result["outlet"]["passed"]
    assert result["section"]["late_outward_m3"] == 4


def test_long_tail_runs_to_a_common_later_endpoint(tmp_path, monkeypatch):
    from research.floodwarp_methods_paper.fidelity import long_tail

    directory = tmp_path / "study" / "fidelity"
    folder = directory / "extended_runs"
    folder.mkdir(parents=True)
    (directory / "long_tail.py").write_text("storage sensitivity test")
    options = dict(
        case="mixed",
        dx=10,
        methods=["fv1", "hancock_adaptive"],
        minimum_hours=48,
        tail_window_hours=12,
        extension_hours=12,
        late_volume_fraction=0.01,
    )
    (directory / "long_tail_settings.json").write_text(json.dumps(options))
    native = []
    ids = options["methods"]
    for name in ids:
        native.append(
            dict(
                id=name,
                s=dict(method=name, dx=10, case="mixed"),
                factor=1,
                d=dict(initial_m3=10),
            )
        )
    (folder / "group_native.json").write_text(
        json.dumps(
            dict(
                sources=ids,
                series=ids,
                source="hydraulic",
                end_s=20 * 3600,
                cohort="expanded",
                case="mixed",
                representation="sampled",
            )
        )
    )

    def config(**kwargs):
        return {}

    def observations(**kwargs):
        return {("expanded", "mixed", "sampled"): native}

    def series(*, name, end):
        hours = end // 3600
        last_flow_hour = 48 if name == "fv1" else 36
        frames = []
        for hour in range(1, hours + 1):
            flow = float(hour <= last_flow_hour)
            frames.append(dict(t_s=hour * 3600, section_m3s=flow, outlet_m3s=flow))
        return dict(
            id=name,
            parent_id=name,
            end_s=end,
            spec=dict(method=name, case="mixed"),
            frames=frames,
            storage_m3=5,
            infiltration_m3=0,
            rain_m3=2,
            residual_m3=0,
            checks={"section": {"passed": True}, "outlet": {"passed": True}},
        )

    def early(**kw):
        return series(name=kw["ident"], end=kw["end"])

    calls = []

    def advance(**kw):
        calls.append((kw["it"]["id"], kw["floor_s"]))
        return series(name=kw["it"]["id"], end=kw["floor_s"])

    monkeypatch.setattr(long_tail, "read_config", config)
    monkeypatch.setattr(long_tail.assessment, "archive", observations)
    monkeypatch.setattr(long_tail.receipts, "read", early)
    monkeypatch.setattr(long_tail.recession, "prolong", advance)
    long_tail.run(directory=directory, settings={"forcing_hours": 72})
    assert calls == [(name, h * 3600) for h in (48, 60) for name in ids]
    long_tail.run(directory=directory, settings={"forcing_hours": 72})
    assert len(calls) == 4


def test_reference_requires_physical_rain_timing_through_the_common_end():
    from research.floodwarp_methods_paper.fidelity.assessment import (
        matched_reference_inputs,
    )

    def item(*, times, rates):
        return dict(
            s={"hours": 8},
            d={"forcing": {"rain_times_s": times, "rain_rates_ms": rates}},
        )

    a = item(times=[0, 10800, 21600, 28800], rates=[0, 1, 0, 0])
    b = item(times=[0, 10800, 21600, 720 * 3600], rates=[0, 1, 0, 0])
    matched_reference_inputs(base=a, control=b)
    b["d"]["recession_capture"] = {
        "actual_forcing": {"rainfall": [[0, 7200, 21600, 720 * 3600], [0, 1, 0, 0]]}
    }
    with pytest.raises(AssertionError, match="Different physical hyetographs"):
        matched_reference_inputs(base=a, control=b)


def test_native_horizon_omission_uses_integrated_outward_transport():
    from research.floodwarp_methods_paper.fidelity.native_horizons import station_rows

    data = dict(
        end_s=8,
        checks={"outlet": {"passed": True, "recent_to_volume": 0.01}},
        frames=[
            {"t_s": 1, "outlet_m3s": -2.0},
            {"t_s": 3, "outlet_m3s": 3.0},
            {"t_s": 8, "outlet_m3s": 2.0},
        ],
    )
    measured, = station_rows(data=data, minimum_volume=0.01, original_s=3)
    assert measured["original_outward_m3"] == 6.0
    assert measured["omitted_outward_m3"] == 10.0
    assert measured["omitted_pct"] == 62.5
    assert measured["inward_m3"] == 2.0
    assert measured["original_peak_out_m3s"] == measured["final_peak_out_m3s"] == 3.0
    negligible, = station_rows(data=data, minimum_volume=17.0, original_s=3)
    assert negligible["omitted_pct"] is None
    with pytest.raises(ValueError, match="Missing a measured endpoint"):
        station_rows(data=data, minimum_volume=0.01, original_s=2)


def test_fixed_crest_grid_refines_initial_state_and_rainfall_coordinates():
    from research.floodwarp_methods_paper.fidelity.incised_crest import fixed_fields

    def parent(*, dx, dem):
        assert dx == dem == 10
        state = np.array([[0.0, 0.30], [0.7, 1.0]])
        return (state, 1 - state), {"grid_m": dx}

    child, info = fixed_fields(parent=parent, dx=5, dem=10)
    assert info == {"grid_m": 10}
    for y in range(2):
        for x in range(2):
            np.testing.assert_array_equal(child[0][y::2, x::2], [[0, 0.3], [0.7, 1]])
            np.testing.assert_array_equal(child[1][y::2, x::2], 1 - child[0][y::2, x::2])
    with pytest.raises(ValueError, match="integer subdivision"):
        fixed_fields(parent=parent, dx=3, dem=10)


def test_coarse_cadence_moves_dominant_crest_to_earlier_event(config):
    from research.floodwarp_methods_paper.fidelity.statistics import cadence_observation

    t = np.arange(61) * 60
    q = np.zeros(60)
    q[:15] = 2
    q[30] = 10
    result = cadence_observation(edges=t, flow=q, seconds=900, config=config)
    assert result["peak_bias_pct"] == -80
    assert result["peak_shift_lower_s"] == -1860
    assert result["peak_shift_upper_s"] == -900
    assert result["volume_error_m3"] == pytest.approx(0, abs=1e-12)
    revised = np.array([0, 300, 750, 1200, 1500, 1800])
    assert cadence_observation(edges=revised, flow=np.ones(5), seconds=900,
                              config=config)["cadence_status"] == "missing_observed_boundary"


def test_spatial_peak_and_final_recession_censor_nonunique_and_persistent_water(tmp_path):
    from research.floodwarp_methods_paper.fidelity.transient import timing_fields

    h = np.array([[0, 0, .2, 0], [.2, .2, .2, 0], [.1, 0, .2, 0],
                  [0, .2, .2, 0], [0, 0, .2, 0]])
    t = [0, 900, 1800, 2700, 3600]
    for k, time in enumerate(t):
        np.savez(tmp_path / f"{time}.npz", h=h[k])
    peak, recession = timing_fields(folder=tmp_path, edges=t, interval=900,
                                    peak=h.max(axis=0), threshold=.01)
    np.testing.assert_equal(peak, [900, np.nan, np.nan, np.nan])
    np.testing.assert_equal(recession, [2700, 3600, np.nan, np.nan])


def test_mean_dem_pyramid_uses_native_cells_for_every_landform(tmp_path):
    from research.floodwarp_methods_paper.terrain.generators import mean_dem
    from research.floodwarp_methods_paper.qualification import hydraulics, matrix

    root = Path(__file__).resolve().parents[1]
    study = root / "research/floodwarp_methods_paper"
    from research.floodwarp_methods_paper.plan import read_config

    cfg = read_config(path=study / "configs/study.json")
    template = json.loads((root / "validation/hancock_long_storm_review/cases.json").read_text())
    options = matrix.design(study=study)
    for family in cfg["synthetic"]["landforms"]:
        spec = dict(landform=family, seed=707, initial="channel", dimensions=[80,60], controls=True, initial_grid="parent")
        cache, desc = mean_dem.generate(spec=spec, inputs=dict(cfg=cfg,template=template,options=options), cache_root=tmp_path,strip_rows=20)
        fields, _ = hydraulics.represented(options=options, settings=spec, design=cfg["synthetic"], template=template, family=family,seed=707,initial="channel",dx=1,dem=1)
        five = np.load(cache / "bed_5m.npy")
        ten = np.load(cache / "bed_10m.npy")
        expected = fields[0].reshape(12,5,16,5).mean(axis=(1,3))
        np.testing.assert_allclose(five,expected,rtol=1e-14,atol=1e-12)
        np.testing.assert_array_equal(ten,five.reshape(6,2,8,2).mean(axis=(1,3)))
        assert desc["native_1m_cells"] == 4800


def test_mean_dem_averages_an_offcentre_feature_and_rejects_partial_footprints():
    from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells

    z=np.zeros((10,10))
    z[0,0]=100.
    five=mean_cells(values=z,factor=5)
    np.testing.assert_array_equal(five, [[4,0],[0,0]])
    np.testing.assert_array_equal(mean_cells(values=five,factor=2), [[1.]])
    with pytest.raises(ValueError):
        mean_cells(values=z[:9],factor=5)


@pytest.mark.parametrize(
    ("initial", "storm"),
    (("channel", "spatial"), ("pond", "soil"), ("dry", "moderate")),
)
def test_operational_inputs_start_at_native_metre_cells(tmp_path, initial, storm):
    from research.floodwarp_methods_paper.qualification import hydraulics, matrix
    from research.floodwarp_methods_paper.fidelity import native_event
    from research.floodwarp_methods_paper.plan import read_config
    from research.floodwarp_methods_paper.terrain.generators import native_inputs

    root = Path(__file__).resolve().parents[1]
    study = root / "research/floodwarp_methods_paper"
    cfg = read_config(path=study / "configs/study.json")
    tpl = json.loads(
        (root / "validation/hancock_long_storm_review/cases.json").read_text()
    )
    options = matrix.design(study=study)
    spec = dict(
        landform="incised",
        seed=707,
        initial=initial,
        storm=storm,
        dimensions=[80, 60],
        controls=True,
        initial_grid="native_1m_block_mean",
        terrain_aggregation="native_1m_block_mean",
        multiplier=1.0,
        hours=8,
        method="hancock_adaptive",
        dx=5,
        dem=5,
    )
    bundle, receipt = native_inputs.generate(
        spec=spec,
        inputs=dict(cfg=cfg, template=tpl, options=options),
        cache_root=tmp_path,
        strip_rows=20,
        include_native=True,
        channel_core_m=0.01,
    )
    (z, h, hu, u, v), _ = hydraulics.represented(
        options=options,
        settings=spec,
        design=cfg["synthetic"],
        template=tpl,
        family=spec["landform"],
        seed=spec["seed"],
        initial=initial,
        dx=1,
        dem=1,
    )
    mask = (
        np.maximum(0, 1 - ((u - 0.30) / 0.23) ** 2 - ((v - 0.70) / 0.20) ** 2)
        if storm == "spatial"
        else np.ones_like(z)
    )
    soil = ((6 + 22 * u + 6 * v) if storm == "soil" else np.full_like(z, 2.0)) / 3.6e6
    if initial == "channel":
        hu = np.where(h > 0.01, 0.12 * h, 0.0)
    for key, expected in dict(
        bed=z, depth=h, hu=hu, rain=mask, infiltration=soil
    ).items():
        native = np.load(bundle / f"{key}_1m.npy")
        five = np.load(bundle / f"{key}_5m.npy")
        ten = np.load(bundle / f"{key}_10m.npy")
        np.testing.assert_allclose(native, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(
            five, native.reshape(12, 5, 16, 5).mean(axis=(1, 3))
        )
        np.testing.assert_array_equal(ten, five.reshape(6, 2, 8, 2).mean(axis=(1, 3)))
    kwargs = dict(
        config=cfg, template=tpl, device="cpu", bundle=bundle, receipt=receipt
    )
    for invalid in (dict(initial_grid="parent"), dict(dem=10)):
        with pytest.raises(ValueError, match="native metre"):
            native_event.create(spec=dict(spec, **invalid), **kwargs)
    model, rain_volume, forcing = native_event.create(spec=spec, **kwargs)
    assert (
        model.solver.time == 0
        and forcing["native_inputs"]["native_sha256"] == receipt["native_sha256"]
    )
    np.testing.assert_array_equal(
        model.solver._fields["bed"].numpy().reshape(12, 16),
        np.load(bundle / "bed_5m.npy"),
    )
    np.testing.assert_array_equal(
        model.solver._fields["multiplier"].numpy().reshape(12, 16),
        np.load(bundle / "rain_5m.npy"),
    )
    np.testing.assert_allclose(
        rain_volume,
        mask.sum() * np.trapezoid(forcing["rain_rates_ms"], forcing["rain_times_s"]),
        rtol=1e-14,
    )
    from floodwarp.model import advance, diagnostics, load_checkpoint, save_checkpoint
    from research.floodwarp_methods_paper.fidelity.native_matrix import prepare

    advance(model=model, end_time=30.0, use_cuda_graph=False)
    path = tmp_path / "native.npz"
    save_checkpoint(model=model, path=path)
    fresh, _, _ = native_event.create(spec=spec, **kwargs)
    load_checkpoint(model=fresh, path=path)
    np.testing.assert_array_equal(fresh.solver.q.numpy(), model.solver.q.numpy())
    assert abs(diagnostics(model=fresh)["residual_m3"]) < 1e-8
    del fresh.identity["native_inputs"]
    with pytest.raises(ValueError):
        load_checkpoint(model=fresh, path=path)
    auto, _, forcing_auto = prepare(
        spec=dict(
            spec, method="hancock_dense", requested_method="hancock_auto", factor=1.0
        ),
        **kwargs,
    )
    assert (
        auto.solver.time == 0
        and forcing_auto["execution_selection"]["requested"] == "auto"
    )
    assert auto.identity["native_inputs"] == model.identity["native_inputs"]


def test_practical_extent_separates_final_water_from_the_event_maximum():
    from research.floodwarp_methods_paper.fidelity.final_depth import difference

    ref = (np.zeros((2, 2)), np.array([[0.06, 0.12], [0.04, 0.0]]))
    run = (np.zeros((2, 2)), np.array([[0.09, 0.09], [0.08, 0.0]]))
    d = difference(
        candidate=run, reference=ref, support=5, thresholds=(0.01, 0.05, 0.1)
    )
    assert d["extent_error_5cm"] is None
    assert d["maximum_extent_error_5cm"] == pytest.approx(1 / 3)
    assert d["maximum_extent_error_10cm"] == 1
    assert d["maximum_reference_wet_area_10cm_m2"] == 25
    assert "maximum_extent_error_1cm" not in d


def test_practical_matrix_uses_native_sources_and_matches_fixed_area_across_grids():
    from research.floodwarp_methods_paper.fidelity.native_matrix import specifications
    from research.floodwarp_methods_paper.plan import read_config

    study = Path(__file__).resolve().parents[1] / "research/floodwarp_methods_paper"
    cfg = read_config(path=study / "configs/study.json")
    settings = json.loads((study / "fidelity/native_matrix_settings.json").read_text())
    specs = list(specifications(study=study, config=cfg, design=settings))
    assert all(
        s["dx"] == s["dem"] and s["initial_grid"] == "native_1m_block_mean"
        for s in specs
    )
    assert {
        s["landform"] for s in specs if s["experiment_class"] == "operational"
    } == set(cfg["synthetic"]["landforms"])
    matched = [s for s in specs if s["experiment_class"] == "fixed_physical_domain"]
    assert {tuple(s["dimensions"]) for s in matched} == {(1000, 1000)}
    assert {s["dx"] for s in matched} == {1, 5, 10}


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize(("method", "mode"), (("fv2", "dense"), ("hancock", "dense"), ("hancock", "tiles")))
def test_coincident_interior_supercritical_discharge_and_rotated_faces(device, method, mode):
    import warp as wp
    from research.floodwarp_methods_paper.fidelity.free_plane import measure

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    folder = Path(__file__).resolve().parents[1] / "research/floodwarp_methods_paper/fidelity"
    settings = json.loads((folder/"free_plane_settings.json").read_text())
    compared={}
    for distance in (0, 400):
        values=[]
        for face in settings["cardinal_faces"]:
            row = measure(settings=settings, dx=10, extension=distance, method=method, face=face,
                          dt=settings["fine_dt_s"], device=device, execution_mode=mode)[-1]
            assert row["reference_froude"] > 1
            assert abs(row["residual_m3"]) < 1e-8
            values.append(row["discharge_bias_pct"])
        np.testing.assert_allclose(values,values[0],atol=1e-10,rtol=0)
        compared[distance]=values[0]
    assert abs(compared[400]) < .05
    assert abs(compared[0]) < 1.0


def test_native_outlet_budget_partitions_routed_water_and_local_drawdown():
    from research.floodwarp_methods_paper.fidelity.native_tail import changes, measurements

    h = np.array([[1.0, 4.0], [2.0, 5.0]])
    soil = np.zeros_like(h)
    rain = np.zeros_like(h)
    start = measurements(h=h, soil=soil, rain=rain, volumes=np.zeros((2, 3)),
                         distances=[0, 1], dx=1)
    h[:, -1] -= 2
    soil[:, -1] = 0.5
    end = measurements(h=h, soil=soil, rain=rain, volumes=np.array([[0, 2, 4], [0, 3, 4]]),
                       distances=[0, 1], dx=1)
    band = changes(before=start, after=end)[1]
    assert band["net_export_m3"] == 8
    assert band["net_into_strip_m3"] == 5
    assert band["fraction_from_strip_storage"] == 3 / 8
    assert band["residual_m3"] == 0
    assert changes(before=start, after=start)[1]["fraction_from_strip_storage"] is None
    end[1]["storage_m3"] += 1
    with pytest.raises(ArithmeticError, match="water balance"):
        changes(before=start, after=end)
    with pytest.raises(ValueError, match="outside or between"):
        measurements(h=h, soil=soil, rain=rain, volumes=np.zeros((2, 3)), distances=[0, 3], dx=2)


def test_post_rain_peak_and_volume_are_compared_at_shared_horizons():
    from research.floodwarp_methods_paper.fidelity.native_tail import crest_windows

    frames = [
        {"t_s": t, "outlet_m3s": q}
        for t, q in ((60, 1), (120, 2), (180, 3), (240, 5), (300, 0))
    ]
    series = [
        dict(id="long", method="fv1", dx=10, frames=frames, settled=True),
        dict(id="short", method="adaptive", dx=5, frames=frames[:3], settled=False),
    ]
    rows = crest_windows(histories=series, rain_end_s=120, original_end_s=180)
    matched = [r for r in rows if r["window"] == "whole_record" and r["end_s"] == 180]
    assert len(matched) == 2
    assert all(r["peak_interval_mean_m3s"] == 3 and r["net_volume_m3"] == 360 for r in matched)
    after = next(r for r in rows if r["window"] == "after_rain" and r["end_s"] == 300)
    assert after["first_peak_interval_end_s"] == 240
    assert after["recession_passed_at_end"]
    assert after["net_volume_m3"] == 480


def test_independent_native_references_hold_out_the_entire_realisation():
    from research.floodwarp_methods_paper.fidelity.native_matrix import in_suite, specifications
    from research.floodwarp_methods_paper.plan import read_config

    study = Path(__file__).resolve().parents[1] / "research/floodwarp_methods_paper"
    cfg = read_config(path=study / "configs/study.json")
    design = json.loads((study / "fidelity/native_matrix_settings.json").read_text())
    specs = list(specifications(study=study, config=cfg, design=design))
    references = [s for s in specs if in_suite(spec=s, suite="references")]
    held = [s for s in specs if in_suite(spec=s, suite="heldout_references")]
    assert held and {s["split"] for s in held} == {"test"}
    assert {s["seed"] for s in held}.isdisjoint(s["seed"] for s in references if s["split"] != "test")
    for seed in {s["seed"] for s in references}:
        group = [s for s in references if s["seed"] == seed]
        assert len({s["split"] for s in group}) == len({tuple(s["dimensions"]) for s in group}) == 1
        assert {s["dx"] for s in group} == {1, 5, 10}
        assert all(s["dx"] == s["dem"] and s["initial_grid"] == "native_1m_block_mean" for s in group)
        assert {s["factor"] for s in group if s["method"] == "adaptive" and s["dx"] == 1} == {0.5, 1}


def test_native_terrain_readback_rejects_child_and_parent_mismatches(tmp_path):
    from research.floodwarp_methods_paper.fidelity.native_audit import read_grids
    from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells

    native = np.arange(400, dtype=float).reshape(20, 20)
    five = mean_cells(values=native, factor=5)
    ten = mean_cells(values=five, factor=2)
    terrain = dict(inputs={"grids_m": [1, 5, 10], "native_m": 1}, native_sha256={"bed": "test"},
                   files={"bed_5m.npy": "five", "bed_10m.npy": "ten"}, max_direct_10m_difference={"bed": 0.})
    for dx, a in ((1, native), (5, five), (10, ten)):
        np.save(tmp_path / f"bed_{dx}m.npy", a)
    assert read_grids(folder=tmp_path, terrain=terrain)[0]["parents_with_different_children"] == 4
    bad = five.copy()
    bad[0, 0] += 4.
    bad[0, 1] -= 4.
    np.save(tmp_path / "bed_5m.npy", bad)
    with pytest.raises(AssertionError):
        read_grids(folder=tmp_path, terrain=terrain)
    np.save(tmp_path / "bed_5m.npy", five)
    np.save(tmp_path / "bed_10m.npy", ten + 1.)
    with pytest.raises(AssertionError):
        read_grids(folder=tmp_path, terrain=terrain)


def test_time_aggregation_can_hide_a_crest_crossing_an_output_edge():
    from research.floodwarp_methods_paper.fidelity.native_tail import cadence_peaks, crest_windows

    frames = [dict(t_s=(i + 1) * 60, outlet_m3s=q) for i, q in enumerate((0, 8, 8, 0))]
    histories = [dict(id="pulse", method="fv1", dx=5, frames=frames, settled=True)]
    windows = crest_windows(histories=histories, rain_end_s=120, original_end_s=240)
    rows = cadence_peaks(histories=histories, windows=windows, minutes=[1, 2])
    assert [r["peak_m3s"] for r in rows] == [8, 4]
    assert rows[-1]["moving_peak_m3s"] == 8
    assert rows[-1]["first_moving_peak_end_s"] == 180
    assert rows[0]["net_volume_m3"] == rows[1]["net_volume_m3"] == 960


def test_infiltration_accounting_includes_cells_outside_the_rain_footprint():
    from research.floodwarp_methods_paper.fidelity.native_infiltration import losses

    soil = np.array([0.002, 0.001])
    kw = dict(rain=np.array([0.01, 0.0]), h=np.zeros(2), multiplier=np.array([1., 0.]),
              dx=10, seconds=3600, capacity=0.002 / 3600)
    regions = losses(soil=soil, **kw)
    assert regions["rain_footprint"]["infiltration_m3"] == pytest.approx(0.2)
    assert regions["outside_rain_footprint"]["infiltration_m3"] == pytest.approx(0.1)
    assert regions["outside_rain_footprint"]["equivalent_area_hours_km2_h"] == pytest.approx(0.00005)
    with pytest.raises(ValueError):
        losses(soil=soil*2, **kw)


def test_native_spill_elevation_requires_a_face_connected_path():
    from research.floodwarp_methods_paper.fidelity.native_features import spill_level

    bed = np.array([[0., 9., 9.], [9., 1., 9.], [9., 9., 0.]])
    assert spill_level(bed=bed) == 9
    bed[1, 0], bed[1, 2] = 2, 3
    assert spill_level(bed=bed) == 3


def test_rebuilt_native_states_validate_soil_and_faces_as_well_as_depth():
    from types import SimpleNamespace
    from research.floodwarp_methods_paper.fidelity.restore_native import verify_fields
    from research.floodwarp_methods_paper.terrain.generators.generate import array_sha

    class Array:
        def __init__(self, values):
            self.values = values

        def numpy(self):
            return self.values

    def q():
        return np.array([[[0.002, 0.001, 0.0]]])

    def maximum():
        return np.array([[0.004]])

    fields = {k: Array(np.ones(2)) for k in ("volume_x", "volume_y", "soil", "rain_sum")}
    solver = SimpleNamespace(_fields=fields, get_conservative_state=q, get_max_water_depth=maximum)
    expected = dict(state=array_sha(q()), maximum=array_sha(maximum()), **{k: array_sha(v.numpy()) for k, v in fields.items()})
    verify_fields(solver=solver, expected=expected)
    fields["volume_x"].values[0] = 0
    with pytest.raises(ValueError, match="volume_x"):
        verify_fields(solver=solver, expected=expected)


def test_native_advisor_freezes_before_scoring_held_out_groups(tmp_path, monkeypatch):
    from research.floodwarp_methods_paper.fidelity import native_advisor

    directory = tmp_path / "research/paper/fidelity"
    directory.mkdir(parents=True)
    setting = dict(fit_percent=10, assessment_percentages=[5, 10], depth_tolerance_m=.01,
                   validation_groups_per_spacing=2, hardware="test GPU", rules=["advisor", "slope", "fv1", "hancock_dense", "oracle"],
                   new_final_terrain_ids=["test0"], practical_pair_m=[5,10], evaluation_wet_edges=[0, .01, .05, .2, .5, 1])
    config = dict(advisor=dict(heldout_family="mixed", neighbours=[2],coverage_quantile=.95,quantile=.9))
    accessed = []

    def sample_rows(*, partitions, **kwargs):
        accessed.append(partitions)
        data=[]
        for part in partitions:
            for i in range(3):
                group=f"{part}{i}"
                for dx in (5,10):
                    for m,work in (("fv1",1),("hancock_dense",2)):
                        f=dict(mean_slope=.01+i*.001,resolution=dx,cells=(1000//dx)**2,area_m2=1e6,
                               rain_mm=50.,forecast_duration_h=8.,critical_connection_lost=False)
                        goals=native_advisor.tolerances(pct=10,depth_m=.01)
                        errors={o:.001*work for o in goals}
                        data.append(dict(terrain_id=group,case=group,landform="plain",storm="moderate",split=part,
                                         run_id=f"{group}-{dx}-{m}",method=m,features=f,hardware="test GPU",
                                         native_sha=group,end_h=8,interval_s=60,reference_id=group+"-fine",
                                         spatial_reference_id=group+"-fine",spatial_support_m=10,
                                         runtime_s=float(work),memory_mib=4.,wet_fraction=.1,sampled_regime=None,
                                         errors=errors,qualified=dict.fromkeys(goals,True)))
        return data

    def revision(*args,**kwargs):
        return "test-revision"

    monkeypatch.setattr(native_advisor, "sample_rows",sample_rows)
    monkeypatch.setattr(native_advisor.subprocess,"check_output",revision)
    args=dict(directory=directory,config=config,design={},settings=setting)
    path=native_advisor.train(**args)
    assert accessed==[("development","validation")]
    frozen=json.loads(path.read_text())
    assert set(frozen["training_terrain_ids"]).isdisjoint(frozen["validation_terrain_ids"])
    assert (path.parent/"frozen.json").is_file()
    test=native_advisor.validate(**args)
    assert accessed[-1]==("test",)
    assert all((test/name).is_file() for name in ("decisions.csv","strata.csv","candidate_predictions.csv"))
    from research.floodwarp_methods_paper.metrics import csv_read
    joint = csv_read(file=test/"joint_summary.csv")
    assert {r["requested_dx"] for r in joint} == {"5","10"}
    assert all(int(r["regret_assessable"]) <= int(r["empirical_feasible_opportunities"]) for r in joint)
    from research.floodwarp_methods_paper.fidelity.native_publication import advisor_report

    publication = directory / "native_publication"
    report, sources = advisor_report(directory=directory, output=publication)
    assert any("Measured oracle" in line for line in report)
    assert test / "strata.csv" in sources
    assert all((publication / ("heldout_advisor." + ext)).is_file() for ext in ("pdf", "svg", "png"))
    (test / "summary.csv").write_text("changed")
    with pytest.raises(ValueError, match="summary.csv"):
        advisor_report(directory=directory, output=publication)
    assert json.loads(path.read_text()) == frozen
    path.write_text("{}")
    with pytest.raises(ValueError,match="checksum"):
        native_advisor.validate(**args)


def test_native_policy_costs_use_matched_totals_and_record_auto(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import native_policies, parity

    groups = {("original-window", "plain", "native"): []}
    for method, times in (("hancock_dense", [5., 6.]), ("hancock_adaptive", [2., 7.]),
                          ("hancock_auto", [1., 7.]), ("fv1", [1., 3.])):
        frames = [dict(t_s=t, steps=10, hydraulic_s=s, outlet_m3s=0., section_m3s=0.)
                  for t, s in zip((900, 1800), times, strict=True)]
        data = dict(frames=frames, hydraulic_s=sum(times), hashes=dict.fromkeys(
            ("state", "maximum", "volume_x", "volume_y", "soil", "rain_sum"), "a" * 64))
        item = dict(id=method, s=dict(dx=1, hours=.5, method=method, dimensions=[10, 20]),
                    d=data, native_sha="same-grid", factor=1.)
        groups["original-window", "plain", "native"].append(item)
    verified = parity.compare(directory=tmp_path, groups=groups, target=tmp_path, original_window=True)
    native_policies.collect(groups=groups, verified=verified, cadence=900, target=tmp_path)
    [record] = metrics.csv_read(file=tmp_path / "policy_cost_events.csv")
    assert record["fastest_measured"] == "auto"
    assert float(record["fastest_over_fv1"]) == 2.
    assert float(record["tiled_over_dense"]) == pytest.approx(9 / 11)
    rows = metrics.csv_read(file=tmp_path / "policy_cost_windows.csv")
    assert [r["fastest_measured"] for r in rows] == ["auto", "dense"]
    with pytest.raises(ValueError, match="Missing exact hydraulic parity"):
        native_policies.collect(groups=groups, verified=[], cadence=900, target=tmp_path)
    del groups["original-window", "plain", "native"][1]["d"]["hashes"]["soil"]
    with pytest.raises(ValueError, match="soil"):
        parity.compare(directory=tmp_path, groups=groups, target=tmp_path, original_window=True)
    for member in groups["original-window", "plain", "native"]:
        member["d"]["hashes"]["soil"] = None
    with pytest.raises(ValueError, match="Invalid SHA-256 fingerprint.*soil"):
        parity.compare(directory=tmp_path, groups=groups, target=tmp_path, original_window=True)
    for member in groups["original-window", "plain", "native"]:
        member["d"]["hashes"]["soil"] = "a" * 64
        member["d"]["frames"] = member["d"]["frames"][:-1]
    with pytest.raises(ValueError, match="Incomplete interval transport history"):
        parity.compare(directory=tmp_path, groups=groups, target=tmp_path, original_window=True)


def test_spill_sensitivity_uses_conservative_nested_beds():
    from research.floodwarp_methods_paper.fidelity.native_control_report import control_levels

    bed = np.zeros((20, 20))
    bed[:, 10:15] = .20
    bed[10:12, 10:15] = 0.
    grids, rows = control_levels(native=bed, heads_cm=[0, 5, 10, 20])
    np.testing.assert_allclose(grids[10], grids[5].reshape(2, 2, 2, 2).mean(axis=(1, 3)))
    table = {(r["dx_m"], r["head_above_native_spill_cm"]): r for r in rows}
    assert table[1, 0]["connected"]
    assert not table[5, 10]["connected"]
    assert table[5, 20]["connected"]
    assert table[5, 10]["shift_cm"] == pytest.approx(12)
    assert table[10, 10]["connected"]


def test_explicit_control_geometry_and_unassessed_connection():
    from affine import Affine
    from floodwarp.terrain_advisor.controls import extract_control, nested_grid
    from floodwarp.solver_advisor.model import descriptors
    from floodwarp.terrain_advisor.diagnostics import analyse_array

    native = np.zeros((20, 20))
    native[:, 10:15] = 1
    native[9, 10:15] = 0
    north = native[::-1]
    represented = nested_grid(north_up=north, native_dx=1., dx=5.)
    transform = Affine.translation(100, 220) * Affine.scale(1, -1)
    control = dict(window_m=[100, 200, 20, 20], spill_head_m=.01, axis="west_east")
    marked = extract_control(native=north, represented=represented, transform=transform, dx=5,
                             control=control, window_m=10)
    assert marked["native_connected"] and not marked["represented_connected"]
    assert marked["represented_spill_m"] == pytest.approx(.8)
    assert marked["max_local_barrier_lowering_m"] == pytest.approx(.2)
    stats, _, _ = analyse_array(elevation=north, native_resolution=1, resolution=5)
    assert descriptors(terrain=stats)["critical_connection_lost"] is None
    assert descriptors(terrain=dict(stats, comparison=marked))["critical_connection_lost"] is True


def test_native_outlet_restart_requires_matching_reconstruction():
    from floodwarp.fv2_solver import FV2Solver
    options = dict(nx=4,ny=3,dx=1.,dy=1.,bed_elevation=np.zeros((3,4)),
                   active_mask=np.ones((3,4), bool), manning_n=np.full((3,4),.03),device="cpu")
    legacy = FV2Solver(**options,free_boundary_reconstruction="constant")
    current = FV2Solver(**options)
    old = legacy.capture_state()
    legacy.restore_state(state=old)
    assert "free_boundary_reconstruction" not in old["configuration"]
    assert current.numerical_configuration()["free_boundary_reconstruction"] == "one_sided"
    with pytest.raises(ValueError,match="configuration"):
        current.restore_state(state=old)


def test_open_face_reconstruction_distinguishes_mask_edges_and_corner_walls():
    from floodwarp import fv2_kernels
    from floodwarp.fv2_solver import FV2Solver

    ny, nx = 4, 7
    active = np.ones((ny,nx), dtype=bool)
    active[:, -1] = False
    bed = np.tile(-.5*np.arange(nx), (ny,1)) - .1*np.arange(ny)[:, None]
    s = FV2Solver(nx=nx,ny=ny,dx=1.,dy=1.,bed_elevation=bed,manning_n=np.full_like(bed,.035),
                  active_mask=active,device="cpu",limiter="mc")
    typ = np.zeros((ny,nx+1),np.int32)
    typ[-1,-2] = 2
    typ[1,-2] = 3
    s.set_boundary_type_qx(boundary_type=typ)
    s.set_water_depth(depth=np.full_like(bed,.03))
    f = s._fields
    s._launch_grid(fv2_kernels.inverse_depth,nx*ny,[s.q,s._inverse,s.dry_depth],layout=0)
    s._launch_grid(fv2_kernels.slopes,nx*ny,[s.q,s._inverse,s._shock_flags,f["bed"],f["active"],
                   *s._slopes,nx,ny,s.dry_depth,s._limiter_code,f["types_x"],f["types_y"],1],layout=0)
    sx,sy = [a.numpy().reshape(ny,nx,4) for a in s._slopes]
    np.testing.assert_allclose(sx[-1,-2],[-.5,-.5,0.,0.],atol=1e-14)
    np.testing.assert_array_equal(sx[:-1,-2],0.)
    np.testing.assert_array_equal(sx[:,-1],0.)
    np.testing.assert_array_equal(sy[-1,-2],0.)


def test_spatial_depth_error_uses_each_practical_wet_union_on_matched_support():
    from research.floodwarp_methods_paper.fidelity.final_depth import difference

    candidate = np.array([[.010, .050, .120, .001]])
    reference = np.array([[.011, .040, .090, .150]])
    fields = difference(candidate=(candidate, candidate), reference=(reference, reference),
                        support=10, thresholds=[.01, .05, .10])
    assert fields["wet_union_area_5cm_m2"] == 300
    assert fields["wet_union_area_10cm_m2"] == 200
    assert fields["wet_union_depth_rmse_5cm_m"] == pytest.approx(np.sqrt((.01**2 + .03**2 + .149**2) / 3))
    assert fields["wet_union_depth_rmse_10cm_m"] == pytest.approx(np.sqrt((.03**2 + .149**2) / 2))
    dry = np.zeros_like(candidate)
    fields = difference(candidate=(dry, dry), reference=(dry, dry), support=10, thresholds=[.05, .10])
    assert fields["wet_union_area_5cm_m2"] == 0
    assert fields["wet_union_depth_rmse_5cm_m"] is None


def test_native_paper_budgets_keep_qualified_event_and_unmet_budget(tmp_path):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.tables.generate import add_decision_tables

    base = dict(cohort="complete-native-metre", case="event", representation="area means", station="section",
                end_h=20, interval_s=60, reference_id="fine", objective="peak_bias_pct", measured_error="",
                reference_sensitivity="", execution="", runtime_s="", steps="", hydraulic_s_per_step="")
    feasible = dict(base, tolerance=5, assessable=3, feasible=0, method="", dx="", id="")
    budget = dict(base, assessable_in_budget=0, budget_dx=10, budget_s=1.5, ranking_status="unassessable",
                  selected_method="", selected_dx="", selected_id="")
    for name, rows in (("least_cost", [feasible, dict(feasible, case="unresolved", assessable=0)]),
                       ("equal_cost", [budget, dict(budget, case="unresolved")])):
        metrics.csv_write(file=tmp_path / f"{name}_fidelity.csv", rows=rows)
        metrics.csv_write(file=tmp_path / f"{name}_depth_extent.csv", rows=[])
    output = {}
    add_decision_tables(output=output, directory=tmp_path, incised=False)
    _, cost = output["21_native_cost_at_tolerance"]
    _, error = output["22_native_error_at_budget"]
    assert len(cost) == len(error) == 1
    assert cost[0]["case"] == error[0]["case"] == "event"
    assert cost[0]["qualified"] == "3" and cost[0]["feasible"] == "0"
    assert error[0]["qualified_in_budget"] == "0" and error[0]["id"] == ""
