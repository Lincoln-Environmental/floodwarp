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
