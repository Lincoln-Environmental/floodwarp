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
