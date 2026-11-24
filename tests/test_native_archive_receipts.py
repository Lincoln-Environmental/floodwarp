import gzip
import json

import pytest

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity import native_archive


def test_completed_attempt_authenticates_fields_and_result(tmp_path):
    result = tmp_path / "result.json"
    spatial = tmp_path / "spatial.npz"
    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    result.write_text('{"steps":2}')
    spatial.write_bytes(b"fields")
    receipt = dict(result_sha256=runner.checksum(path=result), spatial_sha256=runner.checksum(path=spatial))
    (attempt / "complete.json").write_text(json.dumps(receipt))
    assert native_archive.prefix_evidence(folder=tmp_path) == dict(receipt, attempt=attempt.name)
    spatial.write_bytes(b"altered")
    with pytest.raises(ValueError, match="authenticates"):
        native_archive.prefix_evidence(folder=tmp_path)
    spatial.write_bytes(b"fields")
    result.write_text('{"steps":3}')
    with pytest.raises(ValueError, match="authenticates"):
        native_archive.prefix_evidence(folder=tmp_path)


def test_frozen_horizon_detects_changed_gauge_with_same_final_storage(tmp_path):
    ident, parent, end = "series", "prefix", 900
    folder = tmp_path / ident
    horizons = folder / "horizons"
    horizons.mkdir(parents=True)
    saved = dict(id=ident, parent_id=parent, end_s=end, storage_m3=20, frames=[dict(section_m3s=2.0)])
    path = horizons / "900.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(saved, stream)
    cursor = folder / "series.json"
    cursor.write_text(json.dumps(saved))
    expected = dict(horizon_sha256=runner.checksum(path=path))
    assert native_archive.horizon_evidence(root=tmp_path, ident=ident, parent=parent, end_s=end) == expected
    cursor.write_text(json.dumps(dict(saved, frames=[dict(section_m3s=1.5)])))
    with pytest.raises(ValueError, match="frozen measurements"):
        native_archive.horizon_evidence(root=tmp_path, ident=ident, parent=parent, end_s=end)
    cursor.write_text(json.dumps(dict(saved, end_s=1800)))
    assert native_archive.horizon_evidence(root=tmp_path, ident=ident, parent=parent, end_s=end) == expected
    with pytest.raises(ValueError, match="wrong identity"):
        native_archive.horizon_evidence(root=tmp_path, ident=ident, parent="other", end_s=end)


def test_saved_recession_cfl_matches_archived_face_bound():
    from pathlib import Path
    from research.floodwarp_methods_paper.fidelity import recession_cfl

    directory = Path(recession_cfl.__file__).parent
    ident = "42545c16666c1c94109b354a"
    evidence = directory / "wave_clock" / ident / "72000.npz"
    row = recession_cfl.compare(evidence=evidence, endpoint=directory / "endpoint_fields" / ident / evidence.name)
    assert row["gravity_wave_ms"] > 1 and row["axis_velocity_ms"] < 1e-5
    assert abs(row["recorded_face_bound_s"] - row["cell_signal_bound_s"]) < 1e-8
    assert row["next_predictor_s"] == pytest.approx(.95 * row["recorded_face_bound_s"])


def test_native_timing_joins_original_policy_intervals_to_the_measured_tail():
    from research.floodwarp_methods_paper.fidelity.extended_analysis import native_intervals

    first = dict(t_s=60, outlet_m3s=0.1, section_m3s=0.2, steps=10, hydraulic_s=15.0)
    primary = dict(frames=[first], hydraulic_s=15.0)
    other = dict(first, hydraulic_s=3.0)
    tail = dict(other, t_s=120, hydraulic_s=4.0)
    horizon = dict(frames=[other, tail], hydraulic_s=7.0)
    frames, elapsed, detail = native_intervals(primary=primary, horizon=horizon)
    assert elapsed == 19 and [r["hydraulic_s"] for r in frames] == [15, 4]
    assert detail["changed_timing_intervals"] == 1 and detail["raw_horizon_hydraulic_s"] == 7
    assert horizon["frames"][0]["hydraulic_s"] == 3
    for field in ("section_m3s", "outlet_m3s", "steps", "t_s"):
        invalid = dict(horizon, frames=[dict(other, **{field: 0}), tail])
        with pytest.raises(ValueError, match="accepted"):
            native_intervals(primary=primary, horizon=invalid)
    with pytest.raises(ValueError, match="timing total"):
        native_intervals(primary=primary, horizon=dict(horizon, hydraulic_s=1))


def test_native_phase_totals_combine_recorded_prefix_aggregate_and_tail_intervals():
    from research.floodwarp_methods_paper.fidelity.phase_cost import event_diagnostics

    base = dict(id="continued", parent_id="primary")
    frames = [dict(t_s=60, steps=20), dict(t_s=120, steps=18, compilation_s=.09,
                                           graph_capture_s=.03, extraction_s=.4)]
    data = dict(frames=frames, spatial=[dict(live_pool_mib=32)], hydraulic_s=4,
                native_timing_provenance=dict(initial_intervals=1),
                native_prefix_diagnostics=dict(preload_s=.6, graph_capture_s=.2, extraction_s=.5,
                                               setup_s=1.1, live_pool_mib=48))
    row = event_diagnostics(item=dict(base, d=data))
    assert row["graph_capture_s"] == pytest.approx(.23)
    assert row["compilation_s"] == pytest.approx(.69)
    assert row["extraction_s"] == pytest.approx(.9)
    assert row["graph_capture_s_intervals"] == 2
    assert row["sampled_pool_peak_mib"] == 48 and row["source_final_pool_mib"] == 48
    del frames[-1]["compilation_s"]
    row = event_diagnostics(item=dict(base, d=data))
    assert row["compilation_s"] is None and row["compilation_s_intervals"] == 1


def test_paper_reports_distinct_final_and_ever_wet_support():
    from research.floodwarp_methods_paper.tables.generate import spatial_table_support

    field = dict(support_m="10", reference_wet_area_5cm_m2="700", wet_union_area_5cm_m2="800",
                 maximum_reference_wet_area_10cm_m2="200", maximum_wet_union_area_10cm_m2="300")
    final = spatial_table_support(field=field, objective="extent_error_5cm")
    maximum = spatial_table_support(field=field, objective="maximum_extent_error_10cm")
    assert final["reference_wet_cells"] == 7 and final["wet_union_cells"] == 8
    assert maximum["reference_wet_cells"] == 2 and maximum["wet_union_cells"] == 3


def test_native_cell_comparison_requires_fingerprinted_maximum(tmp_path):
    import numpy as np

    from research.floodwarp_methods_paper.fidelity.native_map_comparison import load_maps
    from research.floodwarp_methods_paper.terrain.generators.generate import array_sha

    path = tmp_path / "state.npz"
    q = np.array([[.1, 0., 0.], [.04, 0., 0.]])
    maximum = np.array([.2, .11])
    manifest = dict(sha256=dict(q=array_sha(q), maximum=array_sha(maximum)))
    np.savez(path, q=q, maximum=maximum, _direct_manifest=json.dumps(manifest), control=np.array([3600.]))
    fields = dict(state=array_sha(q), maximum=array_sha(maximum), volume_x="x", soil="soil")
    item = dict(id="ordinary", d=dict(hashes=fields, numerical=dict(ny=1, nx=2)), s=dict(hours=1))
    index = {(fields["state"], "x", "soil"): path}
    depth, peak = load_maps(item=item, index=index)
    np.testing.assert_array_equal(depth, [[.1, .04]])
    np.testing.assert_array_equal(peak, [[.2, .11]])
    fields["maximum"] = "wrong"
    with pytest.raises(ValueError, match="disagree"):
        load_maps(item=item, index=index)


def test_native_map_shared_recession_requires_both_parents_at_same_time(tmp_path):
    from research.floodwarp_methods_paper.fidelity.native_map_endpoints import common_end

    groups = tmp_path / "extended_runs"
    groups.mkdir()
    group = dict(case="plain", cohort="native-metre", sources=["h", "r"], end_s=43200)
    (groups / "group_auc.json").write_text(json.dumps(group))
    assert common_end(directory=tmp_path, case="plain", parents={"h", "r"}) == 43200
    assert common_end(directory=tmp_path, case="plain", parents={"h", "f", "r"}) is None
    (groups / "group_new.json").write_text(json.dumps(dict(group, end_s=86400)))
    with pytest.raises(ValueError, match="Ambiguous"):
        common_end(directory=tmp_path, case="plain", parents={"h", "r"})
