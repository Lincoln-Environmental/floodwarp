import json
from zoneinfo import ZoneInfo

import pytest

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import gpu_sharing, repeat_costs, statistics


def test_sharing_only_counts_same_tick_device_and_active_compute_process():
    lines = [
        "20261125 12:40:00 0 42 C 75",
        "20261125 12:40:00 0 14 C -",
        "20261125 12:40:00 0 15 G 50",
        "20261125 12:40:00 1 14 C 80",
        "20261125 12:40:01 0 14 C 80",
        "20261125 12:40:02 0 42 C 70",
        "20261125 12:40:02 0 15 C 30",
        "20261125 12:40:02 0 16 C 20",
    ]
    samples = gpu_sharing.process_samples(lines=lines, zone=ZoneInfo("Pacific/Auckland"), gpu=0, solver_pid=42)
    start = samples[0][0]
    assert samples == [[start, 75, 0], [start + 1, 0, 80], [start + 2, 70, 30]]
    series = ([s[0] for s in samples], {s[0]: s[1:] for s in samples})
    assert gpu_sharing.window_status(samples=series, windows=[(start, start + 2)])["gpu_shared_samples"] == 0
    assert gpu_sharing.window_status(samples=series, windows=[(start + 2, start + 2.5)])["raw_timing_status"] == "unmonitored"
    result = gpu_sharing.window_status(samples=series, windows=[(start, start + 3), (start + 2, start + 3)])
    assert (result["gpu_active_samples"], result["gpu_shared_samples"]) == (2, 1)


def test_capture_excludes_in_progress_tick_and_checks_content_identity(tmp_path):
    log = tmp_path / "pmon.log"
    log.write_text("20260101 00:00:00 0 1 C 83\n20260101 00:00:00 0 2 C 15\n20260101 00:00:01 0 1 C 95\n")
    opts = dict(directory=tmp_path, log=log, zone=ZoneInfo("UTC"), gpu=0, solver_pid=1,
                case="test", device_uuid="gpu", before=1767225601)
    gpu_sharing.save_capture(**opts)
    gpu_sharing.save_capture(**opts)
    [path] = (tmp_path / "gpu_samples").iterdir()
    assert gpu_sharing.read_samples(directory=tmp_path)["test"][0] == [1767225600]
    data = json.loads(path.read_text())
    data["samples"][0][2] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Altered GPU sample"):
        gpu_sharing.read_samples(directory=tmp_path)


def test_repeat_cost_preserves_shared_raw_id_and_requires_a_usable_clock(tmp_path):
    event = dict(horizon="complete-native-metre", case="river", method="fv1", execution_mode="sparse",
                 dx_m=5, factor=1, interval_s=60, end_h=24, native_sha="terrain", solver_source="solver",
                 hardware="gpu", steps=30, net_outlet_m3=10, infiltration_m3=2, storage_m3=0, outlet_peak_m3s=1)
    shared = dict(event, source_id="shared", hydraulic_s=20, end_to_end_s=21, timing_repeat=0,
                  raw_timing_status="shared GPU observed")
    available = dict(event, source_id="other", hydraulic_s=11, end_to_end_s=12, timing_repeat=1)
    summary = repeat_costs.collect(rows=[shared, available], target=tmp_path)["shared"]
    assert summary["included_run_ids"] == "other"
    assert summary["run_ids"] == "shared other"
    assert summary["independent_timings"] == 1
    assert summary["hydraulic_median_s"] == 11
    assert summary["end_to_end_median_s"] == 12
    only = repeat_costs.collect(rows=[shared], target=tmp_path)["shared"]
    assert only["hydraulic_median_s"] is None
    assert metrics.csv_read(file=tmp_path / "timing_distribution.csv")[0]["shared_gpu_executions"] == "1"
    row = dict(cohort="primary", case="river", representation="bed", station="outlet", method="fv1", dx=5,
               experiment_id="shared", qualified_peak=True, peak=1.2, runtime_s=only["hydraulic_median_s"])
    [result] = statistics.frontiers(rows=[row], objectives=["peak"])
    assert result["absolute_error"] == 1.2
    assert not result["eligible"]
    assert result["nondominated"] is None


def test_recession_inherits_own_prefix_and_checks_only_elapsed_continuations(tmp_path):
    log = tmp_path / "pmon.log"
    log.write_text("20260101 00:00:01 0 4 C 90\n20260101 00:00:09 0 4 C 70\n20260101 00:00:09 0 8 C 30\n")
    gpu_sharing.save_capture(directory=tmp_path, log=log, zone=ZoneInfo("UTC"), gpu=0,
                             solver_pid=4, device_uuid="gpu", case="pulse", before=None)
    base_ns = 1767225600 * 10**9
    attempt = f"attempt_{base_ns}"
    folder = tmp_path / "prefix"
    (folder / attempt).mkdir(parents=True)
    (folder / attempt / "complete.json").write_text(json.dumps({"seconds": 3.5}))
    spec = {"case": "pulse", "hours": 8}
    parent = dict(id="original", s=spec, path=folder)
    short = dict(id="tail", parent_id="original", s=dict(spec, hours=12))
    long = dict(id="tail", parent_id="original", s=dict(spec, hours=16))
    tail = tmp_path / "extension_timings/tail"
    tail.mkdir(parents=True)
    for timestamp, end_h in ((6, 12), (12, 16)):
        (tail / f"{timestamp}.json").write_text(json.dumps(dict(case="pulse", id="tail", end_s=end_h*3600,
               recorded_ns=base_ns + timestamp * 10**9, elapsed_wall_s=4)))
    gpu_sharing.assess(directory=tmp_path, records={"original": parent}, verification=[dict(id="original", attempt=attempt)],
                      groups={("short", "pulse", "native"): [short], ("long", "pulse", "native"): [long]})
    assert parent["gpu_shared_samples"] == short["gpu_shared_samples"] == 0
    assert long["gpu_active_samples"] == 2
    assert long["gpu_shared_samples"] == 1
