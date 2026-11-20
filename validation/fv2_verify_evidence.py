"""Verify the numerical gates, publish measured outcomes and freeze data and source identities."""

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import warp as wp

from validation.fv1_riemann import report_json
from validation.fv2_figures import rows_table


def digest(*, path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_measured_summary(*, directory, grids, methods, max_residual, native, common):
    peaks = []
    entries = {}
    for dx in grids:
        row = [dx]
        for method in methods:
            record = json.loads((directory / "hirds48" / f"{dx}m_{method}.json").read_text())
            entries[dx, method] = record
            row.append(record["peak_outflow_m3_s"])
        peaks.append(row)
    table = rows_table(
        title="Complete 48-hour outlet peaks (m³/s)",
        columns=["Spacing (m)", "Contact-HLL", "Roe/HLL", "HLLC/HLL", "FV2 MC"],
        rows=peaks,
        note=f"All twelve runs received the same 578-node HIRDS forcing. The largest absolute storm balance residual was {max_residual:.2g} m³.",
    )
    finest = min(grids)
    second = entries[finest, "fv2_mc"]
    base = entries[finest, "contact_hll"]
    metrics = next(r["vs_5m_contact"] for r in common if r["dx"] == finest and r["method"] == "fv2_mc")
    minute = int(second["peak_interval_end_s"] / 60)
    storage = second["ledger"]["storage"] - base["ledger"]["storage"]
    text = (
        f"At {finest:g} m the FV2 peak occurred at {minute // 60} h {minute % 60:02d} min. "
        f"Relative to same-grid FV1, final storage changed by {storage:+,.0f} m³. "
        f"Maximum-depth RMSE between the two methods was {metrics['peak_envelope_rmse_m']:.4f} m, and maximum inundation IoU above 1 cm was {metrics['envelope_wet_iou']:.3f}. "
        "The reference is the original 5 m model. The full-domain comparison measures changes to predicted hydraulics."
    )
    reductions = []
    for r in native["comparisons"]:
        m = r["improvements"]["depth_rmse_m"]
        reductions.append(
            [
                r["dx_m"],
                m["improvement"] * 1000,
                m["percent_of_coarse_fv1_error"],
                *[x * 1000 for x in m["reference_range"]],
            ]
        )
    gaps = [
        native[k]["depth_rmse_m"] * 1000
        for k in ("reference_last_mc_refinement", "reference_cross_formulation", "reference_dt_halving")
    ]
    native_table = rows_table(
        title="Did FV2 improve agreement with the native coastal-tile hydraulics?",
        columns=[
            "Spacing (m)",
            "Reduction in final-depth RMSE (mm)",
            "Reduction relative to FV1 error (%)",
            "Smallest reduction across references (mm)",
            "Largest reduction across references (mm)",
        ],
        rows=reductions,
        note=(
            f"Positive values mean FV2 reduced the difference from the full 48-hour reference. The primary reference uses 0.25 m MC hydraulics on the fixed 1 m survey. "
            f"The final MC grid refinement, finest cross-method difference and timestep-cap halving changed final depths by {gaps[0]:.3f}, {gaps[1]:.3f} and {gaps[2]:.3f} mm RMSE on common 1 m support. "
            "The range uses fine MC, fine FV1 and fine MC with the smaller timestep cap. It describes reference sensitivity."
        ),
    )
    begin = "<!-- measured-storm-summary:start -->"
    end = "<!-- measured-storm-summary:end -->"
    path = directory / "REPORT.md"
    report = path.read_text()
    if report.count(begin) != 1 or report.count(end) != 1:
        raise ValueError("Missing unique measured-storm section")
    before, section = report.split(begin, 1)
    _, after = section.split(end, 1)
    path.write_text(
        before + begin + "\n\n" + table + "\n" + text + "\n\n" + native_table + "\n" + end + after
    )


def check(
    *,
    directory,
    grids,
    methods,
    analytical_tolerance,
    balance_tolerance,
    test_log,
    output,
):
    def record(*, name):
        return json.loads((directory / name).read_text())

    all_cases = record(name="analytical.json")
    if len(all_cases) != 96 or min(r["depth_min_m"] for r in all_cases) < 0:
        raise AssertionError("Incomplete analytical matrix or negative depth")
    lake = [r for r in all_cases if r["case"] == "lake"]
    max_drift = max(r["depth_max_error_m"] for r in lake)
    if max_drift > analytical_tolerance:
        raise AssertionError("Lake at rest drift exceeds FP64 tolerance")
    max_balance = max(r["balance"]["normalised_residual"] for r in all_cases)
    if max_balance > balance_tolerance:
        raise AssertionError("Analytical mass-balance gate failed")
    space = record(name="additional.json")["fixed_time_spatial"]
    time = [r for r in record(name="manufactured.json") if r["study"] == "time"]
    p_space = np.log2(np.array(space[-2]["rmse"]) / space[-1]["rmse"])
    p_time = np.log2(np.array(time[-2]["rmse"]) / time[-1]["rmse"])
    if min(*p_space, *p_time) < 1.8:
        raise AssertionError("Smooth convergence gate failed")
    rain = []
    residuals = []
    for dx in grids:
        for method in methods:
            r = record(name=f"hirds48/{dx}m_{method}.json")
            assert (r["dx"], r["method"], r["time_s"], len(r["hydrograph"])) == (
                dx,
                method,
                172800.0,
                2880,
            )
            np.testing.assert_array_equal([h["t"] for h in r["hydrograph"]], np.arange(60, 172801, 60))
            assert r["minimum_depth_m"] >= 0.0
            area = r["forcing"]["integrated_volume_m3"]
            np.testing.assert_allclose(r["ledger"]["rain"], area, rtol=1.0e-10)
            rain.append(r["ledger"]["rain"])
            residuals.append(abs(r["ledger"]["residual"]))
    assert max(residuals) / min(rain) < balance_tolerance
    for r in record(name="hirds48/stage_replay.json"):
        assert max(r["max_absolute_differences"].values()) == 0.0
    for filename in ("common_support.json", "comparison.json", "wet_cost.json"):
        assert len(record(name="hirds48/" + filename)) == len(grids) * len(methods)
    references = record(name="synthetic_reference.json")
    assert len(references) == 14
    assert all(r["reference_grids_m"] == [0.625, 0.3125] for r in references)
    assert len(record(name="hirds48/stage_replay.json")) == len(grids)
    assert len(record(name="hirds48/muscl_fallback.json")) == len(grids) * 3
    assert len(record(name="hirds48/lisflood_comparison.json")["rows"]) == len(rain)
    geom = record(name="storm_geometry.json")
    assert {(r["method"], r["terrain_dx_m"]) for r in geom} == {
        (m, d) for m in ("fv1", "mc") for d in (5, 10)
    }
    native_result = record(name="native_storm_improvement.json")
    assert len(native_result["comparisons"]) == 5 and len(native_result["connectivity"]) == 10
    auxiliary = record(name="native_hirds48.json")
    assert len(auxiliary["rows"]) == 15
    assert all(r["time_s"] == 172800.0 for r in auxiliary["rows"])
    results = dict(
        fv2_terrain_qualification="superseded: MC bed traces can reverse a monotone face",
        corrected_implementation_commit="94fb7df",
        analytical_runs=len(all_cases),
        lake_max_depth_drift_m=max_drift,
        max_normalised_analytical_balance=max_balance,
        spatial_orders=p_space.tolist(),
        temporal_orders=p_time.tolist(),
        storms=len(rain),
        complete_native_tile_storms=len(auxiliary["rows"]),
        fixed_coarse_terrain_tile_storms=len(geom),
        min_max_storm_rain_m3=[min(rain), max(rain)],
        max_absolute_storm_balance_m3=max(residuals),
    )
    if test_log.is_file():
        text = test_log.read_text()
        counts = re.search(r"(\d+) passed, (\d+) skipped, (\d+) warnings in ([0-9.]+)s", text)
        if not counts:
            raise AssertionError("No successful complete-suite summary found")
        results["tests"] = dict(
            passed=int(counts[1]),
            skipped=int(counts[2]),
            warnings=int(counts[3]),
            elapsed_s=float(counts[4]),
            skips=[line for line in text.splitlines() if line.startswith("SKIPPED")],
            log_sha256=digest(path=test_log),
        )
    sources = list(Path("src/floodwarp").glob("fv2*.py")) + list(Path("validation").glob("fv2*.py"))
    sources.extend(
        Path(p)
        for p in (
            "src/floodwarp/full_swe_kernels.py",
            "src/floodwarp/full_swe_solver.py",
            "src/floodwarp/solver_factory.py",
            "src/floodwarp/model.py",
            "tests/test_fv2.py",
        )
    )
    publish_measured_summary(
        directory=directory,
        grids=grids,
        methods=methods,
        max_residual=max(residuals),
        native=native_result,
        common=record(name="hirds48/common_support.json"),
    )
    artifacts = [p for p in directory.rglob("*") if p.is_file() and p != output]
    info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report_json(
        path=output,
        content=dict(
            base_dev="8f15ad7",
            python=sys.version,
            system=platform.platform(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            warp=wp.__version__,
            gpu=info,
            checks=results,
            source_sha256={str(p): digest(path=p) for p in sorted(sources)},
            artifact_sha256={str(p.relative_to(directory)): digest(path=p) for p in sorted(artifacts)},
        ),
    )


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    test_log = Path("outputs/fv2_qualification_tests.log")
    check(
        directory=directory,
        grids=(5, 10, 20),
        methods=("contact_hll", "roe", "hllc", "fv2_mc"),
        analytical_tolerance=1.0e-13,
        balance_tolerance=1.0e-11,
        test_log=test_log,
        output=directory / "manifest.json",
    )
