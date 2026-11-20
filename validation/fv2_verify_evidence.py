"""Check the investigation's numerical gates and freeze its data and source identities."""

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


def digest(*, path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(*, directory, grids, methods, analytical_tolerance, balance_tolerance, test_log, output):
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
            assert (r["dx"], r["method"], r["time_s"], len(r["hydrograph"])) == (dx, method, 172800.0, 2880)
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
    auxiliary = record(name="native_hirds48.json")
    assert all(r["time_s"] == 172800.0 for r in auxiliary["rows"])
    results = dict(
        analytical_runs=len(all_cases),
        lake_max_depth_drift_m=max_drift,
        max_normalised_analytical_balance=max_balance,
        spatial_orders=p_space.tolist(),
        temporal_orders=p_time.tolist(),
        storms=len(rain),
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
    artifacts = [p for p in directory.rglob("*") if p.is_file() and p != output]
    info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
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
