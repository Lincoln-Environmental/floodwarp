"""Verify archived hydraulic horizons, signed objective decisions and paper products without CUDA replay."""

import json
from pathlib import Path
import sqlite3
from research.floodwarp_methods_paper import metrics, runner


def validate(*, directory):
    root = directory.parents[2]
    manifest = json.loads((directory / "outputs.json").read_text())
    for relative, expected in manifest.items():
        path = root / relative
        if runner.checksum(path) != expected:
            raise ValueError(f"Fidelity product checksum mismatch: {relative}")
    results = directory / "results"
    spatial = json.loads((results / "transient_manifest.json").read_text())
    for relative, expected in spatial.items():
        if runner.checksum(directory / relative) != expected:
            raise ValueError(f"Spatial archive checksum mismatch: {relative}")
    with sqlite3.connect(
        "file:" + str(results / "fidelity.sqlite") + "?mode=ro", uri=True
    ) as db:
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Invalid SQLite database")
        for file, sha, rows in db.execute("SELECT file,sha256,rows FROM csv_sources"):
            path = results / file
            if runner.checksum(path) != sha or len(metrics.csv_read(path)) != rows:
                raise ValueError(f"CSV/database disagreement: {file}")
    ends = metrics.csv_read(results / "recession_horizons.csv")
    primary = [g for g in ends if g["cohort"] == "legacy"]
    if len(primary) != 48 or any(float(r["common_end_h"]) < 8 for r in ends):
        raise ArithmeticError("Missing primary storm or recession")
    if any(
        r["passed"] != "True"
        for r in metrics.csv_read(results / "recession_terminal.csv")
    ):
        raise ArithmeticError("Failed terminal station")
    costs = metrics.csv_read(results / "completed_costs.csv")
    operational = [r for r in costs if r["suite"] == "primary-million"]
    if len(operational) != 8 or len({r["landform"] for r in operational}) != 8:
        raise ArithmeticError("Missing million-cell terrain families")
    for r in operational:
        if not all(
            r.get(k + "_id")
            for k in ("hancock_dense", "hancock_adaptive", "hancock_auto")
        ):
            raise ArithmeticError("Missing measured Hancock policy for " + r["case"])
        if float(r["fastest_s"]) != min(
            float(r[k + "_s"]) for k in ("hancock_dense", "hancock_adaptive")
        ):
            raise ArithmeticError("Incorrect measured Hancock cost")
    print(
        f"Verified {len(manifest)} product/source hashes, {len(ends)} common endpoints and eight operational million-cell families."
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    validate(directory=directory)
