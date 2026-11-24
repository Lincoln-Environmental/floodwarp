"""Verify archived hydraulic horizons, signed objective decisions and paper products without CUDA replay."""

import json
from pathlib import Path
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import catalogue


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
    catalogue.verify_database(target=results)
    ends = metrics.csv_read(results / "recession_horizons.csv")
    primary = [g for g in ends if g["cohort"] == "legacy"]
    if len(primary) != 48 or any(float(r["common_end_h"]) < 8 for r in ends):
        raise ArithmeticError("Missing primary storm or recession")
    if any(
        r["passed"] != "True"
        for r in metrics.csv_read(results / "recession_terminal.csv")
    ):
        raise ArithmeticError("Failed terminal station")
    parity = metrics.csv_read(results / "recession_execution_parity.csv")
    if sum(r["cohort"] == "complete-legacy" and r["dx"] == "10" for r in parity) != 48:
        raise ArithmeticError("Missing small-storm execution comparisons")
    if any(
        r["steps"] != "True" or r["outlet_m3s"] != "True" or r["section_m3s"] != "True"
        for r in parity
    ):
        raise ArithmeticError("Endpoint policy transport disagreement")
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
