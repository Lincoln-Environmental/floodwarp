"""Distinguish attempts, timing repetitions and independent physical realisations."""

import json
from collections import Counter
from pathlib import Path

from research.floodwarp_methods_paper.runner import checksum, digest
from research.floodwarp_methods_paper.metrics import csv_write


def scientific_ids(*, record, row):
    s = record["spec"]
    if "forcing" in row:
        f = row["forcing"]
        physical = {
            k: f[k]
            for k in (
                "terrain_sha256",
                "initial_depth_sha256",
                "initial_hu_sha256",
                "rain_mask_sha256",
                "rain_rates_ms",
                "rain_times_s",
                "infiltration_sha256",
                "boundary_x_sha256",
                "boundary_y_sha256",
                "stage_m",
                "shape",
            )
        }
        physical.update(
            case=s["case"],
            dx=s["dx"],
            dem=s["dem"],
            duration_h=s["hours"],
            physical_n=round(f["roughness"] / s["multiplier"], 14),
        )
        settings = dict(
            algorithm=f["numerical"],
            execution=row["mode"],
            model_n=f["roughness"],
            resolution=s["dx"],
        )
    elif "depth_rmse" in row:
        physical = dict(
            case=s["case"],
            domain_checksum=row["terrain_sha256"],
            shape=row["shape"],
            dx=s["dx"],
            duration_s=row["duration_s"],
        )
        settings = dict(
            algorithm=row["config"],
            method=s["method"],
            policy=s["policy"],
            fixed_dt_s=s.get("fixed_dt_s"),
            dx=s["dx"],
        )
    else:
        physical = dict(case=s["case"], group=record["stage"])
        settings = s
    phy = digest(physical)
    return phy, digest(
        dict(physics=phy, settings=settings, source=record["context"]["source_sha256"])
    )


def collect(*, study):
    stages = {
        "check",
        "analytical",
        "convergence",
        "fixed-dt-space",
        "synthetic",
        "long-storms",
        "resolution",
        "terrain-initial-controls",
        "calibration",
        "performance",
    }
    exceptions = study / "manifests/timing_exclusions.json"
    excluded = json.loads(exceptions.read_text()) if exceptions.is_file() else {}
    rows = []
    for initial in sorted((study / "raw_results").glob("*/*/*/attempt.json")):
        origin = json.loads(initial.read_text())
        if origin["stage"] not in stages:
            continue
        receipt = initial.with_name("receipt.json")
        r = json.loads(receipt.read_text()) if receipt.exists() else origin
        finished = r["status"] == "complete"
        result = (
            json.loads(initial.with_name("result.json").read_text()) if finished else {}
        )
        phy, num = scientific_ids(record=r, row=result) if finished else (None, None)
        rows.append(
            dict(
                physical_grid_forcing_id=phy,
                hydraulic_configuration_id=num,
                experiment_id=r["experiment_id"],
                implementation_source_sha256=r["context"]["source_sha256"],
                git_sha=r["context"]["git_sha"],
                attempt=r["attempt"],
                stage=r["stage"],
                case=r["spec"]["case"],
                method=r["spec"]["method"],
                timing_repeat=r["spec"].get("repeat"),
                status=r["status"],
                timing_exclusion=excluded.get(r["attempt"]),
                elapsed_s=r.get("wall_s"),
                receipt_sha256=checksum(receipt) if receipt.exists() else None,
                log_sha256=checksum(study / r["log"])
                if (study / r["log"]).exists()
                else None,
                directory=r["directory"],
            )
        )
    csv_write(file=study / "processed/metrics/attempt_catalog.csv", rows=rows)
    return dict(
        status_counts=dict(Counter(r["status"] for r in rows)),
        unique_physical_grid_forcings=len(
            {
                r["physical_grid_forcing_id"]
                for r in rows
                if r["physical_grid_forcing_id"]
            }
        ),
        unique_hydraulic_configurations=len(
            {
                r["hydraulic_configuration_id"]
                for r in rows
                if r["hydraulic_configuration_id"]
            }
        ),
    )


if __name__ == "__main__":
    study = Path(__file__).resolve().parent
    collect(study=study)
