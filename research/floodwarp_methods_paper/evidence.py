"""Select one reproducible numerical cohort for publication and advisor fitting."""

from functools import lru_cache
import json
import os
from pathlib import Path

from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.runner import archived, digest, source_identity


@lru_cache(maxsize=64)
def implementation(study, stage):
    if stage == "fixed-dt-space":
        from research.floodwarp_methods_paper import fixed_dt

        return fixed_dt.source(root=study.parents[1], study=study, stage=stage)
    if stage == "terrain-initial-controls":
        from research.floodwarp_methods_paper import terrain_controls

        return terrain_controls.source(root=study.parents[1], study=study, stage=stage)
    return source_identity(root=study.parents[1], study=study, stage=stage)


def records(*, study, stages):
    directory = Path(study).resolve()
    config = read_config(
        path=Path(
            os.environ.get("FLOODWARP_PAPER_CONFIG", directory / "configs/study.json")
        )
    )
    config_id = digest(config)
    exceptions = directory / "manifests/timing_exclusions.json"
    excluded = json.loads(exceptions.read_text()) if exceptions.is_file() else {}
    gpu = os.environ.get("FLOODWARP_PAPER_RESULT_GPU", "").casefold()
    rows = [
        r
        for r in archived(study=directory, stages=stages)
        if r["config_sha256"] == config_id
        and r["attempt"] not in excluded
        and gpu in r["context"]["gpu"].casefold()
        and r["context"]["source_sha256"] == implementation(directory, r["stage"])
    ]
    environments = {
        tuple(
            r["context"].get(k)
            for k in ("gpu", "warp", "python", "numpy", "precision", "device")
        )
        for r in rows
    }
    if len(environments) > 1:
        raise ValueError(
            "More than one timing environment matches. Select FLOODWARP_PAPER_RESULT_GPU or analyse separate archived checkouts."
        )
    return rows
