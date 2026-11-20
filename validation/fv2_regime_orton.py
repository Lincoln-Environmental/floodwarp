"""Matched archived wet-checkpoint replays for the terrain-regime investigation."""

from pathlib import Path
import json
import os
from validation.fv2_storm_cost import review


if __name__ == "__main__":
    directory = Path("validation/fv2_review/regimes")
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    revision = os.environ.get("FV2_STUDY_REVISION", "traces")
    methods = os.environ.get("FV2_ORTON_METHODS", "contact_hll,fv2_mc").split(",")
    review(
        checkpoint=checkpoint,
        factors=(4, 1),
        methods=methods,
        device="cuda:0",
        duration=12.0,
        repeats=5,
        kernel_interval=1.0,
        output=directory / ("orton_" + revision + ".json"),
    )
