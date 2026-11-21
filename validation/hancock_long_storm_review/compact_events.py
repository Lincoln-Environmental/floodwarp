"""Compare cell lists over moving, concentrated and widespread complete storms."""

from pathlib import Path
import json
from unittest.mock import patch

import numpy as np

from validation.hancock_long_storm_review.compact import CompactHancock
from validation.hancock_long_storm_review.storms import event


def compare(*, review, fields, config, choices, device):
    directory = review / "compact_events"
    directory.mkdir(exist_ok=True)
    rows = []
    for suite, storm in choices:
        base = json.loads(
            next(
                (review / "events").glob(
                    f"{suite}_5000x5000_plain_{storm}_hancock_tiles_*"
                )
            ).read_text()
        )
        with patch("floodwarp.fv2_solver.HancockSolver", CompactHancock):
            event(
                config=config,
                landform="plain",
                storm=storm,
                method="hancock",
                mode="compact",
                dx=10,
                terrain_dx=10,
                dimensions=base["dimensions_m"],
                hours=base["hours"],
                interval=base["sample_seconds"],
                device=device,
                directory=directory,
                fields=fields,
                label_name=suite,
            )
        row = json.loads(
            next(
                directory.glob(f"{suite}_5000x5000_plain_{storm}_hancock_compact_*")
            ).read_text()
        )
        for a, b in zip(row["samples"], base["samples"], strict=True):
            for key in (
                "steps",
                "dt_min_s",
                "dt_max_s",
                "section_m3s",
                "outlet_m3s",
                "rain_m3",
                "loss_m3",
                "storage_m3",
                "fallback_cells_last_trial",
            ):
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        if suite == "moving" and row["final_sha256"] != base["final_sha256"]:
            raise ArithmeticError("Moving storm final fields differ")
        rows.append(
            dict(
                run=row["name"],
                reference=base["name"],
                hydraulic_s=row["hydraulic_s"],
                tiles_s=base["hydraulic_s"],
                pool_mib=row["process_pool_mib"],
                steps=row["steps"],
                final_m3=row["final_m3"],
                residual_m3=row["residual_m3"],
            )
        )
    (review / "compact_storms.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    fields = Path("outputs/hancock_long_storm_review/fields")
    config = json.loads((review / "cases.json").read_text())
    choices = (("moving", "moving"), ("large", "spatial"), ("large", "moderate"))
    device = "cuda:0"
    compare(review=review, fields=fields, config=config, choices=choices, device=device)
