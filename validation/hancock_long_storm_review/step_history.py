"""Verify each accepted timestep on matched graph replays, including through recession."""

from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


@wp.kernel
def accepted_dt(ctrl: wp.array(dtype=wp.float64), times: wp.array(dtype=wp.float64)):
    if ctrl[7] == wp.float64(0.0) and ctrl[2] <= ctrl[17]:
        index = int(ctrl[3]) - 1
        if index >= 0 and index < times.shape[0]:
            times[index] = ctrl[2]


def collect(*, config, row, device):
    model, _ = storm_model(
        config=config,
        landform=row["landform"],
        storm=row["storm"],
        method=row["method"],
        mode=row["mode"],
        dx=row["dx"],
        terrain_dx=row["terrain_dx"],
        dimensions=row["dimensions_m"],
        duration=row["hours"] * 3600.0,
        device=device,
    )
    solver = model.solver
    data = wp.zeros(row["steps"], dtype=wp.float64, device=device)
    wp.load_module(module=accepted_dt.module, device=device)
    original = solver._enqueue_step

    def record():
        original()
        wp.launch(kernel=accepted_dt, dim=1, inputs=[solver.ctrl, data], device=device)

    with patch.object(solver, "_enqueue_step", record):
        for sample in row["samples"]:
            stat = advance(model=model, end_time=sample["t_s"])
            if stat["steps"] != sample["steps"]:
                raise ArithmeticError("Instrumented trajectory differs")
    result = data.numpy()
    if np.any(result <= 0):
        raise ArithmeticError("Missing accepted timestep")
    np.testing.assert_allclose(result.sum(), row["hours"] * 3600.0, rtol=5e-12)
    return result


def measure(*, review, cases, device):
    config = json.loads((review / "cases.json").read_text())
    metadata = []
    histories = {}
    for land, storm in cases:
        for mode in ("dense", "sparse", "tiles"):
            file = next(
                (review / "events").glob(
                    f"primary_*_{land}_{storm}_hancock_{mode}_*.json"
                )
            )
            row = json.loads(file.read_text())
            dt = collect(config=config, row=row, device=device)
            if mode == "dense":
                histories[land + "_" + storm] = dt
            else:
                np.testing.assert_array_equal(dt, histories[land + "_" + storm])
            metadata.append(
                dict(
                    run=row["name"],
                    steps=len(dt),
                    dt_sha256=hashlib.sha256(dt.tobytes()).hexdigest(),
                )
            )
    np.savez_compressed(review / "accepted_dt.npz", **histories)
    (review / "accepted_dt.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    device = "cuda:0"
    cases = (
        ("plain", "moderate"),
        ("floodplain", "soil"),
        ("coastal", "front"),
        ("fan", "back"),
        ("incised", "intense"),
        ("mixed", "spatial"),
    )
    measure(review=review, cases=cases, device=device)
