"""Count accepted predictor fallback exposure over complete primary storms."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


@wp.kernel
def count_cells(
    flags: wp.array(dtype=wp.uint8),
    ctrl: wp.array(dtype=wp.float64),
    flagged: wp.array(dtype=int),
):
    if (
        ctrl[2] <= ctrl[17]
        and ctrl[7] == wp.float64(0.0)
        and flags[wp.tid()] != wp.uint8(0)
    ):
        wp.atomic_add(flagged, 0, 1)


@wp.kernel
def count_steps(
    flagged: wp.array(dtype=int),
    ctrl: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int64),
):
    if ctrl[2] <= ctrl[17] and ctrl[7] == wp.float64(0.0):
        counts[0] += wp.int64(1)
        counts[1] += wp.int64(flagged[0] > 0)
        counts[2] += wp.int64(flagged[0])
        counts[3] = wp.max(counts[3], wp.int64(flagged[0]))


class FallbackCounter:
    def __init__(self, *, device):
        self.flagged = wp.zeros(1, dtype=int, device=device)
        self.counts = wp.zeros(4, dtype=wp.int64, device=device)
        wp.load_module(module=count_cells.module, device=device)

    def enqueue(self, *, solver):
        self.flagged.zero_()
        solver._launch(
            kernel=count_cells,
            count=solver.nx * solver.ny,
            inputs=[solver._predictor_fallback, solver.ctrl, self.flagged],
        )
        solver._launch(
            kernel=count_steps, count=1, inputs=[self.flagged, solver.ctrl, self.counts]
        )


def audit(*, review, config, device):
    rows = []
    for file in sorted((review / "events").glob("primary_*_hancock_tiles_*.json")):
        run = json.loads(file.read_text())
        model, _ = storm_model(
            config=config,
            landform=run["landform"],
            storm=run["storm"],
            method="hancock",
            mode="tiles",
            dx=run["dx"],
            terrain_dx=run["terrain_dx"],
            dimensions=run["dimensions_m"],
            duration=run["hours"] * 3600,
            device=device,
        )
        probe = FallbackCounter(device=device)
        model.solver._diagnostics = probe
        for s in run["samples"]:
            stats = advance(model=model, end_time=s["t_s"])
            if stats["steps"] != s["steps"]:
                raise ValueError("Instrumented trajectory changed")
        steps, affected, exposures, peak = probe.counts.numpy().tolist()
        if steps != run["steps"]:
            raise AssertionError("Accepted step mismatch")
        np.testing.assert_allclose(
            model.solver.cumulative_accounting()["storage"],
            run["final_m3"],
            rtol=2e-13,
            atol=1e-12,
        )
        rows.append(
            dict(
                run=run["name"],
                steps=steps,
                fallback_steps=affected,
                cell_steps=exposures,
                maximum_cells_on_one_step=peak,
                fraction=exposures / (steps * model.solver.nx * model.solver.ny),
            )
        )
    (review / "fallback.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    device = "cuda:0"
    audit(review=review, config=config, device=device)
