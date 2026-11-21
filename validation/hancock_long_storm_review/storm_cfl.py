"""Locate the limiting signals during selected fine- and coarse-grid rain events."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from floodwarp import fv2_kernels
from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model
from validation.hancock_long_storm_review.timestep import directional_rates


def sample_trial(*, solver):
    original = solver._launch_grid
    found = []

    def trace(kernel, count, inputs, *, layout):
        if kernel is fv2_kernels.donor_budget:
            axes = [directional_rates(solver=solver, axis=axis) for axis in (0, 1)]
            rates = np.array([a["rate"] for a in axes])
            ctrl = solver.ctrl.numpy()
            np.testing.assert_allclose(
                solver.cfl / (2 * rates.max()), ctrl[17], atol=1.0e-13, rtol=5.0e-12
            )
            base = solver._start.numpy().reshape(solver.ny, solver.nx, 3)
            wet = base[..., 0] > solver.dry_depth
            speeds = np.hypot(base[..., 1], base[..., 2])[wet] / base[..., 0][wet]
            found.append(
                dict(
                    axes=axes,
                    limiting_axis=int(rates.argmax()),
                    bound_s=float(ctrl[17]),
                    proposed_s=ctrl[2],
                    fallback_cells=int(solver.predictor_fallback_cells().sum()),
                    cell_speed_max_ms=float(speeds.max(initial=0.0)),
                )
            )
        return original(kernel=kernel, count=count, inputs=inputs, layout=layout)

    state = solver.capture_state()
    with patch.object(solver, "_launch_grid", trace):
        solver.step()
    solver.restore_state(state=state)
    return found


def diagnose(*, review, cases, times, config, device):
    results = []
    for land, storm, dx in cases:
        group = "refinement" if dx == 1 else "primary"
        row = json.loads(
            next(
                (review / "events").glob(
                    f"{group}_*_{land}_{storm}_hancock_tiles_{dx}m_{dx}dem_8h.json"
                )
            ).read_text()
        )
        m, _ = storm_model(
            config=config,
            landform=land,
            storm=storm,
            method="hancock",
            mode="tiles",
            dx=dx,
            terrain_dx=dx,
            dimensions=row["dimensions_m"],
            duration=28800.0,
            device=device,
        )
        series = []
        for s in row["samples"]:
            stats = advance(model=m, end_time=s["t_s"])
            if stats["steps"] != s["steps"]:
                raise ValueError(
                    "Sampled storm trajectory differs from its uninstrumented control"
                )
            if s["t_s"] in times:
                series.append(dict(t_s=s["t_s"], trials=sample_trial(solver=m.solver)))
        results.append(dict(run=row["name"], samples=series))
        (review / "storm_cfl.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    cases = (("incised", "moderate", 1), ("mixed", "spatial", 10))
    times = np.arange(900.0, 28800.0, 1800.0)
    device = "cuda:0"
    diagnose(review=review, cases=cases, times=times, config=config, device=device)
