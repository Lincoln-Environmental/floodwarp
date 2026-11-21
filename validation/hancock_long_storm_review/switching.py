"""Measure an output-time switching experiment, including recapture costs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from scipy.ndimage import maximum_filter

from floodwarp.hancock_tiles import TILE_HEIGHT, TILE_WIDTH

from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


def work_fraction(*, wet, rain):
    seeds = wet | rain
    ny, nx = seeds.shape
    padded = np.pad(seeds, ((0, -ny % TILE_HEIGHT), (0, -nx % TILE_WIDTH)))
    marked = padded.reshape(
        padded.shape[0] // TILE_HEIGHT, TILE_HEIGHT, -1, TILE_WIDTH
    ).any(axis=(1, 3))
    halo = maximum_filter(marked, size=3, mode="constant", cval=0)
    cells = halo.repeat(TILE_HEIGHT, axis=0).repeat(TILE_WIDTH, axis=1)
    return float(cells[:ny, :nx].mean())


def compare(
    *,
    review,
    config,
    regimes,
    dimensions,
    spacing,
    duration,
    interval,
    threshold,
    repeats,
    device,
):
    rows = []
    orders = (("dense", "tiles", "switch"), ("switch", "tiles", "dense"))
    for trial, regime in (
        (trial, regime) for trial in range(repeats) for regime in regimes
    ):
        fields = []
        for policy in orders[trial % len(orders)]:
            selected = "tiles" if policy == "switch" else policy
            m, _ = storm_model(
                config=config,
                landform="plain",
                storm="spatial",
                method="hancock",
                mode=selected,
                dx=spacing,
                terrain_dx=spacing,
                dimensions=dimensions,
                duration=duration,
                device=device,
            )
            solver = m.solver
            patch = solver._fields["multiplier"].numpy().reshape(solver.ny, solver.nx)
            history = []
            for start in np.arange(0, duration, interval):
                # During the transition event the rain footprint expands at hour 2.
                wide = regime == "uniform" or (regime == "expanding" and start >= 7200)
                solver.set_rainfall_spatial_multiplier(
                    multiplier=np.ones_like(patch) if wide else patch
                )
                wp.synchronize_device(device=device)
                began = perf_counter()
                active = None
                if policy == "switch":
                    if wide and start < 21600:
                        active = 1.0
                        requested = "dense"
                    else:
                        h = solver.get_water_depth()
                        # This experiment permits a read at the scheduled output only.
                        active = work_fraction(
                            wet=h > 0.0, rain=(patch > 0.0) & (start < 21600)
                        )
                        requested = "dense" if active >= threshold else "tiles"
                    if requested != selected:
                        solver.configure_execution(mode=requested)
                        selected = requested
                stats = advance(model=m, end_time=start + interval)
                wp.synchronize_device(device=device)
                history.append(
                    dict(
                        t_s=float(start + interval),
                        mode=selected,
                        estimated_work_fraction=active,
                        wall_s=perf_counter() - began,
                        capture_s=stats["graph_capture_seconds"],
                        steps=stats["steps"],
                    )
                )
            state = solver.capture_state()["arrays"]
            if fields:
                for key, value in state.items():
                    np.testing.assert_array_equal(value, fields[0][key], err_msg=key)
            else:
                fields.append(state)
            rows.append(
                dict(
                    regime=regime,
                    repeat=trial,
                    policy=policy,
                    total_s=sum(s["wall_s"] for s in history),
                    samples=history,
                )
            )
            path = review / "switching.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows, indent=2) + "\n")
            temporary.replace(path)


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    regimes = ("spatial", "uniform", "expanding")
    threshold = 0.97
    compare(
        review=review,
        config=config,
        regimes=regimes,
        dimensions=(5000, 5000),
        spacing=10,
        duration=28800,
        interval=1800,
        threshold=threshold,
        repeats=2,
        device="cuda:0",
    )
