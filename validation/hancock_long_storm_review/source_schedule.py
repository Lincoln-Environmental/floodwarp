"""Measure full source passes with a separate post-source hydraulic work list."""

import json
from pathlib import Path
from unittest.mock import patch
from time import perf_counter

import numpy as np
import warp as wp

from floodwarp import fv2_kernels as k
from floodwarp import hancock_tiles as tiles
from floodwarp.fv2_solver import HancockSolver
from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


@wp.kernel
def mark_after_source(
    q: wp.array(dtype=wp.vec3d),
    active: wp.array(dtype=int),
    bx: wp.array(dtype=int),
    by: wp.array(dtype=int),
    seed: wp.array(dtype=int),
    nx: int,
    columns: int,
):
    cell = wp.tid()
    row = cell / nx
    x = cell + row
    if active[cell] != 0 and (
        q[cell][0] != wp.float64(0.0)
        or bx[x] == 3
        or bx[x + 1] == 3
        or by[cell] == 3
        or by[cell + nx] == 3
    ):
        wp.atomic_max(
            seed, row / tiles.TILE_HEIGHT * columns + cell % nx / tiles.TILE_WIDTH, 1
        )


@wp.kernel
def retain_rejected(
    current: wp.array(dtype=int),
    previous: wp.array(dtype=int),
    ctrl: wp.array(dtype=wp.float64),
):
    if ctrl[2] > ctrl[17]:
        i = wp.tid()
        previous[i] |= current[i]


class PostSourceTiles(tiles.HancockTiles):
    def enqueue(self, *, solver, distributed_forcing):
        # Scheduling occurs after the first source pass in this control.
        if distributed_forcing:
            raise ValueError("This validation control uses physical sources only")


class PostSourceHancock(HancockSolver):
    def configure_execution(self, *, mode, **options):
        super().configure_execution(mode="dense", **options)
        self._tiles = PostSourceTiles(nx=self.nx, ny=self.ny, device=self.device)
        self._omit_dry = 1

    def _launch_grid(self, kernel, count, inputs, *, layout):
        if kernel is not k.split_sources:
            return super()._launch_grid(
                kernel=kernel, count=count, inputs=inputs, layout=layout
            )
        self._launch(
            kernel=kernel, count=count, inputs=[*inputs, self._work_indices[0]]
        )
        if inputs[-1] != 0:
            return
        cache = self._tiles
        cache.seed.zero_()
        cache.count.zero_()
        grid = cache.grids[0]
        f = self._fields
        self._launch(
            kernel=mark_after_source,
            count=count,
            inputs=[
                self._start,
                f["active"],
                f["types_x"],
                f["types_y"],
                cache.seed,
                self.nx,
                grid.columns,
            ],
        )
        self._launch(
            kernel=tiles.compact_trial,
            count=cache.size,
            inputs=[cache.seed, cache.previous, cache.current, grid, cache.rows],
        )

    def _enqueue_step(self):
        super()._enqueue_step()
        self._launch(
            kernel=retain_rejected,
            count=self._tiles.size,
            inputs=[self._tiles.current, self._tiles.previous, self.ctrl],
        )


def controls(*, review, config, storms, dimensions, device):
    results = []
    for storm in storms:
        expected = None
        for strategy in ("pre", "post"):
            solver = HancockSolver if strategy == "pre" else PostSourceHancock
            with patch("floodwarp.fv2_solver.HancockSolver", solver):
                model, _ = storm_model(
                    config=config,
                    landform="plain",
                    storm=storm,
                    method="hancock",
                    mode="tiles",
                    dx=10,
                    terrain_dx=10,
                    dimensions=dimensions,
                    duration=28800.0,
                    device=device,
                )
            s = model.solver
            samples = []
            for t in np.arange(900.0, 28801.0, 900.0):
                tick = perf_counter()
                stats = advance(model=model, end_time=t)
                elapsed = (
                    perf_counter()
                    - tick
                    - stats["graph_capture_seconds"]
                    - stats["compilation_seconds"]
                )
                samples.append(
                    dict(
                        t_s=t,
                        hydraulic_s=elapsed,
                        steps=stats["steps"],
                        active=s.execution_diagnostics(),
                    )
                )
            values = s.capture_state()["arrays"]
            if expected is None:
                expected = values
            else:
                for key, value in values.items():
                    np.testing.assert_array_equal(value, expected[key], err_msg=key)
            results.append(
                dict(
                    storm=storm,
                    strategy=strategy,
                    hydraulic_s=sum(a["hydraulic_s"] for a in samples),
                    samples=samples,
                )
            )
            (review / "source_schedules.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    storms = ("soil", "spatial", "moderate")
    dimensions = (5000, 5000)
    device = "cuda:0"
    controls(
        review=review,
        config=config,
        storms=storms,
        dimensions=dimensions,
        device=device,
    )
