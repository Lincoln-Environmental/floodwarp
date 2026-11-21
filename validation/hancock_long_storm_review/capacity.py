"""Benchmark-only constant-capacity specialization of the reference split-source operator."""

from time import perf_counter
from typing import Any
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp import fv2_kernels as hydraulic
from floodwarp.fv2_solver import HancockSolver
from floodwarp.hancock_tiles import scheduled_kernel, scheduled_index
from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


@wp.func
def reference_exchange(
    q: wp.vec3d,
    soil: wp.float64,
    rain: wp.float64,
    dt: wp.float64,
    infil: wp.float64,
    f0: wp.float64,
    fd: wp.float64,
    cond: wp.float64,
):
    # Explicit midpoint for cumulative infiltration, with a finite supply.
    added = rain * dt
    if f0 == cond and infil == wp.float64(0.0) and cond == wp.float64(0.0):
        factor = wp.where(q[0] > wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        return wp.vec4d(q[0] + added, factor, added, wp.float64(0.0))
    initial = (f0 - cond) * wp.exp(-soil / fd)
    base = infil + cond
    mid = wp.float64(0.5) * wp.min(q[0] + added, dt * (base + initial))
    removed = wp.min(q[0] + added, dt * (base + initial * hydraulic.decay(mid / fd)))
    h = (q[0] + added) - removed
    factor = wp.float64(0.0)
    if h > wp.float64(0.0) and q[0] > wp.float64(0.0):
        # Integrate dm/dt=-I*m/h for constant interval-mean rain and loss.
        delta = added - removed
        inverse = wp.float64(2.0) / (q[0] + h)
        if wp.abs(delta) > wp.float64(1.0e-5) * q[0]:
            r = delta / (q[0] + h)
            if wp.abs(r) < wp.float64(0.01):
                # log(h/h0)/delta = 2*atanh(r)/(r*(h+h0)).
                # Relative tail beyond r^6 is below 1.2e-17.
                rr = r * r
                inverse *= wp.float64(1.0) + rr * (
                    wp.float64(1.0 / 3.0)
                    + rr * (wp.float64(1.0 / 5.0) + rr * wp.float64(1.0 / 7.0))
                )
            else:
                inverse = wp.log(h / q[0]) / delta
        factor = hydraulic.decay(removed * inverse)
    return wp.vec4d(h, factor, added, removed)


@scheduled_kernel
def capacity_sources(
    q: wp.array(dtype=wp.vec3d),
    start: wp.array(dtype=wp.vec3d),
    end: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    n: wp.array(dtype=wp.float64),
    soil: wp.array(dtype=wp.float64),
    rain: wp.array(dtype=wp.float64),
    mult: wp.array(dtype=wp.float64),
    infil: wp.array(dtype=wp.float64),
    f0: wp.array(dtype=wp.float64),
    fd: wp.array(dtype=wp.float64),
    cond: wp.array(dtype=wp.float64),
    amounts: wp.array(dtype=wp.vec2d),
    rain_sum: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    g: wp.float64,
    dry: wp.float64,
    stage: int,
    work: Any,
):
    k = scheduled_index(wp.tid(), work)
    if k < 0:
        return
    if active[k] == 0:
        return
    dt = ctrl[2] * wp.float64(0.5)
    state = q[k]
    rate = rain[k] + mult[k] * (
        ctrl[9] + (wp.float64(0.5) + wp.float64(wp.min(stage, 1))) * dt * ctrl[11]
    )
    history = soil[k]
    if stage != 0:
        if ctrl[2] > ctrl[17] or ctrl[7] != wp.float64(0.0):
            return
        state = end[k]
        if stage == 1:
            state = wp.float64(0.5) * (start[k] + end[k])
        state = hydraulic.drag(state, n[k], dt, g, dry)
        history += amounts[k][1]
    change = reference_exchange(
        state, history, rate, dt, infil[k], f0[k], fd[k], cond[k]
    )
    state = wp.vec3d(change[0], state[1] * change[1], state[2] * change[1])
    if stage == 0:
        start[k] = hydraulic.drag(state, n[k], dt, g, dry)
        inverse[k] = hydraulic.wet_inverse(state[0], dry)
        if flags.shape[0] > 0:
            flags[k] = 0
        amounts[k] = wp.vec2d(change[2], change[3])
    else:
        if state[0] <= dry:
            state[1] = wp.float64(0.0)
            state[2] = wp.float64(0.0)
        q[k] = state
        soil[k] += amounts[k][1] + change[3]
        rain_sum[k] += amounts[k][0] + change[2]
        maximum[k] = wp.max(maximum[k], state[0])


class ReferenceHancock(HancockSolver):
    def _launch_grid(self, kernel, count, inputs, *, layout):
        replacement = capacity_sources if kernel is hydraulic.split_sources else kernel
        return super()._launch_grid(
            kernel=replacement, count=count, inputs=inputs, layout=layout
        )

    def _load_extra_step_modules(self):
        super()._load_extra_step_modules()
        wp.load_module(module=capacity_sources.module, device=self.device)


def measure(*, review, config, device, intervals, dimensions, cases, repeats):
    rows = []
    for storm, mode in cases:
        reference = None
        for trial in range(repeats):
            pairs = [("original", ReferenceHancock), ("specialized", HancockSolver)]
            if trial % 2:
                pairs.reverse()
            for name, constructor in pairs:
                with patch("floodwarp.fv2_solver.HancockSolver", constructor):
                    model, _ = storm_model(
                        config=config,
                        landform="plain",
                        storm=storm,
                        method="hancock",
                        mode=mode,
                        dx=10,
                        terrain_dx=10,
                        dimensions=dimensions,
                        duration=28800.0,
                        device=device,
                    )
                parts = []
                for t in intervals:
                    wp.synchronize_device(device=device)
                    start = perf_counter()
                    stats = advance(model=model, end_time=t)
                    wp.synchronize_device(device=device)
                    parts.append(
                        perf_counter()
                        - start
                        - stats["graph_capture_seconds"]
                        - stats["compilation_seconds"]
                    )
                state = model.solver.capture_state()["arrays"]
                if reference is None:
                    reference = state
                else:
                    for key, value in state.items():
                        np.testing.assert_array_equal(
                            value, reference[key], err_msg=key
                        )
                rows.append(
                    dict(
                        trial=trial,
                        storm=storm,
                        mode=mode,
                        operator=name,
                        hydraulic_s=sum(parts),
                        intervals_s=parts,
                    )
                )
                (review / "integrated_capacity.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    device = "cuda:0"
    intervals = np.arange(3600, 28801, 3600)
    dimensions = (5000, 5000)
    cases = (("moderate", "dense"), ("spatial", "tiles"))
    repeats = 2
    measure(
        review=review,
        config=config,
        device=device,
        intervals=intervals,
        dimensions=dimensions,
        cases=cases,
        repeats=repeats,
    )
