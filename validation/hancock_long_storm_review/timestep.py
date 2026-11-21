"""Independent per-trial CFL attribution for the closed, sloping travelling front.

The counterfactual uses the sum of directional global maxima with the same CFL.
For this diagnostic only, inspect the actual GPU midpoint before Euler overwrites it.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from floodwarp import fv2_kernels
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import initialise, measures


def directional_rates(*, solver, axis):
    dim = 1 - axis
    spacing = (solver.dx, solver.dy)[axis]
    shape = (solver.ny, solver.nx)
    q = solver._predicted.numpy().reshape(*shape, 3)
    slopes = solver._slopes[axis].numpy().reshape(*shape, 4)
    z = solver._fields["bed"].numpy().reshape(shape)
    base = solver._start.numpy().reshape(*shape, 3)
    dry = solver.dry_depth
    h = q[..., 0]
    traces = []
    vel = np.divide(q[..., 1 + axis], h, out=np.zeros_like(h), where=h > dry)
    for sign in (1, -1):
        delta = 0.5 * sign * (slopes[..., 0] - slopes[..., 1])
        speed = vel + np.divide(
            0.5 * sign * (h - delta) * slopes[..., 2 + axis],
            h,
            out=np.zeros_like(h),
            where=h > dry,
        )
        traces.append((h + delta, np.where(h + delta > dry, speed, 0.0)))
    # All outer slopes vanish. Walls reflect normal speed and FREE copies it.
    rise = np.diff(z, axis=dim) - 0.5 * (
        np.take(slopes[..., 1], range(z.shape[dim] - 1), axis=dim)
        + np.take(slopes[..., 1], range(1, z.shape[dim]), axis=dim)
    )
    pad = [(0, 0), (0, 0)]
    pad[dim] = (1, 1)
    rise = np.pad(rise, pad)
    hl, ul = traces[0]
    hr, ur = traces[1]
    left = np.maximum(
        0.0,
        np.concatenate((np.take(hr, [0], axis=dim), hl), axis=dim)
        - np.maximum(0.0, rise),
    )
    right = np.maximum(
        0.0,
        np.concatenate((hr, np.take(hl, [-1], axis=dim)), axis=dim)
        - np.maximum(0.0, -rise),
    )
    ul, ur = (
        np.concatenate((-np.take(ur, [0], axis=dim), ul), axis=dim),
        np.concatenate((ur, -np.take(ul, [-1], axis=dim)), axis=dim),
    )
    types = solver._fields["types_" + ("x", "y")[axis]].numpy().reshape(left.shape)
    if np.any(types == 3):
        raise ValueError("This attribution supports walls and FREE boundaries")
    first, last = [slice(None)] * 2, [slice(None)] * 2
    first[dim], last[dim] = 0, -1
    first, last = tuple(first), tuple(last)
    ul[first] = np.where(types[first] == 2, ur[first], ul[first])
    ur[last] = np.where(types[last] == 2, ul[last], ur[last])
    ul, ur = np.where(left > dry, ul, 0.0), np.where(right > dry, ur, 0.0)
    cl, cr = np.sqrt(solver.gravity * left), np.sqrt(solver.gravity * right)
    low, high = np.minimum(ul - cl, ur - cr), np.maximum(ul + cl, ur + cr)
    low, high = (
        np.where(right <= dry, ul - cl, low),
        np.where(right <= dry, ul + 2 * cl, high),
    )
    low, high = (
        np.where(left <= dry, ur - 2 * cr, low),
        np.where(left <= dry, ur + cr, high),
    )
    speeds = np.maximum(abs(low), abs(high))
    point = np.unravel_index(np.argmax(speeds), speeds.shape)
    h0 = base[..., 0]
    u0 = np.divide(base[..., 1 + axis], h0, out=np.zeros_like(h0), where=h0 > dry)
    model_speed = np.where(h0 > dry, abs(u0) + np.sqrt(solver.gravity * h0), 0.0)
    if model_speed.max() > speeds[point]:
        point = np.unravel_index(np.argmax(model_speed), model_speed.shape)
        return dict(
            location=list(map(int, point)),
            kind="cell",
            rate=float(model_speed[point] / spacing),
            depth_m=[h0[point]],
            normal_speed_ms=[u0[point]],
        )
    return dict(
        location=list(map(int, point)),
        kind="face",
        rate=float(speeds[point] / spacing),
        depth_m=[left[point], right[point]],
        normal_speed_ms=[ul[point], ur[point]],
    )


def investigate(*, case, dx, method, directional, settings):
    solver, reference, exact, times, x = initialise(
        case=case,
        dx=dx,
        method=method,
        policy="production",
        multiplier=1.0,
        settings=settings,
    )
    launch = solver._launch if method.startswith("hancock") else None
    trials = []

    def record_trial(kernel, count, inputs):
        if kernel is fv2_kernels.donor_budget:
            directional_record()
        launch(kernel=kernel, count=count, inputs=inputs)

    def directional_record():
        axes = [directional_rates(solver=solver, axis=a) for a in (0, 1)]
        rates = [a["rate"] for a in axes]
        ctrl = solver.ctrl.numpy()
        old = solver.cfl / (2 * max(rates))
        np.testing.assert_allclose(old, ctrl[17], rtol=2.0e-13, atol=1.0e-15)
        new = solver.cfl / sum(rates)
        trials.append(
            dict(
                t_s=solver.time,
                proposed_s=ctrl[2],
                legacy_s=old,
                directional_s=new,
                fallback_cells=int(solver.predictor_fallback_cells().sum()),
                axes=axes,
            )
        )
        if directional:
            ctrl[17] = new
            solver.ctrl.assign(ctrl)

    from contextlib import nullcontext
    from validation.fv1_diffusion_matrix import face_transport

    states = [reference[0].T]
    volumes = [np.zeros(len(case.gauges))]
    with patch.object(solver, "_launch", record_trial) if launch else nullcontext():
        for target in times[1:]:
            while solver.time < target:
                solver.step(remaining_time=target - solver.time)
            states.append(solver.get_conservative_state()[solver.ny // 2])
            volumes.append(
                face_transport(solver=solver, case=case, dx=dx, x=x, li=False)
            )
    metrics = measures(
        result=np.stack(states).transpose(0, 2, 1),
        states=reference,
        discharge=np.diff(volumes, axis=0) / np.diff(times)[:, None],
        exact=exact,
        times=times,
        x=x,
        dx=dx,
        case=case,
        settings=settings,
    )
    return dict(
        method=method,
        directional=directional,
        steps=solver.steps,
        trials=trials,
        rejected=len(trials) - solver.steps if trials else 0,
        metrics=metrics,
        residual_m3=solver.cumulative_accounting()["residual"],
    )


if __name__ == "__main__":
    root = Path(__file__).parent
    settings = dict(
        device="cpu",
        gravity=9.80665,
        samples=12,
        order=16,
        max_dt=5.0,
        wet=(0.001, 0.01, 0.05),
        li_floor=1.0e-8,
    )
    case = next(c for c in design() if c.name == "front_very_steep")
    dx = 5.0
    methods = (
        ("fv1", False),
        ("hancock_mc", False),
        ("hancock_adaptive", False),
        ("hancock_adaptive", True),
    )
    rows = [
        investigate(case=case, dx=dx, method=m, directional=d, settings=settings)
        for m, d in methods
    ]
    (root / "steep_cfl.json").write_text(json.dumps(rows, separators=(",", ":")) + "\n")
