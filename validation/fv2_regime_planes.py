"""Shared exact SWE and local-inertia plane flows and flat wetting/loss controls.

For a uniform sheet, u' = g*S - g*n²*u*abs(u)/h^(4/3).
The three interior sections exclude boundary signals for the sampled durations.
"""

from pathlib import Path
import numpy as np
from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import coordinates
from validation.fv2_local_inertia import build, read_state


def fields(*, grid, slope, depth, roughness, datum, start, gravity, rain, **unused):
    x, y = grid
    initial = np.zeros((*x.shape, 3))
    terminal = depth ** (2 / 3) * np.sqrt(slope) / roughness
    initial[..., 0] = depth
    if start == "steady":
        initial[..., 1] = depth * terminal
    qx = np.zeros((x.shape[0], x.shape[1] + 1))
    qx[:, 1:-1] = initial[:, :-1, 1]
    return dict(
        initial=initial,
        z=datum - slope * x,
        n=np.full_like(x, roughness),
        qx=qx,
        qy=np.zeros((x.shape[0] + 1, x.shape[1])),
        rain=np.full_like(y, rain / 3.6e6),
    )


def exact(*, time, slope, depth, roughness, start, gravity, rain, loss, **unused):
    terminal = depth ** (2 / 3) * np.sqrt(slope) / roughness
    if slope == 0:
        return max(0.0, depth + time * (rain - loss) / 3.6e6), 0.0, 0.0
    if start == "steady":
        return depth, depth * terminal, time * depth * terminal
    rate = gravity * slope / terminal
    volume = depth * terminal * np.logaddexp(rate * time, -rate * time) / rate
    volume -= depth * terminal * np.log(2.0) / rate
    return depth, depth * terminal * np.tanh(rate * time), volume


def run(*, parameters, method, dx, length, width, duration, interval, device, wet, sections):
    grid = coordinates(length=length, width=width, dx=dx)
    d = fields(grid=grid, **parameters)
    s = build(
        d=d, method=method, dx=dx, cap=min(interval, dx / 32), device=device, gravity=parameters["gravity"]
    )
    # Set both arrays explicitly as the LI and SWE setters have different optional arguments.
    s.set_source_rates(rain=d["rain"], infil=np.full_like(d["rain"], parameters["loss"] / 3.6e6))
    indices = np.rint((np.asarray(sections) + length / 2) / dx).astype(int)
    window = abs(grid[0]) < length / 4
    volumes = [np.zeros(len(indices))]
    reference = [0.0]
    times = []
    steps = 0
    for t in np.arange(interval, duration + interval / 2, interval):
        stats = s.run_device_loop(duration=interval)
        times.append(stats["loop_seconds"])
        steps += stats["steps"]
        data = s.get_cumulative_face_volumes()
        v = data["cumulative_net_qx_volume_m3" if method == "li" else "qx"]
        volumes.append(v[:, indices].mean(axis=0) / dx)
        h, m, cum = exact(time=t, **parameters)
        reference.append(cum)
    target = np.zeros_like(d["initial"])
    target[..., 0], target[..., 1] = h, m
    q = read_state(solver=s, li=method == "li")
    actual = np.diff(volumes, axis=0) / interval
    truth = np.diff(reference) / interval
    diff = actual - truth[:, None]
    ledger = s.cumulative_accounting()
    mass = (
        s.initial_volume
        + ledger["rain"]
        - ledger["infiltration"]
        + ledger["boundary_inflow"]
        - ledger["boundary_outflow"]
        - q[..., 0].sum() * dx**2
    )
    if not np.isfinite(q).all() or q[..., 0].min() < 0:
        raise ArithmeticError("Invalid sheet-flow state")
    iou = {}
    for threshold in wet:
        a, b = q[..., 0][window] > threshold, target[..., 0][window] > threshold
        iou[str(threshold)] = float(np.sum(a & b) / np.sum(a | b)) if np.any(a | b) else 1.0
    return dict(
        **parameters,
        method=method,
        dx=dx,
        steps=steps,
        loop_s=sum(times),
        duration=duration,
        froude=m / (h * np.sqrt(parameters["gravity"] * h)) if h > 0 else 0.0,
        metrics=errors(result=q, expected=target, wet_depth=wet[0], dx=dx, window=window),
        slope_rmse=float(np.sqrt(np.mean(np.gradient(q[..., 0], dx, axis=1)[window] ** 2))),
        wet_iou=iou,
        sections_m=sections,
        interval_s=interval,
        exact_q=truth.tolist(),
        discharge=actual.tolist(),
        q_rmse=float(np.sqrt(np.mean(diff**2))),
        peak_error=(actual.max(axis=0) - truth.max()).tolist(),
        peak_time_error_s=((actual.argmax(axis=0) - truth.argmax()) * interval).tolist()
        if np.ptp(truth) > 1e-10
        else None,
        integrated_error=(volumes[-1] - reference[-1]).tolist(),
        volume_residual_m3=float(mass),
    )


def matrix(*, output, grades, depths, roughness, rain_rates, methods, grids, gravity, **options):
    rows = []
    for i, slope in enumerate(grades):
        for j, start in enumerate(("steady", "accelerating")):
            base = dict(
                slope=slope / 100,
                depth=depths[(i + 2 * j) % len(depths)],
                roughness=roughness[(i // 2 + j) % len(roughness)],
                gravity=gravity,
                start=start,
                rain=0.0,
                loss=0.0,
            )
            for datum in (0.0, 100.0, 2.0e6) if i in (0, 3, 7) and j == 0 else (0.0,):
                for dx in grids:
                    for method in methods:
                        rows.append(run(parameters=dict(base, datum=datum), dx=dx, method=method, **options))
        report_json(path=output, content=rows)
        print("slope percent", slope, flush=True)
    for rain in rain_rates:
        for loss in (0.0,) if rain else (100.0,):
            initial = 0.0 if rain else 0.0005
            for method in methods:
                rows.append(
                    run(
                        parameters=dict(
                            slope=0.0,
                            depth=initial,
                            roughness=0.04,
                            datum=100.0,
                            gravity=gravity,
                            start="dry_rain" if rain else "exhaustion",
                            rain=rain,
                            loss=loss,
                        ),
                        dx=grids[0],
                        method=method,
                        **options,
                    )
                )
    report_json(path=output, content=rows)


if __name__ == "__main__":
    matrix(
        output=Path("validation/fv2_review/regimes/planes.json"),
        grades=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        depths=(0.005, 0.02, 0.1, 0.3, 1.0),
        roughness=(0.02, 0.04, 0.08, 0.15),
        rain_rates=(0.0, 5.0, 20.0, 50.0, 100.0),
        methods=("contact_hll", "minmod", "adaptive", "li"),
        grids=(10.0, 5.0, 2.5),
        gravity=9.80665,
        length=1600.0,
        width=40.0,
        duration=24.0,
        interval=3.0,
        wet=(0.001, 0.01, 0.1),
        sections=(-100.0, 0.0, 100.0),
        device="cuda:0",
    )
