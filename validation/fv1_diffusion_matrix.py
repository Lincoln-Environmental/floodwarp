"""FP64 accepted-face comparisons with common physical support and independent truth."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from validation.fv1_diffusion_domains import design, prepare, front_depth
from validation.fv2_local_inertia import build, read_state
from validation.fv1_riemann import report_json


def initialise(*, case, dx, method, policy, multiplier, settings):
    data, states, exact, times, x = prepare(
        case=case, dx=dx, gravity=settings["gravity"], samples=settings["samples"], order=settings["order"]
    )
    # LI requires strictly positive roughness. Record its frictionless-limit substitution.
    actual = np.maximum(data["n"] * multiplier, settings["li_floor"] if method == "li" else 0.0)
    cap = dx * settings["common_cap_per_m"] if policy == "common" else settings["max_dt"]
    solver = build(
        d=dict(data, n=actual),
        method=method,
        dx=dx,
        cap=cap,
        device=settings["device"],
        gravity=settings["gravity"],
    )
    if case.infiltration_mm_h:
        solver.set_source_rates(infil=np.full_like(data["z"], case.infiltration_mm_h / 3.6e6))
    return solver, states, exact, times, x


def face_transport(*, solver, case, dx, x, li):
    volumes = solver.get_cumulative_face_volumes()
    if case.family in ("pulse", "damped_pulse"):
        a = volumes["cumulative_net_qy_volume_m3" if li else "qy"][solver.ny // 2]
        return np.array([a[np.abs(x - g) < 10].sum() / 20 for g in case.gauges])
    a = volumes["cumulative_net_qx_volume_m3" if li else "qx"]
    faces = np.rint((np.array(case.gauges) + case.length / 2) / dx).astype(int)
    return a[:, faces].sum(axis=0) / case.width


def step_interval(*, solver, case, a, b):
    if case.rain_mm_h and abs(a - case.rain_end) < 1e-9:
        solver.set_source_rates(rain=np.zeros((solver.ny, solver.nx)))
    return solver.run_device_loop(duration=b - a)


def sample(*, case, solver, states, times, x, dx, li):
    rows = solver.ny // 2
    volumes = [np.zeros(len(case.gauges))]
    snapshots = [states[0].T]
    stats = []
    minimum = states[0, 0].min()
    for a, b in zip(times[:-1], times[1:], strict=True):
        stats.append(step_interval(solver=solver, case=case, a=a, b=b))
        volumes.append(face_transport(solver=solver, case=case, dx=dx, x=x, li=li))
        field = read_state(solver=solver, li=li)
        minimum = min(minimum, field[..., 0].min())
        if minimum < 0 or not np.isfinite(field).all():
            raise ArithmeticError("Nonphysical state in " + case.name)
        snapshots.append(field[rows].copy())
    ledger = solver.cumulative_accounting()
    residual = (
        solver.initial_volume
        + ledger["rain"]
        + ledger["boundary_inflow"]
        - ledger["boundary_outflow"]
        - ledger["infiltration"]
        - field[..., 0].sum() * dx**2
    )
    if abs(residual) > 1e-9 * max(solver.initial_volume, 1.0):
        raise ArithmeticError(f"Water balance residual {residual}")
    return (
        np.array(snapshots).transpose(0, 2, 1),
        np.diff(volumes, axis=0) / np.diff(times)[:, None],
        dict(
            steps=sum(v["steps"] for v in stats),
            dt_min_s=min(v["dt_min_s"] for v in stats),
            rejects=sum(v.get("rejected_cfl_trials", 0) for v in stats),
            min_depth_m=float(minimum),
            residual_m3=float(residual),
            graph_capture_s=sum(v["graph_capture_seconds"] for v in stats),
        ),
    )


def rms(a):
    return float(np.mean(a * a) ** 0.5)


def measures(*, result, states, discharge, exact, times, x, dx, case, settings):
    # Interior support is outside the physical boundary signal cone over the integration.
    radius = case.length / 6
    if case.family == "front":
        upstream = np.array([case.initial_tip + case.speed * case.duration + case.length / 2])
        maximum = front_depth(distance=upstream, case=case)[0]
        radius = min(
            radius, case.length / 2 - case.duration * (case.speed + np.sqrt(settings["gravity"] * maximum))
        )
        if max(abs(g) for g in case.gauges) > radius:
            raise ValueError("Analytical gauge lies inside the conservative wall-signal cone")
    selected = np.abs(x) <= radius
    h = result[:, 0, selected]
    ref = states[:, 0, selected]
    delta = h - ref
    wet = ref > settings["wet"][0]
    qerr = discharge - exact
    dt = times[1] - times[0]
    spatial = {}
    for threshold in settings["wet"]:
        m = h > threshold
        e = ref > threshold
        union = (m | e).sum()
        both = m.any(axis=0) & e.any(axis=0)
        spatial[str(threshold)] = dict(
            iou=float((m & e).sum() / union) if union else 1.0,
            extent_rmse_m=rms((m.sum(axis=1) - e.sum(axis=1)) * dx),
            last_wet_rmse_s=rms((m[::-1, both].argmax(axis=0) - e[::-1, both].argmax(axis=0)) * dt)
            if both.any()
            else None,
            shoreline_rmse_m=rms((m.sum(axis=1) - e.sum(axis=1)) * dx) if case.family == "front" else None,
            arrival_rmse_s=rms((m[:, both].argmax(axis=0) - e[:, both].argmax(axis=0)) * dt)
            if both.any()
            else None,
        )
    gauges = []
    for j, loc in enumerate(case.gauges):
        num, truth = discharge[:, j], exact[:, j]
        stationary = np.ptp(truth) <= 1e-12 * max(1.0, np.max(abs(truth)))
        gauges.append(
            dict(
                location_m=loc,
                exact_peak=float(truth.max()),
                model_peak=float(num.max()),
                peak_error=float(num.max() - truth.max()),
                timing_error_s=None if stationary else float(dt * (np.argmax(num) - np.argmax(truth))),
                width_delta_s=float(dt * ((num > num.max() / 2).sum() - (truth > truth.max() / 2).sum())),
                volume_error_m2=float(np.sum(num - truth) * dt),
            )
        )
    return dict(
        interior_radius_m=float(radius),
        depth_rmse=rms(delta),
        wet_depth_rmse=rms(delta[wet]) if wet.any() else None,
        depth_linf=float(np.max(abs(delta))),
        maximum_depth_rmse=rms(h.max(axis=0) - ref.max(axis=0)),
        momentum_x_rmse=rms(result[:, 1, selected] - states[:, 1, selected]),
        momentum_y_rmse=rms(result[:, 2, selected] - states[:, 2, selected]),
        extent=spatial,
        hydro_rmse=rms(qerr),
        gauges=gauges,
        peak_error_rmse=rms(num_array(gauges, "peak_error")),
        timing_error_rmse=rms(
            np.array([g["timing_error_s"] for g in gauges if g["timing_error_s"] is not None])
        )
        if any(g["timing_error_s"] is not None for g in gauges)
        else None,
        volume_error_rmse=rms(num_array(gauges, "volume_error_m2")),
    )


def regime(*, case, dx, settings, radius, multiplier, li):
    data, states, _, _, x = prepare(
        case=case, dx=dx, gravity=settings["gravity"], samples=settings["samples"], order=settings["order"]
    )
    inside = np.abs(x) <= radius
    h = states[:, 0, inside]
    wet = h > settings["wet"][0]
    speed = np.hypot(states[:, 1, inside], states[:, 2, inside])
    fr = np.divide(speed, np.sqrt(settings["gravity"]) * h**1.5, where=wet, out=np.zeros_like(h))
    n = data["n"]
    floor = settings["li_floor"] if li else 0.0
    return dict(
        fr_wet_median=float(np.median(fr[wet])) if wet.any() else None,
        fr_wet_max=float(fr.max()),
        mean_abs_grade=float(np.mean(abs(np.diff(data["z"][0]) / dx))),
        physical_n_range=[float(n.min()), float(n.max())],
        model_n_range=[max(float(n.min()) * multiplier, floor), max(float(n.max()) * multiplier, floor)],
    )


def num_array(rows, key):
    return np.array([i[key] for i in rows])


def forecast(*, case, dx, method, policy, multiplier, settings, keep_series=True, repeats=0):
    solver, states, exact, times, x = initialise(
        case=case, dx=dx, method=method, policy=policy, multiplier=multiplier, settings=settings
    )
    init = solver.capture_state() if repeats else None
    li = method == "li"
    result, q, stats = sample(case=case, solver=solver, states=states, times=times, x=x, dx=dx, li=li)
    metrics = measures(
        result=result,
        states=states,
        discharge=q,
        exact=exact,
        times=times,
        x=x,
        dx=dx,
        case=case,
        settings=settings,
    )
    row = dict(
        case=case.name,
        dx=dx,
        method=method,
        policy=policy,
        multiplier=multiplier,
        model_n=max(case.n * multiplier, settings["li_floor"] if li else 0.0),
        **stats,
        **metrics,
        **regime(
            case=case,
            dx=dx,
            settings=settings,
            radius=metrics["interior_radius_m"],
            multiplier=multiplier,
            li=li,
        ),
    )
    if keep_series:
        row.update(interval_end_s=times[1:].tolist(), exact_m2_s=exact.tolist(), model_m2_s=q.tolist())
    if repeats:
        walls = []
        loops = []
        for _ in range(repeats):
            if li:
                solver.restore_state(state=init)
            else:
                from validation.fv2_cost import reset

                reset(solver=solver, state=init)
            if case.rain_mm_h and case.rain_end < case.duration:
                solver.set_source_rates(rain=np.full((solver.ny, solver.nx), case.rain_mm_h / 3.6e6))
            wp.synchronize_device(device=settings["device"])
            start = perf_counter()
            calls = [
                step_interval(solver=solver, case=case, a=a, b=b)
                for a, b in zip(times[:-1], times[1:], strict=True)
            ]
            wp.synchronize_device(device=settings["device"])
            walls.append(perf_counter() - start)
            loops.append(sum(call["loop_seconds"] for call in calls))
        row.update(
            warm_wall_s=walls,
            warm_device_loop_s=loops,
            median_wall_s=float(np.median(walls)),
            cell_steps_s=solver.nx * solver.ny * stats["steps"] / float(np.median(walls)),
            dt_mean_s=case.duration / stats["steps"],
        )
    return row


def matrix(*, cases, spacings, methods, policies, settings, output, repeats):
    rows = []
    for case in cases:
        for dx in spacings:
            for method in methods:
                for policy in policies:
                    rows.append(
                        forecast(
                            case=case,
                            dx=dx,
                            method=method,
                            policy=policy,
                            multiplier=1.0,
                            settings=settings,
                            repeats=repeats,
                        )
                    )
            print(case.name, dx, len(rows), flush=True)
        report_json(path=output / content_name(case=case), content=rows)
        rows = []


def content_name(*, case):
    return case.name + ".json"


if __name__ == "__main__":
    output = Path("validation/fv1_diffusion_review/matrix")
    settings = dict(
        gravity=9.80665,
        samples=12,
        order=16,
        device="cuda:0",
        li_floor=1e-8,
        common_cap_per_m=1 / 32,
        max_dt=5.0,
        wet=(0.001, 0.01, 0.05),
    )
    methods = (
        "fv1",
        "low_froude_hll",
        "euler_minmod",
        "euler_mc",
        "hancock_minmod",
        "hancock_mc",
        "hancock_adaptive",
        "hancock_low_froude",
        "minmod",
        "mc",
        "adaptive",
        "li",
    )
    spacings = (10.0, 5.0, 1.0)
    policies = ("common", "production")
    repeats = 3
    cases = design()
    report_json(
        path=output / "configuration.json",
        content=dict(settings=settings, cases=[asdict(c) for c in cases], repeats=repeats),
    )
    matrix(
        cases=cases,
        spacings=spacings,
        methods=methods,
        policies=policies,
        settings=settings,
        output=output,
        repeats=repeats,
    )
