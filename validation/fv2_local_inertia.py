"""Matched GPU benchmarks against full-SWE and reduced local-inertia analytical solutions."""

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
import warp as wp

from floodwarp.solver_factory import create_overland_solver
from validation.fv1_riemann import crossing, errors, report_json
from validation.fv1_riemann_cases import coordinates, stoker_star
from validation.fv2_cost import reset
from validation.fv2_rain_flow import steady_fields


def dam_state(*, left, right, gravity, reduced):
    c = np.sqrt(gravity * left)
    if reduced:

        def mismatch(h):
            return 2 * np.sqrt(gravity) * (left**1.5 - h**1.5) / 3 - (h - right) * np.sqrt(
                gravity * (h + right) / 2
            )

        h = brentq(f=mismatch, a=right, b=left, xtol=1e-14)
        speed = np.sqrt(gravity * (h + right) / 2)
        flux = (h - right) * speed
        tail = -np.sqrt(gravity * h)
        np.testing.assert_allclose(speed * flux, gravity * (h * h - right * right) / 2, rtol=2e-12)
    elif right > 0:
        h, u, speed = stoker_star(left=left, right=right, gravity=gravity)
        flux = h * u
        tail = u - np.sqrt(gravity * h)
        np.testing.assert_allclose(
            speed * flux, h * u * u + gravity * (h * h - right * right) / 2, rtol=2e-12
        )
    else:
        h, flux, speed, tail = 0.0, 0.0, 2 * c, 2 * c
    return h, flux, (-c, tail, speed)


def dam_average(*, x, dx, duration, left, right, gravity, reduced):
    hstar, qstar, speeds = dam_state(left=left, right=right, gravity=gravity, reduced=reduced)
    lo, hi = x - dx / 2, x + dx / 2
    boundaries = [lo, *[np.clip(s * duration, lo, hi) for s in speeds], hi]
    state = np.zeros((*x.shape, 3))
    points, weights = np.polynomial.legendre.leggauss(4)
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        for node, weight in zip(points, weights, strict=True):
            xi = (a + b + (b - a) * node) / (2 * duration)
            hfan = xi**2 / gravity if reduced else (2 * np.sqrt(gravity * left) - xi) ** 2 / (9 * gravity)
            qfan = (
                2 * np.sqrt(gravity) * (left**1.5 - hfan**1.5) / 3
                if reduced
                else hfan * (2 / 3) * (np.sqrt(gravity * left) + xi)
            )
            h = np.select(
                [xi < speeds[0], xi < speeds[1], xi < speeds[2]], [left, hfan, hstar], default=right
            )
            q = np.select([xi < speeds[0], xi < speeds[1], xi < speeds[2]], [0.0, qfan, qstar], default=0.0)
            state[..., 0] += h * weight * (b - a) / (2 * dx)
            state[..., 1] += q * weight * (b - a) / (2 * dx)
    return state, dict(star_depth_m=hstar, star_discharge_m2_s=qstar, right_wave_m_s=speeds[2])


def domain(*, case, dx, duration, gravity, near_zero_n, length, width):
    x, y = coordinates(length=length, width=width, dx=dx)
    z = np.zeros_like(x)
    n = np.full_like(x, 0.03)
    rain = np.zeros_like(x)
    q = np.zeros((*x.shape, 3))
    qx = np.zeros((x.shape[0], x.shape[1] + 1))
    qy = np.zeros((x.shape[0] + 1, x.shape[1]))
    window = (np.abs(x) < length / 4) & (np.abs(y) < width / 4)
    info = {}
    if case in ("weak_dam", "stoker", "ritter"):
        left, right = {"weak_dam": (1.05, 1.0), "stoker": (2.0, 0.4), "ritter": (2.0, 0.0)}[case]
        n.fill(near_zero_n)
        q[..., 0] = np.where(x < 0, left, right)
        refs = {}
        for label in ("swe", "li"):
            refs[label], info[label] = dam_average(
                x=x, dx=dx, duration=duration, left=left, right=right, gravity=gravity, reduced=label == "li"
            )
        info["reference"] = "inviscid limiting solutions"
    elif case == "damped_contact":
        h, u, v, nval = 1.0, 1.5, 0.7, 0.025
        n.fill(nval)
        k = gravity * nval**2 * np.hypot(u, v) / h ** (4 / 3)
        displacement = u * np.log1p(k * duration) / k
        q[..., 0] = h
        q[..., 1] = h * u
        q[..., 2] = h * v * np.where(x < 0, 1.0, -1.0)
        qx[:] = h * u
        qy[:] = h * v * np.where(x[0] < 0, 1.0, -1.0)
        refs = {}
        for label, position in (("swe", displacement), ("li", 0.0)):
            r = q.copy()
            r[..., 1:] /= 1 + k * duration
            fraction = np.clip((position - (x - dx / 2)) / dx, 0, 1)
            r[..., 2] = h * v * (2 * fraction - 1) / (1 + k * duration)
            refs[label] = r
            info[label] = dict(contact_position_m=position)
        info["reference"] = "distinct exact positive-friction solutions"
    elif case in ("thin_rain", "deep_rain"):
        h, q0 = (0.03, 0.004) if case == "thin_rain" else (0.15, 0.03)
        z, n, rain, qmean, profile, _ = steady_fields(
            x=x,
            dx=dx,
            length=length,
            depth=h,
            inflow=q0,
            mean_rain=2e-5,
            manning=0.025,
            roughness_change=0.015,
            gravity=gravity,
        )
        q[..., 0] = h
        q[..., 1] = qmean
        qx[:] = profile(np.arange(qx.shape[1]) * dx / length)
        refs = {"swe": q.copy()}
        info.update(
            reference="exact full-SWE rain and friction equilibrium",
            gauge_volume_per_width_m2=float(profile(0.5) * duration),
        )
    elif case == "accelerating_plane":
        h, grade, nval = 0.5, 0.002, 0.035
        n.fill(nval)
        z = -grade * x
        q[..., 0] = h
        terminal = h ** (2 / 3) * np.sqrt(grade) / nval
        rate = gravity * grade / terminal
        r = q.copy()
        r[..., 1] = h * terminal * np.tanh(rate * duration)
        refs = {"swe": r, "li": r}
        info.update(
            reference="shared exact positive-friction accelerating flow",
            gauge_volume_per_width_m2=float(h * terminal * np.log(np.cosh(rate * duration)) / rate),
        )
    elif case == "lake":
        z = 1000.0 + 0.43 * np.cos(x / 41) + 0.39 * np.sin(y / 33) + 0.18 * np.sin((x + 2 * y) / 53)
        q[..., 0] = np.maximum(1000.1 - z, 0)
        refs = {"swe": q.copy(), "li": q.copy()}
        info.update(reference="shared exact partially dry lake", gauge_volume_per_width_m2=0.0)
    else:
        raise ValueError(case)
    qx[:, [0, -1]] = 0.0
    qy[[0, -1], :] = 0.0
    return dict(x=x, window=window, z=z, n=n, rain=rain, initial=q, qx=qx, qy=qy, refs=refs, info=info)


def build(*, d, method, dx, cap, device, gravity):
    li = method == "li"
    key = method if li else ("fv2" if method in ("mc", "minmod") else "fv1")
    kw = (
        dict(
            track_cumulative_face_volumes=True,
            track_absolute_face_volumes=False,
            track_diagnostic_max_depth=False,
        )
        if li
        else dict(execution_mode="dense", gravity=gravity)
    )
    if key == "fv2":
        kw["limiter"] = method
    if key == "fv1":
        kw["flux_scheme"] = method
    s = create_overland_solver(
        solver_method=key,
        ny=d["z"].shape[0],
        nx=d["z"].shape[1],
        dx=dx,
        dy=dx,
        bed_elevation=d["z"],
        manning_n=d["n"],
        active_mask=np.ones_like(d["z"], bool),
        device=device,
        explicit_minimum_dt=1e-8,
        explicit_maximum_dt=cap,
        **kw,
    )
    if not np.isclose(float(s.gravity), gravity, rtol=0, atol=1e-13):
        raise ValueError("Mismatched gravity")
    s.set_water_depth(depth=d["initial"][..., 0].copy())
    if li:
        s.set_face_fluxes(qx=d["qx"], qy=d["qy"])
    else:
        s.set_momentum(hu=d["initial"][..., 1], hv=d["initial"][..., 2])
    s.set_source_rates(rain=d["rain"])
    return s


def read_state(*, solver, li):
    if not li:
        return solver.get_conservative_state()
    qx, qy = solver.get_qx(), solver.get_qy()
    return np.stack(
        (solver.get_water_depth(), (qx[:, 1:] + qx[:, :-1]) / 2, (qy[1:, :] + qy[:-1, :]) / 2), axis=-1
    )


def experiment(
    *,
    directory,
    cases,
    spacings,
    methods,
    n_values,
    gravity,
    duration,
    length,
    width,
    cap_ratio,
    device,
    repeats,
    wet,
):
    rows = []
    for case in cases:
        inviscid = case in ("weak_dam", "stoker", "ritter")
        for dx in spacings:
            for n in n_values if inviscid else (n_values[-1],):
                d = domain(
                    case=case,
                    dx=dx,
                    duration=duration,
                    gravity=gravity,
                    near_zero_n=n,
                    length=length,
                    width=width,
                )
                for cap in (
                    (dx * cap_ratio, dx * cap_ratio / 2)
                    if dx == 5 and n == n_values[-1]
                    else (dx * cap_ratio,)
                ):
                    for method in methods:
                        s = build(d=d, method=method, dx=dx, cap=cap, device=device, gravity=gravity)
                        li = method == "li"
                        saved = None if li else s.capture_state()
                        times = []
                        wall = []
                        captures = []
                        timed_count = repeats if cap == dx * cap_ratio and n == n_values[-1] else 0
                        for trial in range(timed_count + 1):
                            if trial and li:
                                s.set_water_depth(depth=d["initial"][..., 0].copy())
                                s.set_face_fluxes(qx=d["qx"], qy=d["qy"])
                            elif trial:
                                reset(solver=s, state=saved)
                            wp.synchronize_device(device=device)
                            start = perf_counter()
                            run = s.run_device_loop(duration=duration)
                            wp.synchronize_device(device=device)
                            elapsed = perf_counter() - start
                            captures.append(run["graph_capture_seconds"])
                            if trial:
                                times.append(run["loop_seconds"])
                                wall.append(elapsed)
                        result = read_state(solver=s, li=li)
                        ledger = s.cumulative_accounting()
                        storage = float((result[..., 0].sum() * dx**2))
                        residual = (
                            s.initial_volume
                            + ledger["rain"]
                            + ledger["boundary_inflow"]
                            - ledger["boundary_outflow"]
                            - ledger["infiltration"]
                            - storage
                        )
                        if (
                            not np.isfinite(result).all()
                            or result[..., 0].min() < 0
                            or abs(residual) > 1e-9 * max(s.initial_volume, 1.0)
                        ):
                            raise ArithmeticError(f"Invalid {case} {dx} {method}: volume residual {residual}")
                        raw = s.get_cumulative_face_volumes()
                        transport = raw["cumulative_net_qx_volume_m3" if li else "qx"]
                        # Average only over the central half-width, beyond wall influence.
                        section_rows = np.abs((np.arange(s.ny) + 0.5) * dx - width / 2) < width / 4
                        gauge = float(transport[section_rows, s.nx // 2].mean() / dx)
                        metrics = {
                            key: errors(
                                result=result, expected=truth, wet_depth=wet, dx=dx, window=d["window"]
                            )
                            for key, truth in d["refs"].items()
                        }
                        position = None
                        if case == "damped_contact":
                            mid = s.ny // 2
                            position = crossing(
                                x=d["x"][mid],
                                profile=result[mid, :, 2],
                                level=0.0,
                                target=d["info"]["swe"]["contact_position_m"],
                            )
                        rows.append(
                            dict(
                                case=case,
                                dx=dx,
                                method=method,
                                duration_s=duration,
                                dt_cap_s=cap,
                                baseline_dt_cap_s=dx * cap_ratio,
                                gravity=gravity,
                                near_zero_manning=n if inviscid else None,
                                exact=d["info"],
                                errors=metrics,
                                gauge_volume_per_width_m2=gauge,
                                contact_position_m=position,
                                domain_min_depth_m=float(result[..., 0].min()),
                                budget_residual_m3=residual,
                                initial_volume_m3=s.initial_volume,
                                numerical=s.numerical_configuration(),
                                steps=run["steps"],
                                min_dt_s=run["dt_min_s"],
                                mean_dt_s=run["dt_mean_s"],
                                warm_loop_s=times,
                                warm_wall_s=wall,
                                graph_capture_s=captures,
                            )
                        )
            report_json(path=directory / "local_inertia.json", content=rows)
            print(f"{case} {dx:g} m complete, {len(rows)} records", flush=True)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    cases = (
        "damped_contact",
        "accelerating_plane",
        "thin_rain",
        "deep_rain",
        "weak_dam",
        "stoker",
        "ritter",
        "lake",
    )
    spacings = (20.0, 10.0, 5.0, 2.5)
    methods = ("li", "contact_hll", "roe", "hllc", "minmod", "mc")
    experiment(
        directory=directory,
        cases=cases,
        spacings=spacings,
        methods=methods,
        n_values=(1e-5, 1e-7),
        gravity=9.80665,
        duration=24.0,
        length=960.0,
        width=640.0,
        cap_ratio=1 / 64,
        device="cuda:0",
        repeats=5,
        wet=0.01,
    )
