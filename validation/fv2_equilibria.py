"""Steady flows, oblique shocks, shear, and nonlinear physical source convergence."""

from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

from floodwarp.fv2_solver import FV2Solver
from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import Domain, coordinates
from validation.fv1_riemann_order import exact_average
from validation.fv2_accuracy import configured


def oblique(*, x, y, dx, g, angle, offset):
    co, si = np.cos(angle), np.sin(angle)
    h = 0.4
    froude = 2.25
    discharge = froude * h * np.sqrt(g * h)
    downstream = h * (np.sqrt(1 + 8 * froude**2) - 1) / 2
    a, b = dx * co, dx * si
    t = offset - (x * co + y * si) + (a + b) / 2
    fraction = (
        np.maximum(t, 0.0) ** 2
        - np.maximum(t - a, 0.0) ** 2
        - np.maximum(t - b, 0.0) ** 2
        + np.maximum(t - a - b, 0.0) ** 2
    ) / (2 * a * b)
    depth = downstream + (h - downstream) * np.clip(fraction, 0.0, 1.0)
    return np.stack((depth, np.full_like(x, discharge * co), np.full_like(x, discharge * si)), axis=-1)


def steady(*, case, dx, length, width, g):
    x, y = coordinates(length=length, width=width, dx=dx)
    h = 1.0 + (0.0007 * x if case == "gradual" else np.zeros_like(x))
    m = 0.8
    n = 0.035
    slope = n**2 * m**2
    if case == "gradual":
        bed = -h - m**2 / (2 * g * h**2) + 3 * slope * (h ** (-7 / 3) - 1) / (7 * 0.0007)
    else:
        bed = -slope * x
    q = np.stack((h, np.full_like(x, m), np.zeros_like(x)), axis=-1)
    return Domain(dx=dx, bed=bed, initial=q, duration=16.0, manning=n, stage_x=tuple((bed + h)[0, [0, -1]]))


def controls(*, spacings, device):
    rows = []
    for dx in spacings:
        for case in ("uniform", "gradual", "oblique", "smooth_shear"):
            if case == "oblique":
                x, y = coordinates(length=640.0, width=640.0, dx=dx)
                q = oblique(x=x, y=y, dx=dx, g=9.81, angle=np.pi / 6, offset=13.75)
                d = Domain(dx=dx, bed=np.zeros_like(x), initial=q, duration=16.0)
                window = (np.abs(x) < 120.0) & (np.abs(y) < 120.0)
                expected = q
            elif case == "smooth_shear":
                x, y = coordinates(length=2560.0, width=40.0, dx=dx)
                h = np.ones_like(x)
                v = exact_average(x=x, dx=dx, speed=1.5, length=640.0, time=0.0)
                d = Domain(
                    dx=dx,
                    bed=0.0 * x,
                    initial=np.stack((h, 1.5 * h, v), axis=-1),
                    duration=160.0,
                    stage_x=(1.0, 1.0),
                    stage_y=1.0,
                )
                expected = d.initial.copy()
                expected[..., 2] = exact_average(x=x, dx=dx, speed=1.5, length=640.0, time=d.duration)
                window = np.abs(x) < 320.0
            else:
                d = steady(case=case, dx=dx, length=640.0, width=40.0, g=9.81)
                x, y = coordinates(length=640.0, width=40.0, dx=dx)
                window = np.abs(x) < 120.0
                expected = d.initial
            for method in ("fv1", "minmod", "mc"):
                s = configured(domain=d, method=method, cap=dx / 32.0, device=device)
                stats = s.run_device_loop(duration=d.duration)
                result = s.get_conservative_state()
                row = dict(
                    case=case,
                    dx=dx,
                    method=method,
                    stats=stats,
                    balance=s.cumulative_accounting(),
                    **errors(result=result, expected=expected, wet_depth=0.01, dx=dx, window=window),
                )
                if case == "oblique":
                    depth = result[..., 0]
                    crossings = (depth[:, 1:] - np.mean([q[..., 0].max(), q[..., 0].min()])) * (
                        depth[:, :-1] - np.mean([q[..., 0].max(), q[..., 0].min()])
                    ) <= 0.0
                    ys, xs = np.where(crossings & window[:, 1:] & window[:, :-1])
                    target = 0.5 * (q[..., 0].max() + q[..., 0].min())
                    pos = x[ys, xs] + dx * (target - depth[ys, xs]) / (depth[ys, xs + 1] - depth[ys, xs])
                    row["mean_normal_jump_offset_m"] = float(
                        np.mean(pos * np.cos(np.pi / 6) + y[ys, xs] * 0.5 - 13.75)
                    )
                rows.append(row)
    return rows


def physical_sources(*, dts, device):
    n = 0.12
    q0 = np.array([0.8, 0.28, -0.16, 0.0])

    def rhs(t, y):
        h, m, v, soil = y
        cap = 0.02 + 0.08 * np.exp(-soil / 0.04)
        sink = cap / h + 9.81 * n * n * np.hypot(m, v) / h ** (7 / 3)
        return [0.03 + 0.01 * t - cap, -sink * m, -sink * v, cap]

    truth = solve_ivp(fun=rhs, t_span=(0.0, 2.0), y0=q0, method="DOP853", rtol=1.0e-13, atol=1.0e-15).y[:, -1]
    rows = []
    for dt in dts:
        s = FV2Solver(
            ny=31,
            nx=31,
            dx=500.0,
            dy=500.0,
            bed_elevation=0.0,
            manning_n=n,
            active_mask=np.ones((31, 31), bool),
            maximum_dt=dt,
            device=device,
        )
        s.set_water_depth(depth=q0[0])
        s.set_momentum(hu=q0[1], hv=q0[2])
        s.set_exponential_infiltration(f0=0.1, fd=0.04, conductivity=0.02)
        s.set_rainfall_hyetograph(times_s=np.array([0.0, 2.0]), rates_m_s=np.array([0.03, 0.05]))
        # Distant reflecting boundaries isolate the sampled interior.
        stats = s.run_device_loop(duration=2.0)
        q = s.get_conservative_state()[15, 15]
        y = np.r_[q, s.cumulative_infiltration_m()[15, 15]]
        rows.append(dict(dt=dt, errors=np.abs(y - truth).tolist(), truth=truth.tolist(), stats=stats))
    return rows


if __name__ == "__main__":
    output = Path("validation/fv2_review/equilibria.json")
    spacings = (20.0, 10.0, 5.0, 2.5)
    device = "cuda:0"
    dts = (0.25, 0.125, 0.0625, 0.03125)
    report_json(
        path=output,
        content=dict(
            rows=controls(spacings=spacings, device=device),
            coupled_source=physical_sources(dts=dts, device=device),
        ),
    )
