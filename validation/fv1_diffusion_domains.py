"""Exact travelling Manning fronts, advected shear pulses and uniform sheet events."""

import json
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.special import erf, hyp2f1
from validation.fv1_riemann_cases import coordinates


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    slope: float
    n: float
    speed: float
    depth: float
    rain_mm_h: float
    rain_end: float
    length: float
    width: float
    duration: float
    initial_tip: float
    gauges: tuple
    infiltration_mm_h: float = 0.0


def front_depth(*, distance, case):
    """Invert the separable travelling-wave ODE using its upstream distance integral."""
    d = np.maximum(distance, 0.0)
    a = (case.n * case.speed) ** 2
    if case.slope == 0.0:
        return (7 * a * d / 3) ** (3 / 7)
    hn = (a / case.slope) ** 0.75
    lower = np.zeros_like(d)
    upper = np.ones_like(d)
    for _ in range(54):
        t = np.minimum((lower + upper) / 2, np.nextafter(1.0, 0.0))
        # t=(h/hn)^(1/3). Series prevents cancellation near the dry front.
        integral = (np.arctanh(t) - np.arctan(t)) / 2 - t**3 / 3
        small = t < 0.15
        integral[small] = t[small] ** 7 / 7 * hyp2f1(1.0, 1.75, 2.75, t[small] ** 4)
        above = 3 * hn / case.slope * integral > d
        lower = np.where(above, lower, t)
        upper = np.where(above, t, upper)
    return np.where(d > 0, hn * ((lower + upper) / 2) ** 3, 0.0)


def front_mean(*, low, high, case, order):
    """Cell or time average including the dry fraction. Transform D=a+(b-a)s^7."""
    lo = np.maximum(low, 0.0)
    hi = np.maximum(high, 0.0)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    s = (1 + nodes) / 2
    distance = lo[..., None] + (hi - lo)[..., None] * s**7
    mean = (
        (hi - lo)
        / (high - low)
        * np.sum(front_depth(distance=distance, case=case) * 7 * s**6 * weights / 2, axis=-1)
    )
    return mean


def sheet(*, case, t, gravity):
    if case.slope:
        terminal = case.depth ** (2 / 3) * np.sqrt(case.slope) / case.n
        tanh = np.tanh(gravity * case.slope / terminal * t)
        velocity = (case.speed + terminal * tanh) / (1 + case.speed / terminal * tanh)
        return case.depth, case.depth * velocity
    rain = case.rain_mm_h / 3.6e6
    end = min(t, case.rain_end)
    h = case.depth + rain * end
    # No initial rain momentum. Friction integrates in closed form for constant rain.
    drag = gravity * case.n**2
    integral = (
        -3 / 4 * case.depth ** (-4 / 3) * np.expm1(-4 / 3 * np.log1p(rain * end / case.depth)) / rain
        if rain
        else end / case.depth ** (7 / 3)
    )
    if case.infiltration_mm_h:
        # Pure infiltration removes momentum in proportion to the pre-withdrawal velocity.
        rate = case.infiltration_mm_h / 3.6e6
        h = case.depth - rate * t
        integral_u = 3 * (h ** (-1 / 3) - case.depth ** (-1 / 3)) / rate
        return h, h / (1 / case.speed + drag * integral_u)
    return h, 1 / (1 / (case.speed * case.depth) + drag * (integral + (t - end) / h ** (7 / 3)))


def pulse_integral(*, left, right, centre, sigma):
    return (
        sigma
        * np.sqrt(np.pi / 2)
        * (erf((right - centre) / (np.sqrt(2) * sigma)) - erf((left - centre) / (np.sqrt(2) * sigma)))
    )


def simple_wave(*, case, x, t, gravity):
    """Invert the strictly increasing right-family characteristic map before breaking."""
    base = np.sqrt(gravity * case.depth)
    amplitude = case.speed / 2
    sigma = 12.0
    if 3 * amplitude * t * np.exp(-0.5) >= sigma:
        raise ValueError("Gravity pulse has reached characteristic crossing")
    lo = x - (base + 3 * amplitude) * t
    hi = x - base * t
    for _ in range(45):
        xi = (lo + hi) / 2
        c = base + amplitude * np.exp(-0.5 * ((xi - case.initial_tip) / sigma) ** 2)
        error = xi + (3 * c - 2 * base) * t - x
        hi = np.where(error >= 0, xi, hi)
        lo = np.where(error < 0, xi, lo)
    c = base + amplitude * np.exp(-0.5 * (((lo + hi) / 2 - case.initial_tip) / sigma) ** 2)
    return c * c / gravity, 2 * (c - base) * c * c / gravity


def damped_pulse(*, case, left, right, t, gravity):
    transverse = 0.5 * case.speed
    rate = gravity * case.n**2 * np.hypot(case.speed, transverse) / case.depth ** (4 / 3)
    centre = case.initial_tip + case.speed * np.log1p(rate * t) / rate
    overlap = np.maximum(0.0, np.minimum(right, centre + 10) - np.maximum(left, centre - 10))
    factor = 1 / (1 + rate * t)
    return (
        case.depth * case.speed * factor,
        case.depth * transverse * factor * (2 * overlap / (right - left) - 1),
    )


def fields(*, case, x, dx, t, gravity, order):
    if case.family == "damped_pulse":
        u, v = damped_pulse(case=case, left=x - dx / 2, right=x + dx / 2, t=t, gravity=gravity)
        return np.full_like(x, case.depth), np.full_like(x, u), v
    if case.family == "gravity_pulse":
        nodes, weights = np.polynomial.legendre.leggauss(order)
        h, m = simple_wave(case=case, x=x[:, None] + dx / 2 * nodes, t=t, gravity=gravity)
        return h @ weights / 2, m @ weights / 2, np.zeros_like(x)
    if case.family == "front":
        tip = case.initial_tip + case.speed * t
        h = front_mean(low=tip - x - dx / 2, high=tip - x + dx / 2, case=case, order=order)
        return h, h * case.speed, np.zeros_like(h)
    if case.family == "pulse":
        h = np.full_like(x, case.depth)
        v = (
            pulse_integral(
                left=x - dx / 2, right=x + dx / 2, centre=case.initial_tip + case.speed * t, sigma=7.0
            )
            / dx
        )
        return h, h * case.speed, h * v
    depth, mom = sheet(case=case, t=t, gravity=gravity)
    return np.full_like(x, depth), np.full_like(x, mom), np.zeros_like(x)


@lru_cache(maxsize=128)
def prepare(*, case, dx, gravity, samples, order):
    x, y = coordinates(length=case.length, width=case.width, dx=dx)
    xs = x[0]
    times = np.linspace(0.0, case.duration, samples + 1)
    if case.family in ("macdonald", "rain_equilibrium"):
        return steady_domain(case=case, x=x, dx=dx, times=times, gravity=gravity)
    states = np.array([fields(case=case, x=xs, dx=dx, t=t, gravity=gravity, order=order) for t in times])
    init = np.broadcast_to(states[0].T, (len(y), len(xs), 3)).copy()
    xf = np.linspace(-case.length / 2, case.length / 2, len(xs) + 1)
    qx = np.full((len(y), len(xf)), init[0, 0, 1])
    if case.family == "front":
        qx[:] = case.speed * front_depth(distance=case.initial_tip - xf, case=case)
    if case.family == "gravity_pulse":
        _, qx[:] = simple_wave(case=case, x=xf, t=0.0, gravity=gravity)
    qy = np.broadcast_to(init[0, :, 2], (len(y) + 1, len(xs))).copy()
    qx[:, [0, -1]] = 0.0
    qy[[0, -1], :] = 0.0
    data = dict(
        z=-case.slope * x,
        n=np.full_like(x, case.n),
        rain=np.full_like(x, case.rain_mm_h / 3.6e6),
        initial=init,
        qx=qx,
        qy=qy,
    )
    gauges = np.asarray(case.gauges)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    expected = []
    for a, b in zip(times[:-1], times[1:], strict=True):
        if case.family == "front":
            q = case.speed * front_mean(
                low=case.initial_tip + case.speed * a - gauges,
                high=case.initial_tip + case.speed * b - gauges,
                case=case,
                order=order,
            )
        else:
            values = []
            for t in (a + b) / 2 + (b - a) / 2 * nodes:
                if case.family == "damped_pulse":
                    values.append(
                        damped_pulse(case=case, left=gauges - 10, right=gauges + 10, t=t, gravity=gravity)[1]
                    )
                elif case.family == "gravity_pulse":
                    values.append(simple_wave(case=case, x=gauges, t=t, gravity=gravity)[1])
                elif case.family == "pulse":
                    values.append(
                        case.depth
                        * pulse_integral(
                            left=gauges - 10,
                            right=gauges + 10,
                            centre=case.initial_tip + case.speed * t,
                            sigma=7.0,
                        )
                        / 20
                    )
                else:
                    values.append(
                        np.full_like(gauges, sheet(case=case, t=t, gravity=gravity)[1], dtype=float)
                    )
            q = weights @ np.array(values) / 2
            if case.family == "damped_pulse":
                rate = gravity * case.n**2 * np.hypot(case.speed, case.speed / 2) / case.depth ** (4 / 3)
                for j, gauge in enumerate(gauges):
                    hits = (
                        np.expm1(
                            rate * (gauge + np.array([-20.0, 0.0, 20.0]) - case.initial_tip) / case.speed
                        )
                        / rate
                    )
                    segments = np.r_[a, hits[(hits > a) & (hits < b)], b]
                    total = 0.0
                    for start, stop in zip(segments[:-1], segments[1:], strict=True):
                        pts = (start + stop) / 2 + (stop - start) / 2 * nodes
                        _, section = damped_pulse(
                            case=case, left=gauge - 10, right=gauge + 10, t=pts, gravity=gravity
                        )
                        total += (stop - start) / 2 * (section @ weights)
                    q[j] = total / (b - a)
        expected.append(q)
    return data, states, np.array(expected), times, xs


def steady_domain(*, case, x, dx, times, gravity):
    from validation.fv2_macdonald_manning import averaged_fields
    from validation.fv2_rain_flow import steady_fields

    ny, nx = x.shape
    xf = (np.arange(nx + 1) * dx - case.length / 2) / case.length + 0.5
    if case.family == "macdonald":
        bed, q = averaged_fields(
            x=x,
            dx=dx,
            depth_at_zero=case.depth,
            decay=0.001,
            discharge=case.depth * case.speed,
            manning=case.n,
            gravity=gravity,
        )
        n = np.full_like(x, case.n)
        rain = np.zeros_like(x)
        face = np.full(nx + 1, case.depth * case.speed)
        section = np.full(len(case.gauges), case.depth * case.speed)
    else:
        bed, n, rain, qmean, profile, _ = steady_fields(
            x=x,
            dx=dx,
            length=case.length,
            depth=case.depth,
            inflow=case.depth * case.speed,
            mean_rain=case.rain_mm_h / 3.6e6,
            manning=case.n,
            roughness_change=0.015,
            gravity=gravity,
        )
        q = np.stack((np.full_like(x, case.depth), qmean, np.zeros_like(x)), axis=-1)
        face = profile(xf)
        section = profile(np.array(case.gauges) / case.length + 0.5)
    qx = np.broadcast_to(face, (ny, nx + 1)).copy()
    qx[:, [0, -1]] = 0.0
    data = dict(z=bed, n=n, rain=rain, initial=q, qx=qx, qy=np.zeros((ny + 1, nx)))
    return (
        data,
        np.broadcast_to(q[0].T, (len(times), 3, nx)),
        np.broadcast_to(section, (len(times) - 1, len(section))),
        times,
        x[0],
    )


def design(*, path=None):
    """Load the common physical controls. A custom manifest can be supplied for reproduction."""
    source = Path(__file__).with_name("fv1_diffusion_review") / "domains.json" if path is None else Path(path)
    return tuple(
        Case(**dict(entry, gauges=tuple(entry["gauges"]))) for entry in json.loads(source.read_text())
    )
