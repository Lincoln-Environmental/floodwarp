"""Discharge verification on accelerating wetting fronts and steady hydraulic controls."""

from pathlib import Path

import numpy as np

from validation.fv1_riemann import crossing, errors, report_json
from validation.fv1_riemann_cases import Domain, coordinates
from validation.fv2_accuracy import configured


def ritter_integral(*, left, right, time, slope, head, gravity):
    """Integrate h and hu from left to right on a frictionless constant slope."""
    if time == 0:
        return head * np.maximum(np.minimum(right, 0) - left, 0), np.zeros_like(left)
    acceleration = gravity * slope
    translation = 0.5 * acceleration * time**2
    c = np.sqrt(gravity * head)
    head_x = translation - c * time
    front = translation + 2 * c * time
    upstream = head * np.maximum(np.minimum(right, head_x) - left, 0)
    near = np.clip(front - left, 0, front - head_x)
    far = np.clip(front - right, 0, front - head_x)
    fan = (near**3 - far**3) / (27 * gravity * time**2)
    fan_momentum = 2 * c * fan - (near**4 - far**4) / (54 * gravity * time**3)
    return upstream + fan, acceleration * time * (upstream + fan) + fan_momentum


def bump(*, x, height, half_width):
    return np.where(np.abs(x) < half_width, height * np.cos(np.pi * x / (2 * half_width)) ** 2, 0.0)


def bernoulli_depth(*, energy, bed, discharge, gravity, shallow):
    available = energy - bed
    cosine = 1 - 27 * discharge**2 / (4 * gravity * available**3)
    if np.any(cosine < -1 - 4e-14) or np.any(cosine > 1 + 4e-14):
        raise ValueError("No physical depth at this specific energy")
    angle = np.arccos(np.clip(cosine, -1, 1)) / 3
    angle = angle - np.where(shallow, 2 * np.pi / 3, 0)
    h = available * (1 + 2 * np.cos(angle)) / 3
    np.testing.assert_allclose(h + discharge**2 / (2 * gravity * h**2), available, rtol=2e-13)
    return h


def bump_state(*, x, regime, height, half_width, discharge, gravity, jump):
    z = bump(x=x, height=height, half_width=half_width)
    hc = (discharge**2 / gravity) ** (1 / 3)
    critical_head = 1.5 * hc + height
    energy = critical_head if regime.startswith("transcritical") else critical_head + height
    shallow = (x > 0) if regime.startswith("transcritical") else (regime == "supercritical")
    if regime == "transcritical_jump":
        zj = bump(x=jump, height=height, half_width=half_width)
        before = bernoulli_depth(energy=energy, bed=zj, discharge=discharge, gravity=gravity, shallow=True)
        after = before * (np.sqrt(1 + 8 * discharge**2 / (gravity * before**3)) - 1) / 2
        np.testing.assert_allclose(
            discharge**2 / before + gravity * before**2 / 2,
            discharge**2 / after + gravity * after**2 / 2,
            rtol=1e-14,
        )
        downstream = after + zj + discharge**2 / (2 * gravity * after**2)
        if downstream >= energy:
            raise ValueError("Hydraulic jump must dissipate specific energy")
        energy = np.where(x > jump, downstream, energy)
        shallow = shallow & (x <= jump)
    return z, bernoulli_depth(energy=energy, bed=z, discharge=discharge, gravity=gravity, shallow=shallow)


def bump_cell_averages(*, centres, dx, regime, height, half_width, discharge, gravity, jump, quadrature):
    points, weights = np.polynomial.legendre.leggauss(quadrature)
    limits = sorted((-half_width, 0.0, jump, half_width))
    start = centres - dx / 2
    end = centres + dx / 2
    total = np.zeros((2, len(centres)))
    for edge in [*limits, np.inf]:
        upper = np.maximum(start, np.minimum(end, edge))
        xs = (upper + start)[:, None] / 2 + (upper - start)[:, None] * points / 2
        z, h = bump_state(
            x=xs,
            regime=regime,
            height=height,
            half_width=half_width,
            discharge=discharge,
            gravity=gravity,
            jump=jump,
        )
        # Gauss nodes at zero-width segments can coincide with the discontinuity.
        total += np.stack((z @ weights, h @ weights)) * ((upper - start) / (2 * dx))
        start = upper
    return total


def case_data(
    *,
    case,
    x,
    dx,
    duration,
    gravity,
    head,
    height,
    half_width,
    discharge,
    jump,
    quadrature,
):
    name, slope = case
    if name == "sloping_ritter":
        bed = -slope * x
        h = np.where(x < 0, head, 0.0)
        q = np.stack((h, np.zeros_like(x), np.zeros_like(x)), axis=-1)
        h_int, m_int = ritter_integral(
            left=x - dx / 2,
            right=x + dx / 2,
            time=duration,
            slope=slope,
            head=head,
            gravity=gravity,
        )
        target = np.stack((h_int / dx, m_int / dx, np.zeros_like(x)), axis=-1)
    else:
        z, h = bump_cell_averages(
            centres=x[0],
            dx=dx,
            regime=name,
            height=height,
            half_width=half_width,
            discharge=discharge,
            gravity=gravity,
            jump=jump,
            quadrature=quadrature,
        )
        bed = np.broadcast_to(z, x.shape).copy()
        q = np.stack(
            (np.broadcast_to(h, x.shape), np.full_like(x, discharge), np.zeros_like(x)),
            axis=-1,
        )
        target = q.copy()
    return Domain(dx=dx, bed=bed, initial=q, duration=duration), target


def run(
    *,
    cases,
    spacings,
    methods,
    gauges,
    duration,
    interval,
    length,
    width,
    gravity,
    head,
    height,
    half_width,
    discharge,
    jump,
    quadrature,
    timestep_ratio,
    window,
    wet,
    device,
    output,
):
    rows = []
    times = np.arange(0, duration + interval, interval)
    for name, slope in cases:
        for dx in spacings:
            x, _ = coordinates(length=length, width=width, dx=dx)
            d, target = case_data(
                case=(name, slope),
                x=x,
                dx=dx,
                duration=duration,
                gravity=gravity,
                head=head,
                height=height,
                half_width=half_width,
                discharge=discharge,
                jump=jump,
                quadrature=quadrature,
            )
            face = np.rint((np.asarray(gauges) + length / 2) / dx).astype(int)
            np.testing.assert_allclose(face * dx - length / 2, gauges, rtol=0, atol=0)
            expected = []
            for t in times:
                if name == "sloping_ritter":
                    vol, _ = ritter_integral(
                        left=np.array(gauges),
                        right=length / 2,
                        time=t,
                        slope=slope,
                        head=head,
                        gravity=gravity,
                    )
                else:
                    vol = np.full(len(gauges), discharge * t)
                expected.append(vol)
            exact = np.array(expected)
            for method in methods:
                s = configured(domain=d, method=method, cap=dx * timestep_ratio, device=device)
                volumes = [np.zeros(len(face))]
                for _ in range(len(times) - 1):
                    s.run_device_loop(duration=interval)
                    volumes.append(s.get_cumulative_face_volumes()["qx"][:, face].sum(axis=0) / width)
                q = s.get_conservative_state()
                if not np.all(np.isfinite(q)) or np.min(q[..., 0]) < 0:
                    raise FloatingPointError("Nonphysical state")
                ledger = s.cumulative_accounting()
                if abs(ledger["residual"]) > 1e-11 * s.initial_volume:
                    raise AssertionError("Global volume residual exceeds tolerance")
                numerical = np.diff(volumes, axis=0) / interval
                theory = np.diff(exact, axis=0) / interval
                hydro = []
                for i, gauge in enumerate(gauges):
                    hydro.append(
                        dict(
                            x_m=gauge,
                            exact_interval_mean_m2_s=theory[:, i].tolist(),
                            model_interval_mean_m2_s=numerical[:, i].tolist(),
                            rmse_m2_s=float(np.sqrt(np.mean((numerical[:, i] - theory[:, i]) ** 2))),
                            signed_peak_error_m2_s=float(numerical[:, i].max() - theory[:, i].max()),
                            integrated_transport_error_m2=float(volumes[-1][i] - exact[-1, i]),
                            exact_transport_m2=float(exact[-1, i]),
                        )
                    )
                row = dict(
                    case=name,
                    slope=slope,
                    setup=dict(
                        gravity=gravity,
                        upstream_depth_m=head,
                        crest_height_m=height,
                        crest_halfwidth_m=half_width,
                        stationary_discharge_m2_s=discharge,
                        jump_x_m=jump,
                        width_m=width,
                        length_m=length,
                        wet_threshold_m=wet,
                    ),
                    dx=dx,
                    method=method,
                    duration_s=duration,
                    interval_s=interval,
                    steps=s.steps,
                    min_depth_domain_m=float(q[..., 0].min()),
                    ledger=ledger,
                    gauges=hydro,
                    **errors(
                        result=q,
                        expected=target,
                        wet_depth=wet,
                        dx=dx,
                        window=(x > window[0]) & (x < window[1]),
                    ),
                )
                if name == "sloping_ritter":
                    analytical = (
                        gravity * slope * duration**2 / 2
                        + (2 * np.sqrt(gravity * head) - 3 * np.sqrt(gravity * wet)) * duration
                    )
                    where = crossing(x=x[0], profile=q[0, :, 0], level=wet, target=analytical)
                    row["front_at_threshold_error_m"] = None if where is None else where - analytical
                rows.append(row)
            report_json(path=output, content=rows)
            print(f"{name}, slope {slope:g}, {dx:g} m complete", flush=True)


if __name__ == "__main__":
    cases = (
        ("sloping_ritter", 0.0),
        ("sloping_ritter", 0.02),
        ("sloping_ritter", 0.08),
        ("sloping_ritter", -0.02),
        ("subcritical", 0.0),
        ("supercritical", 0.0),
        ("transcritical", 0.0),
        ("transcritical_jump", 0.0),
    )
    spacings = (20.0, 10.0, 5.0, 2.5)
    methods = ("fv1", "roe", "hllc", "minmod", "mc")
    run(
        cases=cases,
        spacings=spacings,
        methods=methods,
        gauges=(0.0, 40.0, 80.0),
        duration=20.0,
        interval=1.0,
        length=1600.0,
        width=40.0,
        gravity=9.81,
        head=2.0,
        height=0.4,
        half_width=60.0,
        discharge=1.5,
        jump=22.5,
        quadrature=16,
        timestep_ratio=1 / 160,
        window=(-180.0, 400.0),
        wet=0.01,
        device="cuda:0",
        output=Path("validation/fv2_review/flow_truth.json"),
    )
