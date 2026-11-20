"""Exact signed section hydrographs for a genuinely two-dimensional oscillating basin."""

from pathlib import Path

import numpy as np

from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import Domain, coordinates
from validation.fv2_accuracy import configured


def radius(*, time, omega, first_radius, second_radius):
    return np.hypot(first_radius * np.cos(omega * time), second_radius * np.sin(omega * time))


def right_storage(*, gauge, radius_m, centre_depth):
    """Integrate a wet paraboloid in the half-plane x >= gauge."""
    t = np.clip(np.asarray(gauge) / radius_m, -1, 1)
    primitive = (t * np.sqrt(1 - t * t) * (5 - 2 * t * t) + 3 * np.arcsin(t)) / 8
    return (4 / 3) * centre_depth * radius_m**2 * (3 * np.pi / 16 - primitive)


def wet_cells(*, x, y, dx, radius_m, depth, expansion, quadrature):
    """Integrate the wet paraboloid exactly in y and by split Gaussian integration in x."""
    west, east = x - dx / 2, x + dx / 2
    south, north = y - dx / 2, y + dx / 2
    r2 = radius_m**2
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    limits = [west, east]
    for edge in (south, north):
        cut = np.sqrt(np.maximum(r2 - edge**2, 0))
        limits.extend([np.clip(-cut, west, east), np.clip(cut, west, east)])
    limits.extend([np.clip(-radius_m, west, east), np.clip(radius_m, west, east)])
    edges = np.sort(limits, axis=0)
    h = np.zeros_like(x)
    hu = np.zeros_like(x)
    hv = np.zeros_like(x)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        for node, weight in zip(nodes, weights, strict=True):
            px = (b + a + (b - a) * node) / 2
            cap = np.sqrt(np.maximum(r2 - px**2, 0))
            ly = np.maximum(-cap, south)
            ry = np.maximum(ly, np.minimum(cap, north))
            dist = ry - ly
            scale = weight * (b - a) * depth / (2 * dx**2 * r2)
            line = dist * (r2 - px**2 - (ry**2 + ry * ly + ly**2) / 3)
            h += scale * line
            hu += scale * expansion * px * line
            hv += scale * expansion * dist * (ry + ly) * ((r2 - px**2) / 2 - (ry**2 + ly**2) / 4)
    return np.stack((h, hu, hv), axis=-1)


def run(
    *,
    spacings,
    methods,
    gauges,
    gravity,
    curvature,
    first_radius,
    second_radius,
    length,
    cycles,
    samples,
    quadrature,
    wet,
    cap_ratio,
    device,
    output,
):
    omega = np.sqrt(2 * gravity * curvature)
    period = np.pi / omega
    dt = period / samples
    times = np.arange(1, cycles * samples + 1) * dt
    head = curvature * second_radius**2
    total = np.pi * head * first_radius**2 / 2
    initial_mass = right_storage(gauge=gauges, radius_m=first_radius, centre_depth=head)
    exact = []
    for t in np.r_[0, times]:
        r = radius(time=t, omega=omega, first_radius=first_radius, second_radius=second_radius)
        exact.append(
            right_storage(gauge=gauges, radius_m=r, centre_depth=head * (first_radius / r) ** 2)
            - initial_mass
        )
    truth = np.diff(exact, axis=0) / dt
    rows = []
    for dx in spacings:
        x, y = coordinates(length=length, width=length, dx=dx)
        q = wet_cells(
            x=x,
            y=y,
            dx=dx,
            radius_m=first_radius,
            depth=head,
            expansion=0.0,
            quadrature=quadrature,
        )
        double = wet_cells(
            x=x,
            y=y,
            dx=dx,
            radius_m=first_radius,
            depth=head,
            expansion=0.0,
            quadrature=2 * quadrature,
        )
        cell_error = np.max(np.abs(q - double))
        relative_volume = abs(q[..., 0].sum() * dx**2 - total) / total
        if relative_volume > 1e-9 or cell_error > 1e-7 or q[..., 0].min() < 0:
            raise AssertionError(f"Initial condition quadrature failed: {cell_error=}, {relative_volume=}")
        domain = Domain(
            dx=dx,
            bed=curvature * (x * x + y * y + dx * dx / 6),
            initial=q,
            duration=cycles * period,
        )
        faces = np.rint((np.array(gauges) + length / 2) / dx).astype(int)
        np.testing.assert_allclose(faces * dx - length / 2, gauges, rtol=0, atol=0)
        for method in methods:
            s = configured(domain=domain, method=method, cap=dx * cap_ratio, device=device)
            volumes = [np.zeros(len(gauges))]
            symmetry = 0.0
            minimum = float(q[..., 0].min())
            for t in times:
                s.run_device_loop(duration=t - s.time)
                transport = s.get_cumulative_face_volumes()
                x_vol = transport["qx"][:, faces].sum(axis=0)
                y_vol = transport["qy"][faces, :].sum(axis=1)
                symmetry = max(symmetry, float(np.max(np.abs(x_vol - y_vol))))
                volumes.append(x_vol)
                minimum = min(minimum, float(s.get_water_depth().min()))
            hydro = np.diff(volumes, axis=0) / dt
            result = s.get_conservative_state()
            ledger = s.cumulative_accounting()
            if minimum < 0 or not np.all(np.isfinite(result)) or abs(ledger["residual"]) > 1e-11 * total:
                raise FloatingPointError("Invalid basin state or budget")
            stats = []
            for i, gauge in enumerate(gauges):
                # Report timing and amplitude of the second-cycle peak independently of the first cycle.
                section = slice(-samples, None)
                measured = hydro[section, i]
                expected = truth[section, i]
                stats.append(
                    dict(
                        x_m=gauge,
                        exact_m3_s=truth[:, i].tolist(),
                        model_m3_s=hydro[:, i].tolist(),
                        cycle2_peak_exact_m3_s=float(expected.max()),
                        cycle2_peak_model_m3_s=float(measured.max()),
                        cycle2_peak_time_error_s=(
                            float((np.argmax(measured) - np.argmax(expected)) * dt)
                            if measured.max() > 0
                            else None
                        ),
                        cycle2_positive_peak_resolved=bool(measured.max() > 0),
                        hydrograph_rmse_m3_s=float(np.sqrt(np.mean((hydro[:, i] - truth[:, i]) ** 2))),
                        gross_positive_volume_error_m3=float(
                            dt * (np.maximum(0, hydro[:, i]) - np.maximum(0, truth[:, i])).sum()
                        ),
                        signed_net_transport_m3=float(volumes[-1][i]),
                    )
                )
            rows.append(
                dict(
                    dx=dx,
                    method=method,
                    cycles=cycles,
                    setup=dict(
                        gravity=gravity,
                        curvature=curvature,
                        first_radius_m=first_radius,
                        second_radius_m=second_radius,
                        initial_centre_depth_m=head,
                        length_m=length,
                        wet_threshold_m=wet,
                    ),
                    period_s=period,
                    interval_s=dt,
                    steps=s.steps,
                    gauges=stats,
                    initial_quadrature_cell_error_m=float(cell_error),
                    initial_relative_volume_error=float(relative_volume),
                    initial_volume_m3=total,
                    minimum_depth_m=minimum,
                    max_section_symmetry_difference_m3=symmetry,
                    ledger=ledger,
                    **errors(
                        result=result,
                        expected=q,
                        dx=dx,
                        wet_depth=wet,
                        window=(x * x + y * y <= second_radius**2),
                    ),
                )
            )
        report_json(path=output, content=rows)
        print(
            f"Radial basin {dx:g} m, {cycles} cycles, all five methods complete",
            flush=True,
        )


if __name__ == "__main__":
    spacings = (20.0, 10.0, 5.0, 2.5)
    methods = ("fv1", "roe", "hllc", "minmod", "mc")
    run(
        spacings=spacings,
        methods=methods,
        gauges=(40.0, 80.0, 120.0),
        gravity=9.81,
        curvature=0.0001,
        first_radius=80.0,
        second_radius=160.0,
        length=480.0,
        cycles=2,
        samples=64,
        quadrature=16,
        wet=0.01,
        cap_ratio=1 / 32,
        device="cuda:0",
        output=Path("validation/fv2_review/radial_flow.json"),
    )
