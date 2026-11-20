"""Exact moving-water equilibrium with spatial rain, Manning friction and vertical rainfall momentum."""

from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import Domain, coordinates
from validation.fv2_accuracy import configured


def steady_fields(*, x, dx, length, depth, inflow, mean_rain, manning, roughness_change, gravity):
    # The independent coordinate is the fraction of the distance from the left boundary.
    xi = (x + length / 2) / length
    rain = Polynomial([mean_rain / 2, mean_rain])
    rough = Polynomial([manning, roughness_change])
    q = (length * rain).integ(k=inflow)
    gradient = -2 * rain * q / (gravity * depth**2) - rough**2 * q**2 / (depth ** (10 / 3))
    z = (length * gradient).integ()

    def average(*, poly):
        primitive = poly.integ()
        delta = dx / length
        return (primitive(xi + delta / 2) - primitive(xi - delta / 2)) / delta

    # Check the steady water and normal-momentum equations in physical x units.
    np.testing.assert_allclose(q.deriv()(xi) / length, rain(xi), rtol=2e-14)
    np.testing.assert_allclose(
        (q * q / depth).deriv()(xi) / length,
        -gravity * depth * (z.deriv()(xi) / length + rough(xi) ** 2 * q(xi) ** 2 / depth ** (10 / 3)),
        rtol=5e-11,
        atol=2e-14,
    )
    return (
        average(poly=z),
        average(poly=rough),
        average(poly=rain),
        average(poly=q),
        q,
        gradient,
    )


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
    mean_rain,
    manning,
    roughness_change,
    gravity,
    window,
    cap_ratio,
    wet,
    device,
    output,
):
    rows = []
    for name, depth, inflow in cases:
        for dx in spacings:
            x, _ = coordinates(length=length, width=width, dx=dx)
            bed, n, rain, q, profile, grade = steady_fields(
                x=x,
                dx=dx,
                length=length,
                depth=depth,
                inflow=inflow,
                mean_rain=mean_rain,
                manning=manning,
                roughness_change=roughness_change,
                gravity=gravity,
            )
            initial = np.stack((np.full_like(x, depth), q, np.zeros_like(x)), axis=-1)
            d = Domain(dx=dx, bed=bed, initial=initial, duration=duration, manning=n, rain=rain)
            faces = np.rint((np.asarray(gauges) + length / 2) / dx).astype(int)
            np.testing.assert_array_equal(faces * dx - length / 2, gauges)
            exact = profile((np.asarray(gauges) + length / 2) / length)
            for method in methods:
                solver = configured(domain=d, method=method, cap=dx * cap_ratio, device=device)
                volume = [np.zeros(len(faces))]
                for _ in range(round(duration / interval)):
                    solver.run_device_loop(duration=interval)
                    volume.append(solver.get_cumulative_face_volumes()["qx"][:, faces].sum(axis=0) / width)
                hydro = np.diff(volume, axis=0) / interval
                result = solver.get_conservative_state()
                ledger = solver.cumulative_accounting()
                if (
                    not np.isfinite(result).all()
                    or result[..., 0].min() < 0
                    or abs(ledger["residual"]) > 1e-11 * (solver.initial_volume + ledger["rain"])
                ):
                    raise FloatingPointError("Invalid rain-fed state or budget")
                np.testing.assert_allclose(ledger["rain"], mean_rain * length * width * duration, rtol=1e-11)
                records = []
                for i, gauge in enumerate(gauges):
                    records.append(
                        dict(
                            x_m=gauge,
                            exact_discharge_m2_s=float(exact[i]),
                            model_interval_mean_m2_s=hydro[:, i].tolist(),
                            rmse_m2_s=float(np.sqrt(np.mean((hydro[:, i] - exact[i]) ** 2))),
                            integrated_relative_bias=float(volume[-1][i] / (duration * exact[i]) - 1),
                        )
                    )
                rows.append(
                    dict(
                        case=name,
                        depth_m=depth,
                        setup=dict(
                            gravity=gravity,
                            inflow_m2_s=inflow,
                            initial_roughness=manning,
                            roughness_change=roughness_change,
                            mean_rain_m_s=mean_rain,
                            length_m=length,
                            width_m=width,
                            wet_threshold_m=wet,
                        ),
                        dx=dx,
                        method=method,
                        duration_s=duration,
                        interval_s=interval,
                        steps=solver.steps,
                        ledger=ledger,
                        minimum_depth_domain_m=float(result[..., 0].min()),
                        gauges=records,
                        centre_grade=float(grade(0.5)),
                        centre_froude=float(profile(0.5) / (depth * np.sqrt(gravity * depth))),
                        minimum_maximum_rain_mm_hr=[
                            mean_rain * 0.5 * 3.6e6,
                            mean_rain * 1.5 * 3.6e6,
                        ],
                        **errors(
                            result=result,
                            expected=initial,
                            dx=dx,
                            wet_depth=wet,
                            window=np.abs(x) < window,
                        ),
                    )
                )
            report_json(path=output, content=rows)
            print(f"{name}, {dx:g} m, all five methods complete", flush=True)


if __name__ == "__main__":
    cases = (("thin_transcritical", 0.03, 0.004), ("deep_subcritical", 0.15, 0.03))
    spacings = (20.0, 10.0, 5.0, 2.5)
    methods = ("fv1", "roe", "hllc", "minmod", "mc")
    run(
        cases=cases,
        spacings=spacings,
        methods=methods,
        gauges=(-80.0, 0.0, 80.0),
        duration=40.0,
        interval=2.0,
        length=1600.0,
        width=40.0,
        mean_rain=2e-5,
        manning=0.025,
        roughness_change=0.015,
        gravity=9.81,
        window=160.0,
        cap_ratio=1 / 64,
        wet=0.01,
        device="cuda:0",
        output=Path("validation/fv2_review/rain_flow.json"),
    )
