"""Exact varying-depth steady shallow-water flow with Manning resistance.

MacDonald-style manufactured geometry without artificial momentum forcing:
q is constant, h varies exponentially, and z(x) is integrated analytically
from the steady full-SWE momentum equation. The exact section hydrograph is
q(t)=q0 and the depth field is stationary. Reflecting walls are sufficiently
remote from the central error window for the simulated duration.

This is an additional benchmark, distinct from constant-depth rain-fed flow.
"""
from pathlib import Path
import numpy as np

from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import Domain, coordinates
from validation.fv2_accuracy import configured


def exp_cell_average(*, rate, centre, dx):
    half = rate * dx * 0.5
    if abs(half) < 1.0e-8:
        factor = 1.0 + half * half / 6.0 + half**4 / 120.0
    else:
        factor = np.sinh(half) / half
    return np.exp(rate * centre) * factor


def exact_fields(*, x, depth_at_zero, decay, discharge, manning, gravity):
    depth = depth_at_zero * np.exp(-decay * x)
    momentum = np.full_like(depth, discharge)
    bed = (
        -depth
        - (discharge**2 / (2.0 * gravity * depth_at_zero**2)) * np.exp(2.0 * decay * x)
        - (3.0 * manning**2 * discharge * abs(discharge)
           / (10.0 * decay * depth_at_zero**(10.0 / 3.0)))
        * np.exp((10.0 * decay / 3.0) * x)
    )
    return bed, depth, momentum


def averaged_fields(*, x, dx, depth_at_zero, decay, discharge, manning, gravity):
    height = depth_at_zero * exp_cell_average(rate=-decay, centre=x, dx=dx)
    kinetic = (discharge**2 / (2.0 * gravity * depth_at_zero**2)) * exp_cell_average(
        rate=2.0 * decay, centre=x, dx=dx
    )
    friction = (
        3.0 * manning**2 * discharge * abs(discharge)
        / (10.0 * decay * depth_at_zero**(10.0 / 3.0))
        * exp_cell_average(rate=(10.0 * decay / 3.0), centre=x, dx=dx)
    )
    bed = -height - kinetic - friction
    state = np.stack((height, np.full_like(x, discharge), np.zeros_like(x)), axis=-1)
    return bed, state


def run(*, output, device, methods, spacings, gauges, length, width, duration,
        interval, depth_at_zero, decay, discharge, manning, gravity):
    rows = []
    for dx in spacings:
        x, _ = coordinates(length=length, width=width, dx=dx)
        bed, state = averaged_fields(
            x=x, dx=dx, depth_at_zero=depth_at_zero, decay=decay,
            discharge=discharge, manning=manning, gravity=gravity
        )
        domain = Domain(dx=dx, bed=bed, initial=state, duration=duration, manning=manning)
        face_indices = np.rint((np.asarray(gauges) + length / 2.0) / dx).astype(int)
        np.testing.assert_allclose(face_indices * dx - length / 2.0, gauges, rtol=0.0, atol=1.e-12)
        for method in methods:
            solver = configured(domain=domain, method=method, cap=dx / 128.0, device=device)
            volumes = [np.zeros(len(gauges))]
            for _ in range(round(duration / interval)):
                solver.run_device_loop(duration=interval)
                total = solver.get_cumulative_face_volumes()["qx"][:, face_indices].sum(axis=0) / width
                volumes.append(total)
            field = solver.get_conservative_state()
            ledgers = solver.cumulative_accounting()
            if not np.isfinite(field).all() or field[..., 0].min() < 0.0:
                raise ArithmeticError("Nonphysical state in varying-depth Manning equilibrium")
            flux = np.diff(volumes, axis=0) / interval
            exact = np.full_like(flux, discharge)
            window = np.abs(x) <= 120.0
            metrics = errors(result=field, expected=state, wet_depth=0.01, dx=dx, window=window)
            rows.append(dict(
                case="varying_depth_manning_macdonald",
                dx_m=dx, method=method, steps=solver.steps,
                duration_s=duration, gravity_m_s2=gravity,
                depth_at_zero_m=depth_at_zero, discharge_m2_s=discharge,
                roughness_manning=manning, depth_decay_per_m=decay,
                section_positions_m=list(gauges), section_discharge=flux.tolist(),
                exact_section_discharge_m2_s=discharge,
                discharge_rmse_m2_s=float(np.sqrt(np.mean((flux - exact)**2))),
                cumulative_signed_transport_error_m2=(volumes[-1] - discharge * duration).tolist(),
                ledger=ledgers, **metrics
            ))
            del solver
        report_json(path=output, content=dict(rows=rows))
        print(f"MacDonald Manning equilibrium at {dx:g} m complete", flush=True)


if __name__ == "__main__":
    run(
        output=Path("validation/fv2_review/macdonald_manning.json"),
        device="cuda:0",
        methods=("fv1", "roe", "hllc", "minmod", "mc"),
        spacings=(20.0, 10.0, 5.0, 2.5),
        gauges=(-80.0, 0.0, 80.0),
        length=640.0, width=40.0, duration=20.0, interval=5.0,
        depth_at_zero=0.5, decay=0.001, discharge=0.5,
        manning=0.025, gravity=9.81,
    )
