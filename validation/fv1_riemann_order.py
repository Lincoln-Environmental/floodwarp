"""Isolate MUSCL/SSP-RK2 on the exact flat, constant-depth SWE shear subproblem."""

from pathlib import Path
from time import perf_counter

import numpy as np

from validation.fv1_riemann import report_json, rms


def exact_average(*, x, dx, speed, length, time):
    return .4 + .7 * np.sinc(dx / length) * np.sin(2. * np.pi * (x - speed * time) / length)


def spatial_rhs(*, momentum, dx, speed, use_muscl):
    # At h=1 and constant positive u, contact-HLL reduces to F_hv=u*(hv)_upwind.
    face = momentum
    if use_muscl:
        west = momentum - np.roll(momentum, shift=1)
        east = np.roll(momentum, shift=-1) - momentum
        slope = np.where(west * east > 0., np.sign(west) * np.minimum(np.abs(west), np.abs(east)), 0.)
        face = momentum + .5 * slope
    return speed / dx * (np.roll(face, shift=1) - face)


def evolve(*, momentum, steps, dt, dx, speed, second_order):
    for _ in range(steps):
        stage = momentum + dt * spatial_rhs(momentum=momentum, dx=dx, speed=speed, use_muscl=second_order)
        if second_order:
            momentum = .5 * (momentum + stage + dt * spatial_rhs(momentum=stage, dx=dx, speed=speed, use_muscl=True))
        else:
            momentum = stage
    return momentum


def run(*, output, length, speed, duration, dx_values, courant, repeats):
    rows = []
    for dx in dx_values:
        steps = int(np.ceil(duration * speed / (dx * courant)))
        dt = duration / steps
        x = (np.arange(round(length / dx)) + .5) * dx
        initial = exact_average(x=x, dx=dx, length=length, speed=speed, time=0.)
        expected = exact_average(x=x, dx=dx, length=length, speed=speed, time=duration)
        for second_order in (False, True):
            options = dict(momentum=initial, steps=steps, dt=dt, dx=dx, speed=speed, second_order=second_order)
            evolve(**options)
            samples = []
            for _ in range(repeats):
                began = perf_counter()
                result = evolve(**options)
                samples.append(perf_counter() - began)
            rows.append(dict(dx_m=dx, second_order=second_order, steps=steps, dt_s=dt,
                rmse_m2_s=rms(value=result - expected), max_abs_error_m2_s=float(np.abs(result - expected).max()),
                transverse_momentum_integral_residual=float((result - initial).sum() * dx),
                diffuse_amplitude_loss=float(expected.max() - expected.min() - result.max() + result.min()),
                median_numpy_seconds=float(np.median(samples)), timings_s=samples))
    for row in rows:
        matches = [earlier for earlier in rows if earlier['second_order'] == row['second_order'] and earlier['dx_m'] == 2. * row['dx_m']]
        row['pairwise_order'] = float(np.log2(matches[0]['rmse_m2_s'] / row['rmse_m2_s'])) if matches else None
    report_json(path=output, content=dict(length_m=length, speed_m_s=speed, time_s=duration, cfl=courant, rows=rows))


if __name__ == "__main__":
    output = Path('validation/fv1_riemann_review/order.json')
    length, speed, duration = 640., 1.5, 160.
    dx_values = (20., 10., 5., 2.5, 1.25)
    courant, repeats = .4, 15
    run(output=output, length=length, speed=speed, duration=duration, dx_values=dx_values, courant=courant, repeats=repeats)
