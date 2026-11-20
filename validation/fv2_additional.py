"""Independent spatial convergence, Thacker phase and shallow-velocity reconstruction checks."""

from pathlib import Path

import numpy as np

from validation.fv1_riemann import report_json
from validation.fv1_riemann_cases import analytical_domain, cell_averages, coordinates
from validation.fv2_accuracy import configured, manufactured_average, manufactured_run
from floodwarp.fv2_reference import limited, traces


def fixed_time_space(*, grids, dt, length, duration, device, bed_amplitude=0.0):
    rows = []
    for n in grids:
        reference = manufactured_average(n=n, t=duration)
        solutions = []
        for cap in (dt, dt / 2):
            q, stats = manufactured_run(
                n=n, dt=cap, length=length, duration=duration, device=device, bed_amplitude=bed_amplitude
            )
            solutions.append(q)
        r = np.sqrt(np.mean((solutions[-1] - reference) ** 2, axis=(0, 1)))
        rows.append(
            dict(
                n=n,
                dt_s=dt / 2,
                steps=stats["steps"],
                rmse=r.tolist(),
                temporal_change_over_error=(
                    np.sqrt(np.mean((solutions[0] - solutions[1]) ** 2, axis=(0, 1))) / r
                ).tolist(),
            )
        )
    return rows


def basin_cycles(*, spacings, methods, cycles, samples_per_cycle, gravity, length, width, device):
    results = []
    alpha = 0.0001
    omega = np.sqrt(2 * gravity * alpha)
    period = 2 * np.pi / omega
    for dx in spacings:
        d = analytical_domain(name="thacker", dx=dx, length=length, width=width, gravity=gravity)
        x, _ = coordinates(length=length, width=width, dx=dx)
        times = np.arange(1, cycles * samples_per_cycle + 1) * period / samples_per_cycle
        exact = cell_averages(name="thacker", x=x, dx=dx, time=times[-1], gravity=gravity)
        for method in methods:
            s = configured(domain=d, method=method, cap=dx / 32, device=device)
            series = []
            worst_min = 0.0
            for t in times:
                s.run_device_loop(duration=t - s.time)
                q = s.get_conservative_state()
                depth = q[..., 0]
                series.append(np.sum(x * depth) / depth.sum())
                worst_min = min(worst_min, depth.min())
            basis = np.column_stack((np.cos(omega * times), np.sin(omega * times), np.ones_like(times)))
            cosine, sine, bias = np.linalg.lstsq(basis, series, rcond=None)[0]
            results.append(
                dict(
                    dx=dx,
                    method=method,
                    samples_per_cycle=samples_per_cycle,
                    cycles=cycles,
                    period_s=period,
                    phase_lag_s=float(np.arctan2(sine, cosine) / omega),
                    mean_shift_m=float(bias),
                    fitted_amplitude_m=float(np.hypot(cosine, sine)),
                    exact_amplitude_m=48.0,
                    depth_rmse_m=float(np.sqrt(np.mean((q[..., 0] - exact[..., 0]) ** 2))),
                    momentum_rmse_m2_s=np.sqrt(
                        np.mean((q[..., 1:] - exact[..., 1:]) ** 2, axis=(0, 1))
                    ).tolist(),
                    min_depth_m=float(worst_min),
                    residual_m3=s.cumulative_accounting()["residual"],
                )
            )
    return results


def shallow_traces(*, seed, count, dry, limiter):
    rng = np.random.default_rng(seed=seed)
    bed = rng.uniform(-0.2, 0.2, (count, 3))
    h = np.exp(rng.uniform(np.log(1.01 * dry), np.log(0.2), (count, 3)))
    # A spatially constant advecting velocity must survive arbitrary bounded wet depth gradients.
    u = rng.uniform(-5.0, 5.0, (count, 1))
    q = np.stack((h, h * u, 0.3 * h * u), axis=-1)
    left, right, _, _ = traces(q=q, z=bed, dry=dry, axis=1, limiter=limiter)
    delta = limited(left=q[:, 1, 1] - q[:, 0, 1], right=q[:, 2, 1] - q[:, 1, 1], limiter=limiter)
    dz = np.diff(bed, axis=1)
    deta = np.diff(h, axis=1) + dz
    bed_slope = limited(left=dz[:, 0], right=dz[:, 1], limiter=limiter)
    bound = np.min(np.abs(dz), axis=1)
    factor = np.divide(bound, abs(bed_slope), out=np.ones_like(bound), where=abs(bed_slope) > bound)
    depth_slope = factor * (limited(left=deta[:, 0], right=deta[:, 1], limiter=limiter) - bed_slope)
    delta[np.abs(depth_slope) >= 2 * h[:, 1]] = 0.0
    reference = u[:, 0]
    records = {}
    for name, sign, states in (("left", -1, left), ("right", 1, right)):
        face = states[:, 1]
        wet = face[:, 0] > dry
        v = face[wet, 1] / face[wet, 0]
        alternative = (q[wet, 1, 1] + sign * 0.5 * delta[wet]) / face[wet, 0]
        records[name] = dict(
            wet_faces=int(wet.sum()),
            velocity_trace_max_error_m_s=float(np.max(np.abs(v - reference[wet]))),
            independent_momentum_trace_p99_error_m_s=float(
                np.percentile(np.abs(alternative - reference[wet]), 99)
            ),
            independent_momentum_trace_max_error_m_s=float(np.max(np.abs(alternative - reference[wet]))),
        )
    return records


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    grid_sizes = (16, 32, 64, 128)
    spacings = (20.0, 10.0, 5.0, 2.5)
    methods = ("fv1", "minmod", "mc")
    device = "cuda:0"
    spatial = fixed_time_space(grids=grid_sizes, dt=0.0009765625, length=8.0, duration=0.5, device=device)
    bed_amplitude = 0.12
    curved = fixed_time_space(
        grids=grid_sizes,
        dt=0.0009765625,
        length=8.0,
        duration=0.5,
        device=device,
        bed_amplitude=bed_amplitude,
    )
    from validation.fv2_accuracy import convergence

    curved_time = convergence(
        grids=(),
        temporal_grid=16,
        timesteps=(1 / 64, 1 / 128, 1 / 256, 1 / 512),
        length=8.0,
        duration=0.5,
        device=device,
        bed_amplitude=bed_amplitude,
    )
    thacker = basin_cycles(
        spacings=spacings,
        methods=methods,
        cycles=2,
        samples_per_cycle=32,
        gravity=9.81,
        length=640.0,
        width=40.0,
        device=device,
    )
    shallow = shallow_traces(seed=94216, count=40000, dry=0.0001, limiter="mc")
    report_json(
        path=directory / "additional.json",
        content=dict(
            fixed_time_spatial=spatial,
            curved_bed_spatial=curved,
            curved_bed_temporal=curved_time,
            bed_amplitude_m=bed_amplitude,
            thacker=thacker,
            shallow_velocity=shallow,
        ),
    )
