"""Full GPU solver verification against analytical and manufactured solutions."""

from pathlib import Path

import numpy as np
import warp as wp

from floodwarp.solver_factory import create_overland_solver
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.fv2_solver import FV2Solver, MUSCLEulerSolver, HancockSolver
from validation.fv1_riemann import analytical_metrics, cap_at_grid, report_json, trial
from validation.fv1_riemann_cases import analytical_domain


def configured(*, domain, method, cap, device, mode="dense", mask=None):
    bed = domain.bed
    options = dict(
        ny=bed.shape[0],
        nx=bed.shape[1],
        dx=domain.dx,
        dy=domain.dx,
        bed_elevation=bed,
        manning_n=domain.manning,
        active_mask=np.ones_like(bed, bool) if mask is None else mask,
        maximum_dt=cap,
        device=device,
        execution_mode=mode,
    )
    if method == "default":
        # Match the analytical timestep policy while exercising omitted numerical settings.
        solver = create_overland_solver(explicit_maximum_dt=options.pop("maximum_dt"),
                                       explicit_minimum_dt=0., **options)
    elif method in ("fv1", "roe", "hllc", "low_froude_hll"):
        flux = "contact_hll" if method == "fv1" else method
        solver = FullSWESolver(flux_scheme=flux, **options)
    elif method.startswith(("hancock_", "euler_")):
        integrator, limiter = method.split("_", 1)
        factory = HancockSolver if integrator == "hancock" else MUSCLEulerSolver
        flux = "low_froude_hll" if limiter == "low_froude" else "contact_hll"
        solver = factory(limiter="mc" if limiter == "low_froude" else limiter, flux_scheme=flux, **options)
    else:
        solver = FV2Solver(limiter=method, **options)
    solver.set_water_depth(depth=domain.initial[..., 0])
    wet = domain.initial[..., 0] > solver.dry_depth
    solver.set_momentum(
        hu=np.where(wet, domain.initial[..., 1], 0.0), hv=np.where(wet, domain.initial[..., 2], 0.0)
    )
    solver.set_source_rates(rain=domain.rain)
    for axis, stage in enumerate((domain.stage_x, domain.stage_y)):
        if stage is None:
            continue
        shape = list(bed.shape)
        shape[1 - axis] += 1
        codes = np.zeros(shape, np.int32)
        levels = np.zeros(shape)
        if axis == 0:
            codes[:, [0, -1]] = 3
            levels[:, [0, -1]] = stage
            solver.set_boundary_type_qx(boundary_type=codes)
            solver.set_boundary_stage_qx(stage=levels)
        else:
            codes[[0, -1], :] = 3
            levels[[0, -1], :] = stage
            solver.set_boundary_type_qy(boundary_type=codes)
            solver.set_boundary_stage_qy(stage=levels)
    return solver


@wp.kernel
def manufactured_source(
    out: wp.array(dtype=wp.vec3d),
    ctrl: wp.array(dtype=wp.float64),
    nx: int,
    dx: wp.float64,
    length: wp.float64,
    g: wp.float64,
    bed_amplitude: wp.float64,
    stage: int,
):
    i = wp.tid()
    instant = wp.where(stage == 2, wp.float64(.5), wp.float64(wp.max(stage, 0)))
    t = ctrl[0] + instant * ctrl[2]
    freq = wp.float64(3.141592653589793) / length
    x = (wp.float64(i % nx) + wp.float64(0.5)) * dx * freq
    y = (wp.float64(i / nx) + wp.float64(0.5)) * dx * freq
    sx = wp.sin(x)
    sy = wp.sin(y)
    p = sx * sx * sx * sx * sy * sy * sy * sy
    px = wp.float64(4.0) * freq * sx * sx * sx * wp.cos(x) * sy * sy * sy * sy
    py = wp.float64(4.0) * freq * sy * sy * sy * wp.cos(y) * sx * sx * sx * sx
    t2 = wp.float64(2.0) * t
    a = wp.float64(0.125) * wp.sin(t2 + wp.float64(0.5))
    b = wp.float64(0.25) * wp.cos(t2 + wp.float64(0.25))
    c = wp.float64(0.1875) * wp.sin(t2 + wp.float64(0.75))
    h = wp.float64(1.0) + a * p
    # d(phi^2/h)/d(phi) for both advective momentum fluxes.
    deriv = p * (wp.float64(2.0) + a * p) / (h * h)
    source = wp.vec3d(
        wp.float64(0.25) * wp.cos(t2 + wp.float64(0.5)) * p + b * px + c * py,
        -wp.float64(0.5) * wp.sin(t2 + wp.float64(0.25)) * p + deriv * b * (b * px + c * py) + g * h * a * px,
        wp.float64(0.375) * wp.cos(t2 + wp.float64(0.75)) * p
        + deriv * c * (b * px + c * py)
        + g * h * a * py,
    )
    # Prescribed force compensates the physical -g h grad(z) in the SWE.
    two = wp.float64(2.0)
    source[1] -= g * h * bed_amplitude * two * freq * wp.sin(two * x) * wp.cos(two * y)
    source[2] -= g * h * bed_amplitude * two * freq * wp.cos(two * x) * wp.sin(two * y)
    weight = wp.where(stage == -1,wp.float64(.5),wp.float64(1.))
    out[i] += weight * ctrl[2] * source


class Manufactured(FV2Solver):
    def __init__(self, *, bed_amplitude, **options):
        self.bed_amplitude = bed_amplitude
        super().__init__(**options)

    def _stage_forcing(self, *, dest, stage):
        self._launch(
            manufactured_source,
            self.nx * self.ny,
            [dest, self.ctrl, self.nx, self.dx, self.dx * self.nx, self.gravity, self.bed_amplitude, stage],
        )


class ManufacturedHancock(Manufactured):
    method_name = HancockSolver.method_name
    integration = HancockSolver.integration


class ManufacturedEuler(Manufactured):
    method_name = MUSCLEulerSolver.method_name
    integration = MUSCLEulerSolver.integration


def manufactured_average(*, n, t):
    x = (np.arange(n) + 0.5) / n
    line = (3.0 - 4.0 * np.cos(2 * np.pi * x) * np.sinc(1 / n) + np.cos(4 * np.pi * x) * np.sinc(2 / n)) / 8.0
    p = line[:, None] * line[None, :]
    return np.stack(
        (
            1 + 0.125 * np.sin(2 * t + 0.5) * p,
            0.25 * np.cos(2 * t + 0.25) * p,
            0.1875 * np.sin(2 * t + 0.75) * p,
        ),
        axis=-1,
    )


def manufactured_run(*, n, dt, length, duration, device, bed_amplitude=0.0, limiter="mc", integrator="fv2"):
    coord = (np.arange(n) + 0.5) / n
    cos_mean = np.cos(2 * np.pi * coord) * np.sinc(1 / n)
    factory = {"fv2": Manufactured, "hancock": ManufacturedHancock, "euler": ManufacturedEuler}[integrator]
    s = factory(
        limiter=limiter,
        bed_amplitude=bed_amplitude,
        ny=n,
        nx=n,
        dx=length / n,
        dy=length / n,
        bed_elevation=bed_amplitude * np.outer(cos_mean, cos_mean),
        manning_n=0.0,
        active_mask=np.ones((n, n), bool),
        maximum_dt=dt,
        device=device,
    )
    q = manufactured_average(n=n, t=0.0)
    s.set_water_depth(depth=q[..., 0])
    s.set_momentum(hu=q[..., 1], hv=q[..., 2])
    stats = s.run_device_loop(duration=duration)
    return s.get_conservative_state(), stats


def convergence(*, grids, temporal_grid, timesteps, length, duration, device, bed_amplitude=0.0, limiter="mc", integrator="fv2"):
    rows = []
    for n in grids:
        dt = duration / (4 * n)
        q, stats = manufactured_run(
            n=n, dt=dt, length=length, duration=duration, device=device, bed_amplitude=bed_amplitude, limiter=limiter, integrator=integrator
        )
        error = q - manufactured_average(n=n, t=duration)
        rows.append(
            dict(
                study="space", n=n, dt=dt, stats=stats, rmse=np.sqrt(np.mean(error**2, axis=(0, 1))).tolist()
            )
        )
    ref, _ = manufactured_run(
        n=temporal_grid,
        dt=timesteps[-1] / 16,
        length=length,
        duration=duration,
        device=device,
        bed_amplitude=bed_amplitude, limiter=limiter, integrator=integrator,
    )
    for dt in timesteps:
        q, stats = manufactured_run(
            n=temporal_grid,
            dt=dt,
            length=length,
            duration=duration,
            device=device,
            bed_amplitude=bed_amplitude, limiter=limiter, integrator=integrator,
        )
        rows.append(
            dict(
                study="time",
                n=temporal_grid,
                dt=dt,
                stats=stats,
                rmse=np.sqrt(np.mean((q - ref) ** 2, axis=(0, 1))).tolist(),
            )
        )
    return rows


def analytical(*, output, names, methods, spacings, device, length, width):
    rows = []
    for name in names:
        for dx in spacings:
            domain = analytical_domain(name=name, dx=dx, length=length, width=width, gravity=9.81)
            for method in methods:
                solver = configured(
                    domain=domain, method=method, cap=cap_at_grid(dx=dx, scale=0.015625), device=device
                )
                result = trial(solver=solver, duration=domain.duration, device=device)
                metrics = analytical_metrics(
                    domain=domain,
                    name=name,
                    states=result["states"],
                    times=result["times"],
                    gravity=solver.gravity,
                    wet_depth=0.01,
                )
                if name == "thacker":
                    x = (np.arange(solver.nx) + 0.5) * dx - length / 2
                    for i, state in enumerate(result["states"]):
                        metrics["centroid_" + str(i) + "_m"] = float(
                            np.sum(x * state[..., 0]) / np.sum(state[..., 0])
                        )
                rows.append(
                    dict(
                        case=name,
                        dx=dx,
                        method=method,
                        balance=result["balance"],
                        stats=result["stats"],
                        **metrics,
                    )
                )
        report_json(path=output, content=rows)
        print(name, flush=True)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    device = "cuda:0"
    grids = (16, 32, 64, 128)
    time_grid = 16
    timesteps = (0.015625, 0.0078125, 0.00390625, 0.001953125)
    manufactured = convergence(
        grids=grids, temporal_grid=time_grid, timesteps=timesteps, length=8.0, duration=0.5, device=device
    )
    report_json(path=directory / "manufactured.json", content=manufactured)
    names = ("lake", "ritter", "stoker", "thacker", "shear", "jump", "jump_offset", "manufactured")
    methods = ("fv1", "minmod", "mc")
    spacings = (20.0, 10.0, 5.0, 2.5)
    analytical(
        output=directory / "analytical.json",
        names=names,
        methods=methods,
        spacings=spacings,
        device=device,
        length=640.0,
        width=40.0,
    )
