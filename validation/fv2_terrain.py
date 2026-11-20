"""Fix terrain independently of numerical refinement and compare cell averages."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from scipy.ndimage import label

from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import Domain, coordinates, synthetic_domain
from validation.fv2_accuracy import configured


def blocks(*, value, factor):
    ny, nx = value.shape[:2]
    return value.reshape(ny // factor, factor, nx // factor, factor, *value.shape[2:]).mean(axis=(1, 3))


def prolong(*, value, factor):
    if np.ndim(value) < 2:
        return value
    return np.repeat(np.repeat(value, factor, axis=0), factor, axis=1)


def refined(*, domain, factor):
    return replace(
        domain,
        dx=domain.dx / factor,
        bed=prolong(value=domain.bed, factor=factor),
        initial=prolong(value=domain.initial, factor=factor),
        rain=prolong(value=domain.rain, factor=factor),
        manning=prolong(value=domain.manning, factor=factor),
    )


def open_boundaries(*, solver, active, north_only):
    ny, nx = active.shape
    for axis, shape in ((0, (ny + 1, nx)), (1, (ny, nx + 1))):
        p = [(0, 0)] * 2
        p[axis] = (1, 1)
        sides = np.pad(active, p)
        a = np.take(sides, range(shape[axis]), axis=axis)
        b = np.take(sides, range(1, shape[axis] + 1), axis=axis)
        selected = a != b
        if north_only:
            selected = (a & ~b) if axis == 0 else np.zeros_like(a)
        types = 2 * selected.astype(np.int32)
        if axis == 0:
            solver.set_boundary_type_qy(boundary_type=types)
        else:
            solver.set_boundary_type_qx(boundary_type=types)


def evaluate(*, solver, duration, samples, wet, diagnostic=False):
    if diagnostic:
        probe = solver.track_reconstruction(depth_edges=(0.001, 0.01, 0.1, 1.0))
    stats = []
    hydro = []
    states = []
    previous = solver.cumulative_accounting()["boundary_outflow"]
    start_time = solver.time
    for target in start_time + np.linspace(duration / samples, duration, samples):
        wp.synchronize_device(device=solver.device)
        began = perf_counter()
        step = solver.run_device_loop(duration=float(target - solver.time))
        wp.synchronize_device(device=solver.device)
        step["wall_s"] = perf_counter() - began
        stats.append(step)
        exported = solver.cumulative_accounting()["boundary_outflow"]
        hydro.append((exported - previous) * samples / duration)
        previous = exported
        states.append(solver.get_conservative_state())
    maximum = solver.get_max_water_depth()
    final = states[-1]
    ledger = solver.cumulative_accounting()
    counts = label(final[..., 0] > wet)[1]
    row = dict(
        stats=stats,
        ledger=ledger,
        export_m3=sum(hydro) * duration / samples,
        peak_m3_s=max(hydro),
        peak_time_s=(np.argmax(hydro) + 0.5) * duration / samples,
        hydrograph=hydro,
        wet_components=int(counts),
        minimum_depth_m=float(final[..., 0].min()),
        maximum_depth_m=float(maximum.max()),
        wet_area_m2=float(np.sum(final[..., 0] > wet) * solver.dx * solver.dy),
    )
    if diagnostic:
        row["reconstruction"] = probe.summary()
    return row, states, maximum


def compare(*, result, reference, wet, dx, duration):
    row, states, maximum = result
    ref, snapshots, envelope = reference
    ratio = snapshots[-1].shape[0] // states[-1].shape[0]
    exact = blocks(value=snapshots[-1], factor=ratio)
    values = errors(result=states[-1], expected=exact, wet_depth=wet, dx=dx)
    actual = states[-1][..., 0] > wet
    expected = exact[..., 0] > wet
    union = np.count_nonzero(expected | actual)
    values["wet_iou"] = np.count_nonzero(actual & expected) / union if union else 1.0
    values["peak_envelope_rmse_m"] = float(
        np.sqrt(np.mean((maximum - blocks(value=envelope, factor=ratio)) ** 2))
    )
    for name in ("export_m3", "peak_m3_s", "peak_time_s"):
        values[name + "_difference"] = float(row[name] - ref[name])
    values["wet_component_difference"] = int(label(actual)[1] - label(expected)[1])
    observed = np.asarray([s[..., 0] > wet for s in states])
    comparison = np.asarray([blocks(value=s[..., 0], factor=ratio) > wet for s in snapshots])
    both = observed.any(axis=0) & comparison.any(axis=0)
    values["sampled_arrival_mae_s"] = (
        float(
            np.abs(observed.argmax(axis=0) - comparison.argmax(axis=0))[both].mean() * duration / len(states)
        )
        if both.any()
        else None
    )
    return values


def run_domain(
    *,
    domain,
    method,
    cap,
    device,
    samples,
    wet,
    mask=None,
    rain_series=None,
    north_only=False,
    infiltration=None,
    diagnostic=False,
):
    s = configured(domain=domain, method=method, cap=cap, device=device, mask=mask)
    active = s._active
    open_boundaries(solver=s, active=active, north_only=north_only)
    if rain_series is not None:
        s.set_rainfall_hyetograph(times_s=rain_series[:, 0], rates_m_s=rain_series[:, 1])
    if infiltration is not None:
        s.set_exponential_infiltration(**infiltration)
    return evaluate(solver=s, duration=domain.duration, samples=samples, wet=wet, diagnostic=diagnostic)


def terrain(*, name, dx, length, width):
    if name not in ("barrier", "oblique_channel"):
        return synthetic_domain(name=name, dx=dx, length=length, width=width)
    x, y = coordinates(length=length, width=width, dx=dx)
    if name == "barrier":
        bed = np.where((np.abs(x) < 6.0) & (np.abs(y - 20.0) > 4.0), 0.75, 0.0) - 0.12 * np.exp(
            -((x + 30.0) ** 2 + y**2) / 200.0
        )
        h = np.maximum(0.0, np.where(x < -6.0, 0.55, 0.0) - bed)
    else:
        bed = -0.003 * x + 0.55 * (1.0 - np.exp(-(((y - 0.3 * x) / 5.0) ** 2)))
        h = np.maximum(0.0, 0.24 - bed)
    u, v = (0.0, 0.0) if name == "barrier" else (1.2, 0.36)
    return Domain(dx=dx, bed=bed, initial=np.stack((h, h * u, h * v), axis=-1), duration=16.0, manning=0.025)


def synthetic(*, output, names, spacings, fine, device, samples, wet, length, width):
    rows = []
    for name in names:
        runs = {}
        for dx in (*spacings, *fine):
            domain = terrain(name=name, dx=dx, length=length, width=width)
            for method in ("fv1", "mc"):
                cap = dx / 32.0
                runs[dx, method] = run_domain(
                    domain=domain, method=method, cap=cap, device=device, samples=samples, wet=wet
                )
        for (dx, method), result in runs.items():
            row = dict(case=name, dx=dx, method=method, **result[0])
            row["fine_comparisons"] = {
                m: compare(
                    result=result, reference=runs[fine[-1], m], wet=wet, dx=dx, duration=domain.duration
                )
                for m in ("fv1", "mc")
            }
            rows.append(row)
        domain = terrain(name=name, dx=fine[-1], length=length, width=width)
        small_dt = run_domain(
            domain=domain, method="mc", cap=fine[-1] / 64, device=device, samples=samples, wet=wet
        )
        rows[-1]["finest_dt_check"] = compare(
            result=runs[fine[-1], "mc"], reference=small_dt, wet=wet, dx=fine[-1], duration=domain.duration
        )
        # Identical represented 5 m terrain, forcing and initial averages on nested subcells.
        fixed = terrain(name=name, dx=5.0, length=length, width=width)
        fine_fixed = {}
        for factor in (2, 4):
            for method in ("mc", "fv1"):
                finer = refined(domain=fixed, factor=factor)
                fine_fixed[factor, method] = run_domain(
                    domain=finer, method=method, cap=finer.dx / 32.0, device=device, samples=samples, wet=wet
                )
        for method in ("fv1", "mc"):
            result = runs[5.0, method]
            rows.append(
                dict(
                    case=name,
                    dx=5.0,
                    method=method,
                    terrain_fixed_5m=True,
                    refined={
                        m: compare(
                            result=result,
                            reference=fine_fixed[4, m],
                            wet=wet,
                            dx=5.0,
                            duration=fixed.duration,
                        )
                        for m in ("fv1", "mc")
                    },
                    reference_gap=compare(
                        result=fine_fixed[2, method],
                        reference=fine_fixed[4, method],
                        wet=wet,
                        dx=2.5,
                        duration=fixed.duration,
                    ),
                )
            )
        report_json(path=output, content=rows)
        print(name, flush=True)


if __name__ == "__main__":
    output = Path("validation/fv2_review/synthetic.json")
    names = ("plane", "irregular", "steep", "expansion", "rainfall", "barrier", "oblique_channel")
    spacings = (20.0, 10.0, 5.0, 2.5)
    fine = (1.25, 0.625)
    synthetic(
        output=output,
        names=names,
        spacings=spacings,
        fine=fine,
        device="cuda:0",
        samples=4,
        wet=0.01,
        length=320.0,
        width=160.0,
    )
