"""Matched lowland hydrographs, terrain aggregation and fixed-geometry refinement."""

from pathlib import Path
import os
from time import perf_counter
import warp as wp
import numpy as np
from scipy.ndimage import label
from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import coordinates
from validation.fv2_local_inertia import build, read_state
from validation.fv2_terrain import blocks, prolong


def geometry(*, case, dx, length, width):
    x, y = coordinates(length=length, width=width, dx=dx)
    grade = 0.0005
    bed = -grade * x + 0.002 * np.cos(x / 13) * np.cos(y / 8)
    level = 0.035 + 0.03 * np.exp(-(((x + length / 4) / 12) ** 2))
    rain = np.zeros_like(x)
    rough = 0.04 + 0.005 * np.cos(y / 12)
    if case == "backwater":
        bed += 0.018 * np.exp(-((x / 14) ** 2))
        level = 0.040 + 0 * x
    elif case == "depression":
        bed -= 0.05 * np.exp(-((x - 12) ** 2 + y**2) / 130)
        level = 0.065 * np.exp(-((x + 15) ** 2 + y**2) / 180) - 0.015
    elif case == "overbank":
        bed += 0.10 * (1 - np.exp(-((y / 8) ** 4)))
        level = 0.075 + 0.11 * np.exp(-(((x + length / 4) / 15) ** 2))
    elif case == "embankment":
        bed += 0.07 * ((abs(x) < 1.8) & (abs(y - 8) > 1.5))
        level = np.where(x < 0, 0.09, -0.06)
    elif case == "lateral_exchange":
        bed += 0.07 * (1 - np.exp(-(((abs(y) - 12) / 5) ** 2))) * (1 - 0.7 * np.exp(-(((x - 13) / 4) ** 2)))
        level = 0.048 + 0.05 * np.exp(-((x + 25) ** 2 + (y + 12) ** 2) / 120)
    elif case in ("spatial_rain", "rain_loss"):
        level = bed
        rain = (50 + 45 * np.sin(x / 24) * np.cos(y / 11)) / 3.6e6
    elif case == "stage":
        level = 0.024 + 0 * x
    elif case != "wide_wave":
        raise ValueError(case)
    return dict(bed=bed, h=np.maximum(level - bed, 0.0), rain=rain, rough=rough)


def raster(*, native, ratio, subdivide):
    fields = {}
    for name, field in native.items():
        coarse = blocks(value=field, factor=ratio)
        fields[name] = prolong(value=coarse, factor=subdivide)
    return fields


def simulate(*, fields, case, dx, method, cap, interval, count, device, length, sections, gravity):
    h = fields["h"]
    initial = np.zeros((*h.shape, 3))
    initial[..., 0] = h
    d = dict(
        z=fields["bed"],
        initial=initial,
        rain=fields["rain"],
        n=fields["rough"],
        qx=np.zeros((h.shape[0], h.shape[1] + 1)),
        qy=np.zeros((h.shape[0] + 1, h.shape[1])),
    )
    s = build(d=d, method=method, dx=dx, cap=cap, device=device, gravity=gravity)
    if case == "rain_loss":
        s.set_source_rates(rain=d["rain"], infil=np.full_like(h, 20 / 3.6e6))
    if case in ("backwater", "stage"):
        types = np.zeros_like(d["qx"], dtype=np.int32)
        types[:, 0] = 1
        types[:, -1] = 3
        s.set_boundary_type_qx(boundary_type=types)
    indices = np.rint((np.asarray(sections) + length / 2) / dx).astype(int)
    volumes = [np.zeros(len(indices))]
    duration = 0.0
    states = [initial]
    stats = []
    capture = 0.0
    compilation = 0.0
    hydraulic = 0.0
    for i in range(count):
        if case in ("backwater", "stage"):
            stage = np.zeros_like(d["qx"])
            stage[:, -1] = (
                0.055 if case == "backwater" else 0.024 + 0.018 * np.sin(2 * np.pi * (i + 0.5) / count)
            )
            s.set_boundary_stage_qx(stage=stage)
        wp.synchronize_device(device=device)
        began = perf_counter()
        step = s.run_device_loop(duration=interval)
        wp.synchronize_device(device=device)
        hydraulic += perf_counter() - began
        capture += step["graph_capture_seconds"]
        compilation += step.get("compilation_seconds", 0.0)
        duration += step["loop_seconds"]
        stats.append(step["steps"])
        states.append(read_state(solver=s, li=method == "li"))
        key = "cumulative_net_qx_volume_m3" if method == "li" else "qx"
        volumes.append(s.get_cumulative_face_volumes()[key][:, indices].sum(axis=0))
    final = states[-1]
    ledger = s.cumulative_accounting()
    residual = (
        s.initial_volume
        + ledger["rain"]
        + ledger["boundary_inflow"]
        - ledger["boundary_outflow"]
        - ledger["infiltration"]
        - final[..., 0].sum() * dx**2
    )
    if not np.isfinite(states).all() or min(a[..., 0].min() for a in states) < 0.0:
        raise ArithmeticError(f"Invalid state {case} {method} {dx}")
    return dict(
        states=states,
        field=final,
        hydro=np.diff(volumes, axis=0) / interval,
        seconds=duration,
        hydraulic=hydraulic,
        capture=capture,
        compilation=compilation,
        steps=sum(stats),
        ledger=ledger,
        residual=float(residual),
    )


def difference(*, run, truth, dx, wet, interval):
    ratio = truth["field"].shape[0] // run["field"].shape[0]
    expected = blocks(value=truth["field"], factor=ratio)
    actual = run["field"]
    delta = actual[..., 0] - expected[..., 0]
    metrics = errors(result=actual, expected=expected, dx=dx, wet_depth=wet[1])
    metrics["surface_gradient_rmse"] = float(np.sqrt(np.mean(np.square(np.gradient(delta, dx)))))
    observed = run["hydro"]
    reference = truth["hydro"]
    metrics.update(
        hydro_rmse=np.sqrt(np.mean((observed - reference) ** 2, axis=0)).tolist(),
        peak_error=(observed.max(axis=0) - reference.max(axis=0)).tolist(),
        peak_time_difference_s=((observed.argmax(axis=0) - reference.argmax(axis=0)) * interval).tolist(),
        cumulative_error_m3=((observed - reference).sum(axis=0) * interval).tolist(),
        flow_reversals=(np.sum(np.diff(np.sign(observed), axis=0) != 0, axis=0)).tolist(),
    )
    all_h = np.stack([v[..., 0] for v in run["states"]])
    exact_h = np.stack([blocks(value=v[..., 0], factor=ratio) for v in truth["states"]])
    metrics["sampled_maximum_depth_rmse"] = float(
        np.sqrt(np.mean((all_h.max(axis=0) - exact_h.max(axis=0)) ** 2))
    )
    metrics["extent"] = {}
    for gate in wet:
        a, b = all_h > gate, exact_h > gate
        shared = a.any(axis=0) & b.any(axis=0)
        union = (a[-1] | b[-1]).sum()
        metrics["extent"][str(gate)] = dict(
            iou=float((a[-1] & b[-1]).sum() / union) if union else 1.0,
            components=int(label(a[-1])[1] - label(b[-1])[1]),
            arrival_mae_s=float(abs(a.argmax(axis=0) - b.argmax(axis=0))[shared].mean() * interval)
            if shared.any()
            else None,
            last_wet_mae_s=float(
                abs(a[::-1].argmax(axis=0) - b[::-1].argmax(axis=0))[shared].mean() * interval
            )
            if shared.any()
            else None,
        )
    return metrics


def compare(
    *,
    output,
    cases,
    methods,
    native_dx,
    fine,
    grids,
    fixed_dx,
    length,
    width,
    interval,
    count,
    wet,
    sections,
    device,
):
    rows = []
    for case in cases:
        native = geometry(case=case, dx=native_dx, length=length, width=width)
        native_truth = None
        for fixed in (False, True):

            def execute(*, dx, method, dt_factor):
                represented = fixed_dx if fixed else max(dx, native_dx)
                fields = raster(
                    native=native, ratio=round(represented / native_dx), subdivide=round(represented / dx)
                )
                return simulate(
                    fields=fields,
                    case=case,
                    method=method,
                    dx=dx,
                    cap=dx / dt_factor,
                    interval=interval,
                    count=count,
                    device=device,
                    length=length,
                    sections=sections,
                    gravity=9.80665,
                )

            refs = {}
            for dx in fine:
                for method in ("contact_hll", "adaptive"):
                    refs[dx, method] = execute(dx=dx, method=method, dt_factor=8)
            finest = fine[-1]
            truth = refs[finest, "adaptive"]
            redone = execute(dx=finest, method="adaptive", dt_factor=16)
            record = dict(
                case=case,
                fixed_terrain=fixed,
                reference_dx=finest,
                section_x_m=sections,
                interval_s=interval,
                reference_flux_m3_s=truth["hydro"].tolist(),
                reference_checks={},
                runs=[],
            )
            if fixed:
                record["representation_change_at_same_hydraulic_dx"] = difference(
                    run=truth, truth=native_truth, dx=finest, wet=wet, interval=interval
                )
                coarse = raster(
                    native=native, ratio=round(fixed_dx / native_dx), subdivide=round(fixed_dx / native_dx)
                )
                dz = coarse["bed"] - native["bed"]
                record["aggregation"] = dict(
                    native_dx=native_dx,
                    represented_dx=fixed_dx,
                    bed_rmse_m=float(np.mean(dz**2) ** 0.5),
                    maximum_lowering_m=float(-dz.min()),
                    maximum_raising_m=float(dz.max()),
                    storage_change_at_0p04m_m3=float(
                        native_dx**2
                        * (
                            np.maximum(0.04 - coarse["bed"], 0.0) - np.maximum(0.04 - native["bed"], 0.0)
                        ).sum()
                    ),
                )
            else:
                native_truth = truth
            for name, test, dx in (
                ("half_dt", redone, finest),
                ("formulation", refs[finest, "contact_hll"], finest),
                ("grid_fv1", refs[fine[0], "contact_hll"], fine[0]),
                ("grid_fv2", refs[fine[0], "adaptive"], fine[0]),
            ):
                record["reference_checks"][name] = difference(
                    run=test, truth=truth, dx=dx, wet=wet, interval=interval
                )
            for dx in (fixed_dx,) if fixed else grids:
                for method in methods:
                    run = execute(dx=dx, method=method, dt_factor=8)
                    record["runs"].append(
                        dict(
                            dx=dx,
                            method=method,
                            steps=run["steps"],
                            seconds=run["seconds"],
                            ledger=run["ledger"],
                            volume_residual_m3=run["residual"],
                            flux_m3_s=run["hydro"].tolist(),
                            errors=difference(run=run, truth=truth, dx=dx, wet=wet, interval=interval),
                        )
                    )
            rows.append(record)
            report_json(path=output, content=rows)
            print(case, "fixed", fixed, flush=True)


if __name__ == "__main__":
    refined = os.environ.get("FV2_FINE_REFERENCE") == "1"
    filename = "lowland_refined.json" if refined else "lowland.json"
    fine = (0.625, 0.3125) if refined else (1.25, 0.625)
    compare(
        output=Path("validation/fv2_review/regimes") / filename,
        cases=(
            "wide_wave",
            "backwater",
            "depression",
            "overbank",
            "embankment",
            "lateral_exchange",
            "spatial_rain",
            "rain_loss",
            "stage",
        ),
        methods=("contact_hll", "minmod", "adaptive", "li"),
        native_dx=0.625,
        fine=fine,
        grids=(10.0, 5.0, 2.5),
        fixed_dx=5.0,
        length=160.0,
        width=80.0,
        interval=10.0,
        count=12,
        wet=(0.001, 0.01, 0.05),
        sections=(-40.0, 0.0, 40.0),
        device="cuda:0",
    )
