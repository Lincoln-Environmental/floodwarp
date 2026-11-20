"""Warmed, replicated dense/sparse FV1 and FV2 hydraulic cost at equal error."""

from pathlib import Path
from time import perf_counter
import json
import gc

import numpy as np
import warp as wp

from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_cases import analytical_domain, cell_averages, coordinates
from validation.fv2_accuracy import configured
from validation.fv2_native import aggregate, source_fields
from validation.fv2_terrain import blocks, open_boundaries
from validation.fv1_riemann_cases import Domain


def reset(*, solver, state):
    for name, values in state["arrays"].items():
        solver._fields[name].assign(values)
    solver.time = state["time"]
    solver.steps = state["steps"]


def timed(*, solver, duration, samples):
    wp.synchronize_device(device=solver.device)
    start = perf_counter()
    start_time = solver.time
    calls = []
    for t in np.linspace(duration / samples, duration, samples):
        calls.append(solver.run_device_loop(duration=start_time + t - solver.time))
    count = sum(call["steps"] for call in calls)
    stats = dict(
        compilation_seconds=sum(call["compilation_seconds"] for call in calls),
        steps=count,
        dt_min_s=min(call["dt_min_s"] for call in calls),
        dt_mean_s=duration / count,
        graph_capture_seconds=sum(call["graph_capture_seconds"] for call in calls),
        rejected_cfl_trials=sum(call.get("rejected_cfl_trials", 0) for call in calls),
    )
    wp.synchronize_device(device=solver.device)
    stats["full_hydraulic_s"] = perf_counter() - start
    return stats


def benchmark(*, domain, method, mode, device, repeats, exact, mask=None, rain=None, profile=False):
    s = configured(
        domain=domain, method=method, cap=1.0 if rain is None else 2.0, device=device, mode=mode, mask=mask
    )
    if rain is not None:
        open_boundaries(solver=s, active=s._active, north_only=True)
        s.set_rainfall_hyetograph(times_s=rain[:, 0], rates_m_s=rain[:, 1])
    initial = s.capture_state()
    warm = timed(solver=s, duration=domain.duration, samples=1 if rain is None else 6)
    gc.collect()
    samples = []
    for _ in range(repeats):
        reset(solver=s, state=initial)
        samples.append(timed(solver=s, duration=domain.duration, samples=1 if rain is None else 6))
    q = s.get_conservative_state()
    wall = float(np.median([r["full_hydraulic_s"] for r in samples]))
    extra = 0
    if method != "fv1":
        arrays = [s._start, s._stage, s._end, s._amounts, *s._slopes, *s._flux[0], *s._flux[1], *s._samples]
        for v in s._stage_series.values():
            arrays.extend(v)
        extra = sum(v.capacity for v in arrays)
    total = (
        sum(v.capacity for v in s._fields.values())
        + sum(v.capacity for v in (s.ctrl, s.condition, s._rain_rates, s._rain_times))
        + extra
        + s.execution_diagnostics()["scheduling_bytes"]
    )
    row = dict(
        dx=domain.dx,
        method=method,
        mode=mode,
        shape=list(domain.bed.shape),
        wall_s=wall,
        repeats=samples,
        module_preload_s=warm["compilation_seconds"],
        pool_used_bytes=int(wp.get_mempool_used_mem_current(device=device)),
        process_pool_high_water_bytes=int(wp.get_mempool_used_mem_high(device=device)),
        graph_capture_s=warm["graph_capture_seconds"],
        cell_steps_s=q.shape[0] * q.shape[1] * samples[0]["steps"] / wall,
        device_array_bytes=total,
        fv2_scratch_and_stage_forcing_bytes=extra,
        steps=samples[0]["steps"],
        min_dt=samples[0]["dt_min_s"],
        mean_dt=samples[0]["dt_mean_s"],
    )
    if exact is not None:
        row["errors"] = errors(result=q, expected=exact, wet_depth=0.01, dx=domain.dx)
    if profile:
        kernels = []
        for _ in range(3):
            reset(solver=s, state=initial)
            with wp.ScopedTimer(
                name="uncaptured FV2 kernels", cuda_filter=wp.TIMING_KERNEL, print=False
            ) as timer:
                st = s.run_device_loop(duration=min(domain.duration, 2.0), use_cuda_graph=False)
            groups = {}
            for event in timer.timing_results:
                key = event.name.split("_cuda")[0]
                groups[key] = groups.get(key, 0.0) + 1000 * event.elapsed / st["steps"]
            kernels.append(groups)
        row["kernel_us_per_step"] = kernels
    return row, q


def cost(*, output, spacings, methods, modes, device, repeats, site_dir):
    rows = []
    z, active, n, _, rain, _ = source_fields(
        site="orton", crop_dir=site_dir, purau=None, purau_window=None, duration=240.0
    )
    native = json.loads((output.parent / "orton_terrain.json").read_text())
    for case in ("stoker", "orton_crop"):
        for dx in spacings[case]:
            if case == "stoker":
                domain = analytical_domain(name=case, dx=dx, length=640.0, width=320.0, gravity=9.81)
                x, _ = coordinates(length=640.0, width=320.0, dx=dx)
                exact = cell_averages(name=case, x=x, dx=dx, time=domain.duration, gravity=9.81)
                mask = None
                forcing = None
            else:
                mask = blocks(value=active.astype(float), factor=dx) >= 0.5
                bed = aggregate(value=z, mask=active, factor=dx)
                rough = aggregate(value=n, mask=active, factor=dx)
                domain = Domain(
                    dx=dx, bed=bed, initial=np.zeros((*bed.shape, 3)), manning=rough, duration=240.0
                )
                exact = None
                forcing = rain
            for method in methods:
                for mode in modes:
                    row, q = benchmark(
                        domain=domain,
                        method=method,
                        mode=mode,
                        device=device,
                        repeats=repeats,
                        exact=exact,
                        mask=mask,
                        rain=forcing,
                        profile=(case == "stoker" and dx == 5.0),
                    )
                    if mode == "dense":
                        baseline = q
                    else:
                        np.testing.assert_array_equal(baseline, q)
                    if case == "orton_crop":
                        reference = next(
                            r
                            for r in native["rows"]
                            if r["hydraulic_dx"] == dx and r["terrain_dx"] == dx and r["method"] == method
                        )
                        np.testing.assert_allclose(
                            q[..., 0].max(),
                            reference["depth_max_m"]
                            if "depth_max_m" in reference
                            else reference["vs_fine"]["depth_max_m"],
                            atol=2.0e-13,
                            rtol=1.0e-13,
                        )
                        row["common_1m_error_m"] = reference["common_1m"]["depth_rmse_m"]
                    row["case"] = case
                    rows.append(row)
        report_json(path=output, content=rows)


if __name__ == "__main__":
    output = Path("validation/fv2_review/cost.json")
    spacings = dict(stoker=(20.0, 10.0, 5.0, 2.5), orton_crop=(20, 10, 5, 2, 1))
    cost(
        output=output,
        spacings=spacings,
        methods=("fv1", "mc"),
        modes=("dense", "sparse"),
        device="cuda:0",
        repeats=5,
        site_dir=Path("examples/purau"),
    )
