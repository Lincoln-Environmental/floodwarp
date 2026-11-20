"""Warmed CUDA comparisons from the same wet Orton Bradley state at storm hour 25."""

from pathlib import Path
import gc
import json

import numpy as np
import warp as wp

from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_cost import reset, timed
from validation.fv2_orton_matrix import geometry, padded, replay_solver
from validation.fv2_terrain import blocks


def profile(*, solver, state, duration):
    reset(solver=solver, state=state)
    with wp.ScopedTimer(name="Orton kernels", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
        stats = solver.run_device_loop(duration=duration, use_cuda_graph=False)
    result = {}
    for event in timer.timing_results:
        name = event.name.split("_cuda")[0]
        result[name] = result.get(name, 0.0) + 1000 * event.elapsed / stats["steps"]
    return result


def record(*, state, factor, method, device, duration, repeats, kernel_interval, mode="dense"):
    data, info = geometry(state=state, factor=factor)
    s = replay_solver(data=data, state=state, factor=factor, method=method, device=device)
    if mode != "dense":
        s.configure_execution(mode=mode)
    start = s.capture_state()
    warm = timed(solver=s, duration=duration, samples=1)
    expected = s.get_conservative_state()
    trials = []
    for _ in range(repeats):
        reset(solver=s, state=start)
        trials.append(timed(solver=s, duration=duration, samples=1))
    np.testing.assert_array_equal(s.get_conservative_state(), expected)
    kernels = [profile(solver=s, state=start, duration=kernel_interval) for _ in range(repeats)]
    elapsed = float(np.median([x["full_hydraulic_s"] for x in trials]))
    row = dict(
        dx=s.dx,
        method=method,
        mode=mode,
        shape=s._shape,
        geometry=info,
        start_s=state["time"],
        duration_s=duration,
        steps=trials[0]["steps"],
        mean_dt_s=trials[0]["dt_mean_s"],
        min_dt_s=trials[0]["dt_min_s"],
        median_wall_s=elapsed,
        replicates=trials,
        kernel_us_per_step=kernels,
        module_preload_s=warm["compilation_seconds"],
        graph_capture_s=warm["graph_capture_seconds"],
        pool_used_bytes=wp.get_mempool_used_mem_current(device=device),
        process_pool_high_water_bytes=wp.get_mempool_used_mem_high(device=device),
        cell_steps_s=s.nx * s.ny * trials[0]["steps"] / elapsed,
    )
    return row, expected


def review(*, checkpoint, factors, methods, device, duration, repeats, kernel_interval, output):
    base = checkpoint_solver(scheme="contact_hll", device="cpu", path=checkpoint).capture_state()
    rows = []
    states = {}
    for factor in factors:
        for method in methods:
            gc.collect()
            row, q = record(
                state=base,
                factor=factor,
                method=method,
                device=device,
                duration=duration,
                repeats=repeats,
                kernel_interval=kernel_interval,
            )
            states[factor, method] = q
            rows.append(row)
            print(f"Timed {factor * 5} m {method}, {row['median_wall_s']:.3f} s", flush=True)
    for r in rows:
        factor = round(r["dx"] / 5)
        q = states[factor, r["method"]]
        reference = blocks(value=padded(a=states[1, "contact_hll"], multiple=factor), factor=factor)
        r["vs_5m_contact"] = errors(result=q, expected=reference, wet_depth=0.01, dx=r["dx"])
    report_json(path=output, content=rows)


if __name__ == "__main__":
    output = Path("validation/fv2_review/hirds48/wet_cost.json")
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    factors = (4, 2, 1)
    methods = ("contact_hll", "roe", "hllc", "fv2_mc")
    device = "cuda:0"
    duration = 12.0
    repeats = 5
    kernel_interval = 1.0
    review(
        checkpoint=checkpoint,
        factors=factors,
        methods=methods,
        device=device,
        duration=duration,
        repeats=repeats,
        kernel_interval=kernel_interval,
        output=output,
    )
