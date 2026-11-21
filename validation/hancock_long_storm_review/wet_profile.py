"""Separate uncaptured kernel attribution at peak forcing and during recession."""

import json
from pathlib import Path

import warp as wp

from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


def sample(*, model, steps):
    s = model.solver
    state = s.capture_state()
    with wp.ScopedTimer(
        name="accepted hydraulic steps", cuda_filter=wp.TIMING_KERNEL, print=False
    ) as timer:
        for _ in range(steps):
            s.step()
    kernels = {}
    for event in timer.timing_results:
        name = event.name.split("kernel ")[-1].split("_cuda")[0].rsplit("_", 1)[0]
        kernels[name] = kernels.get(name, 0.0) + 1000.0 * event.elapsed / steps
    s.restore_state(state=state)
    return kernels


def profile(*, review, config, methods, times, dimensions, steps, device):
    rows = []
    for storm in ("spatial", "moderate"):
        for method, mode in methods:
            m, _ = storm_model(
                config=config,
                landform="plain",
                storm=storm,
                method=method,
                mode=mode,
                dx=10,
                terrain_dx=10,
                dimensions=dimensions,
                duration=28800.0,
                device=device,
            )
            for t in times:
                stats = advance(model=m, end_time=t)
                rows.append(
                    dict(
                        storm=storm,
                        method=method,
                        mode=mode,
                        t_s=t,
                        steps=steps,
                        mean_previous_dt=stats["dt_mean_s"],
                        schedule=m.solver.execution_diagnostics(),
                        us_per_accepted_step=sample(model=m, steps=steps),
                    )
                )
    (review / "large_phase_kernels.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    methods = (("fv1", "sparse"), ("hancock", "dense"), ("hancock", "tiles"))
    times = (3 * 3600.0, 7 * 3600.0)
    profile(
        review=review,
        config=config,
        methods=methods,
        times=times,
        dimensions=(5000, 5000),
        steps=16,
        device="cuda:0",
    )
