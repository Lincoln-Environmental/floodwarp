"""Check the FV2 fixed-mask work list against dense execution at wet and dry Orton states."""

from pathlib import Path
import json

import numpy as np
from validation.fv1_riemann import report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_cost import reset, timed
from validation.fv2_orton_matrix import geometry, replay_solver
from validation.sparse_fv1_benchmark import select_checkpoints


def benchmark(*, checkpoint, device, factors, hours, duration, repeats, output):
    rows = []
    for hour, path in zip(hours, select_checkpoints(directory=checkpoint.parent, hours=hours), strict=True):
        original = checkpoint_solver(device="cpu", scheme="contact_hll", path=path).capture_state()
        for factor in factors:
            data, _ = geometry(state=original, factor=factor)
            s = replay_solver(data=data, state=original, factor=factor, method="fv2_mc", device=device)
            initial = s.capture_state()
            times = {"dense": [], "sparse": []}
            configs = {}
            reference = None
            for index in range(repeats):
                order = ("dense", "sparse") if index % 2 == 0 else ("sparse", "dense")
                for mode in order:
                    s.configure_execution(mode=mode)
                    reset(solver=s, state=initial)
                    timed(solver=s, duration=duration, samples=1)
                    reset(solver=s, state=initial)
                    stats = timed(solver=s, duration=duration, samples=1)
                    times[mode].append(stats)
                    values = [
                        s._fields[name].numpy()
                        for name in (
                            "q",
                            "soil",
                            "rain_sum",
                            "maximum",
                            "flux_x",
                            "flux_y",
                            "volume_x",
                            "volume_y",
                            "outflow",
                            "inflow",
                            "dt_histogram",
                            "fv2_cfl_predictor",
                        )
                    ]
                    if reference is None:
                        reference = values
                    for actual, expected in zip(values, reference, strict=True):
                        np.testing.assert_array_equal(actual, expected)
                    configs[mode] = s.execution_diagnostics()
            rows.append(
                dict(
                    hour=hour,
                    dx_m=5 * factor,
                    duration_s=duration,
                    configuration=configs,
                    trials=times,
                    dense_median_s=float(np.median([t["full_hydraulic_s"] for t in times["dense"]])),
                    sparse_median_s=float(np.median([t["full_hydraulic_s"] for t in times["sparse"]])),
                )
            )
            print(f"{5 * factor} m, hour {hour}, compact-index parity verified", flush=True)
            report_json(path=output, content=rows)


if __name__ == "__main__":
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    output = Path("validation/fv2_review/scheduling.json")
    benchmark(
        checkpoint=checkpoint,
        device="cuda:0",
        factors=(4, 2, 1),
        hours=(1, 25),
        duration=8.0,
        repeats=5,
        output=output,
    )
