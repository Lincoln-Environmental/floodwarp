"""Measure operator changes against the audited solver at the same accepted times."""

from pathlib import Path
import json
import os
import numpy as np
from validation.fv1_riemann import report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_orton_matrix import geometry, replay_solver
from validation.fv2_terrain import terrain
from validation.fv2_accuracy import configured


def fields(*, solver, duration):
    run = solver.run_device_loop(duration=duration)
    q = solver.get_conservative_state()
    faces = solver.get_cumulative_face_volumes()
    return dict(
        h=q[..., 0],
        hu=q[..., 1],
        hv=q[..., 2],
        maximum=solver.get_max_water_depth(),
        soil=solver.cumulative_infiltration_m(),
        qx_volume=faces["qx"],
        qy_volume=faces["qy"],
        steps=run["steps"],
        rejections=run["rejected_cfl_trials"],
        dt_min=run["dt_min_s"],
        dt_mean=run["dt_mean_s"],
    )


def measure(*, baseline, checkpoint, directory, output, device):
    state = checkpoint_solver(scheme="contact_hll", path=checkpoint, device="cpu").capture_state()
    rows = []
    directory.mkdir(parents=True, exist_ok=True)
    for case in ("orton", "steep", "rainfall", "irregular", "barrier"):
        if case == "orton":
            data, _ = geometry(state=state, factor=1)
            solver = replay_solver(data=data, state=state, factor=1, method="fv2_mc", device=device)
            duration = 12.0
        else:
            d = terrain(name=case, length=320.0, width=160.0, dx=2.5)
            solver = configured(domain=d, method="mc", cap=d.dx / 32, device=device)
            duration = d.duration
        result = fields(solver=solver, duration=duration)
        path = directory / (case + ".npz")
        if baseline:
            np.savez_compressed(file=path, **result)
        else:
            with np.load(path) as ref:
                metrics = {}
                for key, values in result.items():
                    diff = np.array(values) - ref[key]
                    metrics[key] = dict(
                        reference_max=float(np.max(np.abs(ref[key]))),
                        max_abs=float(np.max(np.abs(diff))),
                        rmse=float(np.sqrt(np.mean(diff**2))),
                    )
                rows.append(dict(case=case, metrics=metrics))
    if not baseline:
        report_json(path=output, content=rows)


if __name__ == "__main__":
    measure(
        baseline=os.environ.get("FV2_PARITY_ROLE") == "baseline",
        checkpoint=Path(
            json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())[
                "checkpoint"
            ]
        ),
        directory=Path("outputs/fv2_regime_parity"),
        output=Path("validation/fv2_review/regimes/parity.json"),
        device="cuda:0",
    )
