"""Distinguish reconstruction from time and source changes on identical wet Orton states."""

from pathlib import Path
import json

import numpy as np

from floodwarp import fv2_kernels
from floodwarp.fv2_solver import FV2Solver
from validation.fv1_riemann import report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_orton_matrix import geometry, replay_solver
from validation.sparse_fv1_benchmark import select_checkpoints


class ConstantSpaceRK2(FV2Solver):
    """Experimental control using RK2/source splitting with piecewise-constant cell traces."""

    def _launch_grid(self, kernel, count, inputs, *, layout):
        if kernel is not fv2_kernels.slopes:
            super()._launch_grid(kernel=kernel, count=count, inputs=inputs, layout=layout)


def snapshot(*, state, data, device, method, cap, duration):
    fixed = dict(state, configuration=dict(state["configuration"], maximum_dt=cap))
    solver = replay_solver(
        data=data,
        state=fixed,
        factor=1,
        method="contact_hll" if method == "fv1" else "fv2_mc",
        device=device,
        fv2_class=ConstantSpaceRK2 if method == "constant_rk2" else FV2Solver,
    )
    start = solver.cumulative_accounting()
    stats = solver.run_device_loop(duration=duration)
    end = solver.cumulative_accounting()
    if method == "constant_rk2":
        if any(np.any(a.numpy()) for a in solver._slopes):
            raise ValueError("Piecewise-constant control developed a reconstructed slope")
    field = solver.get_conservative_state()
    if abs(end["residual"]) > 1e-5 or not np.isfinite(field).all() or field[..., 0].min() < 0:
        raise ArithmeticError("Failed physical admissibility or water balance")
    row = dict(
        steps=stats["steps"],
        min_dt_s=stats["dt_min_s"],
        mean_dt_s=stats["dt_mean_s"],
        ledger_change_m3={k: end[k] - start[k] for k in end},
        min_depth_m=float(field[..., 0].min()),
    )
    return field, row


def investigate(*, directory, hours, caps, duration, device, output):
    records = []
    for hour, path in zip(hours, select_checkpoints(directory=directory, hours=hours), strict=True):
        state = checkpoint_solver(scheme="contact_hll", device="cpu", path=path).capture_state()
        data, _ = geometry(state=state, factor=1)
        previous = {}
        active = data["active"]
        for cap in caps:
            fields = {}
            runs = {}
            for method in ("fv1", "constant_rk2", "mc_rk2"):
                fields[method], runs[method] = snapshot(
                    state=state, data=data, device=device, method=method, cap=cap, duration=duration
                )
            changes = {}
            for a, b in (("constant_rk2", "fv1"), ("mc_rk2", "constant_rk2"), ("mc_rk2", "fv1")):
                diff = (fields[a] - fields[b])[active]
                changes[a + "_minus_" + b] = dict(
                    component_rmse=np.sqrt(np.mean(diff**2, axis=0)).tolist(),
                    outlet_volume_difference_m3=runs[a]["ledger_change_m3"]["boundary_outflow"]
                    - runs[b]["ledger_change_m3"]["boundary_outflow"],
                )
            records.append(
                dict(
                    start_hour=hour,
                    source=str(path),
                    duration_s=duration,
                    cap_s=cap,
                    steps_identical=len({r["steps"] for r in runs.values()}) == 1,
                    runs=runs,
                    contrasts=changes,
                    cap_halving_rmse={
                        m: np.sqrt(np.mean((fields[m] - previous[m])[active] ** 2, axis=0)).tolist()
                        for m in previous
                    },
                )
            )
            report_json(path=output, content=records)
            previous = fields
            print(f"Hour {hour}, cap {cap:g} s, three matched operators complete", flush=True)


if __name__ == "__main__":
    original = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    investigate(
        directory=original.parent,
        hours=(16, 24),
        caps=(1 / 64, 1 / 128),
        duration=32.0,
        device="cuda:0",
        output=Path("validation/fv2_review/operators.json"),
    )
