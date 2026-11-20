"""Restart an explicit physical-state experiment from the archived false-step snapshot."""

import hashlib
import json
from pathlib import Path

import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_orton_matrix import dry_initial, geometry, replay_solver


def sample(*, solver, cell, base_time, initial, wet):
    q = solver.get_conservative_state()
    velocity = np.divide(
        np.linalg.norm(q[..., 1:], axis=-1),
        q[..., 0],
        out=np.zeros_like(q[..., 0]),
        where=q[..., 0] > solver.dry_depth,
    )
    y, x = cell
    total = solver.cumulative_accounting()
    return dict(
        seconds=solver.time - base_time,
        tagged_depth_m=float(q[y, x, 0]),
        tagged_speed_m_s=float(velocity[y, x]),
        max_speed_m_s=float(velocity.max()),
        wet_speed_p95_m_s=float(np.percentile(velocity[q[..., 0] > wet], 95)),
        outlet_m3=total["boundary_outflow"] - initial["boundary_outflow"],
        residual_m3=total["residual"],
    )


def release(*, inputs, snapshot, cell, targets, device, directory, wet):
    state = dry_initial(checkpoint=inputs)
    data, _ = geometry(state=state, factor=1)
    with np.load(snapshot) as archive:
        meta = json.loads(str(archive["metadata"]))
        if meta["configuration"].get("reconstruction") != "muscl_eta_bed_velocity_v1":
            raise ValueError("This transfer experiment requires the superseded archive")
        state["time"] = meta["time"]
        state["steps"] = meta["steps"]
        state["initial_volume"] = meta["initial_volume"]
        for key in ("q", "rain_sum", "soil", "inflow", "outflow"):
            data[key] = archive[key].reshape(data[key].shape)
    solver = replay_solver(data=data, state=state, factor=1, method="fv2_mc", device=device)
    np.testing.assert_array_equal(solver.get_conservative_state(), data["q"])
    initial = solver.cumulative_accounting()
    rows = []
    for t in targets:
        if t > 0:
            solver.run_device_loop(duration=state["time"] + t - solver.time)
        rows.append(sample(solver=solver, cell=cell, base_time=state["time"], initial=initial, wet=wet))
    revision = solver.numerical_configuration()["reconstruction"]
    with snapshot.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    report_json(
        path=directory / ("bed_release_" + revision + ".json"),
        content=dict(
            reconstruction=revision,
            physical_state_sha256=identity,
            tagged_south_first_cell=cell,
            start_s=state["time"],
            samples=rows,
        ),
    )


if __name__ == "__main__":
    inputs = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    directory = Path("validation/fv2_review")
    snapshot = Path("outputs/fv2_hirds48_v1/5m_fv2_mc/hour-24.npz")
    release(
        inputs=inputs,
        snapshot=snapshot,
        cell=(363, 379),
        targets=(0.0, 1.0, 10.0, 30.0, 60.0),
        device="cuda:0",
        directory=directory,
        wet=0.01,
    )
