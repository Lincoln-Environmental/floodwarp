"""Check masked-stage compatibility and replay a storm-peak checkpoint interval."""

from pathlib import Path
import json
import hashlib
import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_orton_matrix import dry_initial, geometry, replay_solver, release_initial


def stage_replay(*, state, dx, work, hours, duration, device):
    data, info = geometry(state=state, factor=dx // 5)
    y, x = np.where(data["by"] == 3)
    clearance = data["bed"][y - 1, x] - data["sy"][y, x]
    if clearance.min() < 0:
        raise ValueError("A wet masked exterior requires rerunning the complete event")
    solver = replay_solver(data=data, state=state, factor=dx // 5, method="fv2_mc", device=device)
    directory = work / f"{dx}m_fv2_mc"
    solver.load_checkpoint(path=directory / f"hour-{hours[0]:02d}.npz")
    steps = solver.steps
    count = round((hours[1] - hours[0]) * 3600 / duration)
    for _ in range(count):
        solver.run_device_loop(duration=duration)
    final = solver.capture_state()
    keys = (
        "q",
        "maximum",
        "soil",
        "rain_sum",
        "flux_x",
        "flux_y",
        "volume_x",
        "volume_y",
        "inflow",
        "outflow",
        "dt_histogram",
        "fv2_cfl_predictor",
    )
    differences = {}
    with np.load(directory / f"hour-{hours[1]:02d}.npz") as archive:
        meta = json.loads(str(archive["metadata"]))
        if solver.steps != meta["steps"] or solver.time != meta["time"]:
            raise ArithmeticError("Checkpoint time or accepted timestep count changed")
        for k in keys:
            actual = final["arrays"][k]
            expected = archive[k]
            np.testing.assert_array_equal(actual, expected, err_msg=k)
            differences[k] = float(np.max(np.abs(actual - expected)))
    return dict(
        dx_m=dx,
        start_end_hours=hours,
        steps=solver.steps - steps,
        minimum_stage_bed_clearance_m=float(clearance.min()),
        geometry=info,
        max_absolute_differences=differences,
    )


def input_parity(*, state, published, grids):
    checks = []
    for dx in grids:
        archive, info = geometry(state=state, factor=dx // 5)
        ordinary, other = geometry(state=published, factor=dx // 5)
        if info != other:
            raise ValueError("The public release and local checkpoint give different stage openings")
        digests = {}
        for name, values in archive.items():
            np.testing.assert_array_equal(values, ordinary[name], err_msg=name)
            digests[name] = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        y, x = np.where(ordinary["by"] == 3)
        checks.append(
            dict(
                dx_m=dx,
                physical_field_sha256=digests,
                stage_openings=info,
                minimum_bed_above_coastal_stage_m=float(
                    np.min(ordinary["bed"][y - 1, x] - ordinary["sy"][y, x])
                ),
            )
        )
    np.testing.assert_array_equal(state["rain_rates"], published["rain_rates"])
    np.testing.assert_array_equal(state["rain_times"], published["rain_times"])
    return checks


def audit(*, checkpoint, release, work, output, grids, hours, duration, device):
    state = dry_initial(checkpoint=checkpoint) if checkpoint.exists() else release_initial(directory=release)
    if release.is_dir() and checkpoint.exists():
        report_json(
            path=output.parent / "public_input_parity.json",
            content=input_parity(
                state=state, published=release_initial(directory=release), grids=(5, 10, 20)
            ),
        )
    rows = []
    for dx in grids:
        rows.append(
            stage_replay(state=state, dx=dx, work=work, hours=hours, duration=duration, device=device)
        )
        print(f"{dx} m masked-stage peak replay matched {rows[-1]['steps']} steps bitwise", flush=True)
    report_json(path=output, content=rows)


if __name__ == "__main__":
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    release = Path("outputs/release/orton_bradley_v0.1.0")
    work = Path("outputs/fv2_hirds48_v1")
    output = Path("validation/fv2_review/hirds48/stage_replay.json")
    grids = (20, 10, 5)
    hours = (24, 26)
    duration = 60.0
    device = "cuda:0"
    audit(
        checkpoint=checkpoint,
        release=release,
        work=work,
        output=output,
        grids=grids,
        hours=hours,
        duration=duration,
        device=device,
    )
