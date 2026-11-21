"""Create a checkpoint with the prior release of dev and verify it on the current solver."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import floodwarp
from floodwarp.inputs import terrain_from_array
from floodwarp.model import advance, create_model, load_checkpoint, save_checkpoint
import numpy as np


def prepare(*, mode, device):
    y, x = np.indices((13, 79))
    terrain = terrain_from_array(
        elevation=-0.002 * x,
        dx=2.0,
        dy=2.0,
        row_order="south_to_north",
        vertical_datum="local",
    )
    return create_model(
        terrain=terrain,
        solver_method="hancock",
        execution_mode=mode,
        initial_depth=np.where(x < 16, 0.4, 0.0),
        rainfall_series=([0.0, 6.0, 12.0, 24.0], [0.0, 2e-5, 0.0, 0.0]),
        infiltration=5e-7,
        manning_n=0.03,
        device=device,
    )


def replay(*, directory, make, historical, deadline, checkpoint_s, device):
    implementation = Path(floodwarp.__file__).resolve()
    identity = subprocess.check_output(
        ["git", "-C", str(implementation.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    if identity.startswith(historical) != make:
        raise ValueError(
            "Historical fixture must be generated with the archived implementation"
        )
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "historical_rain_front.npz"
    expected = directory / "historical_continuation.npz"
    if make:
        model = prepare(mode="dense", device=device)
        advance(model=model, end_time=checkpoint_s)
        save_checkpoint(model=model, path=file)
        advance(model=model, end_time=deadline)
        np.savez_compressed(expected, **model.solver.capture_state()["arrays"])
        return
    results = []
    with np.load(expected) as peer:
        for mode in ("dense", "sparse", "dynamic", "tiles"):
            model = prepare(mode=mode, device=device)
            load_checkpoint(model=model, path=file)
            advance(model=model, end_time=deadline)
            arrays = model.solver.capture_state()["arrays"]
            if set(arrays) != set(peer.files):
                raise ValueError("Checkpoint array set changed")
            for key in peer:
                np.testing.assert_array_equal(peer[key], arrays[key], err_msg=key)
            results.append(
                dict(
                    mode=mode,
                    arrays=len(arrays),
                    time_s=model.solver.time,
                    steps=model.solver.steps,
                )
            )
    data = dict(
        old_sha=historical,
        tested_sha=identity,
        checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
        replays=results,
    )
    (directory / "results.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    directory = Path(
        os.environ.get(
            "FLOODWARP_CHECKPOINT_AUDIT",
            "validation/hancock_long_storm_review/checkpoint",
        )
    )
    create_fixture = os.environ.get("FLOODWARP_HISTORICAL_FIXTURE") == "1"
    historical = "018032761146a915bd6089f858a97500715e3758"
    checkpoint_s = 6.0
    deadline = 24.0
    device = "cpu"
    replay(
        directory=directory,
        make=create_fixture,
        historical=historical,
        deadline=deadline,
        checkpoint_s=checkpoint_s,
        device=device,
    )
