"""Compare the recorded Hancock face CFL bound with cell wave speeds in a saved recession."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def capture(*, checkpoint, destination):
    if destination.is_file():
        return
    with np.load(checkpoint, allow_pickle=False) as state:
        fields = {key: state[key] for key in
                  ("metadata", "_direct_manifest", "q", "active", "control", "fv2_cfl_predictor")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        pending = Path(temporary) / destination.name
        np.savez_compressed(pending, **fields)
        pending.replace(destination)


def compare(*, evidence, endpoint):
    with np.load(endpoint, allow_pickle=False) as saved:
        receipt = json.loads(saved["receipt"].item())
    with np.load(evidence, allow_pickle=False) as state:
        manifest = json.loads(state["_direct_manifest"].item())
        hashes = manifest["sha256"]
        if (runner.digest(hashes) != receipt["checkpoint_array_set_sha256"]
                or runner.digest(manifest["identity"]) != receipt["identity_sha256"]):
            raise ValueError("CFL measurements differ from the archived hydraulic checkpoint")
        for key in ("q", "active", "control", "fv2_cfl_predictor", "metadata"):
            if hashes[key] != array_sha(state[key]):
                raise ValueError(f"Changed CFL checkpoint field {key}")
        meta = json.loads(state["metadata"].item())
        cfg = meta["configuration"]
        ctrl = state["control"].ravel()
        if (cfg["method"] != "fv1_hancock" or len(ctrl) != 18 or cfg["dx"] != cfg["dy"]
                or ctrl[0] != receipt["end_s"] or cfg["dx"] != receipt["dx"]):
            raise ValueError("Expected a square-grid legacy Hancock CFL control at the recorded time")
        q = state["q"]
        h = q[:, 0]
        u = np.divide(q[:, 1:], h[:, None], out=np.zeros_like(q[:, 1:]),
                      where=h[:, None] > cfg["dry_depth"])
        wave = np.sqrt(cfg["gravity"] * h)
        signal = wave + np.max(np.abs(u), axis=1)
        active_signal = np.where(state["active"].ravel() != 0, signal, -np.inf)
        cell = int(np.argmax(active_signal))
        if not np.isfinite(active_signal[cell]) or active_signal[cell] <= 0:
            raise ValueError("No positive characteristic speed in the saved state")
        row, col = divmod(cell, cfg["nx"])
        return dict(case=receipt["case"], run_id=receipt["id"], end_s=receipt["end_s"],
                    dx_m=cfg["dx"], row=row, column=col, domain_x_m=(col + .5) * cfg["dx"],
                    domain_y_m=(row + .5) * cfg["dy"], depth_m=float(h[cell]),
                    axis_velocity_ms=float(np.max(np.abs(u[cell]))), gravity_wave_ms=float(wave[cell]),
                    cell_signal_bound_s=.5 * cfg["dx"] * cfg["cfl"] / float(signal[cell]),
                    recorded_face_bound_s=float(ctrl[17]), next_predictor_s=float(state["fv2_cfl_predictor"][0]),
                    state_sha256=hashes["q"], evidence_sha256=runner.checksum(path=evidence))


def publish(*, directory, evidence):
    endpoint = directory / "endpoint_fields" / evidence.parent.name / evidence.name
    observation = compare(evidence=evidence, endpoint=endpoint)
    destination = evidence.with_suffix(".json")
    runner.write_json(path=destination, data=observation)
    return observation


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    ident = "42545c16666c1c94109b354a"
    filename = "72000.npz"
    evidence = directory / "wave_clock" / ident / filename
    checkpoint = directory.parents[2] / ".fidelity-restarts" / ident / filename
    capture(checkpoint=checkpoint, destination=evidence)
    print(publish(directory=directory, evidence=evidence))
