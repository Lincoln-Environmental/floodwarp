"""Replay both FP64 surface storage reductions when their final rounded totals differ."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def totals(*, state, receipt):
    direct = json.loads(state["_direct_manifest"].item())
    hashes = direct["sha256"]
    if (runner.digest(hashes) != receipt["checkpoint_array_set_sha256"]
            or runner.digest(direct["identity"]) != receipt["identity_sha256"]):
        raise ValueError("Storage evidence differs from the hydraulic checkpoint identity")
    values = {key: state[key] for key in ("q", "active", "metadata", "control")}
    if any(array_sha(value) != hashes[key] for key, value in values.items()):
        raise ValueError("Changed checkpoint field in storage reduction")
    cfg = json.loads(values["metadata"].item())["configuration"]
    if (cfg["dx"] != receipt["dx"] or cfg["dy"] != receipt["dx"]
            or values["control"].ravel()[0] != receipt["end_s"]):
        raise ValueError("Storage evidence uses a different grid or time")
    h = values["q"][:, 0].reshape(cfg["ny"], cfg["nx"])
    active = values["active"].reshape(h.shape) != 0
    if not np.all(np.isfinite(h)) or np.any(h < 0) or np.any(h[~active] != 0):
        raise ValueError("Invalid depth or stored water outside the active footprint")
    return float(np.sum(h) * cfg["dx"]**2), float(np.sum(h[active]) * cfg["dx"] * cfg["dy"])


def verify(*, directory, receipt, series):
    if receipt["storage_m3"] == series["storage_m3"]:
        return 0.
    ident, end = receipt["id"], receipt["end_s"]
    expected = receipt["storage_m3"], series["storage_m3"]
    file = directory / "checkpoint_storage" / ident / f"{end}.npz"
    if not file.is_file():
        root = directory.parents[2] / ".fidelity-restarts"
        source = root / ident / file.name
        if not source.is_file():
            source = root / "retained_native_fields" / ident / file.name
        with np.load(source, allow_pickle=False) as state:
            if totals(state=state, receipt=receipt) != expected:
                raise ValueError("Checkpoint and diagnostic storage totals do not reproduce exactly")
            evidence = {key: state[key] for key in ("_direct_manifest", "metadata", "q", "active", "control")}
        file.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=file.parent) as folder:
            candidate = Path(folder) / file.name
            np.savez_compressed(candidate, **evidence)
            candidate.replace(file)
    with np.load(file, allow_pickle=False) as state:
        if totals(state=state, receipt=receipt) != expected:
            raise ValueError("Archived state does not reproduce both storage totals exactly")
    return expected[0] - expected[1]
