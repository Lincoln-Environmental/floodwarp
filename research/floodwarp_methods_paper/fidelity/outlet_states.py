"""Verify workstation checkpoints and restore missing incised fields from their physical start."""

import json
import os
from pathlib import Path
import sys

import numpy as np

from floodwarp.model import validate_checkpoint_archive
from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity import outlet_storms


def state_receipt(*, path, ident, end):
    with np.load(file=path, allow_pickle=False) as archive:
        manifest = json.loads(archive["_direct_manifest"].item())
        metadata = json.loads(archive["metadata"].item())
    if metadata["time"] != end or not validate_checkpoint_archive(path=path, identity=manifest["identity"]):
        raise ValueError("Incised checkpoint failed its physical time or integrity check")
    return dict(primary=ident, end_s=end, steps=metadata["steps"],
                model_identity=manifest["identity"], field_sha256=manifest["sha256"])


def check_replay(*, original, replay):
    ignored = {"hydraulic_s", "graph_capture_s", "compilation_s"}
    expected = json.loads(original.read_text())
    measured = json.loads(replay.read_text())
    if (expected["start_s"], expected["end_s"]) != (measured["start_s"], measured["end_s"]):
        raise ValueError("Incised checkpoint reconstruction used a different time window")
    for first, second in zip(expected["frames"], measured["frames"], strict=True):
        if ({k: v for k, v in first.items() if k not in ignored}
                != {k: v for k, v in second.items() if k not in ignored}):
            raise ValueError("Incised checkpoint reconstruction differs in accepted hydraulic measurements")


def restore(*, directory, base, identity, missing, cache, control):
    contract = dict(primary=base.name, initial=identity, restoration=runner.checksum(Path(__file__)))
    recovery = directory / "outlet_reconstruction" / runner.digest(contract)[:24]
    outlet_storms.trajectory(directory=directory, control=control, dx=identity["dx"],
                            factor=identity["cfl_factor"], extension=identity["extension_m"],
                            end=max(missing), settings=identity["settings"], stopping=identity["stopping"],
                            run_dir=recovery, identity=contract)
    for end in missing:
        check_replay(original=base/f"{end}.json", replay=recovery/f"{end}.json")
        saved = cache / recovery.name / f"{end}.npz"
        proof = state_receipt(path=saved, ident=base.name, end=end)
        if proof != json.loads((base / f"state_{end}.json").read_text()):
            raise ValueError("Reconstructed incised fields differ from the original checkpoint")
        destination = cache / base.name / saved.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(src=saved, dst=destination)


def run(*, directory, dry):
    controls = outlet_storms.archived(directory=directory)
    root = directory.parents[2]
    source = runner.source_identity(root=root, study=directory.parent, stage="long-storms")
    driver = runner.checksum(Path(outlet_storms.__file__))
    cache = root / ".fidelity-restarts/outlet_storms"
    counts = dict(present=0, missing=0, receipts=0)
    for path in sorted((directory/"outlet_storms").glob("*/inputs.json")):
        base = path.parent
        identity = json.loads(path.read_text())["identity"]
        if (identity["source"], identity["numerical"]) != (driver, source):
            continue
        if base.name != runner.digest(identity)[:24]:
            raise ValueError("Primary incised checkpoint contract changed")
        missing = []
        for window in sorted(base.glob("[0-9]*.json")):
            end = int(window.stem)
            receipt = base / f"state_{end}.json"
            saved = cache / base.name / f"{end}.npz"
            counts["receipts"] += int(receipt.exists())
            if saved.exists():
                counts["present"] += 1
                if not dry:
                    proof = state_receipt(path=saved, ident=base.name, end=end)
                    if receipt.exists():
                        if proof != json.loads(receipt.read_text()):
                            raise ValueError("Incised checkpoint differs from its published field hashes")
                    else:
                        runner.write_json(path=receipt, data=proof)
            else:
                counts["missing"] += 1
                if not dry and not receipt.is_file():
                    raise ValueError(f"Missing original field hashes and checkpoint at {base.name}/{end}")
                missing.append(end)
        if missing:
            print(base.name, "requires hydraulic replay from initial state to", max(missing)/3600, "hours", flush=True)
            if not dry:
                restore(directory=directory, base=base, identity=identity, missing=missing, cache=cache,
                        control=controls[identity["method"], identity["dx"]])
    print("Incised checkpoint inventory at entry", counts, flush=True)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    arguments = set(sys.argv[1:])
    if arguments.difference({"--dry-run", "--resume"}):
        raise ValueError(f"Unknown outlet state options {arguments}")
    run(directory=directory, dry="--dry-run" in arguments)
