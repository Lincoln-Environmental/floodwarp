"""Reuse the verified primary windows collected before the LI recession diagnostic repair."""

import json

from research.floodwarp_methods_paper import runner


def previous_prefix(*, directory, output, definition):
    proof = json.loads((directory / "native_prefix_compatibility.json").read_text())
    if definition["spec"]["case"] != proof["case"]:
        return None
    sources = definition["sources"].copy()
    for name, revisions in proof["sources"].items():
        if sources.get(name) != revisions["repaired"]:
            return None
        sources[name] = revisions["original"]
    prior = dict(definition, sources=sources)
    ident = runner.digest(prior)[:24]
    if ident not in proof["prefix_ids"]:
        return None
    folder = output / ident
    path = folder / "result.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record["inputs"] != prior:
        raise ValueError("Historical launch contract differs from the selected native input")
    receipts = [json.loads(p.read_text()) for p in folder.glob("attempt_*/complete.json")]
    files = (runner.checksum(path), runner.checksum(folder / "spatial.npz"))
    if not any((r["result_sha256"], r["spatial_sha256"]) == files for r in receipts):
        raise ValueError("Historical native prefix fields or completed hydraulic measurements changed")
    return dict(id=ident, data=record["data"])
