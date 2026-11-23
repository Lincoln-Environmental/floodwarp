"""Preserve immutable measured horizon receipts independently of the restart cursor."""

import gzip
import json
from research.floodwarp_methods_paper.fidelity import recession


def freeze(*, folder, data):
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (str(data["end_s"]) + ".json.gz")
    if dest.exists():
        with gzip.open(dest, "rt") as saved:
            existing = json.load(saved)
        recession.verify_prefix(old=existing, new=data["frames"])
        return
    raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    tmp = dest.with_suffix(".pending")
    tmp.write_bytes(gzip.compress(raw, compresslevel=3, mtime=0))
    tmp.replace(dest)


def preserve(*, original):
    def publish(*, path, payload):
        if path.name == "series.json":
            freeze(folder=path.parent / "horizons", data=payload)
        original(path=path, payload=payload)

    return publish


def read(*, root, ident, end):
    folder = root / ident
    data = json.loads((folder / "series.json").read_text())
    if data["end_s"] == end:
        return data
    path = folder / "horizons" / (str(end) + ".json.gz")
    with gzip.open(path, "rt") as saved:
        return json.load(saved)
