"""Authenticate native prefixes and the matched horizon used in each published comparison."""

import gzip
import json

from research.floodwarp_methods_paper import runner


def prefix_evidence(*, folder):
    files = {
        "result_sha256": runner.checksum(path=folder / "result.json"),
        "spatial_sha256": runner.checksum(path=folder / "spatial.npz"),
    }
    for path in sorted(folder.glob("attempt_*/complete.json")):
        receipt = json.loads(path.read_text())
        if all(receipt.get(key) == value for key, value in files.items()):
            return dict(attempt=path.parent.name, **files)
    raise ValueError(f"No completed attempt authenticates the native fields and result at {folder}")


def horizon_evidence(*, root, ident, parent, end_s):
    folder = root / ident
    path = folder / "horizons" / f"{end_s}.json.gz"
    with gzip.open(path, "rt") as stream:
        frozen = json.load(stream)
    if (frozen["id"], frozen["parent_id"], frozen["end_s"]) != (ident, parent, end_s):
        raise ValueError(f"Frozen native horizon has the wrong identity at {path}")
    cursor = json.loads((folder / "series.json").read_text())
    if cursor["id"] != ident or cursor["parent_id"] != parent or cursor["end_s"] < end_s:
        raise ValueError(f"Native restart cursor disagrees with the matched horizon at {path}")
    if cursor["end_s"] == end_s and cursor != frozen:
        raise ValueError(f"Native restart cursor differs from its frozen measurements at {path}")
    return dict(horizon_sha256=runner.checksum(path=path))


def observations(*, directory, records, groups):
    rows = []
    for ident, item in sorted(records.items()):
        if item["path"].parent.parent != directory / "native_matrix":
            continue
        rows.append(dict(id=ident, case=item["s"]["case"], evidence="completed initial window",
                         end_s=item["d"]["frames"][-1]["t_s"], **prefix_evidence(folder=item["path"])))
    for (cohort, case, _), items in sorted(groups.items()):
        if not cohort.startswith("complete-native-"):
            continue
        for item in items:
            parent = item.get("parent_id", item["id"])
            if parent == item["id"]:
                continue
            end_s = item["d"]["frames"][-1]["t_s"]
            if end_s != int(end_s):
                raise ValueError("Expected a whole-second native recession endpoint")
            evidence = horizon_evidence(root=directory / "extended_runs", ident=item["id"],
                                        parent=parent, end_s=int(end_s))
            rows.append(dict(id=item["id"], parent_id=parent, case=case, evidence="matched recession",
                             end_s=end_s, **evidence))
    return rows
