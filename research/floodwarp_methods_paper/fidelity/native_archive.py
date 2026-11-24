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


def native_grid_differences(*, directory, records):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import native_map_endpoints

    measurements = {}
    receipts = sorted((directory / "native_grid_maps").glob("*/receipt.json"))
    if not receipts:
        return []
    implementation = {name: runner.checksum(path=directory / name)
                      for name in ("native_map_comparison.py", "final_depth.py", "native_map_endpoints.py")}
    for receipt in receipts:
        archive = json.loads(receipt.read_text())
        contract = archive["identity"]
        if contract["implementation"] != implementation:
            continue
        observations = contract["observations"]
        if not observations.keys() <= records.keys():
            continue
        ident = receipt.parent.name
        file = receipt.parent / "differences.csv"
        if runner.digest(contract)[:24] != ident or runner.checksum(path=file) != archive["table_sha256"]:
            raise ValueError("Changed native grid field comparison")
        endpoints = {parent: raw["endpoint"] for parent, raw in observations.items() if "endpoint" in raw}
        if endpoints:
            live = native_map_endpoints.observations(directory=directory, parents={p: records[p] for p in observations},
                                                     end_s=round(contract["end_h"] * 3600))
            if live != endpoints:
                raise ValueError("Archived map measurement disagrees with the original endpoint fingerprints")
        for key, raw in observations.items():
            item = records[key]
            if (raw["result_sha256"] != runner.checksum(path=item["path"] / "result.json")
                    or raw["fields"] != item["d"]["hashes"]):
                raise ValueError("Original native-grid fields disagree with the comparison identity")
        rows = metrics.csv_read(file=file)
        if ({r["id"] for r in rows} != observations.keys() - {contract["normal_rk2"]}
                or len(rows) != len(observations) - 1):
            raise ValueError("Incomplete native-grid field comparison")
        for row in rows:
            key = (row["id"], row["comparator_id"])
            if key[0] not in observations or key[1] != contract["normal_rk2"]:
                raise ValueError("Field comparison has different participants from its receipt")
            if float(row["end_h"]) != contract["end_h"]:
                raise ValueError("Native maps measured at a different time from their receipt")
            key = (*key, contract["end_h"])
            if key in measurements and measurements[key] != row:
                raise ValueError("Contradictory map comparisons for the same forecast pair")
            measurements[key] = row
    return [measurements[key] for key in sorted(measurements)]
