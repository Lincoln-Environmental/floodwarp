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


def regional_water(*, directory, completed):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import reach_balances

    events = {r["source_id"]: r for r in completed if r["horizon"] == "complete-native-metre"}
    implementation = runner.checksum(path=directory / "reach_balances.py")
    result = {}
    for path in sorted((directory / "reach_balances").glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        identity = receipt["identity"]
        event = identity["event"]
        ident = event["source_id"]
        if ident not in events or identity["analysis_sha"] != implementation:
            continue
        source = events[ident]
        table = path.with_name("water.csv")
        endpoint = directory / "endpoint_fields" / ident / f"{int(float(event['end_h']) * 3600)}.npz"
        if (runner.digest(identity)[:24] != path.parent.name or runner.checksum(path=table) != receipt["water_sha"]
                or runner.checksum(path=endpoint) != identity["endpoint_sha"]
                or identity["hydraulic_source"] != source["solver_source"]
                or any(str(source[key]) != value for key, value in event.items())):
            raise ValueError("Regional water account differs from the archived event")
        rows = metrics.csv_read(file=table)
        if ident in result or [r["reach"] for r in rows] != ["upstream", "downstream"]:
            raise ValueError("Duplicate or missing regional water account")
        reaches = [dict(reach=r["reach"], **{k: float(v) for k, v in r.items() if k != "reach"}) for r in rows]
        reach_balances.verify_totals(reaches=reaches, event=source)
        result[ident] = [dict(**event, **r, physical_sha=identity["physical_sha"], receipt_id=path.parent.name)
                         for r in reaches]
    return [row for ident in sorted(result) for row in result[ident]]


def embankment_observations(*, directory, windows):
    from research.floodwarp_methods_paper import metrics
    from research.floodwarp_methods_paper.fidelity import embankment_transport

    events = {(r["source_id"], r["horizon"]): r for r in windows}
    files = sorted((directory / "embankment_transport").glob("*/profile.json"))
    if not files:
        return []
    features = {(r["case"], int(r["dx_m"])): r
                for r in metrics.csv_read(file=directory / "native_features/observations.csv")}
    implementation = runner.checksum(path=directory / "embankment_transport.py")
    rows = {}
    for path in files:
        saved = json.loads(path.read_text())
        identity = saved["identity"]
        archived = identity["event"]
        key = (archived["source_id"], archived["horizon"])
        if key not in events or identity["analysis_sha"] != implementation:
            continue
        source = events[key]
        dx = int(archived["dx_m"])
        normalized = {k: str(v) for k, v in source.items()}
        expected, _, _, _ = embankment_transport.contracts(directory=directory, event=normalized,
            feature=features[archived["case"], dx], face=identity["face_m"])
        if path.parent.name != runner.digest(identity)[:24] or runner.digest(saved["profile"]) != saved["profile_sha"]:
            raise ValueError("Embankment measurement does not match the verified forecast and terrain")
        if expected != identity:
            historical_identity = dict(identity, feature_sha=expected["feature_sha"])
            if historical_identity == expected:
                continue
            raise ValueError("Embankment measurement does not match the verified forecast and terrain")
        key = (*key, identity["face_m"])
        if key in rows:
            raise ValueError("Duplicate fixed-transect measurement")
        values = embankment_transport.summarize(data=saved["profile"], dx=dx, window=identity["window_m"])
        west, south, width, height = identity["window_m"]
        rows[key] = dict(**archived, face_x_m=identity["face_m"], band_south_m=south, band_north_m=south + height,
                         crest_strip_west_m=west, crest_strip_east_m=west + width, **values,
                         physical_sha=identity["physical_sha"], receipt_id=path.parent.name)
    return [rows[k] for k in sorted(rows)]
