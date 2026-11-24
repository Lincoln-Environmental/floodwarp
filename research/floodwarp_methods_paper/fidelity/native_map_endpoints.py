"""Bind native-cell map comparisons to a measured recession horizon and checkpoint."""

import gzip
import json

import numpy as np

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity import storage_reductions


def common_end(*, directory, case, parents):
    endpoints = set()
    for path in (directory / "extended_runs").glob("group_*.json"):
        group = json.loads(path.read_text())
        if group["case"] == case and group["cohort"] == "native-metre" and parents <= set(group["sources"]):
            endpoints.add(group["end_s"])
    if len(endpoints) > 1:
        raise ValueError("Ambiguous common recession endpoint")
    return next(iter(endpoints), None)


def observations(*, directory, parents, end_s):
    result = {}
    for p in (directory / "endpoint_fields").glob(f"*/{end_s}.npz"):
        with np.load(p, allow_pickle=False) as saved:
            receipt = json.loads(saved["receipt"].item())
        parent = receipt["parent_id"]
        if parent not in parents:
            continue
        ident = p.parent.name
        frozen = directory / "extended_runs" / ident / "horizons" / f"{end_s}.json.gz"
        with gzip.open(frozen, "rt") as stream:
            horizon = json.load(stream)
        if (receipt["id"] != ident or receipt["end_s"] != end_s or horizon["id"] != ident
                or horizon["end_s"] != end_s or horizon["parent_id"] != parent
                or horizon["spec"] != parents[parent]["s"] or horizon["cohort"] != "native-metre"
                or not horizon["settled"]):
            raise ValueError("Native map endpoint and settled hydraulic horizon disagree")
        storage_reductions.verify(directory=directory, receipt=receipt, series=horizon)
        if parent in result:
            raise ValueError("Multiple recession trajectories for the same native map")
        result[parent] = dict(series_id=ident, endpoint_sha256=runner.checksum(p),
                              horizon_sha256=runner.checksum(frozen), end_s=end_s,
                              checkpoint_array_set_sha256=receipt["checkpoint_array_set_sha256"],
                              identity_sha256=receipt["identity_sha256"], fields=receipt["hashes"])
    if result.keys() != parents.keys():
        raise ValueError("Missing a settled endpoint at the requested time for one or more peers")
    return result


def checkpoint(*, directory, evidence):
    cache = directory.parents[2] / ".fidelity-restarts"
    name = f"{evidence['end_s']}.npz"
    for base in (cache, cache / "retained_native_fields"):
        path = base / evidence["series_id"] / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No local native-grid checkpoint for {evidence['series_id']} at {name}")
