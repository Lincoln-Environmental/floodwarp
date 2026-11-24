"""Compare equal-grid hydraulic cells using authenticated original forecast checkpoints."""

import getopt
import json
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import final_depth, native_archive, native_map_endpoints, native_products, restore_native
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def load_maps(*, item, index, directory=None, endpoint=None):
    if endpoint is None:
        expected = item["d"]["hashes"]
        path = index.get((expected["state"], expected["volume_x"], expected["soil"]))
        state_hash, max_hash = expected["state"], expected["maximum"]
        end = item["s"]["hours"] * 3600
    else:
        path = native_map_endpoints.checkpoint(directory=directory, evidence=endpoint)
        state_hash, max_hash = endpoint["fields"]["q"], endpoint["fields"]["maximum"]
        end = endpoint["end_s"]
    if path is None:
        raise FileNotFoundError(f"Missing verified initial-window checkpoint for {item['id']}")
    with np.load(path, allow_pickle=False) as state:
        direct = json.loads(state["_direct_manifest"].item())
        manifest = direct["sha256"]
        if endpoint is not None and (runner.digest(manifest) != endpoint["checkpoint_array_set_sha256"]
                                     or runner.digest(direct["identity"]) != endpoint["identity_sha256"]):
            raise ValueError("Native-grid checkpoint differs from its settled endpoint")
        q, maximum = state["q"], state["maximum"]
        if (manifest["q"] != array_sha(q) or manifest["q"] != state_hash
                or manifest["maximum"] != array_sha(maximum) or manifest["maximum"] != max_hash):
            raise ValueError("Checkpoint maps disagree with the completed forecast")
        numerical = item["d"]["numerical"]
        shape = (numerical["ny"], numerical["nx"])
        if state["control"].ravel()[0] != end:
            raise ValueError("Checkpoint maps have the wrong physical end time")
        return q[:, 0].copy().reshape(shape), maximum.reshape(shape)


def compare(*, directory, case, dx, method, thresholds, dry, repeat, horizon):
    current = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    records = [item for item in native_products.prefixes(directory=directory).values()
               if item["s"]["case"] == case and item["s"]["dx"] == dx and item["solver_source"] == current
               and not item["s"].get("timing_repeat")
               and item["s"].get("requested_method", item["s"]["method"]) in ("fv1", "hancock_dense", "adaptive")
               and (method is None or item["s"]["method"] == method
                    or (item["s"]["method"] == "adaptive" and item["factor"] == 1))]
    paths = {}
    for item in records:
        key = (item["s"]["method"], item["factor"])
        if key in paths:
            raise ValueError(f"Ambiguous map source for {key}")
        paths[key] = item
    reference = paths.get(("adaptive", 1.0))
    parents = {item["id"]: item for item in records}
    if horizon == "matched":
        end_s = native_map_endpoints.common_end(directory=directory, case=case, parents=set(parents))
    elif horizon == "initial":
        end_s = reference["s"]["hours"] * 3600 if reference else None
    else:
        value = float(horizon) * 3600
        if not value.is_integer() or value <= 0:
            raise ValueError("Native map endpoint must be a positive whole number of seconds")
        end_s = int(value)
    if dry:
        print(case, dx, "m native hydraulic support; available", sorted(paths), horizon, "end (s)", end_s)
        return
    if reference is None or len(paths) < 2:
        raise ValueError("Matched ordinary-CFL RK2 and at least one peer are required")
    if end_s is None:
        print(case, "waiting for the common recession endpoint")
        return
    spec = reference["s"]
    ends = (native_map_endpoints.observations(directory=directory, parents=parents, end_s=end_s)
            if end_s != spec["hours"] * 3600 else {})
    raw = {}
    for item in records:
        if (item["s"]["hours"] != spec["hours"] or item["s"]["interval"] != spec["interval"]
                or item["native_sha"] != reference["native_sha"] or item["d"]["forcing"] != reference["d"]["forcing"]):
            raise ValueError("Native map controls differ in physical inputs, cadence or endpoint")
        receipt = native_archive.prefix_evidence(folder=item["path"])
        raw[item["id"]] = dict(result_sha256=receipt["result_sha256"], fields=item["d"]["hashes"])
        if ends:
            raw[item["id"]]["endpoint"] = ends[item["id"]]
    identity = dict(case=case, dx_m=dx, end_h=end_s / 3600, cadence_s=spec["interval"], thresholds_m=thresholds,
                    native_fields=reference["native_sha"], normal_rk2=reference["id"], observations=raw,
                    implementation={p.name: runner.checksum(p) for p in (Path(__file__), Path(final_depth.__file__), Path(native_map_endpoints.__file__))})
    dest = directory / "native_grid_maps" / runner.digest(identity)[:24]
    contract = dest / "receipt.json"
    table = dest / "differences.csv"
    if contract.is_file():
        saved = json.loads(contract.read_text())
        if saved["identity"] != identity or runner.checksum(table) != saved["table_sha256"]:
            raise ValueError("Native-grid map comparison was changed after archival")
        if not repeat:
            print("Verified archived native-grid field comparisons", dest.name)
            return
    index = {} if ends else restore_native.prefix_index(root=directory.parents[2] / ".fidelity-restarts")
    ref = load_maps(item=reference, index=index, directory=directory, endpoint=ends.get(reference["id"]))
    rows = []
    for (method, factor), item in sorted(paths.items()):
        if item["id"] == reference["id"]:
            continue
        candidate = load_maps(item=item, index=index, directory=directory, endpoint=ends.get(item["id"]))
        values = final_depth.difference(candidate=candidate, reference=ref,
                                        support=dx, thresholds=thresholds)
        rows.append(dict(case=case, end_h=end_s / 3600, cadence_s=spec["interval"], dx_m=dx,
                         native_fields=reference["native_sha"], method=method, cfl_factor=factor, id=item["id"],
                         comparator_id=reference["id"],
                         state_source=ends[item["id"]]["series_id"] if ends else item["id"],
                         comparator_state_source=ends[reference["id"]]["series_id"] if ends else reference["id"],
                         **values))
    if contract.is_file():
        archived = metrics.csv_read(file=table)
        if archived != [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]:
            raise ValueError("Recalculated native hydraulic fields disagree with archived comparisons")
    else:
        metrics.csv_write(file=table, rows=rows)
        runner.write_json(path=contract, data=dict(identity=identity, table_sha256=runner.checksum(table)))
    print("Measured", len(rows), "native-grid comparisons at", dest)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    flags, extra = getopt.getopt(sys.argv[1:], "", ["case=", "dx=", "end-h=", "method=", "resume", "dry-run", "recompute"])
    if extra:
        raise ValueError(f"Unexpected native grid comparison arguments {extra}")
    options = dict(flags)
    cases = ([options["--case"]] if "--case" in options else
             ["hill-8500-back-1000x1000", "plain-709-spatial-1000x1000", "plain-709-moderate-1000x1000"])
    dx = float(options.get("--dx", 1))
    thresholds = [t for t in json.loads((directory / "application_settings.json").read_text())["endpoint_depth"]["thresholds_m"] if t >= 0.05]
    horizons = [options["--end-h"]] if "--end-h" in options else ["initial", "matched"]
    for case in cases:
        for horizon in horizons:
            compare(directory=directory, case=case, dx=dx, method=options.get("--method"), thresholds=thresholds,
                    dry="--dry-run" in options, repeat="--recompute" in options, horizon=horizon)
