"""Exact state and accepted-transport comparisons at matched recession endpoints."""

import json
import re
import numpy as np
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity.endpoints import endpoint_path
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def fingerprints(*, directory, item):
    end = round(item["s"]["hours"] * 3600)
    archive = endpoint_path(directory=directory, ident=item["id"], end_s=end)
    d = item["d"]
    if archive.is_file():
        with np.load(archive) as f:
            hashes = json.loads(f["receipt"].item())["hashes"]
        result = dict(hashes, state=hashes["q"])
        if d.get("accepted_history_sha256"):
            result["accepted_dt"] = d["accepted_history_sha256"]
        return result
    if "hashes" in d:
        result = dict(d["hashes"])
        if d.get("accepted_history_sha256"):
            result["accepted_dt"] = d["accepted_history_sha256"]
        return result
    with np.load(item["path"] / (d["name"] + ".npz")) as f:
        out = {k: array_sha(f[k]) for k in ("depth", "maximum")}
    if d.get("accepted_history_sha256"):
        out["accepted_dt"] = d["accepted_history_sha256"]
    return out


def validate_interval_history(*, run, observations, case):
    times = np.array([f["t_s"] for f in observations])
    spans = np.diff(np.r_[0., times])
    if not len(times) or times[-1] != round(run["s"]["hours"] * 3600) or not np.all(spans > 0):
        raise ValueError(f"Incomplete interval transport history for {case}")
    cadence = run["s"].get("interval")
    if cadence and np.any(spans != cadence):
        raise ValueError(f"Missing scheduled interval transport for {case}")
    if not all(np.isfinite([f[k] for f in observations]).all() for k in ("outlet_m3s", "section_m3s")):
        raise ValueError(f"Nonfinite interval transport for {case}")


def compare(*, directory, groups, target, original_window=False):
    out = []
    for (cohort, case, rep), items in sorted(groups.items()):
        if not original_window and not cohort.startswith("complete-"):
            continue
        for dx in sorted({i["s"]["dx"] for i in items}):
            peers = {
                i["s"]["method"]: i
                for i in items
                if i["s"]["dx"] == dx and i["factor"] == 1
            }
            if "hancock_dense" not in peers:
                continue
            dense = peers["hancock_dense"]
            a = fingerprints(directory=directory, item=dense)
            frames = dense["d"].get("frames", dense["d"].get("samples"))
            for mode in ("hancock_adaptive", "hancock_auto"):
                if mode not in peers:
                    continue
                other = peers[mode]
                b = fingerprints(directory=directory, item=other)
                more = other["d"].get("frames", other["d"].get("samples"))
                shared = sorted(set(a) & set(b))
                if ("accepted_dt" in a) != ("accepted_dt" in b):
                    raise ValueError("An ordered accepted-step fingerprint is missing from one peer")
                if cohort in ("original-window", "complete-native-metre") or cohort.startswith("complete-native-repeat-"):
                    required = {"state", "maximum", "volume_x", "volume_y", "soil", "rain_sum"}
                    missing = required.difference(shared)
                    if missing:
                        raise ValueError(f"Missing native policy parity fields for {case}: {sorted(missing)}")
                elif "maximum" not in shared or not {"state", "depth"}.intersection(shared):
                    raise ValueError(f"Missing final state/depth or maximum fingerprint for {case}")
                if set(a) != set(b):
                    raise ValueError(f"Asymmetric parity evidence for {case}: {sorted(set(a) ^ set(b))}")
                for name in shared:
                    if any(not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
                           for value in (a[name], b[name])):
                        raise ValueError(f"Invalid SHA-256 fingerprint for {case}: {name}")
                for run, observations in ((dense, frames), (other, more)):
                    validate_interval_history(run=run, observations=observations, case=case)
                fields = {k: a[k] == b[k] for k in shared if k != "q"}
                fields.update(
                    {
                        k: np.array_equal([r[k] for r in frames], [r[k] for r in more])
                        for k in ("steps", "t_s", "outlet_m3s", "section_m3s")
                    }
                )
                if not all(fields.values()):
                    raise ArithmeticError(
                        f"Dense/alternative FP64 discrepancy for {case} at {dx} m: {fields}"
                    )
                out.append(
                    dict(
                        cohort=cohort,
                        case=case,
                        representation=rep,
                        dx=dx,
                        end_h=dense["s"]["hours"],
                        dense_id=dense["id"],
                        other_id=other["id"],
                        other_method=mode,
                        final_state_equal=fields.get("state"),
                        interval_transport_equal=all(fields[k] for k in ("t_s", "outlet_m3s", "section_m3s")),
                        accepted_timestep_history_equal=fields.get("accepted_dt"),
                        ordered_history_status="verified" if "accepted_dt" in fields else "unrecorded",
                        **fields,
                    )
                )
    name = "original_execution_parity.csv" if original_window else "recession_execution_parity.csv"
    metrics.csv_write(file=target / name, rows=out)
    return out
