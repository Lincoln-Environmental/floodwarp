"""Exact state and accepted-transport comparisons at matched recession endpoints."""

import json
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
        return dict(hashes, state=hashes["q"])
    if "hashes" in d:
        return d["hashes"]
    with np.load(item["path"] / (d["name"] + ".npz")) as f:
        out = {k: array_sha(f[k]) for k in ("depth", "maximum")}
    if d.get("accepted_history_sha256"):
        out["accepted_dt"] = d["accepted_history_sha256"]
    return out


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
                        **fields,
                    )
                )
    name = "original_execution_parity.csv" if original_window else "recession_execution_parity.csv"
    metrics.csv_write(file=target / name, rows=out)
    return out
