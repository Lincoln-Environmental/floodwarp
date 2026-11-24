"""Compare measured Hancock execution costs over identical hydraulic trajectories."""

from collections import defaultdict

import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import phase_cost


def matched_costs(*, dense, tiled, automatic, baseline):
    rows = [dense, tiled]
    if automatic:
        rows.append(automatic)
    if any(r["steps"] != dense["steps"] for r in rows):
        raise ValueError("Hancock policies used different accepted step counts")
    seconds = [r["seconds"] for r in rows]
    if not all(np.isfinite(t) and t > 0 for t in seconds):
        raise ValueError("Invalid measured hydraulic runtime")
    policies = ["dense", "tiles", "auto"][: len(seconds)]
    best = min(seconds)
    explicit = min(seconds[:2])
    return dict(
        dense_s=seconds[0],
        tiled_s=seconds[1],
        auto_s=automatic["seconds"] if automatic else None,
        fv1_s=baseline["seconds"] if baseline else None,
        hancock_steps=dense["steps"],
        fv1_steps=baseline["steps"] if baseline else None,
        fastest_measured=",".join(
            p for p, t in zip(policies, seconds, strict=True) if t == best
        ),
        fastest_s=best,
        fastest_over_fv1=best / baseline["seconds"] if baseline else None,
        tiled_over_dense=seconds[1] / seconds[0],
        auto_over_fastest_explicit=automatic["seconds"] / explicit
        if automatic
        else None,
    )


def collect(*, groups, verified, cadence, target):
    pairs = {(r["dense_id"], r["other_id"]) for r in verified}
    windows, events = [], []
    for (cohort, case, rep), items in sorted(groups.items()):
        grids = defaultdict(dict)
        for it in items:
            if it["factor"] != 1 or it.get("raw_timing_status") == "shared GPU observed":
                continue
            spec = it["s"]
            key = (
                spec["dx"],
                spec["hours"],
                it["native_sha"],
                spec.get("timing_repeat", 0),
            )
            method = spec.get("requested_method", spec["method"])
            if method in grids[key]:
                raise ValueError("Duplicate policy on the same physical trajectory")
            grids[key][method] = it
        for (dx, hours, physical, repeat), peers in grids.items():
            dense, tiled = peers.get("hancock_dense"), peers.get("hancock_adaptive")
            if not dense or not tiled:
                continue
            auto = peers.get("hancock_auto")
            fv1 = peers.get("fv1")
            for other in (tiled, auto):
                if other and (dense["id"], other["id"]) not in pairs:
                    raise ValueError("Missing exact hydraulic parity for policy costs")
            data = {
                it["id"]: phase_cost.intervals(item=it, cadence=cadence)
                for it in (dense, tiled, auto, fv1)
                if it
            }
            ends = sorted(data[dense["id"]])
            if (
                not ends
                or ends[-1] != hours * 3600
                or any(sorted(v) != ends for v in data.values())
            ):
                raise ValueError(
                    "Execution comparisons require identical physical time windows"
                )
            meta = dict(
                cohort=cohort,
                case=case,
                representation=rep,
                native_sha=physical,
                dx_m=dx,
                cells=round(np.prod(dense["s"]["dimensions"]) / dx**2),
                end_h=hours,
                timing_repeat=repeat,
                dense_id=dense["id"],
                tiled_id=tiled["id"],
                auto_id=auto["id"] if auto else None,
                fv1_id=fv1["id"] if fv1 else None,
            )
            for end in ends:
                d, t = data[dense["id"]][end], data[tiled["id"]][end]
                if (
                    d["wet"] is not None
                    and t["wet"] is not None
                    and d["wet"] != t["wet"]
                ):
                    raise ValueError("Dense and tiled wet footprints disagree")
                costs = matched_costs(
                    dense=d,
                    tiled=t,
                    automatic=data[auto["id"]][end] if auto else None,
                    baseline=data[fv1["id"]][end] if fv1 else None,
                )
                windows.append(
                    dict(
                        **meta,
                        end_s=end,
                        window_s=cadence,
                        wet_fraction=t["wet"],
                        tiled_scheduled_fraction=t["active"],
                        **costs,
                    )
                )
            totals = {
                k: dict(
                    seconds=sum(v["seconds"] for v in values.values()),
                    steps=sum(v["steps"] for v in values.values()),
                )
                for k, values in data.items()
            }
            for it in (dense, tiled, auto, fv1):
                if it and not np.isclose(
                    totals[it["id"]]["seconds"],
                    it["d"]["hydraulic_s"],
                    rtol=1e-10,
                    atol=1e-8,
                ):
                    raise ValueError(
                        "Reporting windows do not recover the measured event cost"
                    )
            events.append(
                dict(
                    **meta,
                    **matched_costs(
                        dense=totals[dense["id"]],
                        tiled=totals[tiled["id"]],
                        automatic=totals[auto["id"]] if auto else None,
                        baseline=totals[fv1["id"]] if fv1 else None,
                    ),
                )
            )
    metrics.csv_write(file=target / "policy_cost_windows.csv", rows=windows)
    metrics.csv_write(file=target / "policy_cost_events.csv", rows=events)
