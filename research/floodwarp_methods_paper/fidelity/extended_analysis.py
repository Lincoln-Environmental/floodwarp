"""Join matched, physically completed recessions to the original fixed-horizon observations."""

import json
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import recession, receipts as immutable


def attach(*, directory, groups, partial=False, target=None):
    if target is None:
        target = directory / "results"
    result = {}
    histories = []
    complete = []
    root = directory / "extended_runs"
    receipts = {}
    for p in sorted(root.glob("group_*.json")):
        g = json.loads(p.read_text())
        key = g["cohort"], g["case"], g["representation"], tuple(sorted(g["sources"]))
        if key in receipts:
            raise ValueError(f"Ambiguous recession implementation for {key[:3]}")
        receipts[key] = g
    for key, all_items in groups.items():
        cohort, case, rep = key
        if cohort == "fixed" or rep not in ("sampled", "10m bed"):
            continue
        original = {
            i["id"]: i for i in recession.candidates(items=all_items, cohort=cohort)
        }
        if not original:
            continue
        registered = cohort, case, rep, tuple(sorted(original))
        if registered not in receipts:
            if partial:
                continue
            raise ValueError("Unfinished matched recession " + str(key))
        receipt = receipts[registered]
        label = "complete-" + cohort
        end = receipt["end_s"]
        if not receipt["extended"]:
            result[label, case, rep] = list(original.values())
            for i in original.values():
                d = i["d"]
                frames = d.get("frames", d.get("samples"))
                storage = d.get("final_storage_m3", d.get("final_m3"))
                passed, checks = recession.settled(
                    frames=frames,
                    storage=storage,
                    supply=d["rain_m3"] + d["initial_m3"],
                    config=receipt["settings"],
                )
                if not passed or frames[-1]["t_s"] != end:
                    raise ValueError("Archived recession no longer verifies")
                for station, check in checks.items():
                    histories.append(
                        dict(
                            cohort=cohort,
                            case=case,
                            representation=rep,
                            method=i["s"]["method"],
                            dx=i["s"]["dx"],
                            station=station,
                            id=i["id"],
                            parent_id=i["id"],
                            old_h=end / 3600,
                            end_h=end / 3600,
                            interval_s=frames[0]["t_s"],
                            storage_m3=storage,
                            infiltration_m3=d["infiltration_m3"],
                            residual_m3=d.get("residual_m3", d.get("residual")),
                            **check,
                        )
                    )
        records = []
        for ident in receipt["series"]:
            s = immutable.read(root=root, ident=ident, end=end)
            base = original[s["parent_id"]]
            if s["end_s"] != end or not s["settled"]:
                raise ValueError("Inconsistent group end time for " + case)
            d = dict(base["d"])
            d.pop("samples", None)
            original_spatial = {
                row["t_s"]: row
                for row in base["d"].get("spatial", base["d"].get("samples", []))
                if "wet_fraction" in row
            }
            original_spatial.update(
                {f["t_s"]: f for f in s["frames"] if "wet_fraction" in f}
            )
            samples = list(original_spatial.values()) if end in original_spatial else []
            d.update(
                frames=s["frames"],
                spatial=samples,
                final_m3=s["storage_m3"],
                final_storage_m3=s["storage_m3"],
                hydraulic_s=s["hydraulic_s"],
                hours_completed=end / 3600,
                infiltration_m3=s["infiltration_m3"],
                rain_m3=s["rain_m3"],
                residual_m3=s["residual_m3"],
                forcing=s["forcing"],
                steps=sum(f["steps"] for f in s["frames"]),
            )
            spec = dict(s["spec"], hours=end / 3600)
            records.append(
                dict(
                    base, s=spec, d=d, id=ident, parent_id=s["parent_id"], cohort=label
                )
            )
            for station, check in s["checks"].items():
                histories.append(
                    dict(
                        cohort=cohort,
                        case=case,
                        representation=rep,
                        method=spec["method"],
                        dx=spec["dx"],
                        station=station,
                        id=ident,
                        parent_id=s["parent_id"],
                        old_h=base["s"]["hours"],
                        end_h=end / 3600,
                        interval_s=spec["interval"],
                        **check,
                        storage_m3=s["storage_m3"],
                        infiltration_m3=s["infiltration_m3"],
                        residual_m3=s["residual_m3"],
                    )
                )
        if records:
            result[label, case, rep] = records
        complete.append(
            dict(
                cohort=cohort,
                case=case,
                representation=rep,
                extended=receipt["extended"],
                original_h=max(i["s"]["hours"] for i in original.values()),
                common_end_h=end / 3600,
                configurations=len(original),
            )
        )
    metrics.csv_write(file=target / "recession_horizons.csv", rows=complete)
    metrics.csv_write(file=target / "recession_terminal.csv", rows=histories)
    return result


def available(*, base):
    observed = {}
    for row in base:
        if (
            row["station"] != "outlet"
            or row["representation"] != "sampled"
            or not row["cohort"].startswith("complete-")
        ):
            continue
        ident = row.get("parent_id") or row["experiment_id"]
        observed[ident] = row
    return observed


def horizon_effects(*, directory, rows, cfg):
    previous = {
        (r["cohort"], r["experiment_id"], r["station"]): r
        for r in rows
        if not r["cohort"].startswith("complete-")
    }
    out = []
    for new in rows:
        if not new["cohort"].startswith("complete-"):
            continue
        old = previous.get(
            (new["cohort"].removeprefix("complete-"), new["parent_id"], new["station"])
        )
        if old is None or old["hours"] == new["hours"]:
            continue
        for metric in cfg["reference_limits"]:
            if metric in old and metric in new:
                out.append(
                    dict(
                        case=new["case"],
                        station=new["station"],
                        method=new["method"],
                        dx=new["dx"],
                        metric=metric,
                        old_id=old["experiment_id"],
                        new_id=new["experiment_id"],
                        old_h=old["hours"],
                        new_h=new["hours"],
                        old_value=old[metric],
                        new_value=new[metric],
                        change=new[metric] - old[metric],
                        old_qualified=old.get("qualified_" + metric, False),
                        new_qualified=new.get("qualified_" + metric, False),
                    )
                )
    metrics.csv_write(file=directory / "results/horizon_effects.csv", rows=out)
