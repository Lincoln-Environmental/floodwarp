"""Measure transport and additional crests observed after an earlier archived endpoint."""

import gzip
import json

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import recession


def transport(*, frames, station, cut, settings):
    t = np.asarray([r["t_s"] for r in frames], dtype=float)
    flow = np.asarray([r[station + "_m3s"] for r in frames], dtype=float)
    dt = np.diff(np.r_[0, t])
    if dt.size == 0 or np.any(dt <= 0) or not np.isfinite(flow).all() or cut not in t:
        raise ValueError("Earlier run must end on an observed accepted-transport interval")
    later = t > cut
    outward = np.maximum(flow, 0)
    volume = outward * dt
    gross = float(volume.sum())
    peak = float(outward.max())
    before_peak = float(outward[~later].max())
    tail_peak = float(outward[later].max()) if later.any() else None
    after = float(volume[later].sum()) if later.any() else None
    meaningful = gross >= settings["minimum_volume_m3"] and peak >= settings["minimum_peak_m3s"]
    return dict(early_peak_m3s=before_peak, complete_peak_m3s=peak, followup_peak_m3s=tail_peak,
                larger_crest_during_followup=tail_peak > before_peak if meaningful and later.any() else None,
                complete_outward_m3=gross, followup_outward_m3=after,
                followup_inward_m3=float(np.maximum(-flow[later], 0) @ dt[later]) if later.any() else None,
                followup_net_m3=float(flow[later] @ dt[later]) if later.any() else None,
                followup_pct_of_complete_outward=100 * after / gross if meaningful and later.any() else None)


def first_pass(*, directory, item, prefix, settings):
    data = prefix["d"]
    base = data["frames"][-1]["t_s"]
    end = item["d"]["frames"][-1]["t_s"]
    supply = data["initial_m3"] + data["rain_m3"]
    passed, _ = recession.settled(frames=data["frames"], storage=data["final_storage_m3"], supply=supply, config=settings)
    if passed:
        return base, data["final_storage_m3"], runner.checksum(path=prefix["path"] / "result.json")
    folder = directory / "extended_runs" / item["id"] / "horizons"
    for stop in sorted(int(p.name.split(".")[0]) for p in folder.glob("*.json.gz")):
        if not base < stop < end:
            continue
        path = folder / f"{stop}.json.gz"
        with gzip.open(path, "rt") as stream:
            early = json.load(stream)
        if (early["parent_id"] != prefix["id"] or early["id"] != item["id"] or early["end_s"] != stop
                or early["frames"][-1]["t_s"] != stop):
            raise ValueError("Earlier recession checkpoint has different run identity or endpoint")
        verified, _ = recession.settled(frames=early["frames"], storage=early["storage_m3"], supply=supply, config=settings)
        if verified != early["settled"]:
            raise ValueError("Earlier two-gauge recession decision does not reproduce")
        if verified:
            full = item["d"]["frames"][:len(early["frames"])]
            for field in ("t_s", "steps", "outlet_m3s", "section_m3s"):
                if [r[field] for r in full] != [r[field] for r in early["frames"]]:
                    raise ValueError("Earlier recession receipt differs from the measured trajectory")
            return stop, early["storage_m3"], runner.checksum(path=path)
    return None


def collect(*, directory, groups, records, settings, target):
    rows = []
    for (cohort, case, _), members in groups.items():
        if cohort != "complete-native-metre":
            continue
        for item in members:
            parent = records[item.get("parent_id", item["id"])]
            initial = parent["d"]
            frames = item["d"]["frames"]
            end = frames[-1]["t_s"]
            early = first_pass(directory=directory, item=item, prefix=parent, settings=settings)
            prefix = (initial["frames"][-1]["t_s"], initial["final_storage_m3"],
                      runner.checksum(path=parent["path"] / "result.json"))
            cuts = [("original window", prefix)]
            if early and early[0] < end:
                cuts.append(("earliest archived two-gauge pass", early))
            for label, (cut, storage, source) in cuts:
                for station in ("section", "outlet"):
                    rows.append(dict(case=case, method=item["s"]["method"], dx_m=item["s"]["dx"],
                                     cfl_factor=item["factor"], parent_id=parent["id"], id=item["id"],
                                     station=station, earlier_window=label, earlier_end_h=cut/3600, end_h=end/3600,
                                     followup_h=(end-cut)/3600, cadence_s=item["s"]["interval"], earlier_sha256=source,
                                     earlier_storage_m3=storage, final_storage_m3=item["d"]["final_storage_m3"],
                                     **transport(frames=frames, station=station, cut=cut, settings=settings)))
    metrics.csv_write(file=target / "recession_followup.csv", rows=rows)
    return rows
