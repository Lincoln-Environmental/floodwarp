"""Analyse fixed physical sections and separately conserved regions in completed storm windows."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import native_tail


def crest(*, q, times, count, stride):
    starts = np.arange(0, len(q)-count+1, stride)
    if len(starts) == 0:
        raise ValueError("Incomplete peak reporting interval")
    sums = np.r_[0., np.cumsum(q)]
    means = (sums[starts+count] - sums[starts])/count
    k = int(np.argmax(means))
    return float(means[k]), int(times[starts[k]+count-1])


def strip(*, row):
    return {0:dict(transfer_m3=row["gauge"]), 40:dict(transfer_m3=row["tail_in"],
            storage_m3=row["last40m_storage_m3"], rain_m3=row["last40m_rain_m3"],
            infiltration_m3=row["last40m_infiltration_m3"])}


def strip_change(*, before, after):
    a, b = strip(row=before), strip(row=after)
    for data in (a,b):
        data[0].update(storage_m3=0.,rain_m3=0.,infiltration_m3=0.)
    return native_tail.changes(before=a,after=b)[1]


def checkpoint_fields(*, state_dir, hours, grid, footprint):
    with np.load(state_dir/f"{int(hours*3600)}.npz") as archive:
        width=footprint//grid
        ny=json.loads(archive["metadata"].item())["configuration"]["ny"]
        return (archive["q"][:,0].reshape(ny,-1)[:,:width],
                archive["maximum"].reshape(ny,-1)[:,:width])


def run(*, directory, end, rainfall_end, extent_hour, thresholds):
    summaries, strips, snapshots, peers = [], [], [], []
    for path in sorted((directory/"outlet_storms").glob(f"*/summary_{end}.json")):
        run = path.parent
        contract = json.loads((run/"inputs.json").read_text())
        identity, initial = contract["identity"],contract["initial"]
        case = dict(run_id=run.name, method=identity["method"], dx_m=identity["dx"],
                    extension_m=identity["extension_m"], cfl_factor=identity["cfl_factor"],footprint_width_m=identity["forcing"]["shape"][1]*identity["dx"],end_h=end/3600)
        rows = [r for p in sorted(run.glob("[0-9]*.json"),key=window_end) if int(p.stem)<=end
                for r in json.loads(p.read_text())["frames"]]
        times = np.array([r["t_s"] for r in rows])
        cadence = identity["settings"]["interval_s"]
        np.testing.assert_array_equal(times,np.arange(cadence,end+1,cadence))
        q = np.array([r["outlet_m3s"] for r in rows])
        peaks = {}
        for label, minutes, stride in (("minute",1,1),("fixed_15min",15,15),("moving_15min",15,1)):
            value,t = crest(q=q,times=times,count=minutes*60//cadence,stride=stride*60//cadence)
            peaks[label+"_peak_m3s"],peaks[label+"_end_s"] = value,t
        frames = {r["t_s"]:r for r in rows}
        frames[0]=initial
        for name,start,stop in (("crest_minute",peaks["minute_end_s"]-cadence,peaks["minute_end_s"]),
                                ("post_rain_recession",rainfall_end,end)):
            delta=strip_change(before=frames[start],after=frames[stop])
            strips.append(dict(case,period=name,start_s=start,end_s=stop,**delta))
        final = rows[-1]
        stats = json.loads(path.read_text())
        summaries.append(dict(case,**peaks,initial_storage_m3=initial["footprint_storage_m3"],
                              recession_passed=stats["recession_passed"],hydraulic_s=stats["hydraulic_s"],
                              steps=stats["steps"],retries=sum(r["rejected"] for r in rows),
                              **{k:v for k,v in final.items() if k.startswith(("extension_","footprint_","infiltration_"))},
                              gauge_m3=final["gauge"],outer_boundary_m3=final["boundary"]))
        for hour in (extent_hour,end/3600):
            record=frames[int(hour*3600)]
            snapshots.append(dict(case,hour=hour,**{k:v for k,v in record.items() if "_above_" in k}))
        peers.append(case)
    dest = directory/"outlet_results"
    for name,rows in (("events",summaries),("last_40m",strips),("extent",snapshots)):
        metrics.csv_write(file=dest/f"{name}.csv",rows=rows)
    differences = []
    for a,b in combinations(peers,2):
        if (a["extension_m"],a["dx_m"],a["cfl_factor"]) != (b["extension_m"],b["dx_m"],b["cfl_factor"]):
            continue
        grid = a["dx_m"]
        for hour in (extent_hour,end/3600):
            fields=[]
            for peer in (a,b):
                cache=directory.parents[2]/".fidelity-restarts/outlet_storms"/peer["run_id"]
                fields.append(checkpoint_fields(state_dir=cache,hours=hour,grid=grid,
                                                footprint=peer["footprint_width_m"]))
            for label,first,second in zip(("instant","maximum"),*fields,strict=True):
                for threshold in thresholds:
                    wet=(first>threshold)|(second>threshold)
                    differences.append(dict(a_id=a["run_id"],b_id=b["run_id"],hour=hour, field=label,
                                            dx_m=grid,threshold_m=threshold,common_wet_union_m2=int(wet.sum())*grid**2,
                                            wet_union_depth_rmse_m=float(np.sqrt(np.mean((first[wet]-second[wet])**2)))
                                            if wet.any() else None))
    metrics.csv_write(file=dest/"depth_disagreement.csv",rows=differences)
    from research.floodwarp_methods_paper.fidelity import repaired_outlet_report

    repaired_outlet_report.run(directory=directory)


def window_end(path):
    return int(path.stem)


if __name__ == "__main__":
    directory=Path(__file__).resolve().parent
    end=16*3600
    rainfall_end=6*3600
    extent_hour=8
    thresholds=(.05,.10)
    run(directory=directory,end=end,rainfall_end=rainfall_end,extent_hour=extent_hour,thresholds=thresholds)
