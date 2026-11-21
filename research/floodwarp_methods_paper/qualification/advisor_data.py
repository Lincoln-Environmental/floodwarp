"""Construct pre-run descriptors and independently grouped large-domain evidence."""
import json
from collections import defaultdict

import numpy as np
from floodwarp.solver_advisor.model import descriptors

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.qualification.observations import current


def group_key(spec):
    return (spec["landform"],spec["seed"],tuple(spec["dimensions"]),spec["dx"])


def samples(*,study,config):
    base=study/"processed/million/observations.csv"
    if not base.exists():
        return []
    terrain={}
    for rec in current(study=study,stage="million-terrain",config=config):
        terrain[group_key(rec["spec"])]=json.loads((study/rec["directory"]/"result.json").read_text())
    values={r["experiment_id"]:r for r in metrics.csv_read(base) if r["stage"] in ("million-storms","million-auto")}
    out=[]
    for rec in current(study=study,stage="million-storms",config=config)+current(study=study,stage="million-auto",config=config):
        s=rec["spec"]; row=values.get(rec["experiment_id"])
        if row is None: continue
        info=terrain.get(group_key(s))
        if info is None: raise ValueError(f"Missing represented terrain for {s['case']} at {s['dx']} m")
        d=json.loads((study/rec["directory"]/"result.json").read_text())
        f=d["forcing"]
        hours=np.array(f["rain_times_s"])/3600.
        rates=np.array(f["rain_rates_ms"])*3.6e6
        total=np.trapezoid(rates,hours)
        centroid=np.sum(np.diff(hours)*(hours[:-1]*(2*rates[:-1]+rates[1:])+hours[1:]*(rates[:-1]+2*rates[1:]))/6)/total
        area=np.prod(s["dimensions"])
        c=info["window"][str(int(s["dx"]))]["comparison"]
        physical=dict(rain_centroid_h=centroid,forecast_duration_h=s["hours"],rain_duration_h=6,
            spatial_mean_multiplier=d["expected_full_rain_m3"]*1000/(area*total),
            rain_mm=d["expected_full_rain_m3"]*1000/area,peak_mm_h=rates.max(),initial_mean_m=d["initial_m3"]/area,
            manning_n=f["roughness"],infiltration_mean_mm_h=20 if s["storm"]=="soil" else 2,
            downstream_stage_m=f.get("stage_m"))
        features=descriptors(terrain=dict(native=dict(dx_m=1.),model=info["domain"],comparison=c),hydraulic=physical)
        out.append(dict(case=s["case"],terrain_id=f"{s['landform']}-{s['seed']}",split=s["split"],landform=s["landform"],
            storm=s["storm"],method=s["method"],features=features,runtime_s=d["hydraulic_s"],memory_mib=d["live_pool_mib"],
            errors={k:float(row[v]) if row.get(v) else None for k,v in metrics.OBJECTIVES.items() if k!="memory"},
            qualified={k:row["qualified_"+k]=="True" and row["reference_self"]=="False" for k in config["advisor"]["tolerances"]},
            wet_fraction=d["wet_fraction"],sampled_regime=None,run_id=rec["experiment_id"]))
    return out


def split_groups(*,rows,heldout_family):
    groups={"development":[],"validation":[],"test":[]}
    for r in rows:
        split="test" if r["landform"]==heldout_family else r["split"]
        groups[split].append(r)
    ids={key:{r["terrain_id"] for r in values} for key,values in groups.items()}
    if any(ids[a]&ids[b] for a,b in (("development","validation"),("development","test"),("validation","test"))):
        raise ValueError("Terrain realisation leakage")
    return groups


def interval_calibration(*,model,rows):
    from floodwarp.solver_advisor.model import recommend
    gaps=defaultdict(dict)
    for r in rows:
        for obj,ok in r["qualified"].items():
            if not ok: continue
            prediction=recommend(model=model,features=r["features"],objective=obj)
            candidate=next((c for c in prediction["candidates"] if c["method"]==r["method"]),None)
            if candidate is None: continue
            lo,hi=candidate["error_range"]
            gap=max(lo-r["errors"][obj],r["errors"][obj]-hi,0)
            key=r["terrain_id"]
            gaps[obj][key]=max(gap,gaps[obj].get(key,0))
    model["interval_calibration_groups"]={key:len(values) for key,values in gaps.items()}
    model["interval_padding"]={key:float(np.quantile(list(values.values()),model["quantile"],method="higher")) for key,values in gaps.items()}

