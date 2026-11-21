"""Frozen held-out cost, abstention and predictive-coverage comparisons."""
from collections import defaultdict
from statistics import median

import numpy as np
from floodwarp.solver_advisor.model import recommend
from research.floodwarp_methods_paper.metrics import csv_write


def run_time(row):
    return row["runtime_s"]


def runtime_predictions(*,model,rows):
    by_grid=defaultdict(list)
    for r in rows:
        by_grid[r["case"],r["features"]["resolution"]].append(r)
    predictions=[]
    for (case,dx),group in by_grid.items():
        methods={r["method"]:r for r in group}
        for rule in ("advisor","slope","fv1","hancock_dense","oracle"):
            best=min(group,key=run_time)
            res=None
            if rule=="oracle": chosen=best["method"]
            elif rule in ("fv1","hancock_dense"):chosen=rule
            else:
                res=recommend(model=model,features=group[0]["features"],objective="runtime",slope_only=rule=="slope")
                chosen=res["method"]
            actual=methods.get(chosen)
            target=methods.get("fv1",group[0])
            predictions.append(dict(case=case,dx=dx,cells=target["features"]["cells"],landform=target["landform"],
                wet_fraction_fv1=target.get("wet_fraction"),rule=rule,selected=chosen,oracle=best["method"],
                available_configs=len(group),correct=chosen==best["method"],abstained=chosen is None,
                measured_s=actual["runtime_s"] if actual else None,predicted_s=res.get("runtime_s") if res else None,
                regret_pct=100*(actual["runtime_s"]/best["runtime_s"]-1) if actual else None))
    return predictions


def validate(*,study,model,held,decisions):
    from research.floodwarp_methods_paper.advisor import summary
    groups=defaultdict(list)
    fv_wet={(r["case"],r["features"]["resolution"]):r.get("wet_fraction") for r in held if r["method"]=="fv1"}
    for row in decisions:
        size=row["cells"]
        wet=fv_wet.get((row["case"],row["dx"]))
        wet_bin="unavailable" if wet is None else ["0–1%","1–5%","5–20%","20–50%","50–100%"][np.searchsorted([.01,.05,.2,.5],wet)]
        bins=dict(landform=row["family"],resolution=f"{row['dx']:g} m",cells=str(size),wet_fraction=wet_bin,
                  scale="at least one million" if size>=1000000 else "250k to under one million" if size>=250000 else "under 250k")
        for variable,value in bins.items():
            groups[variable,value,row["rule"]].append(row)
    strata=[]
    for (axis,value,rule),g in groups.items():
        s=summary(rows=g)[0]
        strata.append(dict(axis=axis,group=value,**s))
    csv_write(file=study/"processed/advisor/validation_strata.csv",rows=strata)
    csv_write(file=study/"processed/advisor/runtime_only.csv",rows=runtime_predictions(model=model,rows=held))


def scale_fit(*,model,training,validation):
    """Choose the scale window on absolute held-out log cost, charging an unavailable forecast."""
    from floodwarp.solver_advisor.model import nearest
    trials=[]
    for factor in (1.,2.,4.):
        losses=[]
        for r in validation:
            f=r["features"]
            pool=[a for a in training if a["method"]==r["method"] and a["features"]["resolution"]==f["resolution"] and
                  f["cells"]/factor<=a["features"]["cells"]<=f["cells"]*factor]
            n=nearest(rows=pool,query=f,model=model,k=3)
            if len(n)>=2:
                losses.append(abs(np.log(median(a["runtime_s"] for _,a in n)/r["runtime_s"])))
            else:
                losses.append(np.log(4.))
        trials.append((float(np.mean(losses)),factor))
    model["size_factor"]=min(trials)[1]
    model["validation_scale_losses"]=trials

