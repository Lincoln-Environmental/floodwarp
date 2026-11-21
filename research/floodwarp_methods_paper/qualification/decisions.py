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


def predicted_cost(row):
    return row["predicted_runtime_s"], row["dx"], row["chosen"]


def observed_cost(row):
    return row["observed_runtime_s"], row["dx"], row["chosen"]


def by_spacing(row):
    return row["dx"]


def joint_resolution(*, decisions):
    """Choose across spacings of the same held-out physical event, before inspecting its error."""
    groups = defaultdict(list)
    for row in decisions:
        groups[row["terrain_id"], row["case"], row["objective"]].append(row)
    results = []
    for (terrain, case, objective), group in groups.items():
        grids = sorted({r["dx"] for r in group})
        if len(grids) < 2:
            continue
        feasible = [r for r in group if r["rule"] == "oracle" and r["feasible"]]
        best = min(feasible, key=observed_cost) if feasible else None
        for rule in sorted({r["rule"] for r in group}):
            candidates = [r for r in group if r["rule"] == rule and r["chosen"] is not None]
            if rule == "oracle":
                selected = best
            elif rule in ("advisor", "slope"):
                candidates = [r for r in candidates if r["predicted_runtime_s"] is not None]
                selected = min(candidates, key=predicted_cost) if candidates else None
            else:
                # Fixed methods use the coarsest supplied grid as the baseline.
                selected = max(candidates, key=by_spacing) if candidates else None
            results.append(dict(terrain_id=terrain, case=case, objective=objective, rule=rule,
                available_dx_m=",".join(f"{dx:g}" for dx in grids),
                chosen=selected["chosen"] if selected else None, dx=selected["dx"] if selected else None,
                selected_run=selected["selected_run"] if selected else None,
                feasible=selected["feasible"] if selected else None,
                abstained=selected is None, optimal=best["chosen"] if best else None,
                optimal_dx=best["dx"] if best else None, optimal_run=best["selected_run"] if best else None,
                regret_pct=100*(selected["observed_runtime_s"]/best["observed_runtime_s"]-1)
                    if best and selected and selected["feasible"] else None))
    return results


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
    csv_write(file=study/"processed/advisor/joint_resolution.csv", rows=joint_resolution(decisions=decisions))
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

