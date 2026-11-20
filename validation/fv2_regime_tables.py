"""Build numerical tables with physical units directly from the retained measurements."""
from pathlib import Path
import csv
import json
import numpy as np


def grid_table(*, columns, values):
    rows = ["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for row in values:
        rows.append("| "+" | ".join(format(v,".5g") if isinstance(v,(float,np.floating)) else str(v) for v in row)+" |")
    return "\n".join(rows)


def load(*,d,key):
    return json.loads((d/(key+".json")).read_text())


def tables(*,d,output):
    content=["# Numerical results", "Times are seconds unless labelled otherwise. All hydraulic state arrays use FP64."]
    def section(title,columns,values):
        content.extend(["## "+title,grid_table(columns=columns,values=values)])
    run=load(d=d,key="profile_marked")
    original=load(d=d,key="profile_baseline")
    section("Steady hydraulic timing, seven warmed runs",["Domain","Method","Original ms","Final ms","Final / FV1","Steps","Retries","Mean dt","Minimum dt"],[
        (r["case"],r["method"],next(x["median_s"]*1000 for x in original if x["case"]==r["case"] and x["method"]==("contact_hll" if r["method"]=="contact_hll" else "mc")),
         r["median_s"]*1000,r["median_s"]/next(x["median_s"] for x in run if x["case"]==r["case"] and x["method"]=="contact_hll"),
         r["times"][0]["steps"],r["times"][0]["rejected_cfl_trials"],r["times"][0]["dt_mean_s"],r["times"][0]["dt_min_s"]) for r in run])
    content.append("The original FV2 column uses MC as the common historical baseline for all FV2 rows.")
    rows=[]
    for key in ("baseline","reuse","source","warp","traces","primitive","inv_cache","fused","marked"):
        item=load(d=d,key="profile_"+key)
        rows.append([key,*[next(r["median_s"]*1000 for r in item if r["method"]=="mc" and r["case"]==case) for case in ("stoker","wet_large","dry_patch")]])
    section("Incremental implementation timings for FV2 MC",["Revision","Stoker ms","Large wet ms","Dry patch ms"],rows)
    kernel=[]
    for key,method in (("baseline","fv2_mc"),("marked","fv2_mc"),("marked","fv2_adaptive"),("marked","contact_hll")):
        r=next(v for v in load(d=d,key="orton_"+key) if v["dx"]==5 and v["method"]==method)
        groups={}
        for name in r["kernel_us_per_step"][0]:
            label=name.removeprefix("forward kernel ").rsplit("_",1)[0]
            groups[label]=np.median([record[name] for record in r["kernel_us_per_step"]])
        kernel.append((key,method,groups))
    all_keys=sorted(set().union(*(v.keys() for _,_,v in kernel)))
    section("Orton 5 m kernel microseconds per accepted step, uncaptured",["Kernel",*[k+" "+m for k,m,_ in kernel]],[(name,*[v.get(name,0.) for _,_,v in kernel]) for name in all_keys])
    sparse=load(d=d,key="sparse")
    section("Dynamic scheduling, identical final fields",["Case","dx","Mode","Median ms"],[(r.get("case","Orton"),r["dx"],r["mode"],1000*r.get("median_s",r.get("median_wall_s"))) for r in sparse])
    cases=load(d=d,key="analytical")
    section("Analytical errors at 2.5 m",["Case","Method","Depth RMSE m","Wet depth RMSE m","hu RMSE m²/s","hv RMSE m²/s","Volume residual m³"],[(r["case"],r["method"],r["depth_rmse_m"],r["wet_depth_rmse_m"],*r["momentum_rmse_m2_s"],r["balance"]["residual"]) for r in cases if r["dx"]==2.5])
    rhs=load(d=d,key="terrain")
    section("Synthetic 5 m depth RMSE and reference sensitivity, mm",["Case","Fixed 5 m input","FV1","Minmod","Adaptive","MC grid gap","Roe grid gap","Fine formulation gap","Half dt change"],[(r["case"],r["terrain_fixed_5m"],*[next(q["against_mc"]["depth_rmse_m"]*1000 for q in r["candidates"] if q["method"]==m and q["dx"]==5) for m in ("fv1","minmod","adaptive")],
        *[1000*r[k]["depth_rmse_m"] for k in ("mc_refinement_gap","roe_hll_refinement_gap","cross_formulation_spread","finest_mc_dt_halving")]) for r in rhs])
    plains=load(d=d,key="planes")
    rows=[]
    for r in plains:
        if r["dx"]==5 and r["method"]=="contact_hll" and r["datum"]==0 and r["start"]=="steady":
            rows.append((100*r["slope"],r["depth"],r["roughness"],r["froude"],r["exact_q"][0],*[next(x["q_rmse"] for x in plains if x["method"]==m and all(x[k]==r[k] for k in ("dx","slope","start","datum"))) for m in ("contact_hll","minmod","adaptive","li")]))
    section("Uniform Manning sheets at 5 m, accepted transport RMSE in m²/s",["Slope %","h m","n","Fr","Exact q","FV1","Minmod","Adaptive","LI"],rows)
    for key,title in (("lowland","0.625 m reference"),("lowland_refined","hydraulic refinement below native terrain")):
        if not (d/(key+".json")).exists():
            continue
        low=load(d=d,key=key)
        rows=[]
        for r in low:
            if r["fixed_terrain"]:continue
            for s in r["runs"]:
                if s["dx"]!=5: continue
                e=s["errors"]
                rows.append((r["case"],s["method"],1000*e["depth_rmse_m"],1000*(e["wet_depth_rmse_m"] or 0.),e["extent"]["0.01"]["iou"],np.mean(e["hydro_rmse"]),s["volume_residual_m3"],1000*r["reference_checks"]["formulation"]["depth_rmse_m"]))
        section("Lowland hydraulics at 5 m, "+title,["Case","Method","Depth RMSE mm","Wet RMSE mm","1 cm IoU","Mean section RMSE m³/s","Mass residual m³","Fine method spread mm"],rows)
        if key=="lowland_refined":
            section("Loss of terrain detail at constant hydraulic resolution",["Case","Bed RMS mm","Max crest lowering mm","Storage change at 0.04 m stage m³","Fine depth disagreement mm","Mean section disagreement m³/s"],[(r["case"],1000*r["aggregation"]["bed_rmse_m"],1000*r["aggregation"]["maximum_lowering_m"],r["aggregation"]["storage_change_at_0p04m_m3"],1000*r["representation_change_at_same_hydraulic_dx"]["depth_rmse_m"],np.mean(r["representation_change_at_same_hydraulic_dx"]["hydro_rmse"])) for r in low if r["fixed_terrain"]])
    section("Equal measured budget at the FV1 2.5 m runtime",["Case","Method","Budget ms","Best tested grid m","RMSE","Tolerance","Cheapest passing run ms"],[(r["case"],r["method"],r["budget_s"]*1000,r["best"]["dx"] if r["best"] else "—",r["best"]["error"] if r["best"] else "—",r["tolerance"],r["fastest"]["wall"]*1000 if r["fastest"] else "—") for r in load(d=d,key="equal_budget") if r["fv1_budget_dx"]==2.5])
    content.append("This budget selects measured runs that fit. Tolerances are illustrative depth RMSE in metres except shear, which uses transverse momentum in m²/s. Missing entries indicate no qualifying measured configuration.")
    for name in ("baseline","final"):
        csv_path=d/("counters_"+name+".csv")
        with csv_path.open() as f:
            source=csv.DictReader(line for line in f if line.startswith('"'))
            groups={}
            for r in source:
                groups.setdefault((r["ID"],r["Kernel Name"]),{})[r["Metric Name"]]=float(r["Metric Value"].replace(",",""))
        section("Nsight Compute, "+name,["Launch","Kernel","Registers/thread","Active warps %","DRAM % sustained peak","DRAM read GB/s"],[(i,k.split("_cuda")[0],v["launch__registers_per_thread"],v["sm__warps_active.avg.pct_of_peak_sustained_active"],v["dram__throughput.avg.pct_of_peak_sustained_elapsed"],v["dram__bytes_read.sum.per_second"]/1e9) for (i,k),v in groups.items()])
    section("Native 5 m HIRDS continuation, hour 25 to 25 h 10 min",["Method","Export m³","Rain m³","Loss m³","Final wet area m²","Speed p95 m/s","Final depth RMSE vs FV1 m","Mass residual m³"],[(r["method"],r["ledger_change"]["boundary_outflow"],r["ledger_change"]["rain"],r["ledger_change"]["infiltration"],r["wet_area_m2"],r["q95_speed"],r["vs_contact"]["depth_rmse_m"],r["volume_residual_m3"]) for r in load(d=d,key="halo/wet_5m")])
    storms=[]
    for path in sorted(d.glob("*/*m_*_dense.json")):
        r=json.loads(path.read_text())
        if "hydrograph" not in r:continue
        storms.append((path.parent.name,r["method"],r["steps"],sum(s["seconds"] for s in r["hydrograph"]),r["peak_outflow_m3_s"],r["peak_interval_end_s"]/3600,r["ledger"]["residual"]))
    section("Complete HIRDS 48-hour event, matched 20 m terrain",["Version","Method","Steps","Hydraulic loop s","Peak m³/s","Peak interval end h","Residual m³"],storms)
    output.write_text("\n\n".join(content)+"\n")


if __name__ == "__main__":
    directory=Path("validation/fv2_review/regimes")
    tables(d=directory,output=directory/"TABLES.md")
