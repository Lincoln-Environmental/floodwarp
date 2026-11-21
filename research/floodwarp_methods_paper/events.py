"""Construction, time-history replay and archive of prospective rainfall experiments."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp.inputs import terrain_from_array,boundary_faces
from floodwarp.model import create_model,advance
from research.floodwarp_methods_paper.history import checked,history
from research.floodwarp_methods_paper.terrain.generators.generate import fields,array_sha
from validation.hancock_long_storm_review import storms
from validation.hancock_default_profile import measure


def numerical(*, method):
    options={}
    if method=="li":
        return "li","native",dict(track_cumulative_face_volumes=True,track_absolute_face_volumes=False,track_max_water_depth=True)
    if method in ("fv1","low_froude_hll"):
        return "fv1","sparse",dict(flux_scheme="contact_hll" if method=="fv1" else "low_froude_hll")
    if method in ("hancock_adaptive","hancock_dense"):
        return "hancock","dense" if method=="hancock_dense" else "tiles",dict(limiter="adaptive")
    if method=="adaptive": return "fv2","dense",dict(limiter="adaptive")
    if method=="euler_mc": return "fv1_muscl_euler","dense",dict(limiter="mc")
    raise ValueError(method)


def create_event(*, config,template,spec,device):
    rain=config["storms"]
    (z,h,hu,u,v), params=fields(design=config["synthetic"],template=template,family=spec["landform"],seed=spec["seed"],initial=spec["initial"],dx=spec["dx"],dem=spec["dem"])
    terrain=terrain_from_array(elevation=z,dx=spec["dx"],dy=spec["dx"],row_order="south_to_north",vertical_datum="local")
    bx,by=boundary_faces(terrain=terrain)
    coastal=spec["landform"]=="coastal"
    bx[:,-1]=3 if coastal else 2
    t=np.r_[template["hours"],spec["hours"]]*3600.
    rates=np.r_[template["storms_mm_h"][spec["storm"]],0.]/3.6e6
    mask=np.maximum(0.,1-((u-0.30)/0.23)**2-((v-0.70)/0.20)**2) if spec["storm"]=="spatial" else np.ones_like(z)
    infiltration=((6+22*u+6*v) if spec["storm"]=="soil" else np.full_like(z,2.))/3.6e6
    method,mode,options=numerical(method=spec["method"])
    if method!="li": options.update(execution_mode=mode,gravity=rain["gravity"])
    model=create_model(terrain=terrain,solver_method=method,boundary_qx=bx,boundary_qy=by,stage_qx=.015 if coastal else 0.,stage_datum="local",initial_depth=h,manning_n=rain["manning_n"]*spec["multiplier"],rainfall_series=(t,rates),infiltration=infiltration,device=device,**options)
    model.solver.set_rainfall_spatial_multiplier(multiplier=mask)
    if method=="li":
        qx=np.zeros(bx.shape);qx[:,1:-1]=.5*(hu[:,1:]+hu[:,:-1]);qx[:,-1]=hu[:,-1]
        model.solver.set_face_fluxes(qx=qx,qy=np.zeros(by.shape))
    else:
        model.solver.set_momentum(hu=hu,hv=np.zeros_like(hu))
    np.testing.assert_allclose(float(model.solver.gravity),rain["gravity"],rtol=0,atol=1e-13)
    provenance=dict(terrain_sha256=array_sha(z),rain_mask_sha256=array_sha(mask),initial_depth_sha256=array_sha(h),initial_hu_sha256=array_sha(hu),infiltration_sha256=array_sha(infiltration),boundary_x_sha256=array_sha(bx),boundary_y_sha256=array_sha(by),stage_m=.015 if coastal else None,roughness=rain["manning_n"]*spec["multiplier"],shape=list(z.shape),parameters=params,numerical=model.solver.numerical_configuration(),rain_times_s=t.tolist(),rain_rates_ms=rates.tolist())
    return model,float(np.trapezoid(rates,t)*mask.sum()*spec["dx"]**2),provenance


def storm(*,spec,config,template,directory,device):
    method,mode,_=numerical(method=spec["method"])
    def builder(**_kwargs):
        model,volume,_=create_event(config=config,template=template,spec=spec,device=device)
        return model,volume
    with patch.object(storms,"storm_model",builder):
        storms.event(config=template,landform=spec["landform"],storm=spec["storm"],method=method,mode=mode,dx=spec["dx"],terrain_dx=spec["dem"],dimensions=config["synthetic"]["dimensions_m"],hours=spec["hours"],interval=config["storms"]["interval_s"],device=device,directory=directory,fields=directory,label_name="paper")
    file=next(directory.glob("paper_*.json"))
    row=json.loads(file.read_text())
    model,_,provenance=create_event(config=config,template=template,spec=spec,device=device)
    with history(solver=model.solver,capacity=row["steps"]) as get_dt:
        for frame in row["samples"]:
            stats=advance(model=model,end_time=frame["t_s"])
            if stats["steps"]!=frame["steps"]: raise ArithmeticError("Changed step count during history replay")
        dt=get_dt()
    checked(times=dt,count=row["steps"],duration=spec["hours"]*3600)
    if not row["balance_pass"] or abs(row["rain_m3"]-row["expected_rain_m3"])>1e-8*max(1.,row["expected_rain_m3"]):
        raise ArithmeticError("Physical budget failed")
    np.savez_compressed(directory/"accepted_dt.npz",dt_s=dt)
    row.update(hydraulic_method=spec["method"],forcing=provenance,accepted_history_sha256=array_sha(dt),fields_file=row["name"]+".npz")
    return row


def evaluate(*,stage,spec,study,root,config,context,directory):
    original=root/"validation/hancock_long_storm_review"
    template=json.loads((original/"cases.json").read_text())
    if stage=="synthetic":
        rows=[]
        for dx in (1,5,10):
            arrays,params=fields(design=config["synthetic"],template=template,family=spec["landform"],seed=spec["seed"],initial="dry",dx=dx,dem=dx)
            z=arrays[0]
            repeat,_=fields(design=config["synthetic"],template=template,family=spec["landform"],seed=spec["seed"],initial="dry",dx=dx,dem=dx)
            np.testing.assert_array_equal(z,repeat[0])
            if not np.isfinite(z).all() or z.max()<=z.min():raise ValueError("Invalid terrain")
            np.savez_compressed(directory/f"terrain_{dx}m.npz",elevation_m=z,dx_m=dx)
            rows.append(dict(dx=dx,sha256=array_sha(z),shape=list(z.shape),relief_m=float(np.ptp(z)),params=params))
        return dict(landform=spec["landform"],seed=spec["seed"],grids=rows)
    if spec.get("kind")=="scale":
        methods=(("contact_hll","sparse"),("hancock_adaptive","dense"),("hancock_adaptive","tiles"),("adaptive","dense"))
        measure(output=directory/"common_cap.json",cases=("stoker","dry_patch"),methods=methods,length=1024.,width=640.,spacing=1.,cap=1/32,duration=2.,repeats=config["performance"]["timing_repeats"],device=context["device"])
        return dict(common_cap_file="common_cap.json",retrospective_large_matrix="validation/hancock_long_storm_review/event_summary.csv",retrospective_gpu_profiling="validation/hancock_long_storm_review/graph_kernel_totals.csv",implementation=config["performance"]["benchmark_commit"])
    return storm(spec=spec,config=config,template=template,directory=directory,device=context["device"])
