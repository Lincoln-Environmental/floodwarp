"""Matched physical inputs and synchronised hydraulic measurements for large domains."""
import json
from time import perf_counter
from unittest.mock import patch

import numpy as np
import warp as wp
from scipy.ndimage import label

from floodwarp.model import advance, diagnostics
from research.floodwarp_methods_paper import events
from research.floodwarp_methods_paper.terrain.generators.generate import fields, array_sha
from validation.fv2_local_inertia import read_state
from validation.hancock_long_storm_review.switching import work_fraction


def represented(*, options, settings, design, template, family, seed, initial, dx, dem):
    cfg=dict(design,dimensions_m=settings["dimensions"])
    (z,h,hu,u,v),params=fields(design=cfg,template=template,family=family,seed=seed,initial=initial,dx=dx,dem=dem)
    if settings["controls"]:
        length,width=settings["dimensions"]
        x=(np.floor(u*length/dem)+0.5)*dem
        y=(np.floor(v*width/dem)+0.5)*dem
        shoulder=options["family_control_halfwidth_m"]
        height=options["control_bank_height_m"]
        # A continuous narrow outlet connection through a low transverse embankment.
        thalweg=width*(0.47+0.10*np.sin(2*np.pi*x/length+np.random.default_rng(params["phase_seed"]).uniform(-0.3,0.3)))
        gap=np.exp(-((y-thalweg)/shoulder)**2)
        bank=np.exp(-((x-0.875*length)/shoulder)**2)
        z=z+height*bank*(1-gap)-height*gap*np.exp(-((x/length-0.875)/0.06)**2)
    if dem!=dx and initial!="dry" and settings["initial_grid"]=="sampled":
        (_,h,hu,_,_),_=fields(design=cfg,template=template,family=family,seed=seed,initial=initial,dx=dx,dem=dx)
    return (z,h,hu,u,v),params


def average(*, a, factor):
    ny,nx=a.shape
    if nx%factor or ny%factor:
        raise ValueError("Diagnostic support must divide the physical domain")
    return a.reshape(ny//factor,factor,nx//factor,factor).mean(axis=(1,3))


def volumes(*, solver, li):
    field=solver.cumulative_net_qx_volume_m3 if li else solver._fields["volume_x"]
    return field.numpy().reshape(solver.ny,solver.nx+1)[:,[solver.nx//2,-1]].sum(axis=0)


def run_event(*,spec,config,template,options,directory,device,profile=False):
    start=perf_counter()
    def geometry(**kwargs):
        return represented(options=options,settings=spec,**kwargs)
    with patch.object(events,"fields",geometry):
        model,expected,forcing=events.create_event(config=config,template=template,spec=spec,device=device)
    solver=model.solver
    method,mode,_=events.numerical(method=spec["method"])
    li=method=="li"
    wp.synchronize_device(device=device)
    setup_s=perf_counter()-start
    tmax=int(spec.get("end_s",spec["hours"]*3600))
    interval=spec["interval"]
    last=0
    frames=[]
    spatial=[]
    transfer=volumes(solver=solver,li=li)
    fine=spec["dx"]
    factor=int(round(spec["support"]/fine))
    if not np.isclose(factor*fine,spec["support"]):
        raise ValueError("Nonintegral evaluation support")
    rain=solver._fields["multiplier"].numpy().reshape(solver.ny,solver.nx)>0 if not li else None
    if spec["coastal_stage"]:
        shape=(solver.ny,solver.nx+1)
        levels=options["coastal_stage_m"]
        knots=options["coastal_stage_hours"]
        forcing.update(coastal_stage_hours=knots,coastal_stage_m=levels,stage_hold_s=900)
    hydraulic=0.
    total_capture=0.
    total_preload=0.
    extracted=0.
    probe=None
    if profile:
        from validation.hancock_long_storm_review.graph_profile import GraphTimer
        probe=GraphTimer(device=device,limit=256)
    targets=range(interval,tmax+1,interval)
    for end in targets:
        if spec["coastal_stage"] and last%900==0:
            level=float(np.interp(last/3600,knots,levels))
            solver.set_boundary_stage_qx(stage=np.full(shape,level))
        tick=perf_counter()
        if probe is None:
            stats=advance(model=model,end_time=float(end))
        else:
            with patch.object(wp,"launch",probe.launch):
                stats=advance(model=model,end_time=float(end))
        wp.synchronize_device(device=device)
        capture=stats["graph_capture_seconds"]
        preload=stats.get("compilation_seconds",0.)
        wall=perf_counter()-tick
        total_capture+=capture
        total_preload+=preload
        hydraulic+=wall-capture-preload
        tick=perf_counter()
        current=volumes(solver=solver,li=li)
        q=(current-transfer)/(end-last)
        transfer=current
        frames.append(dict(t_s=end,section_m3s=float(q[0]),outlet_m3s=float(q[1]),
                           hydraulic_s=wall-preload-capture,graph=stats.get("cuda_graph",stats.get("execution")=="conditional_cuda"),
                           steps=stats["steps"], rejected=stats.get("rejected_cfl_trials",0),
                           dt_min_s=stats["dt_min_s"],dt_max_s=stats["dt_max_s"]))
        if end%options["spatial_output_s"]==0 or end==tmax:
            h=solver.get_water_depth()
            if not np.isfinite(h).all() or np.min(h)<0:
                raise ArithmeticError("Invalid hydraulic depth")
            info={} if li else solver.execution_diagnostics()
            ledger=diagnostics(model=model)
            spatial.append(dict(t_s=end,wet_fraction=float((h>0).mean()),cm_fraction=float((h>=.01).mean()),
                                active_fraction=info.get("evaluated_cells",h.size)/h.size,
                                tile_work_fraction=work_fraction(wet=h>0,rain=rain if end<21600 and rain is not None else False) if not li else None,
                                residual_m3=ledger["residual_m3"],storage_m3=ledger["storage_m3"],
                                rain_m3=ledger["rain"], infiltration_m3=ledger["infiltration"]))
        extracted+=perf_counter()-tick
        last=end
    if last!=tmax:
        raise RuntimeError("Incomplete hydraulic horizon")
    state=read_state(solver=solver,li=li)
    if not np.isfinite(state).all():
        raise ArithmeticError("Nonfinite final state")
    maximum=solver.get_max_water_depth()
    arrays=dict(depth=average(a=h,factor=factor),maximum=average(a=maximum,factor=factor))
    components={}
    for cm in (1,5,10):
        wet=h>=cm/100
        arrays[f"wet_fraction_{cm}cm"]=average(a=wet,factor=factor)
        components[str(cm)]=int(label(wet)[1])
    np.savez_compressed(directory/"spatial.npz",**arrays)
    hashes=dict(state=array_sha(state),maximum=array_sha(maximum))
    if not li:
        for key in ("volume_x","volume_y","soil","rain_sum"):
            if key in solver._fields:
                hashes[key]=array_sha(solver._fields[key].numpy())
    steps=sum(f["steps"] for f in frames)
    supply=max(1.,solver.initial_volume+ledger["rain"]+ledger["boundary_inflow"])
    if max(abs(f["residual_m3"]) for f in spatial)>1e-8*supply:
        raise ArithmeticError("Water balance exceeded the registered tolerance")
    if tmax>=21600 and abs(ledger["rain"]-expected)>1e-8*max(1.,expected):
        raise ArithmeticError("Rainfall integral differs from prescribed input")
    pool=wp.get_mempool_used_mem_current(device=device)/2**20
    numerical=forcing.pop("numerical")
    return dict(forcing=forcing,numerical=numerical,method=method,execution=mode,cells=h.size,
                hours_completed=last/3600,expected_full_rain_m3=expected,rain_m3=ledger["rain"],infiltration_m3=ledger["infiltration"],
                inflow_m3=ledger["boundary_inflow"],outflow_m3=ledger["boundary_outflow"],initial_m3=solver.initial_volume,
                final_storage_m3=ledger["storage_m3"],residual_m3=ledger["residual_m3"],steps=steps,
                rejected=sum(f["rejected"] for f in frames),dt_mean_s=last/steps,dt_min_s=min(f["dt_min_s"] for f in frames),
                wet_fraction=float(np.mean([f["wet_fraction"] for f in spatial])),
                scheduled_fraction=float(np.mean([f["active_fraction"] for f in spatial])),
                setup_s=setup_s,hydraulic_s=hydraulic,preload_s=total_preload,graph_capture_s=total_capture,
                extraction_s=extracted,end_to_end_s=perf_counter()-start,live_pool_mib=pool,
                stage_seconds=probe.summary() if probe else {},hashes=hashes,components=components,
                frames=frames,spatial=spatial)

