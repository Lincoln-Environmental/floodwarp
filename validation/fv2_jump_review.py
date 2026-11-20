"""Matched sections, time refinement, face orientation and shock position on a steady hydraulic jump."""
from pathlib import Path
import os
import numpy as np
from validation.fv1_riemann import crossing,report_json
from validation.fv1_riemann_cases import coordinates,Domain
from validation.fv2_accuracy import configured
from validation.fv2_flow_truth import case_data,bump_state


def trial(*,dx,method,cap_ratio,axis,case,jump,duration,interval,device):
    x,_=coordinates(length=1600.,width=40.,dx=dx)
    d,target=case_data(case=(case,0),x=x,dx=dx,duration=duration,gravity=9.81,head=2.,height=.4,half_width=60.,discharge=1.5,jump=jump,quadrature=16)
    if axis=="y":
        d=Domain(dx=dx,bed=d.bed.T.copy(),initial=d.initial.transpose(1,0,2)[...,[0,2,1]].copy(),duration=duration)
    s=configured(domain=d,method=method,cap=dx*cap_ratio,device=device)
    faces=np.rint((np.array([0.,40.,80.])+800)/dx).astype(int)
    vols=[np.zeros(3)]
    for _ in range(round(duration/interval)):
        s.run_device_loop(duration=interval)
        v=s.get_cumulative_face_volumes()["q"+axis]
        if axis=="y": v=v.T
        vols.append(v[:,faces].sum(axis=0)/40.)
    flux=np.diff(vols,axis=0)/interval
    q=s.get_conservative_state()
    if axis=="y":q=q.transpose(1,0,2)[...,[0,2,1]]
    win=abs(x)<160.
    profile=q[0]
    section=np.diff(flux,axis=0)
    shock=None
    if case=="transcritical_jump":
        _,sides=bump_state(x=np.array([jump-1e-8,jump+1e-8]),regime=case,height=.4,half_width=60.,discharge=1.5,gravity=9.81,jump=jump)
        shock=crossing(x=x[0],profile=profile[:,0],level=np.mean(sides),target=jump)
    return dict(dx=dx,method=method,axis=axis,cap_ratio=cap_ratio,case=case,jump=jump,duration=duration,steps=s.steps,balance=s.cumulative_accounting(),q_rmse=float(np.sqrt(np.mean((flux-1.5)**2))),late_max_error=float(abs(flux[len(flux)//2:]-1.5).max()),late_step_change=float(abs(section[len(section)//2:]).max()),h_rmse=float(np.sqrt(np.mean((q[...,0]-target[...,0])[win]**2))),hu_rmse=float(np.sqrt(np.mean((q[...,1]-1.5)[win]**2))),min_h=float(q[...,0].min()),shock_location=shock,sections=flux.tolist())


if __name__ == "__main__":
    methods=os.environ.get("FV2_JUMP_METHODS","fv1,minmod,mc").split(",")
    tag=os.environ.get("FV2_STUDY_REVISION","baseline")
    directory=Path("validation/fv2_review/regimes")
    records=[]
    for method in methods:
        for dx in (20.,10.,5.,2.5,1.25):
            for axis,ratio,case,jump in (("x",1/160,"transcritical_jump",22.5),("y",1/160,"transcritical_jump",22.5),("x",1/320,"transcritical_jump",22.5),("x",1/160,"transcritical",22.5),("x",1/160,"transcritical_jump",21.875)):
                records.append(trial(dx=dx,method=method,cap_ratio=ratio,axis=axis,case=case,jump=jump,duration=20.,interval=1.,device="cuda:0"))
        print(method,[round(r["q_rmse"],6) for r in records if r["method"]==method and r["axis"]=="x" and r["cap_ratio"]==1/160 and r["case"]=="transcritical_jump" and r["jump"]==22.5],flush=True)
        report_json(path=directory/("jump_"+tag+".json"),content=records)
