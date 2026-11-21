"""Analytical fields, gauges and warmed timing controls."""

from functools import partial
import hashlib
import json
from unittest.mock import patch
import numpy as np
import warp as wp

from validation import fv1_diffusion_matrix as matrix
from validation.fv1_diffusion_domains import design, Case
from validation.fv1_diffusion_truth import run
from validation.fv1_diffusion_time import temporal
from validation.fv2_accuracy import convergence
from validation.fv2_local_inertia import build
from research.floodwarp_methods_paper.history import checked, history


def production_build(**kwargs):
    solver=build(**kwargs)
    method=kwargs["method"]
    if method in ("fv1","low_froude_hll","hancock_adaptive"):
        solver.configure_execution(mode="tiles" if method=="hancock_adaptive" else "sparse")
    return solver


def evaluate(*,stage,spec,study,root,config,context,directory):
    settings=dict(config["analytical"],device=context["device"])
    if spec["kind"]=="archived-controls":
        run(output=directory,methods=("fv1","low_froude_hll","euler_mc","hancock_adaptive","adaptive"),spacings=tuple(settings["spacings"]),grids=(16,32,64),device=settings["device"],integrations=(),time_steps=(),truth_cases=(("sloping_ritter",0.05),("subcritical",0.0),("transcritical",0.0),("transcritical_jump",0.0)),wave_cases=("lake","ritter","stoker","shear","thacker","jump","jump_offset"))
        return dict(reference="independent exact SWEs, gauges, quadrature and radial solutions",files=sorted(f.name for f in directory.glob("*.json") if f.name!="attempt.json"))
    if spec["kind"]=="temporal":
        case=Case("accelerating","sheet",0.05,0.08,0.12,0.3,0.0,0.0,400.,40.,4.,0.,(0.,))
        temporal(case=case,dx=5.,timesteps=settings["temporal_steps"],methods=("fv1","low_froude_hll","euler_mc","hancock_adaptive","adaptive"),gravity=settings["gravity"],device=settings["device"],output=directory/"fixed_grid.json")
        rows={}
        for integ in ("euler","hancock","rk2"):
            rows[integ]=convergence(grids=(16,32,64,128),temporal_grid=16,timesteps=(1/64,1/128,1/256,1/512),length=8.,duration=0.5,device=settings["device"],bed_amplitude=0.04,limiter="adaptive" if integ!="euler" else "mc",integrator=integ)
        return dict(manufactured=rows)
    case=next(c for c in design() if c.name==spec["case"])
    opts=dict(case=case,dx=spec["dx"],method=spec["method"],policy=spec["policy"],settings=settings,multiplier=1.0)
    with patch.object(matrix,"build",production_build):
        row=matrix.forecast(**opts,repeats=settings["repeats"] if stage=="analytical" else 1)
        solver,truth,qtruth,t,x=matrix.initialise(**opts)
        with history(solver=solver,capacity=row["steps"]) as get_dt:
            states,q,stats=matrix.sample(case=case,solver=solver,states=truth,times=t,x=x,dx=spec["dx"],li=spec["method"]=="li")
            dt=get_dt()
    checked(times=dt,count=row["steps"],duration=case.duration)
    np.testing.assert_array_equal(q,row["model_m2_s"])
    arrays=dict(x_m=x,t_s=t,model_centreline=states,exact_cell_averages=truth,model_gauges=q,exact_gauges=qtruth,accepted_dt_s=dt)
    np.savez_compressed(directory/"fields.npz",**arrays)
    bed=solver.get_bed_elevation() if hasattr(solver,"get_bed_elevation") else None
    if bed is None and hasattr(solver,"_fields"):bed=solver._fields["bed"].numpy()
    if bed is None: bed=solver.bed_elevation.numpy()
    row.update(accepted_dt_sha256=hashlib.sha256(dt.tobytes()).hexdigest(),terrain_sha256=hashlib.sha256(np.asarray(bed).tobytes()).hexdigest(),shape=[solver.ny,solver.nx],config=solver.numerical_configuration(),duration_s=case.duration,initial_m3=solver.initial_volume,ledger=solver.cumulative_accounting(),pool_mib=wp.get_mempool_used_mem_current(device=context["device"])/2**20)
    return row
