"""FV1 wet dam-break convergence against the independent Stoker solution."""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model, diagnostics
from validation.gpu_benchmark import write_json


def star_residual(depth, *, left_depth, right_depth, gravity):
    rarefaction=2.*(np.sqrt(gravity*depth)-np.sqrt(gravity*left_depth))
    shock=(depth-right_depth)*np.sqrt(.5*gravity*(1./depth+1./right_depth))
    return rarefaction+shock


def stoker_depth(*, x, time, left_depth, right_depth, gravity):
    def residual(depth):
        return star_residual(depth=depth,left_depth=left_depth,right_depth=right_depth,gravity=gravity)

    star=brentq(f=residual,a=right_depth,b=left_depth,xtol=1e-14)
    c_left=np.sqrt(gravity*left_depth);c_star=np.sqrt(gravity*star)
    velocity=2.*(c_left-c_star)
    shock=star*velocity/(star-right_depth)
    speed=x/time
    depth=np.full_like(x,right_depth)
    depth[speed<shock]=star
    rare=(speed>-c_left)&(speed<velocity-c_star)
    depth[rare]=(2.*c_left-speed[rare])**2/(9.*gravity)
    depth[speed<=-c_left]=left_depth
    return depth


def run(*, length, spacing_values, cfl_values, time, left_depth, right_depth, roughness,
        gravity, maximum_dt, minimum_dt, dry_depth, output, evidence):
    output.mkdir(parents=True,exist_ok=True);rows=[];depths={}
    for spacing in spacing_values:
        nx=round(length/spacing)
        if nx>100:raise ValueError('Bounded analytical validation is limited to 100 cells')
        x=(np.arange(nx)+.5)*spacing-length/2.
        expected=stoker_depth(x=x,time=time,left_depth=left_depth,right_depth=right_depth,gravity=gravity)
        initial=np.broadcast_to(np.where(x<0.,left_depth,right_depth),(3,nx)).copy()
        terrain=terrain_from_array(elevation=np.zeros((3,nx)),dx=spacing,dy=spacing,
            row_order='south_to_north',vertical_datum='local')
        for cfl in cfl_values:
            model=create_model(terrain=terrain,solver_method='fv1',initial_depth=initial,manning_n=roughness,
                gravity=gravity,dry_depth=dry_depth,cfl=cfl,explicit_maximum_dt=maximum_dt,
                explicit_minimum_dt=minimum_dt,device='cpu')
            result=model.solver.run_device_loop(duration=time,use_cuda_graph=False)
            depth=model.solver.get_water_depth().copy();error=depth[1]-expected
            balance=diagnostics(model=model)
            key=f'dx{spacing}_cfl{cfl}';depths[key]=depth[1]
            np.savez_compressed(file=output/(key+'.npz'),x=x,depth=depth[1],analytical=expected)
            rows.append(dict(spacing_m=spacing,cfl=cfl,steps=result['steps'],dt_mean_s=result['dt_mean_s'],
                l1_depth_error_m=float(np.mean(np.abs(error))),rmse_m=float(np.sqrt(np.mean(error**2))),
                maximum_depth_error_m=float(np.max(np.abs(error))),minimum_depth_m=float(depth.min()),
                normalised_balance_residual=abs(balance['residual_m3'])/max(1.,model.solver.initial_volume)))
    spatial=[]
    for cfl in cfl_values:
        selected=[row for row in rows if row['cfl']==cfl]
        order=float(np.polyfit(x=np.log([row['spacing_m'] for row in selected]),
            y=np.log([row['l1_depth_error_m'] for row in selected]),deg=1)[0])
        spatial.append(dict(cfl=cfl,observed_l1_order=order,
            errors_m=[row['l1_depth_error_m'] for row in selected]))
    temporal=[]
    for spacing in spacing_values:
        final=depths[f'dx{spacing}_cfl{cfl_values[-1]}']
        temporal.append(dict(spacing_m=spacing,reference_cfl=cfl_values[-1],differences=[dict(cfl=cfl,
            rmse_m=float(np.sqrt(np.mean((depths[f'dx{spacing}_cfl{cfl}']-final)**2)))) for cfl in cfl_values[:-1]]))
    report=dict(solution='Stoker wet dam-break: left rarefaction and right shock',duration_s=time,
        left_depth_m=left_depth,right_depth_m=right_depth,gravity_m_s2=gravity,roughness=roughness,
        rows=rows,spatial=spatial,temporal=temporal,
        interpretation='Spatial order is observed over three grids. CFL refinement compares the same spatial discretisation; shocks reduce global convergence order.')
    write_json(path=output/'report.json',value=report)
    write_json(path=evidence,value=report)


if __name__ == "__main__":
    length=20.
    spacing_values=(1.,.5,.25)
    cfl_values=(.4,.2,.1,.05)
    time=.5
    left_depth=1.
    right_depth=.5
    roughness=1e-8
    gravity=9.80665
    maximum_dt=1.
    minimum_dt=.000001
    dry_depth=.0001
    output=Path('outputs/analytical_convergence')
    evidence=Path('validation/repair_review/analytical_convergence.json')
    run(length=length,spacing_values=spacing_values,cfl_values=cfl_values,time=time,
        left_depth=left_depth,right_depth=right_depth,roughness=roughness,gravity=gravity,
        maximum_dt=maximum_dt,minimum_dt=minimum_dt,dry_depth=dry_depth,output=output,evidence=evidence)
