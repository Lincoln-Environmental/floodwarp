"""Component/controller qualification on bounded synthetic native geometries.

These are internal convergence controls, not independent hydraulic validation.
Both SGS formulations remain experimental regardless of the results here.
"""
from pathlib import Path
from time import perf_counter
import os
import json

import numpy as np
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model,diagnostics
from floodwarp.enhanced_geometry import component_volumes
from floodwarp.rainfall import Hyetograph
from validation.benchmark_restart import publish_json,digest_file,sample_targets


def native_geometry(*,shape,factor,scenario,slope,irregularity):
    y,x=np.indices((shape[0]*factor,shape[1]*factor))
    native=slope*(x.max()-x)
    if scenario in ('irregular','coast','steep'):
        native=native+irregularity*(np.sin(1.3*x)+np.cos(.9*y))
    if scenario=='disconnected':
        native=native+irregularity*np.sin(x)
    mask=np.ones(native.shape,dtype=bool)
    if scenario=='disconnected':mask[native.shape[0]//2,:]=False
    return native,mask


def build_case(*,shape,factor,scenario,controller,device,cap,minimum_dt,
               slope,irregularity,rain_times,rain_rates,soil,initial_stage):
    native,mask=native_geometry(shape=shape,factor=factor,scenario=scenario,
        slope=slope,irregularity=irregularity)
    coarse=native.reshape(shape[0],factor,shape[1],factor).mean(axis=(1,3))
    terrain=terrain_from_array(elevation=coarse,dx=float(factor),dy=float(factor),
        row_order='south_to_north',vertical_datum='local')
    f0,k=soil['initial_m_s'],soil['conductivity_m_s']
    if scenario=='infiltrating':f0=k=soil['fully_infiltrating_m_s']
    model=create_model(terrain=terrain,device=device,hydraulic_geometry_mode='enhanced',
        subgrid_bed_elevation=native,subgrid_resolution_m=1.,native_active_mask=mask,
        subgrid_manning_n=.035+np.indices(native.shape)[1]*.0001,
        component_timestep=controller,maximum_stage_m=float(native.max()+2.),
        initial_depth=0.,infiltration_method='exponential',
        infiltration_f0_m_per_s=f0,infiltration_fd_m=soil['decay_depth_m'],
        infiltration_k_m_per_s=k,explicit_maximum_dt=cap,explicit_minimum_dt=minimum_dt,
        adaptive_dt_growth_factor=None,track_max_water_depth=True)
    solver=model.solver
    if scenario=='wet_dry':
        initial=np.where(mask,np.maximum(initial_stage-native,0.),0.)
        solver.set_component_state(volume=component_volumes(geometry=solver._enhanced_geometry,native_depth=initial))
    if scenario=='coast':
        boundary=solver.boundary_type_qy.numpy();boundary[-1,:]=3
        solver.set_boundary_type_qy(boundary_type=boundary)
    rain=Hyetograph(node_times_s=rain_times,node_rates_m_per_s=rain_rates,
        storm_type='controller_qualification',metadata={})
    return model,rain


def hydraulic_groups(*,geometry):
    ends=geometry.endpoints
    internal=ends[(ends[:,0]>=0)&(ends[:,1]>=0)]
    count=len(geometry.state_cell)
    row=np.r_[internal[:,0],internal[:,1]]
    col=np.r_[internal[:,1],internal[:,0]]
    matrix=coo_matrix((np.ones(len(row)),(row,col)),shape=(count,count)).tocsr()
    return connected_components(csgraph=matrix,directed=False)[1]


def simulate(*,model,rain,scenario,duration,interval,stage_times,stage_values,graph):
    solver=model.solver;geometry=solver._enhanced_geometry
    initial=solver.get_component_state()['volume_m3'].copy()
    groups=hydraulic_groups(geometry=geometry)
    group_count=int(groups.max()+1)
    source_area=geometry.source_area
    options=dict(stage_series_qy=(stage_times,stage_values)) if scenario=='coast' else {}
    rows=[];stats=[];maximum_group_residual=0.;maximum_reconstruction_error=0.
    started=perf_counter()
    for target in sample_targets(duration=duration,interval=interval):
        stats.append(solver.run_device_loop(duration=float(target-solver.simulation_time),
            rainfall_hyetograph=rain,use_cuda_graph=graph,**options))
        balance=diagnostics(model=model);state=solver.get_component_state()
        reconstructed=component_volumes(geometry=geometry,native_depth=solver.get_native_water_depth())
        maximum_reconstruction_error=max(maximum_reconstruction_error,
            float(np.max(np.abs(reconstructed-state['volume_m3']))))
        if scenario!='coast':
            supplied=rain.integrated_depth_between(start_time_s=0.,end_time_s=float(target))*source_area
            residual=state['volume_m3']-initial-supplied+state['cumulative_infiltration_m']*source_area
            grouped=np.bincount(groups,weights=residual,minlength=group_count)
            scale=np.bincount(groups,weights=initial+supplied,minlength=group_count)
            maximum_group_residual=max(maximum_group_residual,float(np.max(np.abs(grouped)/np.maximum(1.,scale))))
        assert abs(balance['residual_m3'])<=1e-8*max(1.,solver.initial_volume+balance['rain']+balance['boundary_inflow'])
        depth=solver.get_native_water_depth()
        assert np.isfinite(depth).all() and depth.min()>=0.
        rows.append(dict(seconds=float(target),**balance))
    assert maximum_group_residual<=1e-8
    assert maximum_reconstruction_error<=1e-8*max(1.,float(initial.sum())+rows[-1]['rain']+rows[-1]['boundary_inflow'])
    record=dict(steps=sum(row['steps'] for row in stats),
        hydraulic_seconds=sum(row['loop_seconds'] for row in stats),
        wall_seconds=perf_counter()-started,dt_min_s=min(row['dt_min_s'] for row in stats),
        dt_max_s=max(row['dt_max_s'] for row in stats),groups=group_count,
        component_count=solver._component_count,connection_count=solver._connection_count,
        maximum_normalised_group_balance_residual=maximum_group_residual if scenario!='coast' else None,
        maximum_component_reconstruction_error_m3=maximum_reconstruction_error,
        ledger=rows,final_balance=rows[-1])
    if solver._component_timestep=='provisional':record['controller_statistics']=solver.provisional_timestep_statistics()
    return record,solver.get_native_water_depth().copy(),solver.get_component_state()


def compare(*,candidate,reference,spacing,absolute_depth,relative_depth):
    error=candidate-reference
    rmse=float(np.sqrt(np.mean(error**2)))
    relative=float(np.linalg.norm(error)/max(1e-12,np.linalg.norm(reference)))
    return dict(depth_rmse_m=rmse,depth_relative_l2=relative,
        depth_max_error_m=float(np.max(np.abs(error))),
        storage_error_m3=float(error.sum()*spacing**2),
        depth_accepted=bool(rmse<=absolute_depth or relative<=relative_depth))


def qualify(*,shape,factors,scenarios,controllers,caps,device,minimum_dt,duration,interval,
            slope,steep_slope,irregularity,rain_times,rain_rates,soil,initial_stage,
            stage_times,stage_values,output,absolute_depth,relative_depth):
    wp.init();output.mkdir(parents=True,exist_ok=True)
    source={path.name:digest_file(path=path) for path in Path('src/floodwarp').glob('*.py')}
    signature=dict(shape=shape,factors=factors,scenarios=scenarios,controllers=controllers,
        caps=caps,device=device,minimum_dt=minimum_dt,duration=duration,interval=interval,
        slope=slope,steep_slope=steep_slope,irregularity=irregularity,soil=soil,
        rain_times=rain_times.tolist(),rain_rates=rain_rates.tolist(),initial_stage=initial_stage,
        stage_times=stage_times.tolist(),stage_values=stage_values.tolist(),source=source,
        driver_sha256=digest_file(path=Path(__file__)),absolute_depth_m=absolute_depth,relative_depth=relative_depth)
    signature=json.loads(json.dumps(signature))
    (output/('driver-'+signature['driver_sha256']+'.py')).write_bytes(Path(__file__).read_bytes())
    results=[]
    for factor in factors:
        for scenario in scenarios:
            measurements={};depths={}
            for controller in controllers:
                for cap in caps:
                    name=f'{scenario}-ratio{factor}-{controller}-cap{cap}'
                    path=output/name
                    record_path=path/'record.json'
                    if record_path.exists():
                        record=json.loads(record_path.read_text())
                        # An orchestration-only recovery can reuse measured
                        # physics when every runtime source/input is unchanged.
                        previous=dict(record['signature']);current=dict(signature)
                        previous.pop('driver_sha256');current.pop('driver_sha256')
                        if previous!=current:raise ValueError('SGS qualification source/configuration drift')
                        depth=None
                        if record.get('status','complete')=='complete':
                            with np.load(file=path/'fields.npz') as saved:depth=saved['depth'].copy()
                    else:
                        model,rain=build_case(shape=shape,factor=factor,scenario=scenario,controller=controller,
                            device=device,cap=cap,minimum_dt=minimum_dt,slope=steep_slope if scenario=='steep' else slope,
                            irregularity=irregularity,rain_times=rain_times,rain_rates=rain_rates,soil=soil,initial_stage=initial_stage)
                        path.mkdir(parents=True,exist_ok=True)
                        try:
                            record,depth,state=simulate(model=model,rain=rain,scenario=scenario,duration=duration,
                                interval=interval,stage_times=stage_times,stage_values=stage_values,graph=None)
                        except (ArithmeticError,RuntimeError,AssertionError) as error:
                            record=dict(status='failed',error=type(error).__name__+': '+str(error),
                                time_s=model.solver.simulation_time,hydraulic_seconds=None,steps=None)
                            depth=None
                            np.savez_compressed(file=path/'failure_state.npz',**model.solver.get_component_state())
                        else:
                            record['status']='complete'
                            np.savez_compressed(file=path/'fields.npz',depth=depth,**state)
                        record['signature']=signature
                        publish_json(path=record_path,value=record)
                    measurements[controller,cap]=record;depths[controller,cap]=depth
                    print(name,record['steps'],record['hydraulic_seconds'],flush=True)
            reference=depths['provisional',caps[-1]]
            for controller in controllers:
                row=dict(scenario=scenario,factor=factor,controller=controller,
                    runs=[dict(cap_s=cap,**measurements[controller,cap]) for cap in caps],
                    convergence=[dict(cap_s=cap,**compare(candidate=depths[controller,cap],reference=reference,
                        spacing=1.,absolute_depth=absolute_depth,relative_depth=relative_depth))
                        if depths[controller,cap] is not None and reference is not None else
                        dict(cap_s=cap,status='unavailable_due_to_failed_run') for cap in caps])
                results.append(row)
            publish_json(path=output/'summary.json',value=dict(signature=signature,results=results,
                independent_hydraulic_validation=False,sgs_status='experimental'))


if __name__ == "__main__":
    shape=(3,4)
    factors=(2,4)
    scenarios=('infiltrating','ponding','wet_dry','disconnected','irregular','coast','steep')
    controllers=('storage','wave','provisional')
    caps=(5.,.05,.01)
    device=os.environ.get('FLOODWARP_SGS_DEVICE','cpu')
    minimum_dt=.001
    duration=20.
    interval=2.
    slope=.02
    steep_slope=.2
    irregularity=.08
    rain_times=np.array([0.,6.,12.,20.])
    rain_rates=np.array([0.,.001,.001,0.])
    soil=dict(initial_m_s=.0015,conductivity_m_s=.0001,fully_infiltrating_m_s=.002,decay_depth_m=.002)
    initial_stage=.15
    stage_times=np.array([0.,6.,12.,20.])
    stage_values=np.array([0.,.3,.1,0.])
    output=Path(os.environ.get('FLOODWARP_SGS_OUTPUT',str(Path('outputs')/('sgs_qualification_'+device.replace(':','_')))))
    absolute_depth=.001
    relative_depth=.02
    qualify(shape=shape,factors=factors,scenarios=scenarios,controllers=controllers,caps=caps,
        device=device,minimum_dt=minimum_dt,duration=duration,interval=interval,slope=slope,
        steep_slope=steep_slope,irregularity=irregularity,rain_times=rain_times,rain_rates=rain_rates,
        soil=soil,initial_stage=initial_stage,stage_times=stage_times,stage_values=stage_values,
        output=output,absolute_depth=absolute_depth,relative_depth=relative_depth)
