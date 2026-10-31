"""Qualify actual-source enhanced CFL against the frozen storage controller.

Run separately from timed external comparisons. The two controllers share the
same equations, source kernels, soil parameters, CFL and minimum timestep.
"""
from pathlib import Path
from time import perf_counter
import os

import numpy as np
import warp as wp

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model, diagnostics
from validation.gpu_benchmark import write_json, device_array_bytes


def build_model(*, shape, factor, controller, device, rainfall, infiltration, maximum_dt,
                minimum_dt, method, initial_depth, soil_settings):
    terrain=terrain_from_array(elevation=np.zeros(shape),dx=float(factor),dy=float(factor),
        row_order='south_to_north',vertical_datum='local')
    options={}
    if method=='exponential':options=dict(infiltration_f0_m_per_s=infiltration,
        infiltration_fd_m=soil_settings['decay_depth_m'],infiltration_k_m_per_s=infiltration)
    if method=='green_ampt':options=dict(infiltration_ks_m_per_s=infiltration,
        infiltration_psi_f_m=soil_settings['suction_m'],infiltration_delta_theta=soil_settings['delta_theta'])
    return create_model(terrain=terrain,device=device,hydraulic_geometry_mode='enhanced',
        subgrid_bed_elevation=np.zeros((shape[0]*factor,shape[1]*factor)),subgrid_resolution_m=1.,
        component_timestep=controller,initial_depth=initial_depth,rainfall=rainfall,infiltration=infiltration,
        infiltration_method=method,explicit_maximum_dt=maximum_dt,explicit_minimum_dt=minimum_dt,
        adaptive_dt_growth_factor=None,**options)


def measure(*, model, duration, graph):
    solver=model.solver
    wp.synchronize_device(device=solver.device)
    start=perf_counter()
    with wp.ScopedTimer(name='hydraulic_measure',print=False,use_nvtx=solver.device.is_cuda):
        result=solver.run_device_loop(duration=duration,use_cuda_graph=graph)
    wp.synchronize_device(device=solver.device)
    elapsed=perf_counter()-start
    ledger=diagnostics(model=model)
    record=dict(steps=result['steps'],wall_seconds=elapsed,hydraulic_seconds=result['loop_seconds'],
        capture_seconds=result['graph_capture_seconds'],dt_min_s=result['dt_min_s'],dt_mean_s=result['dt_mean_s'],
        dt_max_s=result['dt_max_s'],execution=result['execution'],device_array_bytes=device_array_bytes(solver=solver),
        ledger=ledger,minimum_depth_m=float(solver.get_water_depth().min()),maximum_depth_m=float(solver.get_water_depth().max()))
    if solver._component_timestep=='provisional':record['controller']=solver.provisional_timestep_statistics()
    return record


def run(*, shape, factor, controllers, device, rainfall, infiltration, maximum_dt, minimum_dt,
        methods, initial_depth, duration, repeats, graph_modes, output, soil_settings):
    wp.init();output.mkdir(parents=True,exist_ok=True)
    records=[]
    for method in methods:
        for controller in controllers:
            for graph in graph_modes:
                model=build_model(shape=shape,factor=factor,controller=controller,device=device,
                    rainfall=rainfall,infiltration=infiltration,maximum_dt=maximum_dt,minimum_dt=minimum_dt,
                    method=method,initial_depth=initial_depth,soil_settings=soil_settings)
                initial=model.solver.capture_state()
                measure(model=model,duration=duration,graph=graph)
                for repeat in range(repeats):
                    model.solver.restore_state(state=initial)
                    result=measure(model=model,duration=duration,graph=graph)
                    result.update(method=method,component_timestep=controller,graph=graph,repeat=repeat)
                    records.append(result)
                    write_json(path=output/'runs.json',value=records)
                    print(method,controller,graph,repeat,result['steps'],result['hydraulic_seconds'],flush=True)
    summaries=[]
    for method in methods:
        for controller in controllers:
            for graph in graph_modes:
                rows=[row for row in records if row['method']==method and row['component_timestep']==controller and row['graph']==graph]
                summaries.append(dict(method=method,controller=controller,graph=graph,
                    median_hydraulic_seconds=float(np.median([row['hydraulic_seconds'] for row in rows])),
                    hydraulic_range_s=[min(row['hydraulic_seconds'] for row in rows),max(row['hydraulic_seconds'] for row in rows)],
                    steps=rows[0]['steps'],dt_mean_s=rows[0]['dt_mean_s'],device_bytes=rows[0]['device_array_bytes'],
                    maximum_balance_residual_m3=max(abs(row['ledger']['residual_m3']) for row in rows)))
    write_json(path=output/'summary.json',value=dict(shape=shape,factor=factor,duration_s=duration,
        rainfall_m_s=rainfall,infiltration_m_s=infiltration,minimum_dt_s=minimum_dt,maximum_dt_s=maximum_dt,
        initial_depth_m=initial_depth,runs=summaries))


if __name__ == "__main__":
    shape=(8,32)
    factor=4
    controllers=('storage','wave','provisional')
    device=os.environ.get('FLOODWARP_SGS_DEVICE','cuda:0')
    rainfall=.0001
    infiltration=.001
    maximum_dt=60.
    minimum_dt=.001
    methods=('constant','exponential','green_ampt')
    initial_depth=0.
    soil_settings=dict(decay_depth_m=.1,suction_m=.1,delta_theta=.3)
    duration=60.
    repeats=3
    graph_modes=(None,False)
    output=Path('outputs')/('sgs_performance_'+device.replace(':','_'))
    if device=='cpu':
        repeats=1
        graph_modes=(False,)
    wet=os.environ.get('FLOODWARP_SGS_WET','0')=='1'
    if wet:
        shape=(64,128)
        rainfall=0.
        infiltration=0.
        initial_depth=.05
        methods=('none',)
        output=Path('outputs')/('sgs_wet_performance_'+device.replace(':','_'))
    profile=os.environ.get('FLOODWARP_SGS_PROFILE','0')
    if profile!='0':
        controllers=(os.environ.get('FLOODWARP_SGS_PROFILE_CONTROLLER','provisional'),)
        methods=('constant',)
        graph_modes=(None,)
        repeats=1
        duration=6000.
        output=Path('outputs/sgs_profile')
    if profile=='wet':
        shape=(64,128)
        rainfall=0.
        infiltration=0.
        initial_depth=.05
        methods=('none',)
        duration=60.
        graph_modes=(False,)
        output=Path('outputs/sgs_wet_profile')
    run(shape=shape,factor=factor,controllers=controllers,device=device,rainfall=rainfall,
        infiltration=infiltration,maximum_dt=maximum_dt,minimum_dt=minimum_dt,methods=methods,
        initial_depth=initial_depth,duration=duration,repeats=repeats,graph_modes=graph_modes,output=output,
        soil_settings=soil_settings)
