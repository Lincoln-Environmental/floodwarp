"""Defect reproductions with hydraulic tolerances and strict balance checks."""
import numpy as np
import pytest
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_kernels import _green_ampt_potential_depth
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.infiltration import green_ampt_infiltration_increment_m
from floodwarp.full_swe_reference import face_fluxes
from floodwarp.inputs import terrain_from_array, boundary_faces
from floodwarp.model import create_model, advance, diagnostics, save_checkpoint, load_checkpoint
from floodwarp.enhanced_geometry import StageRangeError
from tests.test_enhanced_production_state import create
from tests.test_masked_free import _build_orientation_solver


@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param!='cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def flat_model(*, device, method='li', **options):
    terrain=terrain_from_array(elevation=np.zeros((2,4)),dx=1.,dy=1.,
        row_order='south_to_north',vertical_datum='local')
    return create_model(terrain=terrain,device=device,solver_method=method,**options)


@pytest.mark.parametrize('graph',[False,None])
@pytest.mark.parametrize('profile',[None,'lisflood_fp_acc'])
def test_stage_pulse_is_sampled_before_cfl_and_nodes_are_visited(device,graph,profile):
    model=flat_model(device=device,explicit_maximum_dt=60.,adaptive_dt_growth_factor=None,validation_profile=profile)
    solver=model.solver
    boundary=solver.boundary_type_qx.numpy().copy();boundary[:,0]=3
    solver.set_boundary_type_qx(boundary_type=boundary)
    result=solver.run_device_loop(duration=4.,use_cuda_graph=graph,
        stage_series_qx=(np.array([0.,1.,2.,4.]),np.array([0.,1.,0.,0.])),dt_history_capacity=1000)
    times=np.cumsum(result['dt_history'])
    assert np.min(np.abs(times-1.))<1e-12 and np.min(np.abs(times-2.))<1e-12
    assert solver.cumulative_accounting()['boundary_inflow']>.001
    peak_index=int(np.argmin(np.abs(times-1.)))+1
    assert result['dt_history'][peak_index] <= float(solver.cfl_number)/np.sqrt(float(solver.gravity))*(1+1e-12)
    assert abs(diagnostics(model=model)['residual_m3'])<1e-10


@pytest.mark.parametrize('rates',[[.1,.1],[0.,.2]])
def test_first_timestep_accounts_for_new_rainfall(device,rates):
    model=flat_model(device=device,explicit_maximum_dt=60.,adaptive_dt_growth_factor=None)
    from floodwarp.rainfall import Hyetograph
    rain=Hyetograph(node_times_s=np.array([0.,10.]),node_rates_m_per_s=np.asarray(rates),storm_type='test',metadata={})
    result=model.solver.run_device_loop(duration=10.,rainfall_hyetograph=rain,
        use_cuda_graph=None,dt_history_capacity=10000)
    first=result['dt_history'][0]
    upper=float(model.solver.cfl_number)/np.sqrt(float(model.solver.gravity)*max(rates)*10.)
    assert first<=upper*(1+1e-12)
    ledger=diagnostics(model=model)
    assert abs(ledger['residual_m3'])<1e-9*max(1.,ledger['rain'])
    assert ledger['rain']==pytest.approx(8.*10.*np.mean(rates),rel=1e-12)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('side',[0,1])
def test_fv1_free_blocking_has_continuous_pressure(axis,side):
    state=np.zeros((2,2,3));state[...,0]=1.
    shape=(3,2) if axis==0 else (2,3)
    boundary=np.ones(shape,dtype=np.int32)
    for inward in (0.,1e-12,1e-7):
        state[...,2 if axis==0 else 1]=inward*(1. if side==0 else -1.)
        boundary[:]=1
        if axis==0:boundary[side*2,:]=2
        else:boundary[:,side*2]=2
        flux,_,_=face_fluxes(state=state,bed=np.zeros((2,2)),active=np.ones((2,2),bool),
            boundary_type=boundary,boundary_stage=np.zeros(shape),axis=axis,gravity=9.80665,dry_depth=1e-6)
        selected=flux[side*2,:] if axis==0 else flux[:,side*2]
        np.testing.assert_allclose(selected[...,0],0.,atol=1e-14)
        np.testing.assert_allclose(selected[...,2 if axis==0 else 1],.5*9.80665,atol=1e-12)


@wp.kernel
def green_ampt_values(inputs: wp.array2d(dtype=WP_FLOAT), output: wp.array(dtype=WP_FLOAT)):
    i=wp.tid()
    output[i]=_green_ampt_potential_depth(inputs[i,0],inputs[i,1],inputs[i,2],inputs[i,3],inputs[i,4])


def test_green_ampt_converged_root_is_retained(device):
    samples=np.array([[f,k,.1,.3,dt] for f in (0.,1e-6,.001,.1,10.)
        for k in (1e-9,1e-5,.001) for dt in (.001,1.,100.)])
    output=wp.zeros(len(samples),dtype=WP_FLOAT,device=device)
    wp.launch(kernel=green_ampt_values,dim=len(samples),
        inputs=[wp.array(samples,dtype=WP_FLOAT,device=device),output],device=device)
    expected=green_ampt_infiltration_increment_m(cumulative_infiltration_m=samples[:,0],
        ks_m_per_s=samples[:,1],wetting_front_suction_m=samples[:,2],delta_theta=samples[:,3],dt_s=samples[:,4])
    np.testing.assert_allclose(output.numpy(),expected,atol=1e-12,rtol=1e-9)


@pytest.mark.parametrize('orientation',['west','east','north','south'])
@pytest.mark.parametrize('mode',['coarse','traditional','enhanced'])
@pytest.mark.parametrize('stage',[0.,.6])
def test_all_masked_stages_exchange_and_balance(device,orientation,mode,stage):
    solver=_build_orientation_solver(orientation=orientation,device=device,mode='coarse' if mode=='enhanced' else mode)
    if mode=='enhanced':
        original=solver
        active=original.active_mask.numpy().copy()
        native=np.where(active,0.,50.).repeat(2,axis=0).repeat(2,axis=1)
        solver=LocalInertialSolver(ny=4,nx=4,dx=5.,dy=5.,bed_elevation=np.zeros((4,4)),
            manning_n=np.full((4,4),.04),active_mask=active,device=device,maximum_dt=.05,
            hydraulic_geometry_mode='enhanced',subgrid_bed_elevation=native,subgrid_resolution_m=2.5)
        solver.set_boundary_type_qx(boundary_type=original.boundary_type_qx.numpy().copy())
        solver.set_boundary_type_qy(boundary_type=original.boundary_type_qy.numpy().copy())
        solver.set_water_depth(depth=original.get_water_depth().copy())
    codes={'west':(6,5),'east':(7,10),'south':(8,11),'north':(9,4)}
    old,new=codes[orientation]
    for axis in ('x','y'):
        boundary=getattr(solver,'boundary_type_q'+axis).numpy().copy()
        boundary[boundary==old]=new
        getattr(solver,'set_boundary_type_q'+axis)(boundary_type=boundary)
        getattr(solver,'set_boundary_stage_q'+axis)(stage=np.full(boundary.shape,stage))
    solver.run_device_loop(duration=.05,use_cuda_graph=None)
    ledger=solver.cumulative_accounting()
    storage=float(solver.get_water_depth().sum())*float(solver.dx)*float(solver.dy)
    assert ledger['boundary_outflow' if stage==0. else 'boundary_inflow']>0.
    assert abs(solver.initial_volume+ledger['boundary_inflow']-ledger['boundary_outflow']-storage)<1e-9*max(1.,solver.initial_volume)
    assert np.isfinite(solver.get_water_depth()).all() and solver.get_water_depth().min()>=0.


@pytest.mark.parametrize('method',['li','fv1'])
@pytest.mark.parametrize('failure',['write','publish'])
def test_interrupted_checkpoint_preserves_last_restart(method,failure,tmp_path,monkeypatch):
    import floodwarp.model as api
    model=flat_model(device='cpu',method=method,rainfall=.001,explicit_maximum_dt=.05)
    path=tmp_path/'state.npz'
    advance(model=model,end_time=.1)
    save_checkpoint(model=model,path=path)
    last=path.read_bytes()
    advance(model=model,end_time=.2)
    replace=api.os.replace
    save=api.np.savez

    def interrupted_replace(src,dst):
        if dst==path:
            raise OSError('interrupted publication')
        return replace(src=src,dst=dst)

    def interrupted_save(file,**arrays):
        file.write(b'incomplete archive')
        raise OSError('interrupted write')

    with monkeypatch.context() as patch:
        if failure=='publish':patch.setattr(api.os,'replace',interrupted_replace)
        else:patch.setattr(api.np,'savez',interrupted_save)
        with pytest.raises(OSError,match='interrupted'):
            save_checkpoint(model=model,path=path)
    assert path.read_bytes()==last
    fresh=flat_model(device='cpu',method=method,rainfall=.001,explicit_maximum_dt=.05)
    load_checkpoint(model=fresh,path=path)
    assert diagnostics(model=fresh)['time_s']==pytest.approx(.1)


@pytest.mark.parametrize('setter',['depth','stage','component'])
def test_enhanced_stage_rejection_is_transactional(device,setter):
    solver=create(bed=np.zeros((2,4)),factor=2,device=device)
    solver.set_water_depth(depth=np.full((1,2),.1));solver.step(fixed_dt=.01)
    before=solver.capture_state()
    with pytest.raises(StageRangeError):
        if setter=='depth':solver.set_water_depth(depth=np.full((1,2),3.))
        elif setter=='stage':solver.set_water_surface_elevation(stage=np.full((1,2),3.))
        else:solver.set_component_state(volume=np.full(solver._component_count,12.))
    assert solver.simulation_time==before['host']['simulation_time']
    for name,array in before['device'].items():
        if array is not None:np.testing.assert_array_equal(getattr(solver,name).numpy(),array.numpy(),err_msg=name)


def test_native_stage_override_kind_survives_geometry_rebuild(device):
    ports=np.zeros((2,4),bool);ports[-1,:]=True
    solver=create(bed=np.zeros((2,4)),factor=2,device=device,native_north_outlets=ports)
    values=np.full(solver._connection_count,np.nan)
    values[solver._enhanced_geometry.endpoints[:,1]<0]=.2
    solver.set_component_boundary_stages(stage_m=values,native_contacts=True)
    original={tuple(key):value for key,value in zip(solver._enhanced_geometry.connection_keys,values,strict=True) if np.isfinite(value)}
    boundary=solver.boundary_type_qx.numpy().copy();boundary[:,0]=3
    solver.set_boundary_type_qx(boundary_type=boundary)
    for key,kind,head in zip(solver._enhanced_geometry.connection_keys,solver._ec_boundary_override.numpy(),solver.get_component_boundary_stages(),strict=True):
        if tuple(key) in original:
            assert kind==2 and head==pytest.approx(original[tuple(key)])


@pytest.mark.parametrize('enhanced',[False,True])
def test_lts_graph_recaptures_changed_parameters_and_buffers(enhanced):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    if enhanced:
        solver=create(bed=np.zeros((2,4)),factor=2,device='cuda:0',track_max_water_depth=False,track_cumulative_face_volumes=False)
    else:
        solver=flat_model(device='cuda:0',initial_depth=.1,explicit_maximum_dt=.02).solver
    solver.run_spatially_adaptive_loop(duration=.04,max_levels=0,wet_max_level=0,use_device_graph=True)
    first=solver._lts_subcycle_graph
    solver.gravity=WP_FLOAT(8.)
    solver.run_spatially_adaptive_loop(duration=.04,max_levels=0,wet_max_level=0,use_device_graph=True)
    assert solver._lts_subcycle_graph is not first
    second=solver._lts_subcycle_graph
    solver.rainfall_rate=wp.clone(solver.rainfall_rate)
    solver.run_spatially_adaptive_loop(duration=.04,max_levels=0,wet_max_level=0,use_device_graph=True)
    assert solver._lts_subcycle_graph is not second


@pytest.mark.parametrize('method',['li','fv1'])
def test_public_api_automatically_selects_cuda_graphs(method):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    model=flat_model(device='cuda:0',method=method,explicit_maximum_dt=.02)
    result=advance(model=model,end_time=.1)
    if method=='fv1':assert result['cuda_graph']
    else:assert result['execution']=='conditional_cuda'


@pytest.mark.parametrize('method',['constant','exponential','green_ampt'])
def test_provisional_controller_removes_fully_infiltrated_rain_bound(device,method):
    options={}
    if method=='exponential':options=dict(infiltration_f0_m_per_s=.002,infiltration_fd_m=.1,infiltration_k_m_per_s=.001)
    if method=='green_ampt':options=dict(infiltration_ks_m_per_s=.001,infiltration_psi_f_m=.1,infiltration_delta_theta=.3)
    solver=create(bed=np.zeros((2,4)),factor=2,device=device,method=method,
        component_timestep='provisional',maximum_dt=60.,adaptive_dt_growth_factor=None,**options)
    solver.set_source_rates(rain=np.full((1,2),.0001),infil=np.full((1,2),.001))
    result=solver.run_device_loop(duration=120.,use_cuda_graph=None)
    assert result['steps']==2
    ledger=solver.cumulative_accounting()
    assert ledger['infiltration']==pytest.approx(ledger['rain'],rel=1e-12,abs=1e-12)
    np.testing.assert_allclose(solver.get_water_depth(),0.,atol=1e-12)


@pytest.mark.parametrize('threshold',[.01,.10])
def test_engineering_extent_ignores_micrometre_ties_but_keeps_real_errors(threshold):
    from validation.gpu_benchmark import physical_metrics
    reference=np.array([[threshold-5e-7,threshold+.02]])
    candidate=np.array([[threshold+5e-7,threshold+.02]])
    ledger=[dict(seconds=60.,outflow_m3=0.,inflow_m3=0.,rain_m3=1.,residual_m3=0.)]
    result=physical_metrics(candidate=candidate,reference=reference,active=np.ones((1,2),bool),
        thresholds=(threshold,),indifference_m=1e-5,candidate_rows=ledger,reference_rows=ledger)
    assert result['inundation'][str(threshold)]['iou']==pytest.approx(1.)
    candidate[0,0]=threshold+.05
    result=physical_metrics(candidate=candidate,reference=reference,active=np.ones((1,2),bool),
        thresholds=(threshold,),indifference_m=1e-5,candidate_rows=ledger,reference_rows=ledger)
    assert result['inundation'][str(threshold)]['iou']==pytest.approx(.5)


def test_fv1_free_does_not_generate_pressure_momentum(device):
    model=flat_model(device=device,method='fv1',initial_depth=1.,explicit_maximum_dt=.01)
    solver=model.solver
    boundary=np.ones((2,5),np.int32);boundary[:,1:-1]=0;boundary[:,-1]=2
    solver.set_boundary_type_qx(boundary_type=boundary)
    solver.set_momentum(hu=np.full((2,4),-1e-9),hv=np.zeros((2,4)))
    solver.step(fixed_dt=.01)
    hu,hv=solver.get_momentum()
    assert np.max(np.abs(hu))<1e-8 and np.max(np.abs(hv))<1e-8
    assert abs(diagnostics(model=model)['residual_m3'])<1e-10


@pytest.mark.parametrize('orientation',['west','east','north','south'])
def test_masked_stage_cfl_uses_active_bed_datum(device,orientation):
    solver=_build_orientation_solver(orientation=orientation,device=device,mode='coarse')
    active=solver.active_mask.numpy().astype(bool)
    solver.set_bed_elevation(bed_elevation=np.where(active,100.,0.))
    codes={'west':(6,5),'east':(7,10),'south':(8,11),'north':(9,4)}
    old,new=codes[orientation]
    for axis in ('x','y'):
        boundary=getattr(solver,'boundary_type_q'+axis).numpy().copy()
        boundary[boundary==old]=new
        getattr(solver,'set_boundary_type_q'+axis)(boundary_type=boundary)
        getattr(solver,'set_boundary_stage_q'+axis)(stage=np.full(boundary.shape,100.6))
    solver.maximum_dt=60.
    expected=float(solver.cfl_number)*float(solver.dx)/np.sqrt(float(solver.gravity)*.6)
    assert solver.calculate_cfl_dt()==pytest.approx(expected,rel=1e-5,abs=1e-6)
