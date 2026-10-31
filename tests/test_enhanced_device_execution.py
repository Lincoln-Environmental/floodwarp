"""Complete component host/window/conditional stepping with production forcing."""
import numpy as np
import pytest
import warp as wp

from floodwarp.config import NP_FLOAT, WP_FLOAT
from floodwarp import enhanced_kernels
from floodwarp.enhanced_geometry import component_volumes
from floodwarp.rainfall import Hyetograph
from tests.test_enhanced_production_state import create,STATE_RTOL,STATE_ATOL,VOLUME_ATOL
from tests.forcing_helpers import prepare_stage_controller,stage_variation_cap


@pytest.fixture(params=[('cpu',False),('cuda:0',False),('cuda:0',True)])
def execution(request):
    device,graph=request.param
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    return device,graph


def create_model(*,device,formulation,transport,infiltration,after):
    bed=np.ones((5,15));bed[1,:]=0.;bed[3,:]=0.;bed[2,:]=.1
    soil={}
    if infiltration=='exponential':soil=dict(infiltration_f0_m_per_s=.0003,infiltration_fd_m=.01,infiltration_k_m_per_s=.00005)
    if infiltration=='green_ampt':soil=dict(infiltration_ks_m_per_s=.00005,infiltration_psi_f_m=.05,infiltration_delta_theta=.2)
    model=create(bed=bed,factor=5,device=device,method=infiltration,momentum_formulation=formulation,
        momentum_transport=transport,subgrid_manning_n=.03+np.indices(bed.shape)[1]*.0001,**soil)
    model.apply_sources_after_momentum=after
    mask=model.boundary_type_qx.numpy();mask[:,0]=3;mask[:,-1]=2
    model.set_boundary_type_qx(boundary_type=mask)
    model.set_boundary_stage_qx(stage=np.full(mask.shape,.2))
    initial=np.zeros_like(bed);initial[1,:5]=.25
    model.set_component_state(volume=component_volumes(geometry=model._enhanced_geometry,native_depth=initial))
    model.set_source_rates(infil=np.array([[.0001,.0002,.00015]]),other=np.array([[.00001,0.,.00002]]))
    model.set_additional_source_rate(rate=np.array([[0.,.00002,.00001]]))
    model.set_infiltration_exclusion_mask(mask=np.array([[1,0,0]],np.int32))
    model.set_infiltration_storage_limit_m(limit_m=np.array([[.00001,.00002,.000005]]))
    model.set_rainfall_spatial_multiplier(multiplier=np.array([[1.,.7,1.3]]))
    return model


def forcing():
    rain=Hyetograph(node_times_s=np.array([0.,.04,.085,.13]),
        node_rates_m_per_s=np.array([.0002,.0004,.0001,0.]),storm_type='test',metadata={})
    stage=(np.array([0.,.07,.15]),np.array([.2,.15,.27]))
    return rain,stage


def run_host(*,model,duration,rain,stage,checkpoints):
    history=[];fields=[];volume=[];end=model.simulation_time+duration
    initial_rate=rain.rate_at(model.simulation_time);model.set_uniform_rainfall_rate(rainfall_rate=initial_rate)
    index=0
    buffers=prepare_stage_controller(solver=model,series=stage)
    while model.simulation_time<end-1e-13:
        t=model.simulation_time;remaining=end-t
        if t<rain.duration_s:remaining=min(remaining,rain.next_breakpoint_after(t)-t)
        if index<len(checkpoints):remaining=min(remaining,checkpoints[index]-t)
        later=stage[0][stage[0]>t]
        if later.size:remaining=min(remaining,float(later[0]-t))
        boundary=model.boundary_stage_qx.numpy();types=model.boundary_type_qx.numpy()
        boundary[types==3]=np.interp(t,*stage);model.set_boundary_stage_qx(stage=boundary)
        cap=min(model.maximum_dt,remaining)
        if model.adaptive_dt_growth_factor is not None and model._adaptive_dt_reference>0.:
            cap=min(cap,model._adaptive_dt_reference*model.adaptive_dt_growth_factor)
        variation_cap=stage_variation_cap(solver=model,series=stage,buffers=buffers,time=t,cap=cap)
        boundary[types==3]=max(np.interp(t,*stage),np.interp(t+cap,*stage))
        model.set_boundary_stage_qx(stage=boundary)
        bound=max(rain.rate_at(t),rain.rate_at(t+cap)) if t<rain.duration_s else 0.
        model.set_uniform_rainfall_rate(rainfall_rate=bound)
        dt=min(variation_cap,model.calculate_stable_timestep(remaining_time=remaining))
        sample=t if model.apply_sources_after_momentum else t+.5*dt
        rate=rain.rate_at(sample) if sample<rain.duration_s else 0.
        model.set_uniform_rainfall_rate(rainfall_rate=rate)
        boundary=model.boundary_stage_qx.numpy();types=model.boundary_type_qx.numpy()
        boundary[types==3]=np.interp(t+.5*dt,*stage);model.set_boundary_stage_qx(stage=boundary)
        model.step(remaining_time=remaining,fixed_dt=dt);history.append(dt)
        if index<len(checkpoints) and abs(model.simulation_time-checkpoints[index])<1e-12:
            fields.append(model.get_water_depth().copy());volume.append(model.get_component_state()['volume_m3'].sum(dtype=np.float64));index+=1
    assert index==len(checkpoints)
    return np.asarray(history),np.asarray(fields),np.asarray(volume)


@pytest.mark.parametrize('formulation,transport',[('matrix_lumped','discharge'),('transport_star','discharge'),('transport_star','kinetic')])
@pytest.mark.parametrize('infiltration',['constant','exponential','green_ampt','none'])
@pytest.mark.parametrize('after',[False,True])
def test_host_device_complete_forcing_soil_and_checkpoint_parity(execution,formulation,transport,infiltration,after):
    device,graph=execution
    host=create_model(device=device,formulation=formulation,transport=transport,infiltration=infiltration,after=after)
    candidate=create_model(device=device,formulation=formulation,transport=transport,infiltration=infiltration,after=after)
    rain,stage=forcing();checkpoints=np.array([.05,.105,.15])
    expected_dt,expected_fields,expected_volume=run_host(model=host,duration=.15,rain=rain,stage=stage,checkpoints=checkpoints)
    candidate.set_uniform_rainfall_rate(rainfall_rate=rain.rate_at(0.))
    result=candidate.run_device_loop(duration=.15,sync_interval=7,rainfall_hyetograph=rain,stage_series_qx=stage,
        checkpoint_times_s=checkpoints,field_checkpoint_indices=[0,1,2],checkpoint_outlet_rows=[1],
        dt_history_capacity=4096,use_cuda_graph=graph)
    assert result['steps']==len(expected_dt)
    np.testing.assert_allclose(result['dt_history'],expected_dt,rtol=2e-13,atol=1e-14)
    for key,value in host.get_component_state().items():
        np.testing.assert_allclose(candidate.get_component_state()[key],value,rtol=STATE_RTOL,atol=STATE_ATOL,err_msg=key)
    np.testing.assert_allclose(candidate.get_qx(),host.get_qx(),rtol=STATE_RTOL,atol=STATE_ATOL)
    np.testing.assert_array_equal(result['checkpoint_trace']['times_s'],checkpoints)
    np.testing.assert_allclose(result['checkpoint_trace']['field_depths'],expected_fields,rtol=STATE_RTOL,atol=STATE_ATOL)
    np.testing.assert_allclose(result['checkpoint_trace']['volume_m3'],expected_volume,rtol=STATE_RTOL,atol=STATE_ATOL)
    for key,value in host.cumulative_accounting().items():
        assert candidate.cumulative_accounting()[key]==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL)
    for key,value in host.get_cumulative_face_volumes().items():
        actual=candidate.get_cumulative_face_volumes()[key]
        if isinstance(value,np.ndarray):np.testing.assert_allclose(actual,value,rtol=STATE_RTOL,atol=STATE_ATOL)
        elif value is not None:assert actual==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL)
    state=candidate.get_component_state();a=candidate.cumulative_accounting()
    residual=state['volume_m3'].sum(dtype=np.float64)-candidate.initial_volume-a['rain']-a['other_sources']-a['additional_sources']-a['boundary_inflow']+a['infiltration']+a['boundary_outflow']
    assert abs(residual)<VOLUME_ATOL*10
    assert state['volume_m3'].min()>=0.
    assert candidate.simulation_time==pytest.approx(.15,abs=1e-13)


@pytest.mark.parametrize('mode',[0,1,2])
def test_group_free_priority_and_diagnostic_bypass(execution,mode):
    """Two exits compete for one actual component's available water."""
    device,_=execution
    volume=wp.array([1.],dtype=WP_FLOAT,device=device)
    offsets=wp.array([0,2],dtype=int,device=device)
    adjacent=wp.array([1,2],dtype=int,device=device)
    proposed=wp.array([2.,3.],dtype=WP_FLOAT,device=device)
    kinds=wp.array([2,0],dtype=int,device=device)
    alpha=wp.zeros(1,dtype=wp.float64,device=device)
    alpha_free=wp.zeros(1,dtype=wp.float64,device=device)
    control=wp.zeros(18,dtype=wp.float64,device=device)
    status=wp.zeros(1,dtype=int,device=device)
    wp.launch(kernel=enhanced_kernels.set_control,dim=1,inputs=[control,1.],device=device)
    safety=1.000000000000002 if NP_FLOAT==np.float64 else 1.000001
    wp.launch(kernel=enhanced_kernels.demands,dim=1,
        inputs=[volume,offsets,adjacent,proposed,alpha,control,safety,status,kinds,alpha_free,mode],device=device)
    ordinary=alpha.numpy()[0];free=alpha_free.numpy()[0]
    if mode==0:
        assert free==ordinary
        assert free==pytest.approx(1./(5.*safety))
    elif mode==1:
        assert free==pytest.approx(1./(2.*safety))
        assert 2.*free+3.*ordinary<=1.
    else:
        assert free==1.
        assert ordinary==pytest.approx(1./(5.*safety))
        # The explicit diagnostic bypass can overdraw; it is never the default.
        assert 2.*free+3.*ordinary>1.


@pytest.mark.parametrize('mode',[0,1])
def test_device_free_counters_book_actual_group_export(execution,mode):
    device,graph=execution
    if graph:pytest.skip('Production conditional graphs reject per-sync FREE traces')
    model=create(bed=np.zeros((2,4)),factor=2,device=device,free_outflow_limiter_mode=mode)
    boundary=model.boundary_type_qx.numpy();boundary[:,-1]=2
    model.set_boundary_type_qx(boundary_type=boundary)
    model.set_water_depth(depth=np.array([[.3,.2]]))
    result=model.run_device_loop(duration=.1,sync_interval=7,collect_free_outflow_diagnostics=True)
    counters=model.free_outflow_limiter_diagnostics()
    assert counters['free_outflow_samples']==result['steps']
    assert counters['raw_free_outflow_m3']>0.
    assert counters['limited_free_outflow_m3']==pytest.approx(
        model.cumulative_accounting()['boundary_outflow'],rel=STATE_RTOL,abs=STATE_ATOL)
    assert 0.<counters['free_outflow_alpha_min']<=1.
    assert result['free_outflow_limiter_trace']


@pytest.mark.parametrize('varying_rain',[False,True])
def test_device_continuation_and_restore_preserve_independent_state(execution,varying_rain):
    device,graph=execution
    options=dict(device=device,formulation='transport_star',transport='discharge',infiltration='green_ampt',after=False)
    complete=create_model(**options);split=create_model(**options)
    rain,stage=forcing()
    if not varying_rain:
        rain=Hyetograph(node_times_s=np.array([0.,.2]),
            node_rates_m_per_s=np.array([.0002,.0002]),storm_type='test',metadata={})
    complete.run_device_loop(duration=.15,sync_interval=7,rainfall_hyetograph=rain,
        stage_series_qx=stage,checkpoint_times_s=[.07,.15],use_cuda_graph=graph)
    split.run_device_loop(duration=.07,sync_interval=7,rainfall_hyetograph=rain,stage_series_qx=stage,use_cuda_graph=graph)
    saved=split.capture_state()
    split.run_device_loop(duration=.08,sync_interval=7,rainfall_hyetograph=rain,stage_series_qx=stage,use_cuda_graph=graph)
    # As in traditional device execution, entering a new call re-samples the
    # source-aware CFL predictor at its start. With varying rainfall this can
    # change dt relative to an uninterrupted call (not the source integral law).
    # Constant-rain partitions have the same predictor and match tightly.
    if not varying_rain:
        for key,value in complete.get_component_state().items():
            np.testing.assert_allclose(split.get_component_state()[key],value,rtol=STATE_RTOL,atol=STATE_ATOL)
    expected={key:value.copy() for key,value in split.get_component_state().items()}
    split.restore_state(state=saved)
    split.run_device_loop(duration=.08,sync_interval=7,rainfall_hyetograph=rain,stage_series_qx=stage,use_cuda_graph=graph)
    for key,value in expected.items():
        np.testing.assert_allclose(split.get_component_state()[key],value,rtol=STATE_RTOL,atol=STATE_ATOL)


def test_empty_mask_device_loop_remains_dry(execution):
    device,graph=execution
    model=create(bed=np.zeros((2,4)),factor=2,device=device,mask=np.zeros((1,2),dtype=np.int32))
    model.set_uniform_rainfall_rate(rainfall_rate=.01)
    result=model.run_device_loop(duration=.105,sync_interval=7,use_cuda_graph=graph)
    assert result['steps']==6
    np.testing.assert_array_equal(model.get_water_depth(),0.)
    assert model.cumulative_accounting()['rain']==0.


def test_device_subminimum_timestep_raises_before_transport(execution):
    device,graph=execution
    model=create(bed=np.zeros((2,4)),factor=2,device=device)
    model.set_water_depth(depth=np.array([[.3,.2]]));model.minimum_dt=.1
    initial=model.get_component_state()
    with pytest.raises(RuntimeError,match='minimum|sub-minimum'):
        model.run_device_loop(duration=1.,sync_interval=7,use_cuda_graph=graph)
    for key,value in initial.items():
        np.testing.assert_array_equal(model.get_component_state()[key],value)
