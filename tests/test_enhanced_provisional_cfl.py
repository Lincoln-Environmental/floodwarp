"""Actual-source CFL: transactionality, acceptance, scheduling and device parity."""
import numpy as np
import pytest
import warp as wp
from floodwarp import enhanced_kernels as kernels
from tests.test_enhanced_production_state import create,STATE_RTOL,STATE_ATOL,VOLUME_ATOL
from floodwarp.rainfall import Hyetograph


@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    return request.param


def source_model(*,device,method,after=False,momentum_integration='legacy'):
    options={}
    if method=='exponential':options=dict(infiltration_f0_m_per_s=.0008,infiltration_fd_m=.001,infiltration_k_m_per_s=.0001)
    if method=='green_ampt':options=dict(infiltration_ks_m_per_s=.0001,infiltration_psi_f_m=.05,infiltration_delta_theta=.2)
    model=create(bed=np.zeros((5,15)),factor=5,device=device,method=method,component_timestep='provisional',
        maximum_dt=60.,adaptive_dt_growth_factor=None,momentum_integration=momentum_integration,**options)
    model.apply_sources_after_momentum=after
    model.set_source_rates(rain=np.array([[.003,.0015,.003]]),infil=np.full((1,3),.0002))
    return model


@pytest.mark.parametrize('method',['none','constant','exponential','green_ampt'])
def test_rejected_source_trials_have_no_physical_side_effects(device,method):
    model=source_model(device=device,method=method)
    physical=('_ec_volume','_ec_soil','_ec_cap','_ec_q','_ec_previous_d','_ec_maximum_eta',
        '_ec_applied','_ec_source_ledger','cumulative_totals','boundary_stage_qx','boundary_stage_qy')
    before={name:getattr(model,name).numpy().copy() for name in physical}
    dt=model.calculate_stable_timestep(remaining_time=60.)
    assert 0.<dt<60.
    for name,value in before.items():np.testing.assert_array_equal(getattr(model,name).numpy(),value,err_msg=name)
    stats=model.provisional_timestep_statistics()
    assert stats['rejections']>=1 and stats['maximum_accepted_courant']<=float(model.cfl_number)*(1+1e-12)
    # Commit the exact same source law once at the selected dt. This checks
    # trial/accepted arithmetic, not an independently invented source law.
    volume=model._ec_trial_volume.numpy().copy();eta=model._ec_trial_eta.numpy().copy();area=model._ec_trial_area.numpy().copy()
    wp.launch(kernel=kernels.set_control,dim=1,inputs=[model._ec_control,dt],device=device)
    model._component_sources(refresh_stage=True)
    np.testing.assert_array_equal(model._ec_volume.numpy(),volume)
    np.testing.assert_array_equal(model._ec_eta.numpy(),eta)
    np.testing.assert_array_equal(model._ec_area.numpy(),area)


@pytest.mark.parametrize('graph',[False,None])
@pytest.mark.parametrize('method',['none','constant','exponential','green_ampt'])
@pytest.mark.parametrize('after',[False,True])
@pytest.mark.parametrize('momentum_integration',['legacy','analytic'])
def test_global_source_retries_preserve_accounting_and_accepted_cfl(device,graph,method,after,momentum_integration):
    model=source_model(device=device,method=method,after=after,momentum_integration=momentum_integration)
    result=model.run_device_loop(duration=120.,use_cuda_graph=graph,dt_history_capacity=10000)
    stats=model.provisional_timestep_statistics();ledger=model.cumulative_accounting();state=model.get_component_state()
    assert result['steps']>0 and stats['checks']>=result['steps']
    assert stats['maximum_accepted_courant']<=float(model.cfl_number)*(1+1e-12)
    assert state['volume_m3'].min()>=0.
    expected=model.initial_volume+ledger['rain']-ledger['infiltration']+ledger['boundary_inflow']-ledger['boundary_outflow']
    np.testing.assert_allclose(state['volume_m3'].sum(dtype=np.float64),expected,rtol=STATE_RTOL,atol=VOLUME_ATOL)
    assert ledger['rain']==pytest.approx(120.*75.*.0025,rel=STATE_RTOL)
    np.testing.assert_array_equal(model._ec_source_ledger.numpy(),0.)


@pytest.mark.parametrize('momentum_integration',['legacy','analytic'])
def test_hyetograph_stage_series_and_nested_retry_graph_cpu_cuda(momentum_integration):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rain=Hyetograph(node_times_s=np.array([0.,20.,40.]),node_rates_m_per_s=np.array([0.,.003,0.]),storm_type='control',metadata={})
    models=[];results=[]
    for device in ('cpu','cuda:0'):
        bx=np.array([[3,0,0,3]],np.int32)
        model=create(bed=np.zeros((5,15)),factor=5,device=device,method='constant',component_timestep='provisional',
            maximum_dt=60.,adaptive_dt_growth_factor=None,boundary_type_qx=bx,momentum_integration=momentum_integration)
        model.set_source_rates(infil=np.full((1,3),.0001))
        model.set_rainfall_spatial_multiplier(multiplier=np.array([[1.,.5,1.]]))
        result=model.run_device_loop(duration=40.,rainfall_hyetograph=rain,use_cuda_graph=None,
            stage_series_qx=(np.array([0.,40.]),np.array([.01,.04])),dt_history_capacity=10000,
            checkpoint_times_s=[10.,20.,30.,40.],checkpoint_outlet_rows=[1],field_checkpoint_indices=[0,1,2,3])
        models.append(model);results.append(result)
    assert results[0]['steps']==results[1]['steps']
    np.testing.assert_allclose(results[0]['dt_history'],results[1]['dt_history'],rtol=STATE_RTOL,atol=STATE_ATOL)
    for name,value in models[0].get_component_state().items():
        np.testing.assert_allclose(value,models[1].get_component_state()[name],rtol=STATE_RTOL,atol=STATE_ATOL)
    for model in models:assert model.provisional_timestep_statistics()['maximum_accepted_courant']<=float(model.cfl_number)*(1+1e-12)


@pytest.mark.parametrize('momentum_integration',['legacy','analytic'])
def test_host_and_global_constant_source_semantics(device,momentum_integration):
    host=source_model(device=device,method='green_ampt',momentum_integration=momentum_integration)
    global_model=source_model(device=device,method='green_ampt',momentum_integration=momentum_integration)
    duration=10.;steps=[]
    while host.simulation_time<duration:steps.append(host.step(remaining_time=duration-host.simulation_time))
    result=global_model.run_device_loop(duration=duration,use_cuda_graph=None,dt_history_capacity=10000)
    np.testing.assert_allclose(steps,result['dt_history'],rtol=STATE_RTOL,atol=STATE_ATOL)
    for name,value in host.get_component_state().items():
        np.testing.assert_allclose(value,global_model.get_component_state()[name],rtol=STATE_RTOL,atol=STATE_ATOL)


def test_provisional_restart_restores_trial_scratch_and_source_state(device):
    model=source_model(device=device,method='exponential')
    snapshot=model.capture_state()
    first=model.run_device_loop(duration=3.,use_cuda_graph=None,dt_history_capacity=1000)
    expected=model.get_component_state();ledger=model.cumulative_accounting()
    model.restore_state(state=snapshot)
    for name,value in snapshot['device'].items():
        if value is not None:np.testing.assert_array_equal(getattr(model,name).numpy(),value.numpy(),err_msg=name)
    second=model.run_device_loop(duration=3.,use_cuda_graph=None,dt_history_capacity=1000)
    np.testing.assert_array_equal(first['dt_history'],second['dt_history'])
    for name,value in expected.items():np.testing.assert_array_equal(value,model.get_component_state()[name])
    assert ledger==model.cumulative_accounting()


@pytest.mark.parametrize('method',['constant','green_ampt'])
def test_provisional_uptake_exclusion_and_caps_preserve_transaction(device,method):
    model=source_model(device=device,method=method,momentum_integration='analytic')
    model.set_water_depth(depth=np.array([[.000001,.0002,.0001]]))
    model.set_source_rates(rain=np.array([[0.,.0015,.003]]),infil=np.full((1,3),.0002),
        other=np.array([[0.,.0008,0.]]))
    model.set_additional_source_rate(rate=np.array([[0.,.0009,.0004]]))
    model.set_infiltration_exclusion_mask(mask=np.array([[0,1,0]],np.int32))
    model.set_infiltration_storage_limit_m(limit_m=np.array([[.00001,.000007,.000007]]))
    physical=('_ec_volume','_ec_soil','_ec_cap','_ec_q','_ec_maximum_eta',
        '_ec_applied','_ec_source_ledger','cumulative_totals')
    before={name:getattr(model,name).numpy().copy() for name in physical}
    dt=model.calculate_stable_timestep(remaining_time=3.)
    volume=model._ec_trial_volume.numpy().copy();eta=model._ec_trial_eta.numpy().copy();area=model._ec_trial_area.numpy().copy()
    for name,value in before.items():np.testing.assert_array_equal(getattr(model,name).numpy(),value,err_msg=name)
    assert volume[0]==0. and volume.min()>=0.
    wp.launch(kernel=kernels.set_control,dim=1,inputs=[model._ec_control,dt],device=device)
    model._component_sources(refresh_stage=True)
    for name,expected in (('_ec_volume',volume),('_ec_eta',eta),('_ec_area',area)):
        np.testing.assert_array_equal(getattr(model,name).numpy(),expected)
    ledger=model.cumulative_accounting()
    expected=model.initial_volume+ledger['rain']+ledger['other_sources']+ledger['additional_sources']-ledger['infiltration']
    np.testing.assert_allclose(volume.sum(dtype=np.float64),expected,rtol=STATE_RTOL,atol=VOLUME_ATOL)
