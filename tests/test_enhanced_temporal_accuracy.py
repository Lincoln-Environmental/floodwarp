"""Conservation, complete rejection rollback and actual error refinement."""
import numpy as np
import json
import pytest
import warp as wp

from tests.test_enhanced_production_state import create
from floodwarp.rainfall import Hyetograph
from floodwarp.enhanced_geometry import NumericalFailure
from floodwarp.enhanced_cache import load_geometry_cache
from floodwarp.checkpoints import durable_checkpoint


@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param!='cpu' and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def model(*, device, tolerance, method='none', after=False, max_attempts=64,estimator='doubling'):
    defect='analytic'
    wave_fraction=.8
    if estimator=='bounded':
        estimator='embedded';defect='bounded';wave_fraction=.95
    options={}
    if method=='exponential':
        options=dict(infiltration_f0_m_per_s=.0008,infiltration_fd_m=.001,infiltration_k_m_per_s=.0001)
    if method=='green_ampt':
        options=dict(infiltration_ks_m_per_s=.0001,infiltration_psi_f_m=.05,infiltration_delta_theta=.2)
    s=create(bed=np.zeros((2,6)),factor=2,device=device,method=method,maximum_dt=.2,
        momentum_integration='analytic' if estimator=='embedded' else 'legacy',**options)
    s.apply_sources_after_momentum=after
    s.configure_temporal_accuracy(stage_atol_m=tolerance,discharge_atol_m3_s=tolerance,
        relative_tolerance=tolerance,coefficient_relative_tolerance=.05,max_attempts=max_attempts,
        estimator=estimator,defect_evaluation=defect,wave_proposal_fraction=wave_fraction)
    s.set_water_depth(depth=np.array([[.1,.05,.02]]))
    return s


@pytest.mark.parametrize('method',['none','constant','exponential','green_ampt'])
@pytest.mark.parametrize('after',[False,True])
def test_reject_rolls_back_all_physical_state(device,method,after):
    s=model(device=device,tolerance=1.e-14,method=method,after=after,max_attempts=1)
    s.set_source_rates(rain=np.full((1,3),.001),infil=np.full((1,3),.0002))
    s.set_infiltration_storage_limit_m(limit_m=np.full((1,3),.0003))
    s.set_additional_source_rate(rate=np.full((1,3),.0001))
    names=('_ec_volume','_ec_eta','_ec_area','_ec_q','_ec_soil','_ec_cap','_ec_previous_d',
        '_ec_maximum_eta','_ec_applied','_ec_source_ledger','cumulative_totals','max_water_depth_m',
        'cumulative_net_qx_volume_m3','cumulative_net_qy_volume_m3','water_depth')
    before={name:getattr(s,name).numpy().copy() for name in names}
    with pytest.raises(NumericalFailure):
        s.run_device_loop(duration=.2,use_cuda_graph=device!='cpu')
    for name,value in before.items():
        np.testing.assert_array_equal(getattr(s,name).numpy(),value,err_msg=name)
    assert s.simulation_time==0.
    assert s.temporal_accuracy_statistics()['rejected_attempts']==1


@pytest.mark.parametrize('after',[False,True])
def test_sources_conserve_and_restart_exactly(device,after):
    s=model(device=device,tolerance=1.e-4,method='green_ampt',after=after)
    s.set_source_rates(infil=np.full((1,3),.0002))
    s.set_infiltration_exclusion_mask(mask=np.array([[0,1,0]],np.int32))
    s.set_infiltration_storage_limit_m(limit_m=np.full((1,3),.0001))
    h=Hyetograph(node_times_s=np.array([0.,.5,1.]),node_rates_m_per_s=np.array([0.,.001,0.]),storm_type='test',metadata={})
    initial=s.capture_state()
    first=s.run_device_loop(duration=1.,rainfall_hyetograph=h,use_cuda_graph=device!='cpu',dt_history_capacity=10000,
        checkpoint_times_s=[.5,1.],checkpoint_outlet_rows=[1],field_checkpoint_indices=[0,1])
    state=s.get_component_state();ledger=s.cumulative_accounting()
    assert state['volume_m3'].min()>=0
    assert state['volume_m3'].sum()==pytest.approx(s.initial_volume+ledger['rain']-ledger['infiltration'],abs=1.e-12)
    increments=first['dt_history']
    starts=np.r_[0.,np.cumsum(increments)[:-1]]
    offset=0. if after else .25
    rates1=np.interp(starts+increments*offset,h.node_times_s,h.node_rates_m_per_s)
    rates2=np.interp(starts+increments*(offset+.5),h.node_times_s,h.node_rates_m_per_s)
    expected_rain=float(np.sum(increments*.5*(rates1+rates2))*12.)
    assert ledger['rain']==pytest.approx(expected_rain,abs=1.e-12)
    assert ledger['infiltration']<=.0008+1.e-12
    s.restore_state(state=initial)
    second=s.run_device_loop(duration=1.,rainfall_hyetograph=h,use_cuda_graph=device!='cpu',dt_history_capacity=10000,
        checkpoint_times_s=[.5,1.],checkpoint_outlet_rows=[1],field_checkpoint_indices=[0,1])
    np.testing.assert_array_equal(first['dt_history'],second['dt_history'])
    for name,value in state.items():
        np.testing.assert_array_equal(value,s.get_component_state()[name])
    assert ledger==s.cumulative_accounting()
    assert s.temporal_accuracy_statistics()['maximum_accepted_normalized_error']<=1.


@pytest.mark.parametrize('estimator',['doubling','embedded','bounded'])
def test_tolerance_refinement_resolves_momentum_from_rest(device,estimator):
    reference=create(bed=np.zeros((2,6)),factor=2,device=device,method='none',maximum_dt=.00005,
        momentum_integration='analytic' if estimator in ('embedded','bounded') else 'legacy')
    reference.set_water_depth(depth=np.array([[.1,.05,.02]]))
    reference.run_device_loop(duration=2.,use_cuda_graph=device!='cpu')
    expected=reference.get_component_state()
    errors=[];updates=[]
    for tolerance in (1.e-3,1.e-6,1.e-8):
        s=model(device=device,tolerance=tolerance,estimator=estimator)
        s.run_device_loop(duration=2.,use_cuda_graph=device!='cpu')
        actual=s.get_component_state()
        errors.append(float(np.max(np.abs(actual['discharge_m3_per_s']-expected['discharge_m3_per_s']))))
        updates.append(s.temporal_accuracy_statistics()['accepted_hydraulic_updates'])
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<errors[0]*.1
    assert updates[2]>updates[1]>updates[0]


def test_embedded_rejection_is_transactional(device):
    s=model(device=device,tolerance=1.e-14,max_attempts=1,estimator='embedded')
    s.set_source_rates(rain=np.full((1,3),.001))
    names=tuple(s._accuracy_saved)
    before={name:getattr(s,name).numpy().copy() for name in names if name!='_ec_status'}
    with pytest.raises(NumericalFailure):
        s.run_device_loop(duration=.2,use_cuda_graph=device!='cpu')
    for name,value in before.items():
        np.testing.assert_array_equal(getattr(s,name).numpy(),value,err_msg=name)
    assert s.simulation_time==0.


def test_embedded_live_source_guard(device):
    s=model(device=device,tolerance=.001,method='constant',estimator='embedded')
    s.set_source_rates(infil=np.full((1,3),.001))
    with pytest.raises(ValueError,match='zero infiltration'):
        s.run_device_loop(duration=1.)


@pytest.mark.parametrize('estimator',['doubling','embedded','bounded'])
def test_fresh_solver_restart_and_endpoint_closure(device,estimator):
    s=model(device=device,tolerance=.001,estimator=estimator)
    s.maximum_dt=.1
    s.run_device_loop(duration=.100000000001,use_cuda_graph=device!='cpu')
    snapshot=s.capture_state()
    fresh=model(device=device,tolerance=.001,estimator=estimator)
    fresh.maximum_dt=.1
    fresh.restore_state(state=snapshot)
    for name,value in snapshot['device'].items():
        if value is not None:
            np.testing.assert_array_equal(getattr(fresh,name).numpy(),value.numpy(),err_msg=name)
    results=[]
    for solver in (s,fresh):
        results.append(solver.run_device_loop(duration=.2,use_cuda_graph=device!='cpu',dt_history_capacity=1000,
            checkpoint_times_s=[.2,.300000000001],checkpoint_outlet_rows=[1],field_checkpoint_indices=[0,1]))
    np.testing.assert_array_equal(results[0]['dt_history'],results[1]['dt_history'])
    assert results[0]['dt_history'][0]>=s.minimum_dt
    for name,value in s.get_component_state().items():
        np.testing.assert_array_equal(value,fresh.get_component_state()[name])
    assert s.temporal_accuracy_statistics()['maximum_accepted_wave_ratio']<=1.


def test_certified_disk_checkpoint_roundtrip(device,tmp_path):
    s=model(device=device,tolerance=.001,estimator='bounded')
    s.set_source_rates(rain=np.full((1,3),.001))
    s.run_device_loop(duration=.3,use_cuda_graph=device!='cpu')
    path=tmp_path/'checkpoint.npz'
    signature=dict(case='certified_complete_disk_continuation')
    durable_checkpoint(surface=s,path=path,signature=signature,report={})
    fresh=model(device=device,tolerance=.001,estimator='bounded')
    expected=fresh.capture_state()
    with np.load(file=path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata']))
        assert metadata['signature']==signature
        device_fields={name:None for name in metadata['none_fields']}
        for name in saved.files:
            if name=='metadata':continue
            template=expected['device'].get(name)
            device_fields[name]=wp.array(saved[name],dtype=template.dtype,device=device) if template is not None else wp.array(saved[name],device=device)
    host=metadata['host']
    host['_conditioned_bed_array']=fresh.bed_elevation
    geometry,seconds=load_geometry_cache(directory=tmp_path/'geometry',key=metadata['geometry_key'])
    host['enhanced_geometry']=geometry
    fresh.initial_volume=metadata['initial_volume']
    fresh.restore_state(state=dict(host=host,device=device_fields))
    results=[]
    for solver in (s,fresh):
        results.append(solver.run_device_loop(duration=.7,use_cuda_graph=device!='cpu',dt_history_capacity=1000))
    np.testing.assert_array_equal(results[0]['dt_history'],results[1]['dt_history'])
    for name,value in s.get_component_state().items():
        np.testing.assert_array_equal(value,fresh.get_component_state()[name])
    for name in ('cumulative_net_qx_volume_m3','cumulative_net_qy_volume_m3','max_water_depth_m'):
        np.testing.assert_array_equal(getattr(s,name).numpy(),getattr(fresh,name).numpy())
    assert s.cumulative_accounting()==fresh.cumulative_accounting()


def test_certified_cpu_cuda_accepted_history_and_sources():
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rain=Hyetograph(node_times_s=np.array([0.,.4,.8,1.]),node_rates_m_per_s=np.array([0.,.001,.0005,0.]),
        storm_type='CPU CUDA parity',metadata={})
    models=[];results=[]
    for device in ('cpu','cuda:0'):
        s=model(device=device,tolerance=.001,estimator='bounded')
        s.set_rainfall_spatial_multiplier(multiplier=np.array([[1.,.5,1.5]]))
        results.append(s.run_device_loop(duration=1.,rainfall_hyetograph=rain,use_cuda_graph=device!='cpu',dt_history_capacity=1000))
        models.append(s)
    assert results[0]['steps']==results[1]['steps']
    np.testing.assert_allclose(results[0]['dt_history'],results[1]['dt_history'],rtol=2.e-12,atol=1.e-14)
    for name,value in models[0].get_component_state().items():
        np.testing.assert_allclose(value,models[1].get_component_state()[name],rtol=2.e-12,atol=1.e-14)
    a=models[0].cumulative_accounting();b=models[1].cumulative_accounting()
    for name,value in a.items():assert value==pytest.approx(b[name],rel=2.e-12,abs=1.e-14)
