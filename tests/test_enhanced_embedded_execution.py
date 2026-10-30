"""Private trials, forcing, reference parity and exact continuation."""
import numpy as np
import pytest
import warp as wp

from tests.test_enhanced_temporal_accuracy import model
from floodwarp.enhanced_geometry import NumericalFailure
from floodwarp.rainfall import Hyetograph


@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    return request.param


def configured(*,device,execution,tolerance=.001,max_attempts=64):
    solver=model(device=device,tolerance=tolerance,estimator='bounded',max_attempts=max_attempts)
    solver._accuracy_execution=execution
    solver.set_source_rates(rain=np.array([[.001,.0005,.002]]),other=np.array([[.0002,.0001,.0003]]))
    solver.set_additional_source_rate(rate=np.array([[.0003,.0002,.0001]]))
    solver.set_infiltration_storage_limit_m(limit_m=np.array([[.001,.002,.003]]))
    solver.set_infiltration_exclusion_mask(mask=np.array([[1,0,0]],dtype=np.int32))
    types=solver.boundary_type_qx.numpy();types[:,0]=3;types[:,-1]=2
    solver.set_boundary_type_qx(boundary_type=types)
    solver.set_boundary_stage_qx(stage=np.full_like(types,.15,dtype=float))
    return solver


def forcing():
    rain=Hyetograph(node_times_s=np.array([0.,.3,.6,1.]),node_rates_m_per_s=np.array([0.,.002,.001,0.]),storm_type='test',metadata={})
    stage=(np.array([0.,.25,.55,1.]),np.array([.15,.12,.25,.1]))
    return rain,stage


@pytest.mark.parametrize('execution',['optimized'])
def test_private_failure_preserves_accepted_fields(device,execution):
    solver=configured(device=device,execution=execution,tolerance=1.e-14,max_attempts=1)
    names=tuple(name for name in solver._accuracy_saved if name!='_ec_status')
    before={name:getattr(solver,name).numpy().copy() for name in names}
    rain,stage=forcing()
    with pytest.raises(NumericalFailure):
        solver.run_device_loop(duration=.2,rainfall_hyetograph=rain,stage_series_qx=stage,
            use_cuda_graph=False,collect_free_outflow_diagnostics=True)
    assert solver.simulation_time==0.
    assert solver.temporal_accuracy_statistics()['rejected_attempts']==1
    for name,value in before.items():np.testing.assert_array_equal(getattr(solver,name).numpy(),value,err_msg=name)


@pytest.mark.parametrize('execution',['optimized'])
def test_reference_parity_with_dynamic_stages_and_diagnostics(device,execution):
    rain,stage=forcing();models=[];results=[]
    for path in ('reference',execution):
        solver=configured(device=device,execution=path)
        results.append(solver.run_device_loop(duration=1.,rainfall_hyetograph=rain,stage_series_qx=stage,
            use_cuda_graph=False,collect_free_outflow_diagnostics=True,dt_history_capacity=1000,
            checkpoint_times_s=[.3,.6,1.],field_checkpoint_indices=[0,1,2],checkpoint_outlet_rows=[1]))
        models.append(solver)
    np.testing.assert_array_equal(results[0]['dt_history'],results[1]['dt_history'])
    for name,value in models[0].get_component_state().items():
        np.testing.assert_array_equal(value,models[1].get_component_state()[name],err_msg=name)
    for name in ('boundary_stage_qx','boundary_stage_qy','max_water_depth_m','_ec_maximum_eta',
                 'cumulative_net_qx_volume_m3','cumulative_net_qy_volume_m3','_ec_scalar_interval'):
        np.testing.assert_array_equal(getattr(models[0],name).numpy(),getattr(models[1],name).numpy(),err_msg=name)
    a=models[0].cumulative_accounting();b=models[1].cumulative_accounting()
    for name,value in a.items():assert b[name]==pytest.approx(value,rel=2.e-13,abs=2.e-14)
    assert models[1].temporal_accuracy_statistics()['maximum_accepted_wave_ratio']<=1.


@pytest.mark.parametrize('execution',['optimized'])
def test_private_capture_continuation_is_bitwise(device,execution):
    solver=configured(device=device,execution=execution)
    rain,stage=forcing()
    solver.run_device_loop(duration=.3,rainfall_hyetograph=rain,stage_series_qx=stage,use_cuda_graph=device!='cpu')
    snapshot=solver.capture_state()
    fresh=configured(device=device,execution=execution)
    fresh.restore_state(state=snapshot)
    results=[]
    for candidate in (solver,fresh):
        results.append(candidate.run_device_loop(duration=.7,rainfall_hyetograph=rain,stage_series_qx=stage,
            use_cuda_graph=device!='cpu',dt_history_capacity=1000))
    np.testing.assert_array_equal(results[0]['dt_history'],results[1]['dt_history'])
    for name,value in solver.get_component_state().items():np.testing.assert_array_equal(value,fresh.get_component_state()[name],err_msg=name)
    assert solver.cumulative_accounting()==fresh.cumulative_accounting()


def test_old_checkpoint_keeps_reference_execution(device):
    original=configured(device=device,execution='reference')
    snapshot=original.capture_state()
    snapshot['host'].pop('enhanced_accuracy_execution')
    restored=configured(device=device,execution='optimized')
    restored.restore_state(state=snapshot)
    assert restored._accuracy_execution=='reference'
    results=[]
    for solver in (original,restored):results.append(solver.run_device_loop(duration=.2,use_cuda_graph=device!='cpu',dt_history_capacity=1000))
    np.testing.assert_array_equal(results[0]['dt_history'],results[1]['dt_history'])
    for name,value in original.get_component_state().items():np.testing.assert_array_equal(value,restored.get_component_state()[name],err_msg=name)


def test_callback_observes_current_coarse_discharge(device):
    histories=[]
    for method in ('reference','optimized'):
        solver=configured(device=device,execution=method)
        values=[]
        def observe(steps,simulation_time):
            values.append((steps,simulation_time,solver.qx.numpy().copy(),solver.qy.numpy().copy()))
        solver.run_device_loop(duration=.2,sync_interval=1,on_sync=observe,use_cuda_graph=False)
        histories.append(values)
    assert len(histories[0])==len(histories[1])
    assert any(np.any(row[2]!=0.) for row in histories[0])
    for original,fast in zip(histories[0],histories[1]):
        assert original[:2]==fast[:2]
        np.testing.assert_array_equal(original[2],fast[2]);np.testing.assert_array_equal(original[3],fast[3])
