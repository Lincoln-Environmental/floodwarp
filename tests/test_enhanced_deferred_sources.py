"""Source reduction changes accounting order, never accepted hydraulic state."""
import numpy as np
import pytest
import warp as wp

from tests.test_enhanced_device_execution import create_model,forcing
from tests.test_enhanced_production_state import STATE_RTOL,VOLUME_ATOL


@pytest.mark.parametrize('device,graph',[('cpu',False),('cuda:0',False),('cuda:0',True)])
@pytest.mark.parametrize('infiltration',['none','constant','exponential','green_ampt'])
@pytest.mark.parametrize('after',[False,True])
def test_reduced_accepted_source_volumes_and_continuation(monkeypatch,device,graph,infiltration,after):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    models=[]
    for reduced in (False,True):
        monkeypatch.setenv('OVERLANDER_REDUCE_MASS','1' if reduced else '0')
        model=create_model(device=device,formulation='transport_star',transport='discharge',
            infiltration=infiltration,after=after)
        assert model.use_reduced_mass_accounting==reduced
        models.append(model)
    rain,stage=forcing()
    for duration in (.15,.11):
        results=[]
        for model in models:
            results.append(model.run_device_loop(duration=duration,use_cuda_graph=graph,
                rainfall_hyetograph=rain,stage_series_qx=stage,dt_history_capacity=1000,
                checkpoint_times_s=[model.simulation_time+duration],field_checkpoint_indices=[0]))
        np.testing.assert_array_equal(results[0]['dt_history'],results[1]['dt_history'])
        np.testing.assert_array_equal(results[0]['checkpoint_trace']['field_depths'],
                                      results[1]['checkpoint_trace']['field_depths'])
        reference,candidate=models
        for getter in ('get_water_depth','get_qx','get_qy','get_max_water_depth'):
            np.testing.assert_array_equal(getattr(reference,getter)(),getattr(candidate,getter)())
        for key,value in reference.get_component_state().items():
            np.testing.assert_array_equal(value,candidate.get_component_state()[key])
        np.testing.assert_array_equal(reference.cumulative_infiltration_m,candidate.cumulative_infiltration_m)
        for key,value in reference.cumulative_accounting().items():
            assert candidate.cumulative_accounting()[key]==pytest.approx(value,rel=STATE_RTOL,abs=VOLUME_ATOL)
        np.testing.assert_array_equal(candidate._ec_source_ledger.numpy(),0.)
        assert not candidate._defer_component_sources
        # Exact restart includes newly introduced scratch/ledger arrays.
        snapshot=candidate.capture_state()
        before=candidate.cumulative_accounting()
        candidate.step(fixed_dt=.001)
        candidate.restore_state(state=snapshot)
        assert candidate.cumulative_accounting()==before


def test_callback_observes_fully_booked_source_totals(monkeypatch):
    monkeypatch.setenv('OVERLANDER_REDUCE_MASS','1')
    model=create_model(device='cpu',formulation='transport_star',transport='discharge',
        infiltration='none',after=False)
    reference=create_model(device='cpu',formulation='transport_star',transport='discharge',
        infiltration='none',after=False)
    rain,stage=forcing()
    observed=[]
    def observe(steps,time):
        observed.append((time,model.cumulative_accounting()['rain']))
        np.testing.assert_array_equal(model._ec_source_ledger.numpy(),0.)
    model.run_device_loop(duration=.13,sync_interval=1,rainfall_hyetograph=rain,
        stage_series_qx=stage,on_sync=observe,use_cuda_graph=False)
    reference.run_device_loop(duration=.13,sync_interval=1,rainfall_hyetograph=rain,
        stage_series_qx=stage,use_cuda_graph=False)
    assert len(observed)>1
    assert observed[-1][1]==reference.cumulative_accounting()['rain']


@pytest.mark.parametrize('device,graph',[('cpu',False),('cuda:0',False),('cuda:0',True)])
@pytest.mark.parametrize('after',[False,True])
def test_isolated_source_pulse_records_peak_before_complete_recession(device,graph,after):
    """Stage caches must include source changes even when transport net is zero."""
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    from floodwarp.rainfall import Hyetograph
    from tests.sgs_helpers import disconnected_source_model
    from tests.test_enhanced_device_execution import run_host
    from tests.test_enhanced_production_state import STATE_ATOL
    host=disconnected_source_model(device=device);candidate=disconnected_source_model(device=device)
    for model in (host,candidate):
        model.maximum_dt=.001
        model.apply_sources_after_momentum=after
        model.set_source_rates(infil=np.full((1,3),.005))
    rain=Hyetograph(node_times_s=np.array([0.,.004,.005,.02]),
        node_rates_m_per_s=np.array([.01,.01,0.,0.]),storm_type='peak-cache-regression',metadata={})
    times=np.array([0.,.02]);stage=(times,np.zeros(2))
    expected_dt,_,_=run_host(model=host,duration=.02,rain=rain,stage=stage,checkpoints=np.empty(0))
    result=candidate.run_device_loop(duration=.02,rainfall_hyetograph=rain,use_cuda_graph=graph,
        dt_history_capacity=100)
    np.testing.assert_allclose(result['dt_history'],expected_dt,rtol=2.e-13,atol=1.e-14)
    assert np.max(host.get_max_water_depth())>1.e-5
    np.testing.assert_allclose(candidate.get_max_water_depth(),host.get_max_water_depth(),rtol=STATE_RTOL,atol=STATE_ATOL)
    np.testing.assert_allclose(candidate.get_native_water_depth(maximum=True),host.get_native_water_depth(maximum=True),rtol=STATE_RTOL,atol=STATE_ATOL)
    for name,value in host.get_component_state().items():
        np.testing.assert_allclose(candidate.get_component_state()[name],value,rtol=STATE_RTOL,atol=STATE_ATOL)
