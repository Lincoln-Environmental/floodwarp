"""Component momentum subcycling with independent base-clock volume exchange."""
import numpy as np
import pytest

from floodwarp.rainfall import Hyetograph
from tests.test_enhanced_production_state import create,device,STATE_RTOL,STATE_ATOL,VOLUME_ATOL


def make_lts(*,device,method='constant',after=False,maximum_dt=.02,track_max_water_depth=False):
    bed=np.ones((5,20));bed[1,:]=0.;bed[3,:]=0.;bed[2,:]=.1
    options={}
    if method=='exponential':options=dict(infiltration_f0_m_per_s=.0003,infiltration_fd_m=.01,infiltration_k_m_per_s=.00005)
    if method=='green_ampt':options=dict(infiltration_ks_m_per_s=.00005,infiltration_psi_f_m=.05,infiltration_delta_theta=.2)
    model=create(bed=bed,factor=5,device=device,method=method,track_max_water_depth=track_max_water_depth,
        track_cumulative_face_volumes=False,maximum_dt=maximum_dt,adaptive_dt_growth_factor=None,**options)
    model.apply_sources_after_momentum=after
    model.set_component_state(volume=np.array([1.5,.1,.1,0.,0.,0.,0.,0.]))
    model.set_source_rates(rain=np.array([[.0002,.0001,.0003,.0002]]),
        infil=np.array([[.0001,.00015,.0002,.0001]]),other=np.full((1,4),.00001))
    return model


@pytest.mark.parametrize('after',[False,True])
@pytest.mark.parametrize('use_graph',[False,True])
def test_component_lts_tracks_actual_native_and_coarse_maxima(device,after,use_graph):
    host=make_lts(device=device,method='none',after=after,track_max_water_depth=True)
    lts=make_lts(device=device,method='none',after=after,track_max_water_depth=True)
    duration=.1
    while host.simulation_time<duration-1e-12:
        host.step(remaining_time=duration-host.simulation_time)
    lts.run_spatially_adaptive_loop(duration=duration,max_levels=0,wet_max_level=0,use_device_graph=use_graph)
    np.testing.assert_allclose(lts.get_max_water_depth(),host.get_max_water_depth(),rtol=STATE_RTOL,atol=STATE_ATOL)
    np.testing.assert_allclose(lts.get_native_water_depth(maximum=True),
        host.get_native_water_depth(maximum=True),rtol=STATE_RTOL,atol=STATE_ATOL)
    assert np.all(lts.get_native_water_depth(maximum=True)>=lts.get_native_water_depth()-STATE_ATOL)
    assert lts.get_component_state()['volume_m3'].min()>=0.


@pytest.mark.parametrize('method',['none','constant','exponential','green_ampt'])
@pytest.mark.parametrize('after',[False,True])
@pytest.mark.parametrize('use_graph',[False,True])
def test_level_zero_recovers_global_component_step(device,method,after,use_graph):
    host=make_lts(device=device,method=method,after=after)
    lts=make_lts(device=device,method=method,after=after)
    duration=.1
    while host.simulation_time<duration-1e-12:host.step(remaining_time=duration-host.simulation_time)
    result=lts.run_spatially_adaptive_loop(duration=duration,max_levels=0,wet_max_level=0,use_device_graph=use_graph)
    assert result['base_steps']==result['macro_steps']
    for key,value in host.get_component_state().items():
        np.testing.assert_allclose(lts.get_component_state()[key],value,rtol=STATE_RTOL,atol=STATE_ATOL,err_msg=key)
    for key,value in host.cumulative_accounting().items():
        assert lts.cumulative_accounting()[key]==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL)


def test_component_multirate_momentum_is_conservative_and_reduces_updates(device):
    model=create(bed=np.zeros((2,32)),factor=2,device=device,method='none',
        track_max_water_depth=False,track_cumulative_face_volumes=False,maximum_dt=1.,adaptive_dt_growth_factor=None)
    model.set_water_depth(depth=np.array([[1.5]+[.02]*15]))
    observed=[]
    def record_levels(macro_steps,time_s,schedule):
        observed.append(model._ec_lts_levels.numpy().copy())
    result=model.run_spatially_adaptive_loop(duration=2.,max_levels=3,wet_max_level=3,
        drainage_max_level=-1,on_macro_sync=record_levels)
    volume=model.get_component_state()['volume_m3']
    assert volume.min()>=0.
    assert volume.sum(dtype=np.float64)==pytest.approx(model.initial_volume,abs=VOLUME_ATOL*100)
    assert any(np.any(levels>0) for levels in observed)
    assert result['component_momentum_updates']<model._connection_count*result['base_steps']
    assert result['component_volume_updates']==model._component_count*result['base_steps']
    assert result['component_continuity_base_clock']
    assert result['cell_work_fraction']==1.


@pytest.mark.parametrize('after',[False,True])
@pytest.mark.parametrize('use_graph',[False,True])
def test_lts_rainfall_sources_are_accounted_once_over_independent_storage(device,after,use_graph):
    model=make_lts(device=device,method='green_ampt',after=after,maximum_dt=.05)
    rain=Hyetograph(node_times_s=np.array([0.,.04,.1,.2]),
        node_rates_m_per_s=np.array([.0001,.0003,.0002,0.]),storm_type='test',metadata={})
    model.run_spatially_adaptive_loop(duration=.2,rainfall_hyetograph=rain,max_levels=3,wet_max_level=2,use_device_graph=use_graph)
    ledger=model.cumulative_accounting();v=model.get_component_state()['volume_m3']
    residual=v.sum(dtype=np.float64)-model.initial_volume-ledger['rain']-ledger['other_sources']+ledger['infiltration']
    assert abs(residual)<VOLUME_ATOL*20
    assert v.min()>=0.
    assert model.get_component_state()['cumulative_infiltration_m'].max()>0.


def test_lts_never_pools_disconnected_storage(device):
    bed=np.ones((5,15));bed[1,:]=0.;bed[3,:]=0.
    model=create(bed=bed,factor=5,device=device,track_max_water_depth=False,
        track_cumulative_face_volumes=False,maximum_dt=.2)
    model.set_component_state(volume=np.array([1.5,0.,0.,0.,0.,0.]))
    boundary=model.boundary_type_qx.numpy();boundary[:,-1]=2;model.set_boundary_type_qx(boundary_type=boundary)
    model.run_spatially_adaptive_loop(duration=2.,max_levels=3,wet_max_level=2)
    volume=model.get_component_state()['volume_m3']
    np.testing.assert_array_equal(volume[1::2],0.)
    assert volume.sum(dtype=np.float64)+model.cumulative_accounting()['boundary_outflow']==pytest.approx(1.5,abs=VOLUME_ATOL*20)


@pytest.mark.parametrize('after',[False,True])
def test_actual_multirate_graph_and_compact_execution_match(device,after):
    import warp as wp
    options=dict(bed=np.zeros((2,32)),factor=2,device=device,method='green_ampt',
        track_max_water_depth=False,track_cumulative_face_volumes=False,maximum_dt=1.,adaptive_dt_growth_factor=None,
        infiltration_ks_m_per_s=.00005,infiltration_psi_f_m=.05,infiltration_delta_theta=.2)
    compact=create(**options);graph=create(**options)
    results=[]
    for model,use_graph in ((compact,False),(graph,True)):
        model.apply_sources_after_momentum=after
        model.set_water_depth(depth=np.array([[1.5]+[.02]*15]))
        model.set_source_rates(rain=np.full((1,16),.0001),infil=np.full((1,16),.00005))
        results.append(model.run_spatially_adaptive_loop(duration=2.,max_levels=3,wet_max_level=3,
            drainage_max_level=-1,use_device_graph=use_graph))
    if wp.get_device(device).is_cuda:assert results[1]['schedule_mode']=='device_graph_subcycle'
    assert results[0]['base_steps']==results[1]['base_steps']
    assert results[0]['component_momentum_updates']==results[1]['component_momentum_updates']
    assert results[1]['component_momentum_updates']<graph._connection_count*results[1]['base_steps']
    for key,value in compact.get_component_state().items():
        np.testing.assert_allclose(graph.get_component_state()[key],value,rtol=STATE_RTOL,atol=STATE_ATOL)
    for key,value in compact.cumulative_accounting().items():
        assert graph.cumulative_accounting()[key]==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL)
