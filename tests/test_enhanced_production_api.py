"""Same production model inputs, with only hydraulic method selection changed."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.enhanced_solver import EnhancedSGSSolver
from floodwarp.solver_factory import create_overland_solver
from tests.test_enhanced_production_state import device,STATE_RTOL,STATE_ATOL,VOLUME_ATOL


def test_existing_subgrid_selection_uses_production_enhanced_defaults(device):
    options=model_inputs(device=device,method='enhanced',native_roughness=False,infiltration='constant',diagnostics=True)
    explicit=LocalInertialSolver(**options)
    options['hydraulic_geometry_mode']='subgrid'
    default=LocalInertialSolver(**options)
    assert isinstance(default,EnhancedSGSSolver)
    assert default._enhanced_encoding=='scalar'
    assert default._enhanced_coefficient_tolerance==.005
    assert default._enhanced_geometry.formulation=='transport_star'
    assert default._kinetic_transport==0
    for model in (explicit,default):
        model.set_water_depth(depth=np.full((2,3),.03))
        model.step(fixed_dt=.01)
    for name,value in explicit.get_component_state().items():
        np.testing.assert_array_equal(value,default.get_component_state()[name])


def model_inputs(*,device,method,native_roughness,infiltration,diagnostics):
    native=np.indices((4,6))[1]*.01
    shape=(2,3);active=np.array([[1,1,1],[0,1,1]],np.int32)
    bx=np.array([[1,0,0,2],[1,5,0,2]],np.int32)
    by=np.array([[1,1,1],[1,0,0],[1,1,1]],np.int32)
    inputs=dict(ny=2,nx=3,dx=2.,dy=2.,bed_elevation=native.reshape(2,2,3,2).mean(axis=(1,3)),
        manning_n=np.array([[.03,.04,.05],[.04,.03,.05]]),active_mask=active,device=device,
        hydraulic_geometry_mode=method,subgrid_bed_elevation=native,subgrid_resolution_m=1.,
        minimum_dt=1e-8,maximum_dt=.02,accounting_interval=4,boundary_type_qx=bx,boundary_type_qy=by,
        infiltration_method=infiltration,track_max_water_depth=diagnostics,
        track_cumulative_face_volumes=diagnostics)
    if native_roughness:inputs['subgrid_manning_n']=.03+np.indices(native.shape)[0]*.001
    if infiltration=='exponential':inputs.update(infiltration_f0_m_per_s=.0003+np.arange(6).reshape(shape)*.00001,
        infiltration_fd_m=np.full(shape,.01),infiltration_k_m_per_s=np.full(shape,.00005))
    if infiltration=='green_ampt':inputs.update(infiltration_ks_m_per_s=np.full(shape,.00005),
        infiltration_psi_f_m=np.full(shape,.05),infiltration_delta_theta=np.full(shape,.2))
    return inputs


@pytest.mark.parametrize('method',['traditional','enhanced'])
@pytest.mark.parametrize('native_roughness',[False,True])
@pytest.mark.parametrize('infiltration',['none','constant','exponential','green_ampt'])
@pytest.mark.parametrize('execution',['host','device','graph','lts'])
def test_complete_model_method_switch_preserves_supported_functionality(device,method,native_roughness,infiltration,execution,monkeypatch):
    if execution=='graph' and device=='cpu':pytest.skip('Conditional graphs require CUDA for both methods')
    # Exercise the production reduced/deferred account selection, rather than
    # relying on its size threshold to activate this branch in a small model.
    monkeypatch.setenv('OVERLANDER_REDUCE_MASS','1')
    model=LocalInertialSolver(**model_inputs(device=device,method=method,native_roughness=native_roughness,
        infiltration=infiltration,diagnostics=execution!='lts'))
    assert isinstance(model,EnhancedSGSSolver)==(method=='enhanced')
    assert model.hydraulic_geometry_mode==method
    active=model.active_mask.numpy();initial=np.where(active,.1,0.)
    model.set_water_depth(depth=initial)
    model.set_boundary_stage_qx(stage=np.full((2,4),.15))
    rain=np.where(active,.0002,0.)
    model.set_source_rates(rain=rain,infil=np.where(active,.0001,0.),other=np.where(active,.00002,0.))
    if execution!='lts':
        model.set_additional_source_rate(rate=np.where(active,.00001,0.))
        model.set_infiltration_exclusion_mask(mask=np.array([[1,0,0],[0,0,0]],np.int32))
        model.set_infiltration_storage_limit_m(limit_m=np.where(active,.000005,0.))
    duration=.1
    if execution=='host':
        while model.simulation_time<duration-1e-12:model.step(remaining_time=duration-model.simulation_time)
    elif execution=='lts':
        model.run_spatially_adaptive_loop(duration=duration,max_levels=2,wet_max_level=1)
    else:
        model.run_device_loop(duration=duration,sync_interval=7,use_cuda_graph=execution=='graph',
            checkpoint_times_s=[.05,.1],field_checkpoint_indices=[0,1])
    assert model.simulation_time==pytest.approx(duration,abs=1e-12)
    depth=model.get_water_depth();ledger=model.cumulative_accounting()
    assert np.all(np.isfinite(depth)) and depth.min()>=0.
    np.testing.assert_array_equal(depth[active==0],0.)
    residual=depth.sum(dtype=np.float64)*4.-model.initial_volume-ledger['rain']-ledger['other_sources']-ledger['additional_sources']-ledger['boundary_inflow']+ledger['infiltration']+ledger['boundary_outflow']
    assert abs(residual)<VOLUME_ATOL*100
    assert ledger['rain']>0. and ledger['other_sources']>0.
    if infiltration in ('exponential','green_ampt'):
        assert ledger['infiltration']==pytest.approx(model.cumulative_infiltration_m.sum(dtype=np.float64)*4.,
            rel=STATE_RTOL,abs=VOLUME_ATOL)
    if execution!='lts':
        assert ledger['additional_sources']>0.
        assert model.cumulative_infiltration_m[0,0]==0.
        assert model.get_max_water_depth().shape==(2,3)
        assert model.get_cumulative_face_volumes()['cumulative_net_qx_volume_m3'].shape==(2,4)
        saved=model.capture_state()
        model.step(fixed_dt=.01);expected=model.get_water_depth().copy();expected_ledger=model.cumulative_accounting()
        model.restore_state(state=saved);model.step(fixed_dt=.01)
        np.testing.assert_array_equal(model.get_water_depth(),expected)
        for key,value in expected_ledger.items():
            assert model.cumulative_accounting()[key]==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL)
    if method=='enhanced':
        assert model.get_component_state()['volume_m3'].min()>=0.
        native_depth=model.get_native_water_depth()
        np.testing.assert_allclose(native_depth.sum(dtype=np.float64),model.get_water_depth().sum(dtype=np.float64)*4.,
            rtol=STATE_RTOL,atol=VOLUME_ATOL*10)


@pytest.mark.parametrize('method',['traditional','enhanced'])
def test_factory_and_direct_constructor_choose_same_explicit_method(device,method):
    inputs=model_inputs(device=device,method=method,native_roughness=True,infiltration='green_ampt',diagnostics=True)
    maximum_dt=inputs.pop('maximum_dt');minimum_dt=inputs.pop('minimum_dt')
    direct=LocalInertialSolver(**inputs,maximum_dt=maximum_dt,minimum_dt=minimum_dt)
    factory=create_overland_solver(solver_method='explicit',explicit_maximum_dt=maximum_dt,
        explicit_minimum_dt=minimum_dt,**inputs)
    for model in (direct,factory):
        model.set_water_depth(depth=np.array([[.3,.1,.2],[0.,.1,.2]]))
        model.step(fixed_dt=.02)
    np.testing.assert_array_equal(direct.get_water_depth(),factory.get_water_depth())
    np.testing.assert_array_equal(direct.get_qx(),factory.get_qx())


def test_enhanced_dispatch_uses_current_positional_production_constructor():
    native=np.zeros((2,4));bed=np.zeros((1,2));rough=np.full((1,2),.03);mask=np.ones((1,2),np.int32)
    positional=LocalInertialSolver(1,2,2.,2.,bed,rough,mask,'cpu',None,True,1,1e-8,.02,None,
        None,'enhanced',native,1.)
    keyword=LocalInertialSolver(ny=1,nx=2,dx=2.,dy=2.,bed_elevation=bed,manning_n=rough,active_mask=mask,
        hydraulic_geometry_mode='enhanced',subgrid_bed_elevation=native,subgrid_resolution_m=1.,
        minimum_dt=1e-8,maximum_dt=.02,adaptive_dt_growth_factor=None)
    for model in (positional,keyword):
        assert isinstance(model,EnhancedSGSSolver)
        model.set_water_depth(depth=np.array([[.2,.1]]));model.step(fixed_dt=.02)
    np.testing.assert_array_equal(positional.get_water_depth(),keyword.get_water_depth())


def test_explicit_enhanced_selection_requires_native_terrain():
    with pytest.raises(ValueError,match='requires subgrid'):
        LocalInertialSolver(ny=1,nx=2,dx=2.,dy=2.,bed_elevation=np.zeros((1,2)),
            manning_n=np.full((1,2),.03),active_mask=np.ones((1,2),np.int32),hydraulic_geometry_mode='enhanced')
