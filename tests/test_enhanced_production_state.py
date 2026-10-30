"""Production state/source interface controls; factory promotion remains gated."""
import numpy as np
import pytest
import warp as wp

from floodwarp.config import NP_FLOAT, WP_FLOAT
from floodwarp.enhanced_solver import EnhancedSGSSolver
from floodwarp.enhanced_geometry import component_volumes, build_component_geometry
from floodwarp.enhanced_state import coarse_storage_to_components, coarse_face_connections
from floodwarp.local_inertial_solver import LocalInertialSolver


STATE_RTOL = 4e-6 if NP_FLOAT == np.float32 else 2e-12
STATE_ATOL = 1e-8 if NP_FLOAT == np.float32 else 1e-15
VOLUME_ATOL = 2e-6 if NP_FLOAT == np.float32 else 1e-13


@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    return request.param


def create(*, bed, factor, device, method='constant', mask=None, **kwargs):
    shape=tuple(n//factor for n in bed.shape)
    options=dict(ny=shape[0],nx=shape[1],dx=float(factor),dy=float(factor),
        bed_elevation=bed.reshape(shape[0],factor,shape[1],factor).mean(axis=(1,3)),
        manning_n=np.full(shape,.03),active_mask=np.ones(shape,np.int32) if mask is None else mask,
        subgrid_bed_elevation=bed,subgrid_resolution_m=1.,device=device,
        minimum_dt=1e-7,maximum_dt=.02,track_max_water_depth=True,
        track_cumulative_face_volumes=True,infiltration_method=method)
    options.update(kwargs)
    return EnhancedSGSSolver(maximum_stage_m=float(bed.max()+2.),**options)


def parallel():
    bed=np.ones((5,5));bed[1,:]=0.;bed[3,:]=0.
    return bed


def test_complete_subnormal_recession_limits_actual_infiltration_volume(device):
    if NP_FLOAT!=np.float64:pytest.skip('FP64 subnormal-source regression')
    model=create(bed=np.zeros((10,10)),factor=10,device=device,method='constant')
    model.apply_sources_after_momentum=True
    model.set_source_rates(infil=np.full((1,1),1e-5))
    for initial in (1.015408e-315,1e-310,1e-320,np.nextafter(0.,1.)*3):
        model.set_component_state(volume=np.array([initial]))
        model.step(fixed_dt=.02)
        final=float(model.get_component_state()['volume_m3'][0])
        actual=model.cumulative_accounting()['infiltration']
        assert final>=0. and actual>=0.
        assert final+actual==initial


def test_union_storage_initialization_is_exact_and_face_units_are_signed():
    bed=parallel();g=build_component_geometry(bed_m=bed,factor=5,resolution_m=1.,manning_n=.03,
        maximum_stage_m=3.,open_sides=(0,1,2,3),formulation='transport_star',
        interpolation_tolerance=.02,maximum_ports_per_cell=512,maximum_rows_per_port=32768)
    values=coarse_storage_to_components(geometry=g,native_bed=bed,storage_depth=np.array([[.12]]))
    np.testing.assert_allclose(values,[1.5,1.5],atol=1e-15)
    faces,sign,offsets,members=coarse_face_connections(geometry=g)
    assert np.all(sign[g.external_side==0]==-1)
    assert np.all(sign[g.external_side==1]==1)
    assert np.all(faces[g.external_side<0]==-1)
    assert len(members)==np.count_nonzero(faces>=0)
    assert offsets[-1]==len(members)


def test_component_outputs_preserve_actual_dry_channel(device):
    bed=parallel();s=create(bed=bed,factor=5,device=device)
    s.set_component_state(volume=np.array([1.5,0.]))
    native=s.get_native_water_depth();np.testing.assert_array_equal(native[3,:],0.)
    np.testing.assert_allclose(native[1,:],.3)
    assert s.get_water_depth()[0,0]==pytest.approx(.06)
    assert s.get_subgrid_state()['maximum_local_depth'][0,0]==pytest.approx(.3)
    for _ in range(4):s.step(fixed_dt=.02)
    np.testing.assert_array_equal(s.get_component_state()['volume_m3'],[1.5,0.])
    np.testing.assert_array_equal(s.get_native_water_depth(maximum=True),native)
    assert s.get_max_local_water_depth()[0,0]==pytest.approx(.3)
    assert s.get_max_water_depth()[0,0]==pytest.approx(.06)


def test_open_outlet_does_not_draw_from_other_independent_storage(device):
    s=create(bed=parallel(),factor=5,device=device)
    s.set_component_state(volume=np.array([1.5,0.]))
    bx=s.boundary_type_qx.numpy();bx[:,1]=2;s.set_boundary_type_qx(boundary_type=bx)
    for _ in range(20):s.step(fixed_dt=.02)
    state=s.get_component_state();ledger=s.cumulative_accounting()
    assert state['volume_m3'][0]<1.5 and state['volume_m3'][1]==0.
    assert ledger['boundary_outflow']>0.
    assert float(state['volume_m3'].sum(dtype=np.float64))+ledger['boundary_outflow']==pytest.approx(1.5,abs=VOLUME_ATOL)
    assert s.get_qx()[0,1]*5==pytest.approx(state['discharge_m3_per_s'][s._enhanced_geometry.external_side==1].sum())


@pytest.mark.parametrize('method',['constant','exponential','green_ampt','none'])
def test_shared_production_source_law_matches_uniform_traditional(device,method):
    bed=np.zeros((2,4));extra={}
    if method=='exponential':extra=dict(infiltration_f0_m_per_s=.0008,infiltration_fd_m=.001,infiltration_k_m_per_s=.0001)
    if method=='green_ampt':extra=dict(infiltration_ks_m_per_s=.0001,infiltration_psi_f_m=.05,infiltration_delta_theta=.2)
    enhanced=create(bed=bed,factor=2,device=device,method=method,**extra)
    traditional=LocalInertialSolver(ny=1,nx=2,dx=2.,dy=2.,bed_elevation=np.zeros((1,2)),
        manning_n=np.full((1,2),.03),active_mask=np.ones((1,2),np.int32),
        subgrid_bed_elevation=bed,subgrid_resolution_m=1.,hydraulic_geometry_mode='traditional',
        device=device,infiltration_method=method,**extra)
    for s in (enhanced,traditional):
        s.set_water_depth(depth=np.full((1,2),.003))
        s.set_source_rates(rain=np.full((1,2),.0002),infil=np.full((1,2),.0008))
        s.set_additional_source_rate(rate=np.full((1,2),.0001))
        s.set_infiltration_exclusion_mask(mask=np.array([[1,0]],np.int32))
        s.set_infiltration_storage_limit_m(limit_m=np.full((1,2),.000007))
        for _ in range(5):s.step(fixed_dt=.01)
    # Unequal uptake activates transport; both reproduce the same staggered
    # uniform-channel momentum in this case.
    np.testing.assert_allclose(enhanced.get_water_depth(),traditional.get_water_depth(),rtol=STATE_RTOL,atol=STATE_ATOL)
    np.testing.assert_allclose(enhanced.cumulative_infiltration_m,traditional.cumulative_infiltration_m,rtol=STATE_RTOL,atol=STATE_ATOL)
    for name,value in traditional.cumulative_accounting().items():
        assert enhanced.cumulative_accounting()[name]==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL)


def test_actual_component_soil_state_and_caps_are_independent(device):
    s=create(bed=parallel(),factor=5,device=device,method='exponential',
        infiltration_f0_m_per_s=.001,infiltration_fd_m=.01,infiltration_k_m_per_s=.0001)
    s.set_component_state(volume=np.array([1.5,0.]))
    s.set_infiltration_storage_limit_m(limit_m=np.array([[.00001]]))
    for _ in range(5):s.step(fixed_dt=.02)
    states=s.get_component_state()
    assert states['cumulative_infiltration_m'][0]>0
    assert states['cumulative_infiltration_m'][1]==0
    assert states['remaining_infiltration_cap_m'][1]==pytest.approx(.00001)
    assert float(states['volume_m3'].sum(dtype=np.float64))+s.cumulative_accounting()['infiltration']==pytest.approx(1.5,abs=VOLUME_ATOL)


def test_global_checkpoint_covers_component_momentum_peak_and_soil(device):
    bed=parallel();bed[2,:]=.1
    s=create(bed=bed,factor=5,device=device)
    s.set_component_state(volume=np.array([1.5,0.]))
    s.set_source_rates(rain=np.array([[.001]]),infil=np.array([[.0002]]))
    for _ in range(3):s.step(fixed_dt=.02)
    snapshot=s.capture_state();before={name:value.numpy().copy() for name,value in vars(s).items() if isinstance(value,wp.array)}
    s.step(fixed_dt=.01);expected=s.get_component_state()
    s.restore_state(state=snapshot)
    for name,value in before.items():np.testing.assert_array_equal(getattr(s,name).numpy(),value,err_msg=name)
    s.step(fixed_dt=.01);after=s.get_component_state()
    for name,value in expected.items():np.testing.assert_allclose(after[name],value,rtol=STATE_RTOL,atol=STATE_ATOL)


def test_active_mask_and_nonuniform_native_manning_run(device):
    bed=np.zeros((4,6));mask=np.array([[1,1,0],[1,0,1]],np.int32)
    native_n=np.full_like(bed,.03);native_n[:,1::2]=.05
    s=create(bed=bed,factor=2,device=device,mask=mask,subgrid_manning_n=native_n)
    s.set_water_depth(depth=np.full(mask.shape,.1))
    s.set_source_rates(rain=np.full(mask.shape,.001))
    s.step(fixed_dt=.01)
    assert np.all(s.get_water_depth()[mask==0]==0.)
    assert np.all(s.get_native_water_depth()[s._enhanced_geometry.pixel_owner<0]==0.)
    assert s.cumulative_accounting()['rain']==pytest.approx(16.*.001*.01)
























def test_capacity_getter_averages_actual_soil_states(device):
    s=create(bed=parallel(),factor=5,device=device,method='exponential',
        infiltration_f0_m_per_s=.001,infiltration_fd_m=.01,infiltration_k_m_per_s=.0001)
    s.set_component_state(volume=np.array([1.5,0.]))
    for _ in range(5):s.step(fixed_dt=.02)
    state=s.get_component_state();F=state['cumulative_infiltration_m'];g=s._enhanced_geometry
    actual=np.sum((.0001+.0009*np.exp(-F/.01))*g.source_area)/25.
    assert s.current_infiltration_capacity_m_per_s[0,0]==pytest.approx(actual,rel=STATE_RTOL,abs=STATE_ATOL)

















def test_face_initialization_preserves_total_units_and_exact_checkpoint(device):
    bed=np.zeros((2,4));s=create(bed=bed,factor=2,device=device)
    s.set_water_depth(depth=np.full((1,2),.2))
    qx=np.array([[0.,.01,0.]]);qy=np.zeros((2,2))
    s.set_face_fluxes(qx=qx,qy=qy)
    np.testing.assert_allclose(s.get_qx(),qx,atol=STATE_ATOL)
    state=s.get_component_state();g=s._enhanced_geometry
    assert state['discharge_m3_per_s'][g.endpoints[:,1]>=0].sum()==pytest.approx(.02)
    snap=s.capture_state();s.step(fixed_dt=.01);s.restore_state(state=snap)
    np.testing.assert_array_equal(s.get_component_state()['discharge_m3_per_s'],state['discharge_m3_per_s'])





























def test_closed_ports_are_not_equivalent_to_absent_ports_in_reduction():
    """Falsification control: a runtime wall mask alone cannot select tables.

    This is an integration blocker, not permission to change the reference
    transport-star reduction or to advertise closed-port parity.
    """
    from floodwarp.enhanced_geometry import table_value
    bed=parallel();bed[2,:]=.1;values=[]
    for sides in ((),(0,1,2,3)):
        geometry=build_component_geometry(bed_m=bed,factor=5,resolution_m=1.,manning_n=.03,
            maximum_stage_m=3.,open_sides=sides,formulation='transport_star',interpolation_tolerance=.02,
            maximum_ports_per_cell=512,maximum_rows_per_port=32768)
        internal=np.flatnonzero(geometry.endpoints[:,1]>=0)
        inertia=0.
        for p in geometry.halves[internal[0]]:
            inertia+=table_value(stage=.3,sill=geometry.port_sill[p],
                rows=geometry.rows[geometry.offsets[p]:geometry.offsets[p+1]])[0]
        values.append(inertia)
    assert values[0]==pytest.approx(1.442090864032231,rel=1e-7)
    assert values[1]==pytest.approx(1.,rel=1e-7)
    assert abs(values[1]-values[0])>.4


@pytest.mark.parametrize('changed',['terrain','roughness'])
def test_reused_geometry_is_bound_to_physical_inputs(changed):
    bed=np.zeros((2,4))
    first=create(bed=bed,factor=2,device='cpu')
    options={}
    if changed=='terrain':bed[0,0]=.01
    else:options['manning_n']=np.array([[.03,.04]])
    with pytest.raises(ValueError,match='not bound'):
        create(bed=bed,factor=2,device='cpu',enhanced_geometry=first._enhanced_geometry,**options)


def test_constructor_uses_the_actual_closed_port_reduction(device):
    bed=parallel();bed[2,:]=.1
    s=create(bed=bed,factor=5,device=device,coefficient_encoding='uncompressed')
    native=np.maximum(.3-bed,0.)
    s.set_component_state(volume=component_volumes(geometry=s._enhanced_geometry,native_depth=native))
    s._component_stages();s._component_coefficients()
    assert np.all(s._enhanced_geometry.external_side<0)
    np.testing.assert_allclose(s._ec_d.numpy(),[1.4420908607109375],rtol=STATE_RTOL,atol=STATE_ATOL)


def test_boundary_port_changes_preserve_storage_q_and_exact_rollback(device):
    bed=parallel();bed[2,:]=.1
    s=create(bed=bed,factor=5,device=device)
    s.set_component_state(volume=np.array([1.5,0.]))
    s.set_infiltration_storage_limit_m(limit_m=np.array([[.001]]))
    s.set_source_rates(rain=np.array([[.0002]]),infil=np.array([[.0001]]))
    s.step(fixed_dt=.01)
    closed_geometry=s._enhanced_geometry;snapshot=s.capture_state()
    saved_arrays={name:field.numpy().copy() for name,field in vars(s).items() if isinstance(field,wp.array)}
    old=s.get_component_state();old_q=dict(zip(map(tuple,closed_geometry.connection_keys),old['discharge_m3_per_s']))
    boundary=s.boundary_type_qx.numpy();boundary[:,0]=3
    s.set_boundary_type_qx(boundary_type=boundary)
    g=s._enhanced_geometry;current=s.get_component_state()
    np.testing.assert_array_equal(current['volume_m3'],old['volume_m3'])
    np.testing.assert_array_equal(current['cumulative_infiltration_m'],old['cumulative_infiltration_m'])
    np.testing.assert_array_equal(current['remaining_infiltration_cap_m'],old['remaining_infiltration_cap_m'])
    for key,q in zip(g.connection_keys,current['discharge_m3_per_s']):
        assert q==old_q.get(tuple(key),0.)
    assert len(g.endpoints)>len(closed_geometry.endpoints)
    stage=s.boundary_stage_qx.numpy();stage[:,0]=.4;s.set_boundary_stage_qx(stage=stage)
    s.step(fixed_dt=.01)
    assert np.any(s.get_component_state()['discharge_m3_per_s'][g.external_side==0]!=0.)
    s.restore_state(state=snapshot)
    assert s._enhanced_geometry is closed_geometry
    for name,value in saved_arrays.items():
        np.testing.assert_array_equal(getattr(s,name).numpy(),value,err_msg=name)
    s.step(fixed_dt=.01)
    assert s.get_component_state()['volume_m3'].min()>=0.


def test_closing_ports_records_removed_kinetic_metric_without_moving_water(device):
    s=create(bed=np.zeros((2,4)),factor=2,device=device)
    s.set_water_depth(depth=np.array([[.3,.1]]))
    boundary=s.boundary_type_qx.numpy();boundary[:,-1]=2
    s.set_boundary_type_qx(boundary_type=boundary);s.step(fixed_dt=.01)
    g=s._enhanced_geometry;volume=s._ec_volume.numpy().copy();q=s._ec_q.numpy();d=s._ec_previous_d.numpy()
    old_q=dict(zip(map(tuple,g.connection_keys),q))
    removed=g.external_side==1
    expected=.5*np.sum(d[removed]*q[removed].astype(np.float64)**2)
    assert expected>0
    boundary[:,-1]=1;s.set_boundary_type_qx(boundary_type=boundary)
    np.testing.assert_array_equal(s._ec_volume.numpy(),volume)
    assert s._ec_boundary_closure_loss.numpy()[0]==pytest.approx(expected,rel=STATE_RTOL)
    for key,actual in zip(s._enhanced_geometry.connection_keys,s._ec_q.numpy()):assert actual==old_q[tuple(key)]
    assert s.get_qx()[0,-1]==0.


def test_stage_updates_and_steps_do_not_rebuild_native_graph(device,monkeypatch):
    s=create(bed=np.zeros((2,4)),factor=2,device=device)
    s.set_water_depth(depth=np.full((1,2),.2))
    boundary=s.boundary_type_qx.numpy();boundary[:,-1]=2;s.set_boundary_type_qx(boundary_type=boundary)
    geometry=s._enhanced_geometry

    def forbidden_rebuild(*,face_open_x,face_open_y):
        raise AssertionError('native graph rebuild during stage/step')

    monkeypatch.setattr(s,'_build_boundary_geometry',forbidden_rebuild)
    boundary[:,-1]=3;s.set_boundary_type_qx(boundary_type=boundary)
    stages=s.boundary_stage_qx.numpy();stages[:,-1]=.1;s.set_boundary_stage_qx(stage=stages)
    s.step(fixed_dt=.01)
    assert s._enhanced_geometry is geometry


def test_internal_coarse_wall_removes_transport_between_active_cells(device):
    s=create(bed=np.zeros((2,4)),factor=2,device=device)
    s.set_water_depth(depth=np.array([[.3,.1]]));s.step(fixed_dt=.01)
    assert s.get_qx()[0,1]>0
    before=s._ec_volume.numpy()
    boundary=s.boundary_type_qx.numpy();boundary[:,1]=1;s.set_boundary_type_qx(boundary_type=boundary)
    assert s._connection_count==0
    s.step(fixed_dt=.01)
    np.testing.assert_array_equal(s._ec_volume.numpy(),before)
    assert s.get_qx()[0,1]==0.


def test_masked_west_free_has_correct_group_sign_and_conservation(device):
    bed=np.zeros((2,4));s=create(bed=bed,factor=2,device=device,mask=np.array([[0,1]],np.int32))
    s.set_water_depth(depth=np.array([[0.,.2]]))
    boundary=s.boundary_type_qx.numpy();boundary[:,1]=6;s.set_boundary_type_qx(boundary_type=boundary)
    s.step(fixed_dt=.01)
    assert s.get_qx()[0,1]<0 and s.cumulative_accounting()['boundary_outflow']>0
    assert float(s._ec_volume.numpy().sum(dtype=np.float64))+s.cumulative_accounting()['boundary_outflow']==pytest.approx(.8,abs=VOLUME_ATOL)


def test_boundary_rebuild_retains_declared_spacing_in_fp32(device):
    s=create(bed=np.zeros((2,4)),factor=2,device=device,dx=1.9,dy=1.9,subgrid_resolution_m=.95)
    s.set_water_depth(depth=np.full((1,2),.1))
    before=s._ec_volume.numpy();storage=s._enhanced_geometry.storage_rows.copy()
    boundary=s.boundary_type_qx.numpy();boundary[:,-1]=2;s.set_boundary_type_qx(boundary_type=boundary)
    assert s._enhanced_geometry.resolution==.95
    np.testing.assert_array_equal(s._enhanced_geometry.storage_rows,storage)
    np.testing.assert_array_equal(s._ec_volume.numpy(),before)


def test_reset_from_live_component_numpy_views_preserves_inputs(device):
    model=create(bed=np.zeros((2,4)),factor=2,device=device)
    volume=np.array([.2,.4]);discharge=np.full(model._connection_count,.003)
    model.set_component_state(volume=volume,discharge=discharge)
    views=model.get_component_state()
    saved={name:values.copy() for name,values in views.items()}
    model.set_component_state(volume=views['volume_m3'],discharge=views['discharge_m3_per_s'])
    actual=model.get_component_state()
    for name in ('volume_m3','discharge_m3_per_s','stage_m'):
        np.testing.assert_array_equal(actual[name],saved[name])
    assert model.initial_volume==pytest.approx(float(saved['volume_m3'].sum()),abs=VOLUME_ATOL)
