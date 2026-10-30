"""Full component structure transfers with the production hydraulic laws."""
import numpy as np
import pytest

from floodwarp.local_inertial_solver import LocalInertialSolver
from tests.test_enhanced_production_state import create,device,parallel,STATE_RTOL,STATE_ATOL,VOLUME_ATOL
from tests.test_hydraulic_structures import culvert,weir


def structure_record(*,kind):
    if kind=='box':return culvert()
    if kind=='circular':return culvert(shape='circular',diameter_m=1.)
    if kind=='sharp':return weir(crest_elevation_m=.1)
    if kind=='broad':return weir(weir_type='broad_crested',crest_elevation_m=.1)
    return dict(id='bridge',type='bridge',inlet_row=0,inlet_col=0,outlet_row=0,outlet_col=1,
        opening_width_m=1.,opening_height_m=1.,opening_invert_m=0.,length_m=2.,manning_n=.03,
        entrance_loss_coefficient=.5,exit_loss_coefficient=1.)


def close_interfaces(*,model):
    model.set_boundary_type_qx(boundary_type=np.ones((model.ny,model.nx+1),dtype=np.int32))
    model.set_boundary_type_qy(boundary_type=np.ones((model.ny+1,model.nx),dtype=np.int32))


@pytest.mark.parametrize('kind',['box','circular','sharp','broad','bridge'])
@pytest.mark.parametrize('reverse',[False,True])
def test_complete_structure_steps_match_traditional_single_component(device,kind,reverse):
    record=structure_record(kind=kind);bed=np.zeros((2,4));records=(record,)
    enhanced=create(bed=bed,factor=2,device=device,method='none',structures=records)
    traditional=LocalInertialSolver(ny=1,nx=2,dx=2.,dy=2.,bed_elevation=np.zeros((1,2)),
        manning_n=np.full((1,2),.03),active_mask=np.ones((1,2),np.int32),device=device,
        hydraulic_geometry_mode='traditional',subgrid_bed_elevation=bed,subgrid_resolution_m=1.,
        infiltration_method='none',structures=records,track_max_water_depth=True)
    initial=np.array([[.05,.75]]) if reverse else np.array([[.75,.05]])
    for model in (traditional,enhanced):
        close_interfaces(model=model);model.set_water_depth(depth=initial)
        for _ in range(20):model.step(fixed_dt=.01)
    np.testing.assert_allclose(enhanced.get_water_depth(),traditional.get_water_depth(),rtol=STATE_RTOL,atol=STATE_ATOL)
    for key,value in traditional.get_structure_diagnostics()[0].items():
        actual=enhanced.get_structure_diagnostics()[0][key]
        if isinstance(value,(float,int)):
            assert actual==pytest.approx(value,rel=STATE_RTOL,abs=STATE_ATOL),key
        else:assert actual==value
    assert enhanced.get_component_state()['volume_m3'].sum(dtype=np.float64)==pytest.approx(3.2,abs=VOLUME_ATOL*10)
    assert enhanced.get_structure_mass_balance_diagnostics()['internal_transfer_residual_m3']==0.


def test_explicit_structure_endpoints_do_not_draw_from_disconnected_pool(device):
    bed=np.tile(parallel(),(1,2))
    record=culvert(inlet_native_row=3,inlet_native_col=2,outlet_native_row=3,outlet_native_col=7)
    model=create(bed=bed,factor=5,device=device,structures=(record,))
    close_interfaces(model=model)
    # The engineered link joins the lower channel; only the upper one is wet.
    initial=np.array([1.5,0.,0.,0.]);model.set_component_state(volume=initial)
    for _ in range(4):model.step(fixed_dt=.02)
    np.testing.assert_array_equal(model.get_component_state()['volume_m3'],initial)
    assert model.get_structure_diagnostics()[0]['net_transferred_volume_m3']==0.
    model.set_component_state(volume=np.array([1.5,.5,0.,0.]))
    model.step(fixed_dt=.02)
    result=model.get_component_state()['volume_m3']
    assert result[0]==1.5 and result[2]==0.
    assert result[1]<.5 and result[3]>0.
    assert result.sum(dtype=np.float64)==pytest.approx(2.,abs=VOLUME_ATOL)


def test_structure_can_connect_two_actual_components_in_one_cell(device):
    record=culvert(outlet_col=0,allow_same_cell=True,
        inlet_native_row=1,inlet_native_col=2,outlet_native_row=3,outlet_native_col=2)
    model=create(bed=parallel(),factor=5,device=device,structures=(record,))
    model.set_component_state(volume=np.array([1.5,0.]))
    model.step(fixed_dt=.02)
    result=model.get_component_state()['volume_m3']
    assert 0.<result[1]<1.5
    assert result.sum(dtype=np.float64)==pytest.approx(1.5,abs=VOLUME_ATOL)
    assert model.get_structure_diagnostics()[0]['net_transferred_volume_m3']==pytest.approx(result[1],abs=VOLUME_ATOL)


def test_structures_compete_for_only_their_component_donor_volume(device):
    records=tuple(culvert(id=str(k),outlet_col=k+1) for k in range(4))
    bed=np.zeros((2,10));model=create(bed=bed,factor=2,device=device,structures=records)
    close_interfaces(model=model);model.set_water_depth(depth=np.array([[.001,0.,0.,0.,0.]]))
    model._apply_structure_transfers(dt=100.)
    result=model.get_component_state()['volume_m3']
    assert result.min()>=0.
    assert result.sum(dtype=np.float64)==pytest.approx(.004,abs=VOLUME_ATOL)
    np.testing.assert_allclose(result[1:],.001,rtol=STATE_RTOL,atol=STATE_ATOL)
    assert all(record['limiter_activations']==1 for record in model.get_structure_diagnostics())


def test_structure_checkpoint_and_reset_preserve_complete_ledgers(device):
    model=create(bed=np.zeros((2,4)),factor=2,device=device,structures=(culvert(),))
    close_interfaces(model=model);model.set_water_depth(depth=np.array([[.75,0.]]))
    model.step(fixed_dt=.02);saved=model.capture_state()
    model.step(fixed_dt=.02)
    expected=model.get_structure_diagnostics();volume=model.get_component_state()['volume_m3'].copy()
    model.restore_state(state=saved);model.step(fixed_dt=.02)
    assert model.get_structure_diagnostics()==expected
    np.testing.assert_array_equal(model.get_component_state()['volume_m3'],volume)
    model.set_water_depth(depth=np.array([[.2,.1]]))
    assert model.get_structure_diagnostics()[0]['net_transferred_volume_m3']==0.
    assert model.get_structure_diagnostics()[0]['instantaneous_q_m3s']==0.


@pytest.mark.parametrize('fields',[
    dict(inlet_native_row=1),dict(inlet_native_row=.5,inlet_native_col=0),
    dict(inlet_native_row=0,inlet_native_col=4),
])
def test_structure_native_position_is_validated(fields):
    with pytest.raises(ValueError,match='native'):
        create(bed=np.zeros((2,4)),factor=2,device='cpu',structures=(culvert(**fields),))
