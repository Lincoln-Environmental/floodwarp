"""Reusable boundary topology and coarse CFL predictor interface controls."""
import numpy as np
import pytest
import warp as wp

import floodwarp.enhanced_solver as enhanced_module
from floodwarp.enhanced_geometry import build_component_geometry,pack_coefficients
from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_solver import LocalInertialSolver
from tests.test_enhanced_production_state import create,device,parallel,STATE_RTOL,STATE_ATOL


def test_initial_boundary_types_match_existing_setter_semantics(device):
    options=dict(ny=1,nx=2,dx=2.,dy=2.,bed_elevation=np.zeros((1,2)),
        manning_n=np.full((1,2),.03),active_mask=np.ones((1,2),np.int32),device=device)
    bx=np.array([[3,0,2]],np.int32);by=np.ones((2,2),np.int32)
    configured=LocalInertialSolver(**options,boundary_type_qx=bx,boundary_type_qy=by)
    ordinary=LocalInertialSolver(**options)
    ordinary.set_boundary_type_qx(boundary_type=bx);ordinary.set_boundary_type_qy(boundary_type=by)
    for model in (configured,ordinary):
        model.set_water_depth(depth=np.array([[.2,.1]]))
        model.set_boundary_stage_qx(stage=np.full_like(bx,.15,dtype=float))
        for _ in range(4):model.step(fixed_dt=.01)
    np.testing.assert_array_equal(configured.get_water_depth(),ordinary.get_water_depth())
    np.testing.assert_array_equal(configured.get_qx(),ordinary.get_qx())


def test_reused_open_geometry_is_not_rebuilt_for_temporary_default_walls(device,monkeypatch):
    bed=parallel();bx=np.array([[3,2]],np.int32);by=np.ones((2,1),np.int32)
    geometry=build_component_geometry(bed_m=bed,factor=5,resolution_m=1.,manning_n=.03,
        maximum_stage_m=3.,open_sides=(0,1,2,3),formulation='transport_star',
        interpolation_tolerance=.02,maximum_ports_per_cell=512,maximum_rows_per_port=32768,
        face_open_x=bx!=1,face_open_y=by!=1)
    geometry=pack_coefficients(geometry=geometry,relative_tolerance=.005)
    def unexpected_rebuild(**kwargs):
        raise AssertionError('Supplied geometry already has the actual model boundary topology')
    monkeypatch.setattr(enhanced_module,'build_component_geometry',unexpected_rebuild)
    model=create(bed=bed,factor=5,device=device,enhanced_geometry=geometry,
        boundary_type_qx=bx,boundary_type_qy=by)
    assert model._enhanced_geometry is geometry
    model.set_component_state(volume=np.array([1.5,0.]))
    model.step(fixed_dt=.02)
    assert model.get_component_state()['volume_m3'][1]==0.


@pytest.mark.parametrize('requested_depth',[.03,.06,.12])
def test_coarse_cfl_predictor_preserves_live_independent_storage(device,requested_depth):
    model=create(bed=parallel(),factor=5,device=device)
    control=create(bed=parallel(),factor=5,device=device)
    for candidate in (model,control):candidate.set_component_state(volume=np.array([1.5,0.]))
    initial={k:v.copy() for k,v in model.get_component_state().items()}
    field=wp.array([[requested_depth]],dtype=WP_FLOAT,device=device)
    dt=model.calculate_cfl_dt(water_depth=field)
    assert 0.<dt<=model.maximum_dt
    requested=requested_depth*25.;g=model._enhanced_geometry
    expected=(np.array([1.5,0.])+(requested-1.5)*g.source_area/25.
        if requested>=1.5 else np.array([requested,0.]))
    np.testing.assert_allclose(model._ec_predictor_volume.numpy(),expected,rtol=STATE_RTOL,atol=STATE_ATOL)
    for key,value in initial.items():
        np.testing.assert_array_equal(model.get_component_state()[key],value)
    model.step(fixed_dt=.02);control.step(fixed_dt=.02)
    for key,value in control.get_component_state().items():
        np.testing.assert_array_equal(model.get_component_state()[key],value)


@pytest.mark.parametrize('invalid',[-1.,np.nan,np.inf])
def test_invalid_predictor_does_not_poison_live_hydraulic_state(device,invalid):
    model=create(bed=parallel(),factor=5,device=device)
    model.set_component_state(volume=np.array([1.5,0.]))
    with pytest.raises(ValueError,match='finite and nonnegative'):
        model.calculate_cfl_dt(water_depth=wp.array([[invalid]],dtype=WP_FLOAT,device=device))
    model.step(fixed_dt=.02)
    np.testing.assert_array_equal(model.get_component_state()['volume_m3'],[1.5,0.])
