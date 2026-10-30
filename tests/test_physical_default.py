"""Context-specific implicit defaults and unambiguous numerical provenance."""
from pathlib import Path

import numpy as np
import pytest

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.solver_factory import create_overland_solver
from floodwarp.validation_profiles import resolve_validation_profile


def inputs():
    return dict(ny=2,nx=2,dx=2.,dy=2.,bed_elevation=np.zeros((2,2)),
        manning_n=np.full((2,2),.04),active_mask=np.ones((2,2),dtype=np.int32),device='cpu')


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('profile',[None,'default','lisflood_fp_acc','lisflood_fp_acc_parity','physical_candidate'])
def test_ordinary_constructor_and_factory_resolve_profile(factory,profile):
    options=inputs()
    if factory:
        solver=create_overland_solver(solver_method='explicit',validation_profile=profile,**options)
    else:
        solver=LocalInertialSolver(validation_profile=profile,**options)
    assert solver.validation_profile==('physical_candidate' if profile is None else profile)
    assert solver.numerical_configuration()['validation_profile']==solver.validation_profile
    assert solver.numerical_configuration()['free_outflow_closure']==(
        'local_or_prescribed_slope' if profile in ('lisflood_fp_acc','lisflood_fp_acc_parity') else 'dry_exterior_head')


@pytest.mark.parametrize('mode',['traditional','enhanced','subgrid'])
@pytest.mark.parametrize('factory',[False,True])
def test_sgs_preserves_implicit_legacy_profile(mode,factory):
    options=dict(inputs(),hydraulic_geometry_mode=mode,subgrid_bed_elevation=np.zeros((4,4)),subgrid_resolution_m=1.)
    solver=create_overland_solver(solver_method='explicit',**options) if factory else LocalInertialSolver(**options)
    assert solver.validation_profile=='default'












def test_direct_enhanced_base_initialization_does_not_inherit_physical_default():
    from floodwarp.enhanced_solver import EnhancedSGSSolver
    solver=EnhancedSGSSolver(subgrid_bed_elevation=np.zeros((4,4)),subgrid_resolution_m=1.,**inputs())
    assert solver.validation_profile=='default'
    assert solver.hydraulic_geometry_mode=='enhanced'


def test_ordinary_subclass_inherits_physical_default():
    class OrdinarySubclass(LocalInertialSolver):
        pass
    solver=OrdinarySubclass(**inputs())
    assert solver.validation_profile=='physical_candidate'


def test_restore_rejects_missing_or_different_numerical_identity():
    preferred=LocalInertialSolver(**inputs())
    legacy=LocalInertialSolver(validation_profile='default',**inputs())
    state=legacy.capture_state()
    with pytest.raises(ValueError,match='numerical provenance'):
        preferred.restore_state(state=state)
    del state['host']['numerical_configuration']
    with pytest.raises(ValueError,match='ambiguous historical'):
        legacy.restore_state(state=state)


def test_ordinary_implicit_matches_explicit_candidate_and_records_boundary():
    implicit=LocalInertialSolver(free_outflow_closure='local_or_prescribed_slope',**inputs())
    explicit=LocalInertialSolver(validation_profile='physical_candidate',free_outflow_closure='local_or_prescribed_slope',**inputs())
    for solver in (implicit,explicit):
        solver.set_water_depth(depth=np.full((2,2),.2))
        solver.qx.fill_(.01);solver.qy.fill_(.02)
        solver.step(remaining_time=.01,fixed_dt=.01)
    np.testing.assert_array_equal(actual=implicit.get_water_depth(),desired=explicit.get_water_depth())
    np.testing.assert_array_equal(actual=implicit.qx.numpy(),desired=explicit.qx.numpy())
    identity=implicit.capture_state()['host']['numerical_configuration']
    assert identity['use_two_dimensional_friction']==2
    assert identity['manning_face_mode']==2
    assert identity['free_outflow_closure']=='local_or_prescribed_slope'
