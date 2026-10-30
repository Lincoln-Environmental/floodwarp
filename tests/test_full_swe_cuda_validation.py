"""Run the existing independent numerical assertions on CUDA, unchanged."""

import os

import numpy as np
import pytest
import warp as wp

from tests import test_full_swe as controls
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp import full_swe_kernels as kernels


@pytest.mark.skipif(os.environ.get('OVERLANDER_TEST_FULL_SWE_CUDA') != '1', reason='Explicit FV1 CUDA opt-in required')
@pytest.mark.parametrize('name,arguments',[
    ('test_numpy_warp_step_parity_and_budget',dict(axis=axis,boundary=boundary))
    for axis in (0,1) for boundary in (1,2,3)
]+[
    ('test_lake_at_rest_including_dry_shore',dict(datum=datum,irregular=irregular))
    for datum in (0.,1000.) for irregular in (False,True)
]+[
    ('test_zero_stage_hydrostatic_boundary',dict(bed_value=bed,depth=depth))
    for bed,depth in ((.2,0.),(-.2,.2),(-.2,0.),(.2,.2))
]+[
    ('test_dam_break_rotation_positivity_and_transport',dict(right_depth=value)) for value in (0.,.2)
]+[
    (name,{}) for name in (
        'test_mask_and_exact_source_integral','test_vector_manning_decay_in_oblique_flow',
        'test_cfl_includes_both_velocity_components','test_two_dimensional_dam_break_over_step_and_recession',
        'test_complete_disk_restart_and_configuration_rejection','test_endpoint_clipping_is_not_a_minimum_cfl_failure',
        'test_finite_availability_removes_water_and_vector_momentum',
        'test_wet_stage_ghost_limits_initially_dry_inflow',
        'test_shallow_wet_ghost_extrapolated_momentum_participates_in_cfl',
        'test_minimum_dt_aborts_without_overriding_stability',
        'test_spatial_exponential_soil_and_vector_withdrawal_match_numpy',
        'test_spreading_wet_front_recedes_to_dry_without_losing_volume',
    )
])
def test_existing_independent_control_on_cuda(monkeypatch,tmp_path,name,arguments):
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    original=controls.FullSWESolver
    def cuda_solver(**options):
        options['device']='cuda:0'
        return original(**options)
    monkeypatch.setattr(controls,'FullSWESolver',cuda_solver)
    options=dict(arguments)
    if name=='test_complete_disk_restart_and_configuration_rejection':
        options['tmp_path']=tmp_path
    getattr(controls,name)(**options)


@pytest.mark.skipif(os.environ.get('OVERLANDER_TEST_FULL_SWE_CUDA') != '1', reason='Explicit FV1 CUDA opt-in required')
@pytest.mark.parametrize('stage,depth,momentum,duration',[(4.,0.,0.,.2),(.001,.1,-.1,.008)])
def test_hardened_wet_ghost_cfl_matches_cpu_in_conditional_graph(stage,depth,momentum,duration):
    models=[]
    for device,graph in (('cpu',False),('cuda:0',True),('cuda:0',False)):
        solver=FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=3,nx=4,dx=1.,dy=1.,bed_elevation=np.zeros((3,4)),
            manning_n=0.,active_mask=np.ones((3,4),bool),device=device,maximum_dt=10.)
        solver.set_water_depth(depth=np.full((3,4),depth))
        solver.set_momentum(hu=np.zeros((3,4)),hv=np.full((3,4),momentum))
        boundary=np.zeros((4,4),int)
        boundary[-1]=3
        solver.set_boundary_type_qy(boundary_type=boundary)
        solver.set_boundary_stage_qy(stage=np.full((4,4),stage))
        solver.run_device_loop(duration=duration,use_cuda_graph=graph)
        models.append(solver)
    np.testing.assert_allclose(actual=models[1].get_conservative_state(),desired=models[0].get_conservative_state(),rtol=3.e-13,atol=3.e-15)
    np.testing.assert_array_equal(actual=models[1].get_conservative_state(),desired=models[2].get_conservative_state())
    assert models[0].get_timestep_histogram()==models[1].get_timestep_histogram()==models[2].get_timestep_histogram()


@pytest.mark.skipif(os.environ.get('OVERLANDER_TEST_FULL_SWE_CUDA') != '1', reason='Explicit FV1 CUDA opt-in required')
def test_minimum_dt_failure_terminates_cuda_graph_after_one_accepted_step():
    solver=FullSWESolver(stage_boundary_momentum='discharge_extrapolation', flux_scheme='legacy_hll', ny=2,nx=3,dx=1.,dy=1.,bed_elevation=np.zeros((2,3)),manning_n=0.,
        active_mask=np.ones((2,3),bool),device='cuda:0',maximum_dt=1.,minimum_dt=.01)
    solver.set_water_depth(depth=np.ones((2,3)))
    solver.set_momentum(hu=np.full((2,3),10000.),hv=np.zeros((2,3)))
    with pytest.raises(ArithmeticError,match='below configured'):
        solver.run_device_loop(duration=1.,use_cuda_graph=True,max_steps=3)
    assert solver.steps==1 and solver.time<.01


@pytest.mark.skipif(os.environ.get('OVERLANDER_TEST_FULL_SWE_CUDA') != '1', reason='Explicit FV1 CUDA opt-in required')
def test_shallow_donor_limits_the_complete_flux_vector_and_conserves_volume_on_cuda():
    # Isolated limiter fixture, never used as a catchment restart/initial state.
    state=np.array([[5.e-5,0.,0.],[0.,0.,0.]])
    values=np.zeros((3,3))
    values[1]=[.001,.002,-.003]
    q=wp.array(state,dtype=wp.vec3d,device='cuda:0')
    fx=wp.array(values,dtype=wp.vec3d,device='cuda:0')
    fy=wp.zeros(4,dtype=wp.vec3d,device='cuda:0')
    theta=wp.ones(2,dtype=wp.float64,device='cuda:0')
    active=wp.ones(2,dtype=wp.int32,device='cuda:0')
    volume=wp.zeros(3,dtype=wp.float64,device='cuda:0')
    control=np.zeros(15)
    control[2]=.5
    ctrl=wp.array(control,dtype=wp.float64,device='cuda:0')
    wp.launch(kernel=kernels.draining,dim=2,inputs=[q,fx,fy,theta,ctrl,2,1.,1.],device='cuda:0')
    wp.launch(kernel=kernels.limit_faces,dim=3,inputs=[fx,theta,active,volume,ctrl,2,1,0,1.],device='cuda:0')
    limited=fx.numpy()[1]
    assert theta.numpy()[0]<1.
    np.testing.assert_allclose(actual=limited,desired=values[1]*.1,rtol=2.e-15,atol=0.)
    transported=volume.numpy()[1]
    assert transported==pytest.approx(state[0,0],rel=2.e-15)
    assert state[0,0]-transported>=0.
