"""Finite-velocity stage ghosts, consistent CFL, and historical replay."""
import numpy as np
import pytest
import warp as wp

from floodwarp import full_swe_kernels as kernels
from floodwarp.full_swe_reference import numpy_step
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.solver_factory import create_overland_solver


@wp.kernel
def sample_ghost(inside: wp.array(dtype=wp.vec3d),depth: wp.array(dtype=wp.float64),
                 dry: wp.float64,mode: int,result: wp.array(dtype=wp.vec3d)):
    k=wp.tid()
    result[k]=kernels.stage_ghost(inside[k],depth[k],dry,mode)


def solver_inputs(*,shape,device,mode):
    return dict(ny=shape[0],nx=shape[1],dx=1.,dy=1.,bed_elevation=np.zeros(shape),
        manning_n=np.full(shape,.04),active_mask=np.ones(shape,dtype=np.int32),device=device,
        gravity=9.80665,maximum_dt=1.,stage_boundary_momentum=mode,flux_scheme='legacy_hll')


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_velocity_and_finite_energy_are_preserved_when_ghost_depth_shrinks(device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    depths=np.array([1.,.1,.01,.001,0.])
    inside=np.broadcast_to(np.array([1.,2.,-3.]),(len(depths),3)).copy()
    values=wp.array(inside,dtype=wp.vec3d,device=device)
    h=wp.array(depths,dtype=wp.float64,device=device);out=wp.zeros(len(depths),dtype=wp.vec3d,device=device)
    wp.launch(kernel=sample_ghost,dim=len(depths),inputs=[values,h,1e-4,1,out],device=device)
    result=out.numpy()
    np.testing.assert_array_equal(actual=result[:,0],desired=depths)
    np.testing.assert_allclose(actual=result[:-1,1:]/depths[:-1,None],desired=np.tile([2.,-3.],(4,1)),atol=1e-14)
    energy=np.sum(result[:-1,1:]**2,axis=1)/(2*depths[:-1])
    np.testing.assert_allclose(actual=energy,desired=6.5*depths[:-1],rtol=1e-14)
    assert np.all(result[-1]==0.)
    wp.launch(kernel=sample_ghost,dim=len(depths),inputs=[values,h,1e-4,0,out],device=device)
    np.testing.assert_array_equal(actual=out.numpy()[:-1,1:],desired=inside[:-1,1:])


@pytest.mark.parametrize('mode',['discharge_extrapolation','velocity_extrapolation'])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_boundary_flux_and_cfl_use_the_same_state(mode,axis,device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    shape=(4,5);depth=1.0622467979968913;ghost=.02587347030639542
    normal=2.3669314348529102;tangent=-1.4567736219213028;g=9.80665
    solver=FullSWESolver(**solver_inputs(shape=shape,device=device,mode=mode))
    solver.set_water_depth(depth=np.full(shape,depth))
    hu,hv=(normal,tangent) if axis==0 else (tangent,normal)
    solver.set_momentum(hu=np.full(shape,hu),hv=np.full(shape,hv))
    bx=np.ones((4,6),int);by=np.ones((5,5),int);bx[:,1:-1]=0;by[1:-1]=0
    if axis==0:bx[:,-1]=3
    else:by[-1]=3
    solver.set_boundary_type_qx(boundary_type=bx);solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=np.full(bx.shape,ghost));solver.set_boundary_stage_qy(stage=np.full(by.shape,ghost))
    speed=abs(normal)/(depth if mode=='velocity_extrapolation' else ghost)
    expected=.4/(max(abs(normal/depth)+np.sqrt(g*depth),speed+np.sqrt(g*ghost))+abs(tangent/depth)+np.sqrt(g*depth))
    assert solver.step(fixed_dt=1.)==pytest.approx(expected,rel=2e-14)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_velocity_stage_step_matches_independent_numpy(axis,device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(seed=654);shape=(7,9)
    bed=rng.uniform(-.2,.2,shape);depth=rng.uniform(.01,.6,shape)
    options=solver_inputs(shape=shape,device=device,mode='velocity_extrapolation');options['bed_elevation']=bed
    solver=FullSWESolver(**options)
    solver.set_water_depth(depth=depth);solver.set_momentum(hu=rng.uniform(-.05,.05,shape),hv=rng.uniform(-.05,.05,shape))
    bx=np.ones((7,10),int);by=np.ones((8,9),int);bx[:,1:-1]=0;by[1:-1]=0
    if axis==0:bx[:,[0,-1]]=3
    else:by[[0,-1]]=3
    sx=np.full(bx.shape,.1);sy=np.full(by.shape,.1)
    solver.set_boundary_type_qx(boundary_type=bx);solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=sx);solver.set_boundary_stage_qy(stage=sy)
    solver.set_source_rates(rain=3e-5,infil=2e-5)
    original=solver.get_conservative_state();dt=solver.step(fixed_dt=.003)
    expected,budget=numpy_step(state=original,bed=bed,active=np.ones(shape,bool),manning=np.full(shape,.04),
        dx=1.,dy=1.,gravity=9.80665,dry_depth=1e-4,dt=dt,boundary_type_x=bx,boundary_type_y=by,
        boundary_stage_x=sx,boundary_stage_y=sy,rain_rate=np.full(shape,3e-5),infiltration_rate=np.full(shape,2e-5),
        stage_boundary_momentum='velocity_extrapolation',flux_scheme='legacy_hll')
    np.testing.assert_allclose(actual=solver.get_conservative_state(),desired=expected,rtol=3e-13,atol=3e-15)
    for key,value in budget.items():assert solver.cumulative_accounting()[key]==pytest.approx(value,abs=3e-14)


def test_factory_and_restart_record_boundary_semantics():
    legacy=FullSWESolver(**solver_inputs(shape=(2,3),device='cpu',mode='discharge_extrapolation'))
    options=solver_inputs(shape=(2,3),device='cpu',mode='velocity_extrapolation')
    options['explicit_maximum_dt']=options.pop('maximum_dt')
    corrected=create_overland_solver(solver_method='fv1',**options)
    assert corrected.numerical_configuration()['stage_boundary_momentum']=='velocity_extrapolation'
    with pytest.raises(ValueError,match='configuration'):corrected.restore_state(state=legacy.capture_state())
    historical=legacy.capture_state();historical['configuration'].pop('stage_boundary_momentum')
    legacy.restore_state(state=historical)
    with pytest.raises(ValueError,match='configuration'):corrected.restore_state(state=historical)
    with pytest.raises(ValueError,match='stage_boundary_momentum'):
        FullSWESolver(**dict(solver_inputs(shape=(2,3),device='cpu',mode='unsupported')))


def stage_solver(*,device,mode,depth,momentum,stage):
    shape=(8,10)
    solver=FullSWESolver(**solver_inputs(shape=shape,device=device,mode=mode))
    solver.set_water_depth(depth=np.full(shape,depth))
    solver.set_momentum(hu=np.full(shape,momentum),hv=np.zeros(shape))
    bx=np.zeros((8,11),int);by=np.zeros((9,10),int)
    bx[:,0]=1;bx[:,-1]=3;by[[0,-1]]=1
    solver.set_boundary_type_qx(boundary_type=bx);solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=np.full(bx.shape,stage))
    return solver


@pytest.mark.parametrize('depth,momentum,stage',[(1.,2.,.02),(1.,-2.,.02),(0.,0.,.2)])
def test_velocity_stage_cpu_cuda_graph_and_uncaptured_steps(depth,momentum,stage):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    models=[]
    for device,graph in (('cpu',False),('cuda:0',True),('cuda:0',False)):
        solver=stage_solver(device=device,mode='velocity_extrapolation',depth=depth,momentum=momentum,stage=stage)
        solver.run_device_loop(duration=.5,use_cuda_graph=graph,max_steps=1000)
        assert abs(solver.cumulative_accounting()['residual'])<1e-11
        assert np.min(solver.get_water_depth())>=0.
        models.append(solver)
    np.testing.assert_allclose(actual=models[1].get_conservative_state(),desired=models[0].get_conservative_state(),rtol=3e-13,atol=3e-15)
    np.testing.assert_array_equal(actual=models[1].get_conservative_state(),desired=models[2].get_conservative_state())
    assert models[0].get_timestep_histogram()==models[1].get_timestep_histogram()==models[2].get_timestep_histogram()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_velocity_stage_disk_restart_is_exact(tmp_path,device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    options=dict(device=device,mode='velocity_extrapolation',depth=1.,momentum=2.,stage=.02)
    uninterrupted=stage_solver(**options)
    uninterrupted.run_device_loop(duration=.2)
    path=tmp_path/'velocity_stage.npz'
    uninterrupted.save_checkpoint(path=path)
    resumed=stage_solver(**options);resumed.load_checkpoint(path=path)
    for solver in (uninterrupted,resumed):solver.run_device_loop(duration=.3)
    np.testing.assert_array_equal(actual=resumed.get_conservative_state(),desired=uninterrupted.get_conservative_state())
    assert resumed.cumulative_accounting()==uninterrupted.cumulative_accounting()
    assert resumed.get_timestep_histogram()==uninterrupted.get_timestep_histogram()


def test_omitted_momentum_mode_resolves_to_velocity_and_rejects_historical_restore():
    inputs=solver_inputs(shape=(8,10),device='cpu',mode='discharge_extrapolation')
    inputs.pop('stage_boundary_momentum')
    omitted=FullSWESolver(**inputs)
    explicit=stage_solver(device='cpu',mode='velocity_extrapolation',depth=1.,momentum=2.,stage=.02)
    omitted.restore_state(state=explicit.capture_state())
    for solver in (omitted,explicit):solver.run_device_loop(duration=.2)
    np.testing.assert_array_equal(actual=omitted.get_conservative_state(),desired=explicit.get_conservative_state())
    assert omitted.numerical_configuration()['stage_boundary_momentum']=='velocity_extrapolation'
    legacy=stage_solver(device='cpu',mode='discharge_extrapolation',depth=1.,momentum=2.,stage=.02)
    before=omitted.capture_state()
    with pytest.raises(ValueError,match='configuration'):
        omitted.restore_state(state=legacy.capture_state())
    for key,value in before['arrays'].items():
        np.testing.assert_array_equal(actual=omitted.capture_state()['arrays'][key],desired=value)
