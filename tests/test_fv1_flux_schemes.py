"""FV1 release controls: flux physics, production wiring and historical replay."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp

from floodwarp import full_swe_kernels as kernels
from floodwarp import full_swe_reference as reference
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.solver_factory import create_overland_solver


@wp.kernel
def sample_flux(left: wp.array(dtype=wp.vec3d),right: wp.array(dtype=wp.vec3d),
                mode: int,result: wp.array(dtype=wp.vec3d)):
    k=wp.tid()
    result[k]=kernels.hll(left[k],right[k],wp.float64(9.80665),wp.float64(1e-4),mode)


def build(*,scheme,stage_mode,device,shape=(5,8),depth=1.,u=.3,v=.1,cap=.03):
    solver=FullSWESolver(ny=shape[0],nx=shape[1],dx=1.,dy=1.,bed_elevation=np.zeros(shape),
        manning_n=.04,active_mask=np.ones(shape,bool),gravity=9.80665,maximum_dt=cap,
        flux_scheme=scheme,stage_boundary_momentum=stage_mode,device=device)
    solver.set_water_depth(depth=np.full(shape,depth))
    solver.set_momentum(hu=np.full(shape,depth*u),hv=np.full(shape,depth*v))
    bx=np.zeros((shape[0],shape[1]+1),int); bx[:,[0,-1]]=1; bx[::2,-1]=3
    by=np.zeros((shape[0]+1,shape[1]),int); by[[0,-1]]=1
    solver.set_boundary_type_qx(boundary_type=bx);solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=np.full(bx.shape,.1))
    return solver


@pytest.mark.parametrize('scheme',['legacy_hll','contact_hll','star_hll','star_contact_hll'])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_flux_matches_independent_numpy_and_resolves_contacts(scheme,device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(seed=12345)
    left=rng.uniform(-2,2,(300,3));right=rng.uniform(-2,2,(300,3))
    left[:,0]=rng.uniform(.00001,10,300);right[:,0]=rng.uniform(.00001,10,300)
    left[:8]=[[1,0,2],[1,1,2],[1,-1,2],[0,0,0],[1,0,2],[0,0,0],[1,-10,2],[1e-5,0,0]]
    right[:8]=[[1,0,-3],[1,1,-3],[1,-1,-3],[1,0,2],[0,0,0],[0,0,0],[1,10,-3],[1e-5,0,0]]
    out=wp.zeros(len(left),dtype=wp.vec3d,device=device)
    mode=('legacy_hll','contact_hll','star_hll','star_contact_hll').index(scheme)
    wp.launch(kernel=sample_flux,dim=len(left),inputs=[wp.array(left,dtype=wp.vec3d,device=device),wp.array(right,dtype=wp.vec3d,device=device),mode,out],device=device)
    actual=out.numpy();expected=reference.hll_flux(left=left,right=right,gravity=9.80665,dry_depth=1e-4,flux_scheme=scheme)
    np.testing.assert_allclose(actual=actual,desired=expected,rtol=3e-13,atol=3e-13)
    assert np.isfinite(actual).all()
    if 'contact' in scheme:
        np.testing.assert_array_equal(actual=actual[:3,2],desired=np.array([0.,2.,3.]))
        baseline=reference.hll_flux(left=left,right=right,gravity=9.80665,dry_depth=1e-4,
            flux_scheme='star_hll' if mode==3 else 'legacy_hll')
        np.testing.assert_allclose(actual=actual[:,:2],desired=baseline[:,:2],rtol=3e-13,atol=3e-13)
    assert actual[5,2]==0.


@pytest.mark.parametrize('scheme',['legacy_hll','contact_hll','star_hll','star_contact_hll'])
@pytest.mark.parametrize('stage_mode',['discharge_extrapolation','velocity_extrapolation'])
def test_cpu_cuda_graph_disk_restart_all_combinations(tmp_path,scheme,stage_mode):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    states=[]
    for device,graph in (('cpu',False),('cuda:0',False),('cuda:0',True)):
        solver=build(scheme=scheme,stage_mode=stage_mode,device=device)
        solver.run_device_loop(duration=.12,use_cuda_graph=graph)
        path=tmp_path/'restart.npz';solver.save_checkpoint(path=path)
        restored=build(scheme=scheme,stage_mode=stage_mode,device=device);restored.load_checkpoint(path=path)
        for model in (solver,restored):model.run_device_loop(duration=.12,use_cuda_graph=graph)
        for key,value in solver.capture_state()['arrays'].items():
            np.testing.assert_array_equal(actual=restored.capture_state()['arrays'][key],desired=value)
        assert abs(solver.cumulative_accounting()['residual'])<1e-12
        states.append(solver.get_conservative_state())
    np.testing.assert_allclose(actual=states[0],desired=states[1],rtol=3e-13,atol=3e-15)
    np.testing.assert_array_equal(actual=states[1],desired=states[2])


@pytest.mark.parametrize('scheme',['legacy_hll','contact_hll','star_hll','star_contact_hll'])
@pytest.mark.parametrize('side',['west','east','south','north'])
def test_reflecting_wall_pressure_and_mixed_outlet_are_preserved(scheme,side):
    solver=build(scheme=scheme,stage_mode='velocity_extrapolation',device='cpu',u=.8,v=-.6)
    before=solver.get_conservative_state()
    solver.step(fixed_dt=.001)
    fx=solver._fields['flux_x'].numpy().reshape(5,9,3)
    fy=solver._fields['flux_y'].numpy().reshape(6,8,3)
    if side=='west':face=fx[:,0];normal=1;tangent=2;sign=-1.;selected=np.ones(5,bool)
    elif side=='east':face=fx[:,-1];normal=1;tangent=2;sign=1.;selected=np.arange(5)%2==1
    elif side=='south':face=fy[0];normal=2;tangent=1;sign=-1.;selected=np.ones(8,bool)
    else:face=fy[-1];normal=2;tangent=1;sign=1.;selected=np.ones(8,bool)
    np.testing.assert_array_equal(actual=face[selected,0],desired=np.zeros(selected.sum()))
    np.testing.assert_array_equal(actual=face[selected,tangent],desired=np.zeros(selected.sum()))
    interior=before[0,0,[0,normal,tangent]]
    ghost=interior.copy();ghost[1]*=-1
    left,right=(ghost,interior) if sign<0 else (interior,ghost)
    expected=reference.hll_flux(left=left,right=right,gravity=9.80665,dry_depth=1e-4,flux_scheme=scheme)
    np.testing.assert_allclose(actual=face[selected,normal],desired=expected[1],rtol=3e-14,atol=3e-15)
    assert np.any(fx[::2,-1,0]!=0.)


@pytest.mark.parametrize('scheme',['star_hll','star_contact_hll'])
@pytest.mark.parametrize('stage_mode',['discharge_extrapolation','velocity_extrapolation','characteristic'])
@pytest.mark.parametrize('rain,infil',[(0.,0.),(5.,0.),(0.,5.),(5.,5.)])
def test_star_cfl_bounds_actual_source_updated_reconstructed_faces(scheme,stage_mode,rain,infil):
    solver=build(scheme=scheme,stage_mode=stage_mode,device='cpu',depth=1.,u=.3,v=.1,cap=.1)
    solver.set_boundary_stage_qx(stage=np.ones((5,9)))
    solver.set_source_rates(rain=rain,infil=infil)
    original=solver.get_conservative_state()
    dt=solver.step(fixed_dt=.1)
    available=original[...,0]+dt*rain
    q=original.copy();q[...,0]=np.maximum(available-dt*infil,0.)
    q[...,1:] *= (q[...,0]/available)[...,None]
    rates=[]
    original_flux=reference.hll_flux
    def sampled_flux(**options):
        sl,sr,_,_=reference.hll_wave_speeds(**options)
        rates.append(np.maximum(np.abs(sl),np.abs(sr)))
        return original_flux(**options)
    with patch.object(reference,'hll_flux',sampled_flux):
        for axis,boundary,stages in ((1,solver._fields['types_x'].numpy().reshape(5,9),np.ones((5,9))),
                                    (0,solver._fields['types_y'].numpy().reshape(6,8),np.zeros((6,8)))):
            reference.face_fluxes(state=q,bed=np.zeros((5,8)),active=np.ones((5,8),bool),boundary_type=boundary,
                boundary_stage=stages,axis=axis,gravity=9.80665,dry_depth=1e-4,stage_boundary_momentum=stage_mode,flux_scheme=scheme)
    local=np.maximum(rates[0][:,:-1],rates[0][:,1:])+np.maximum(rates[1][:-1],rates[1][1:])
    assert dt*np.max(local)<=.4*(1+3e-14)


def test_exact_c7d8a43_historical_replay_and_missing_field_rejection():
    path=Path(__file__).parent/'fixtures/fv1_c7d8a43_legacy.npz'
    with np.load(file=path) as saved:
        solver=FullSWESolver(ny=6,nx=8,dx=1.,dy=1.,bed_elevation=saved['bed'],manning_n=.04,
            active_mask=np.ones((6,8),bool),device='cpu',gravity=9.80665,maximum_dt=.01,
            stage_boundary_momentum='discharge_extrapolation',flux_scheme='legacy_hll')
        solver.set_water_depth(depth=saved['h']);solver.set_momentum(hu=saved['hu'],hv=saved['hv'])
        solver.set_boundary_type_qx(boundary_type=saved['bx']);solver.set_boundary_type_qy(boundary_type=saved['by'])
        solver.set_boundary_stage_qx(stage=saved['sx']);solver.set_source_rates(rain=.0003,infil=.0002)
        solver.run_device_loop(duration=.1,use_cuda_graph=False)
        legacy=solver.capture_state();legacy['configuration'].pop('flux_scheme');legacy['configuration'].pop('stage_boundary_momentum')
        solver.restore_state(state=legacy)
        solver.run_device_loop(duration=.1,use_cuda_graph=False)
        np.testing.assert_array_equal(actual=solver.get_conservative_state(),desired=saved['final'])
        np.testing.assert_array_equal(actual=solver._fields['flux_x'].numpy(),desired=saved['flux_x'])
        np.testing.assert_array_equal(actual=solver._fields['flux_y'].numpy(),desired=saved['flux_y'])
        assert solver.steps==int(saved['steps'])
        candidate=build(scheme='contact_hll',stage_mode='velocity_extrapolation',device='cpu',shape=(6,8))
        snapshot=candidate.capture_state()
        with pytest.raises(ValueError,match='configuration'):candidate.restore_state(state=legacy)
        for key,value in snapshot['arrays'].items():
            np.testing.assert_array_equal(actual=candidate.capture_state()['arrays'][key],desired=value)


def test_direct_and_factory_default_resolve_to_finite_velocity():
    options=dict(ny=2,nx=3,dx=1.,dy=1.,bed_elevation=np.zeros((2,3)),manning_n=0.,active_mask=np.ones((2,3),bool),device='cpu')
    for solver in (FullSWESolver(**options),create_overland_solver(solver_method='fv1',**options)):
        assert solver.numerical_configuration()['stage_boundary_momentum']=='velocity_extrapolation'
        assert solver.numerical_configuration()['flux_scheme']=='contact_hll'
        assert solver.capture_state()['configuration']['flux_scheme']==solver.flux_scheme
    with pytest.raises(ValueError,match='flux_scheme'):FullSWESolver(flux_scheme='unsupported',**options)


@pytest.mark.parametrize('scheme',['legacy_hll','contact_hll','star_hll','star_contact_hll'])
@pytest.mark.parametrize('side',['west','east','south','north'])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_all_orientations_both_momenta_and_embedded_stage_match_numpy(scheme,side,device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    shape=(6,8);active=np.ones(shape,bool)
    if side=='west':active[:,0]=False
    elif side=='east':active[:,-1]=False
    elif side=='south':active[0]=False
    else:active[-1]=False
    ax=np.pad(active,((0,0),(1,1)));ay=np.pad(active,((1,1),(0,0)))
    bx=np.where(ax[:,:-1]!=ax[:,1:],1,0);by=np.where(ay[:-1]!=ay[1:],1,0)
    if side=='west':bx[::2,1]=3
    elif side=='east':bx[::2,-2]=3
    elif side=='south':by[1,::2]=3
    else:by[-2,::2]=3
    sx=np.full(bx.shape,.02);sy=np.full(by.shape,.02)
    solver=FullSWESolver(ny=6,nx=8,dx=1.,dy=1.,bed_elevation=np.zeros(shape),manning_n=.04,
        active_mask=active,device=device,stage_boundary_momentum='velocity_extrapolation',flux_scheme=scheme,gravity=9.80665)
    solver.set_water_depth(depth=np.full(shape,.6));solver.set_momentum(hu=np.full(shape,-.3),hv=np.full(shape,.2))
    solver.set_boundary_type_qx(boundary_type=bx);solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=sx);solver.set_boundary_stage_qy(stage=sy)
    original=solver.get_conservative_state();dt=solver.step(fixed_dt=.0001)
    expected,budget=reference.numpy_step(state=original,bed=np.zeros(shape),active=active,manning=np.full(shape,.04),
        dx=1.,dy=1.,gravity=9.80665,dry_depth=1e-4,dt=dt,boundary_type_x=bx,boundary_type_y=by,
        boundary_stage_x=sx,boundary_stage_y=sy,rain_rate=np.zeros(shape),infiltration_rate=np.zeros(shape),
        stage_boundary_momentum='velocity_extrapolation',flux_scheme=scheme)
    np.testing.assert_allclose(actual=solver.get_conservative_state(),desired=expected,rtol=3e-13,atol=3e-15)
    assert abs(solver.cumulative_accounting()['residual'])<1e-12


@pytest.mark.parametrize('scheme',['legacy_hll','contact_hll','star_hll','star_contact_hll'])
@pytest.mark.parametrize('u',[0.,1.,-1.])
def test_stationary_and_advected_shear_direction_energy_and_no_overshoot(scheme,u):
    shape=(4,80);h=np.ones(shape);hu=np.full(shape,u);hv=np.where(np.indices(shape)[1]<40,.5,-.5)
    solver=FullSWESolver(ny=4,nx=80,dx=1.,dy=1.,bed_elevation=np.zeros(shape),manning_n=0.,active_mask=np.ones(shape,bool),
        device='cpu',gravity=9.80665,flux_scheme=scheme,stage_boundary_momentum='velocity_extrapolation',maximum_dt=.1)
    solver.set_water_depth(depth=h);solver.set_momentum(hu=hu,hv=hv)
    bx=np.zeros((4,81),int);bx[:,[0,-1]]=3;by=np.zeros((5,80),int);by[[0,-1]]=3
    solver.set_boundary_type_qx(boundary_type=bx);solver.set_boundary_type_qy(boundary_type=by)
    solver.set_boundary_stage_qx(stage=np.ones(bx.shape));solver.set_boundary_stage_qy(stage=np.ones(by.shape))
    initial=solver.get_conservative_state();solver.run_device_loop(duration=1.,use_cuda_graph=False)
    q=solver.get_conservative_state()
    assert q[...,2].min()>=-.5-1e-14 and q[...,2].max()<=.5+1e-14
    assert np.sum(q[...,2]**2)<=np.sum(initial[...,2]**2)+1e-12
    if 'contact' in scheme and u==0.:
        np.testing.assert_array_equal(actual=q,desired=initial)
    if u>0:assert q[1,40,2]>initial[1,40,2]
    if u<0:assert q[1,39,2]<initial[1,39,2]
