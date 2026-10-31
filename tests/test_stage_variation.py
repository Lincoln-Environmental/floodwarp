"""Rapid prescribed stages versus independently timestep-refined controls."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.config import NP_FLOAT,PRECISION_MODE


def make_stage_case(*, orientation, spacing, maximum_dt, mode='coarse',device='cpu'):
    if PRECISION_MODE=='mixed' and mode!='coarse':
        pytest.skip('Traditional/enhanced SGS do not support conditioned mixed precision')
    active=np.ones((4,4),np.int32)
    if orientation=='west':active[:,0]=0
    elif orientation=='east':active[:,-1]=0
    elif orientation=='south':active[0,:]=0
    else:active[-1,:]=0
    options={}
    if mode!='coarse':options=dict(hydraulic_geometry_mode=mode,
        subgrid_bed_elevation=np.zeros((8,8)),subgrid_resolution_m=spacing/2.)
    solver=LocalInertialSolver(ny=4,nx=4,dx=spacing,dy=spacing,bed_elevation=np.zeros((4,4)),
        manning_n=np.full((4,4),.035),active_mask=active,device=device,maximum_dt=maximum_dt,
        minimum_dt=.000001,adaptive_dt_growth_factor=None,infiltration_method='none',
        track_max_water_depth=True,**options)
    if orientation in ('west','east'):
        boundary=solver.boundary_type_qx.numpy().copy()
        boundary[:,1 if orientation=='west' else 3]=5 if orientation=='west' else 10
        solver.set_boundary_type_qx(boundary_type=boundary)
    else:
        boundary=solver.boundary_type_qy.numpy().copy()
        boundary[1 if orientation=='south' else 3,:]=11 if orientation=='south' else 4
        solver.set_boundary_type_qy(boundary_type=boundary)
    return solver


def simulate_stage(*, orientation, spacing, maximum_dt, rise_time, mode='coarse',device='cpu',graph=False):
    solver=make_stage_case(orientation=orientation,spacing=spacing,maximum_dt=maximum_dt,mode=mode,device=device)
    series=(np.array([0.,rise_time,1.,2.]),np.array([0.,1.,0.,0.]))
    options={'stage_series_qx':series} if orientation in ('west','east') else {'stage_series_qy':series}
    rows=[]
    for target in np.arange(.1,2.01,.1):
        solver.run_device_loop(duration=float(target-solver.simulation_time),use_cuda_graph=graph,**options)
        ledger=solver.cumulative_accounting()
        rows.append([float(target),ledger['boundary_inflow'],ledger['boundary_outflow']])
    depth=solver.get_water_depth().copy();ledger=solver.cumulative_accounting()
    residual=ledger['boundary_inflow']-ledger['boundary_outflow']-float(depth.sum(dtype=np.float64))*spacing**2
    # Roundoff scales with supplied m³ and state precision, not a universal
    # dimensionless tolerance. The existing FP64 conservation bound remains.
    tolerance=1e-9*max(1.,ledger['boundary_inflow'])+16.*np.finfo(NP_FLOAT).eps*ledger['boundary_inflow']
    assert abs(residual)<tolerance
    assert np.isfinite(depth).all() and depth.min()>=0.
    return depth,solver.get_max_water_depth().copy(),np.asarray(rows)


@pytest.mark.parametrize('orientation',['west','east','south','north'])
@pytest.mark.parametrize('spacing',[1.,5.,20.])
@pytest.mark.parametrize('rise_time',[.5,.05])
def test_rising_and_triangular_stages_converge(orientation,spacing,rise_time,record_property):
    depth,maximum,rows=simulate_stage(orientation=orientation,spacing=spacing,maximum_dt=60.,rise_time=rise_time)
    reference,reference_max,expected=simulate_stage(orientation=orientation,spacing=spacing,maximum_dt=.002,rise_time=rise_time)
    refined,_,_=simulate_stage(orientation=orientation,spacing=spacing,maximum_dt=.005,rise_time=rise_time)
    # The late reflected wave remains first-order in the hydraulic timestep;
    # refinement must reduce that error independently of boundary quadrature.
    assert np.linalg.norm(refined-reference)<=np.linalg.norm(depth-reference)+1e-6
    np.testing.assert_allclose(refined,reference,atol=.01,rtol=.05)
    np.testing.assert_allclose(maximum,reference_max,atol=.01,rtol=.05)
    volume_budget=1e-5*spacing**2+.05*expected[-1,1]
    assert np.max(np.abs(rows[-1,1:]-expected[-1,1:]))<=volume_budget
    q=np.diff(np.r_[0.,rows[:,1]-rows[:,2]])/.1
    qref=np.diff(np.r_[0.,expected[:,1]-expected[:,2]])/.1
    nrmse=float(np.sqrt(np.mean((q-qref)**2))/max(1e-12,float(np.max(np.abs(qref)))))
    assert nrmse<.05
    assert abs(rows[int(np.argmax(q)),0]-expected[int(np.argmax(qref)),0])<=.1+1e-9
    record_property('default_final_depth_rmse_m',float(np.sqrt(np.mean((depth-reference)**2))))
    record_property('refined_final_depth_rmse_m',float(np.sqrt(np.mean((refined-reference)**2))))
    record_property('default_maximum_envelope_rmse_m',float(np.sqrt(np.mean((maximum-reference_max)**2))))
    record_property('default_hydrograph_nrmse',nrmse)


@pytest.mark.parametrize('mode',['traditional','enhanced'])
@pytest.mark.parametrize('orientation',['west','east','south','north'])
def test_sgs_rapid_stage_refinement(mode,orientation,record_property):
    depth,maximum,rows=simulate_stage(orientation=orientation,spacing=1.,maximum_dt=60.,rise_time=.05,mode=mode)
    reference,reference_max,expected=simulate_stage(orientation=orientation,spacing=1.,maximum_dt=.002,rise_time=.05,mode=mode)
    refined,_,_=simulate_stage(orientation=orientation,spacing=1.,maximum_dt=.005,rise_time=.05,mode=mode)
    # As for the coarse solver, test timestep convergence of the reflected
    # wave separately from the forcing-driven maximum and volume response.
    assert np.linalg.norm(refined-reference)<=np.linalg.norm(depth-reference)+1e-6
    np.testing.assert_allclose(refined,reference,atol=.01,rtol=.05)
    np.testing.assert_allclose(maximum,reference_max,atol=.01,rtol=.05)
    volume_budget=1e-5+.05*expected[-1,1]
    assert np.max(np.abs(rows[-1,1:]-expected[-1,1:]))<=volume_budget
    q=np.diff(np.r_[0.,rows[:,1]-rows[:,2]])/.1
    qref=np.diff(np.r_[0.,expected[:,1]-expected[:,2]])/.1
    assert np.sqrt(np.mean((q-qref)**2))<.05*np.max(np.abs(qref))
    record_property('default_final_depth_rmse_m',float(np.sqrt(np.mean((depth-reference)**2))))
    record_property('refined_final_depth_rmse_m',float(np.sqrt(np.mean((refined-reference)**2))))
    record_property('default_maximum_envelope_rmse_m',float(np.sqrt(np.mean((maximum-reference_max)**2))))


@pytest.mark.parametrize('mode',['coarse','traditional','enhanced'])
@pytest.mark.parametrize('orientation',['west','east','south','north'])
@pytest.mark.parametrize('graph',[False,None])
def test_rapid_stage_cpu_cuda_graph_parity(mode,orientation,graph):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    expected=simulate_stage(orientation=orientation,spacing=1.,maximum_dt=60.,rise_time=.05,mode=mode)
    actual=simulate_stage(orientation=orientation,spacing=1.,maximum_dt=60.,rise_time=.05,mode=mode,
        device='cuda:0',graph=graph)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-5,rtol=1e-4)
    volume_scale=float(np.max(np.abs(expected[2][:,1:])))
    np.testing.assert_allclose(actual[2],expected[2],
        atol=1e-9+16.*np.finfo(NP_FLOAT).eps*volume_scale,rtol=max(1e-8,16.*np.finfo(NP_FLOAT).eps))
