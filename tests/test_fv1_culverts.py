"""Circular FV1 link physics, conservation, accepted maxima and restart tests."""
import copy

import numpy as np
import pytest
import warp as wp

from floodwarp.full_swe_solver import FullSWESolver


def definition(*, diameter=1., inlet=0., outlet=0.):
    return dict(id='pipe', type='culvert', shape='circular', inlet_row=0, inlet_col=0,
        outlet_row=0, outlet_col=2, diameter_m=diameter, inlet_invert_m=inlet,
        outlet_invert_m=outlet, length_m=10., manning_n=.013,
        entrance_loss_coefficient=.5, exit_loss_coefficient=1., one_way=False,
        max_discharge_m3_s=1.e30)


def make_solver(*, device='cpu', structure=None, depth=(1., 0., 0.), bed=(0., 0., 0.), mode='dense'):
    solver=FullSWESolver(ny=1, nx=3, dx=5., dy=5., gravity=9.80665,
        bed_elevation=np.array([bed]), active_mask=np.array([[True, False, True]]),
        manning_n=0., device=device, maximum_dt=.1, execution_mode=mode,
        structures=[definition() if structure is None else structure])
    solver.set_water_depth(depth=np.array([depth]))
    return solver


def test_full_pipe_analytical_capacity_and_internal_conservation():
    solver=make_solver()
    solver.step(fixed_dt=.01)
    area=np.pi/4.
    loss=2.5+2*9.80665*.013**2*10./(.25**(4/3))
    expected=area*np.sqrt(2*9.80665/loss)
    diagnostic=solver.get_structure_diagnostics()[0]
    assert diagnostic['discharge_m3_s']==pytest.approx(expected, rel=1.e-13)
    budget=solver.cumulative_accounting()
    assert abs(budget['residual']) < 1.e-13
    assert budget['boundary_outflow']==budget['boundary_inflow']==0.
    np.testing.assert_array_equal(solver.get_max_water_depth(), np.array([[1., 0., expected*.01/25.]]))


@pytest.mark.parametrize('depth', [(1.,0.,0.), (0.,0.,1.), (.1,0.,0.)])
@pytest.mark.parametrize('mode', ['dense', 'sparse'])
def test_large_barrel_positivity_head_monotonicity_and_recession(depth, mode):
    solver=make_solver(structure=definition(diameter=100.), depth=depth, mode=mode)
    initial=solver.initial_volume
    solver.run_device_loop(duration=100., use_cuda_graph=False)
    h=solver.get_water_depth()
    assert np.min(h)>=0.
    assert np.max(h)<=max(depth)
    assert abs(solver.cumulative_accounting()['residual'])<1.e-11
    assert np.sum(h)*25.==pytest.approx(initial, abs=1.e-11)
    assert abs(h[0,0]-h[0,2])<1.e-12
    assert solver.get_structure_diagnostics()[0]['limited_steps']>0


def test_opening_sill_and_uphill_dry_receiver_block_flow():
    solver=make_solver(structure=definition(inlet=0.,outlet=2.), bed=(0.,0.,2.))
    solver.run_device_loop(duration=1.,use_cuda_graph=False)
    assert solver.get_structure_diagnostics()[0]['forward_volume_m3']==0.
    np.testing.assert_array_equal(solver.get_water_depth(), [[1.,0.,0.]])
    solver=make_solver(structure=definition(inlet=.5),depth=(.3,0.,0.))
    solver.run_device_loop(duration=1.,use_cuda_graph=False)
    assert solver.get_structure_diagnostics()[0]['forward_volume_m3']==0.


def test_one_way_and_below_invert_storage():
    structure=definition(inlet=.5)
    structure['one_way']=True
    solver=make_solver(structure=structure,depth=(0.,0.,1.))
    solver.run_device_loop(duration=2.,use_cuda_graph=False)
    assert solver.get_structure_diagnostics()[0]['reverse_volume_m3']==0.
    solver=make_solver(structure=definition(diameter=100.,inlet=.5))
    solver.run_device_loop(duration=10.,use_cuda_graph=False)
    assert solver.get_water_depth()[0,0]>=.5


def test_maximum_is_accepted_composed_state_with_rain():
    solver=make_solver(depth=(.1,0.,0.),structure=definition(diameter=100.))
    solver.set_uniform_rainfall_rate(rainfall_rate=.1)
    solver.step(fixed_dt=.1)
    h=solver.get_water_depth()
    np.testing.assert_array_equal(solver.get_max_water_depth(),np.maximum([[.1,0.,0.]],h))
    assert solver.get_max_water_depth()[0,0] < .11
    assert abs(solver.cumulative_accounting()['residual'])<1.e-13


def test_momentum_withdrawal_and_zero_jet_mixing():
    solver=make_solver(depth=(1.,0.,.2))
    solver.set_momentum(hu=np.array([[2.,0.,.4]]),hv=np.array([[1.,0.,.2]]))
    ctrl=np.zeros(15); ctrl[2]=.01
    solver.ctrl.assign(ctrl)
    solver._culverts.snapshot(solver=solver)
    solver._culverts.enqueue(solver=solver)
    q=solver.get_conservative_state()
    np.testing.assert_allclose(q[0,0,1:]/q[0,0,0],[2.,1.],rtol=1.e-14)
    np.testing.assert_array_equal(q[0,2,1:],[.4,.2])
    assert q[0,2,0]>.2


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_complete_checkpoint_restart_and_identity(tmp_path,device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    continuous=make_solver(device=device, mode='sparse')
    continuous.run_device_loop(duration=1.)
    checkpoint=tmp_path/'culvert.npz'
    continuous.save_checkpoint(path=checkpoint)
    restored=make_solver(device=device, mode='sparse')
    restored.load_checkpoint(path=checkpoint)
    for solver in (continuous, restored):
        solver.run_device_loop(duration=2.)
    for name,field in continuous._fields.items():
        np.testing.assert_allclose(restored._fields[name].numpy(),field.numpy(),rtol=1.e-13,atol=1.e-13,err_msg=name)
    assert continuous.get_structure_diagnostics()==restored.get_structure_diagnostics()
    wrong=make_solver(device=device,structure=definition(diameter=.9))
    with pytest.raises(ValueError,match='configuration'):
        wrong.load_checkpoint(path=checkpoint)
    corrupt=copy.deepcopy(continuous.capture_state())
    corrupt['arrays']['culvert_parameters'][0,0]=.8
    with pytest.raises(ValueError,match='physical input'):
        restored.restore_state(state=corrupt)


def test_cpu_cuda_control():
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    cpu=make_solver(device='cpu',mode='sparse')
    cuda=make_solver(device='cuda:0',mode='sparse')
    for solver in (cpu,cuda):
        solver.run_device_loop(duration=5.)
    np.testing.assert_allclose(cpu.get_conservative_state(),cuda.get_conservative_state(),rtol=1.e-12,atol=1.e-13)
    np.testing.assert_allclose(cpu._culverts.records.numpy(),cuda._culverts.records.numpy(),rtol=1.e-12,atol=1.e-13)


def test_rejects_overlapping_or_inactive_endpoints():
    bad=definition(); bad['outlet_col']=0
    with pytest.raises(ValueError,match='distinct'):
        make_solver(structure=bad)
    bad=definition(); bad['outlet_col']=1
    with pytest.raises(ValueError,match='active'):
        make_solver(structure=bad)


def test_partially_full_circular_capacity():
    solver=make_solver(depth=(.25,0.,0.))
    solver.step(fixed_dt=.01)
    radius=.5
    angle=np.arccos((radius-.25)/radius)
    area=radius**2*angle-(radius-.25)*np.sqrt(2*radius*.25-.25**2)
    hydraulic_radius=area/(2*radius*angle)
    loss=2.5+2*9.80665*.013**2*10/hydraulic_radius**(4/3)
    expected=area*np.sqrt(2*9.80665*.25/loss)
    assert solver.get_structure_diagnostics()[0]['discharge_m3_s']==pytest.approx(expected,rel=1.e-13)


def test_coupled_surface_and_culvert_timestep_convergence():
    states=[]
    for cap in (.08,.04,.02,.01):
        solver=FullSWESolver(ny=1,nx=3,dx=5.,dy=5.,gravity=9.80665,
            bed_elevation=np.array([[0.,2.,0.]]),active_mask=np.ones((1,3),bool),
            manning_n=.03,maximum_dt=cap,structures=[definition()])
        solver.set_water_depth(depth=np.array([[1.,0.,.1]]))
        solver.set_uniform_rainfall_rate(rainfall_rate=1.e-4)
        solver.run_device_loop(duration=2.,use_cuda_graph=False)
        states.append(solver.get_conservative_state())
        assert abs(solver.cumulative_accounting()['residual'])<1.e-11
    errors=[np.linalg.norm(state-states[-1]) for state in states[:-1]]
    assert errors[0]>errors[1]>errors[2]
    assert errors[1]<.65*errors[0]




def test_equal_water_levels_across_unequal_beds_are_stationary():
    solver=make_solver(depth=(1.,0.,.5),bed=(0.,0.,.5),structure=definition(outlet=.5))
    initial=solver.get_conservative_state()
    solver.run_device_loop(duration=10.,use_cuda_graph=False)
    np.testing.assert_array_equal(solver.get_conservative_state(),initial)
    assert solver.get_structure_diagnostics()[0]['net_volume_m3']==0.
