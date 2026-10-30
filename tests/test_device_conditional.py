"""Unchanged production kernels under conditional CUDA graph execution."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.boundary_conditions import BOUNDARY_SPECIFIED_STAGE, BOUNDARY_FREE_OUTFLOW


def create(*, subgrid, profile, infiltration, native_n, after):
    ny,nx=4,6
    native=np.add.outer(np.arange(ny*5)*.002,np.arange(nx*5)*-.001)
    bed=native.reshape(ny,5,nx,5).mean(axis=(1,3))
    soil={}
    if infiltration=='exponential':soil=dict(infiltration_f0_m_per_s=1e-5,infiltration_fd_m=.03,infiltration_k_m_per_s=1e-6)
    if infiltration=='green_ampt':soil=dict(infiltration_ks_m_per_s=1e-6,infiltration_psi_f_m=.1,infiltration_delta_theta=.2)
    solver=LocalInertialSolver(ny=ny,nx=nx,dx=5.,dy=5.,bed_elevation=bed,
        manning_n=.03+np.indices((ny,nx))[0]*.001,active_mask=np.ones((ny,nx),np.int32),device='cuda:0',maximum_dt=.03,minimum_dt=1e-9,
        hydraulic_geometry_mode='traditional' if subgrid else 'coarse',
        subgrid_bed_elevation=native if subgrid else None,subgrid_resolution_m=1. if subgrid else None,
        subgrid_manning_n=(.03+np.indices(native.shape)[0]*.0001) if native_n else None,
        validation_profile=profile, infiltration_method=infiltration,
        track_cumulative_face_volumes=True,track_max_water_depth=True,**soil)
    solver.set_water_depth(depth=np.full((ny,nx),.08))
    solver.apply_sources_after_momentum=after
    solver.set_source_rates(rain=np.full((ny,nx),2e-5),infil=np.full((ny,nx),1e-6))
    bx=np.zeros((ny,nx+1),np.int32);bx[:,0]=BOUNDARY_SPECIFIED_STAGE;bx[:,-1]=BOUNDARY_FREE_OUTFLOW
    solver.set_boundary_type_qx(boundary_type=bx)
    solver.set_boundary_stage_qx(stage=np.full(bx.shape,.12))
    return solver


@pytest.mark.parametrize('subgrid,native_n',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('profile',['default','lisflood_fp_acc'])
@pytest.mark.parametrize('after',[False,True])
def test_graph_matches_windows_and_continues(subgrid,native_n,profile,after):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    if subgrid and profile!='default':
        with pytest.raises(ValueError,match='native free-outflow convention'):
            create(subgrid=subgrid,profile=profile,infiltration='constant',native_n=native_n,after=after)
        return
    original=create(subgrid=subgrid,profile=profile,infiltration='constant',native_n=native_n,after=after)
    graph=create(subgrid=subgrid,profile=profile,infiltration='constant',native_n=native_n,after=after)
    for duration in (.15,.13):
        times=np.array([original.simulation_time+duration/2,original.simulation_time+duration])
        options=dict(duration=duration,dt_history_capacity=100,checkpoint_times_s=times,field_checkpoint_indices=[0,1])
        a=original.run_device_loop(**options,use_cuda_graph=False);b=graph.run_device_loop(**options,use_cuda_graph=True)
        assert a['steps']==b['steps']
        np.testing.assert_array_equal(a['dt_history'],b['dt_history'])
        np.testing.assert_array_equal(original.get_water_depth(),graph.get_water_depth())
        np.testing.assert_array_equal(original.get_qx(),graph.get_qx())
        np.testing.assert_array_equal(original.get_qy(),graph.get_qy())
        np.testing.assert_array_equal(original.get_max_water_depth(),graph.get_max_water_depth())
        np.testing.assert_array_equal(a['checkpoint_trace']['field_depths'],b['checkpoint_trace']['field_depths'])
        for key,value in original.cumulative_accounting().items():
            assert graph.cumulative_accounting()[key]==pytest.approx(value,rel=1e-12,abs=1e-12)


@pytest.mark.parametrize('infiltration',['exponential','green_ampt'])
def test_graph_soil_sources_and_additional_source(infiltration):
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    sims=[create(subgrid=True,profile='default',infiltration=infiltration,native_n=False,after=False) for _ in range(2)]
    for sim in sims:
        sim.set_additional_source_rate(rate=np.full((4,6),1e-5))
        sim.set_infiltration_exclusion_mask(mask=np.indices((4,6))[0]==0)
    for sim,use in zip(sims,(False,True)):sim.run_device_loop(duration=.21,use_cuda_graph=use)
    np.testing.assert_array_equal(sims[0].get_water_depth(),sims[1].get_water_depth())
    np.testing.assert_array_equal(sims[0].cumulative_infiltration_m,sims[1].cumulative_infiltration_m)


@pytest.mark.parametrize('profile',['default','lisflood_fp_acc','lisflood_fp_acc_parity'])
@pytest.mark.parametrize('reduced',[False,True])
def test_graph_hyetograph_stage_series_and_accounting(profile,reduced,monkeypatch):
    from tests.test_device_loop import build_case,build_hyetograph,STAGE_SERIES
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    monkeypatch.setenv('OVERLANDER_REDUCE_MASS','1' if reduced else '0')
    a=build_case(device='cuda:0',profile=profile);b=build_case(device='cuda:0',profile=profile)
    options=dict(duration=1200.,rainfall_hyetograph=build_hyetograph(),stage_series_qx=STAGE_SERIES,
        dt_history_capacity=20000,checkpoint_times_s=[100.,700.,1200.],field_checkpoint_indices=[0,1,2])
    old=a.run_device_loop(**options,use_cuda_graph=False);new=b.run_device_loop(**options,use_cuda_graph=True)
    np.testing.assert_array_equal(old['dt_history'],new['dt_history'])
    np.testing.assert_array_equal(a.get_water_depth(),b.get_water_depth())
    np.testing.assert_array_equal(a.get_qx(),b.get_qx())
    np.testing.assert_array_equal(a.get_qy(),b.get_qy())
    for key,value in a.cumulative_accounting().items():
        assert b.cumulative_accounting()[key]==pytest.approx(value,rel=2e-12,abs=1e-12)


def test_conditional_callbacks_require_window_path():
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    sim=create(subgrid=False,profile='default',infiltration='constant',native_n=False,after=False)
    with pytest.raises(ValueError,match='per-sync'):
        sim.run_device_loop(duration=.1,use_cuda_graph=True,collect_free_outflow_diagnostics=True)


def selection_case(*, method, device):
    from tests.test_enhanced_production_api import model_inputs
    options=model_inputs(device=device,method=method,
        native_roughness=False,infiltration='constant',diagnostics=True)
    if method=='coarse':
        options['subgrid_bed_elevation']=None
        options['subgrid_resolution_m']=None
    solver=LocalInertialSolver(**options)
    solver.set_water_depth(depth=np.where(solver.active_mask.numpy(),.08,0.))
    solver.set_source_rates(rain=np.full((2,3),2e-5),infil=np.full((2,3),1e-6))
    return solver


def require_device(*, device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')


def assert_same_state(*, reference, candidate):
    for getter in ('get_water_depth','get_qx','get_qy','get_max_water_depth'):
        np.testing.assert_array_equal(getattr(reference,getter)(),getattr(candidate,getter)())
    np.testing.assert_array_equal(reference.cumulative_infiltration_m,candidate.cumulative_infiltration_m)
    for key,value in reference.cumulative_accounting().items():
        assert candidate.cumulative_accounting()[key]==pytest.approx(value,rel=2e-6,abs=1e-12)
    if reference.hydraulic_geometry_mode=='enhanced':
        for key,value in reference.get_component_state().items():
            np.testing.assert_array_equal(value,candidate.get_component_state()[key])


@pytest.mark.parametrize('method',['coarse','traditional','enhanced'])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('selection',['default',None,False,True])
def test_device_graph_selection_and_exact_window_parity(method,device,selection):
    require_device(device=device)
    reference=selection_case(method=method,device=device)
    candidate=selection_case(method=method,device=device)
    supported=False
    if device!='cpu':
        with wp.ScopedDevice(device):supported=wp.is_conditional_graph_supported()
    options={} if selection=='default' else dict(use_cuda_graph=selection)
    if selection is True and not supported:
        with pytest.raises(ValueError,match='conditional-graph support'):
            candidate.run_device_loop(duration=.07,**options)
        assert candidate.simulation_time==0.
        assert_same_state(reference=reference,candidate=candidate)
        return
    expected='conditional_cuda' if supported and selection is not False else 'queued_windows'
    for duration in (.07,.11):
        common=dict(duration=duration,sync_interval=7,dt_history_capacity=100,
            checkpoint_times_s=[candidate.simulation_time+duration],field_checkpoint_indices=[0])
        old=reference.run_device_loop(**common,use_cuda_graph=False)
        result=candidate.run_device_loop(**common,**options)
        assert result['execution']==expected
        assert old['steps']==result['steps']
        np.testing.assert_array_equal(old['dt_history'],result['dt_history'])
        np.testing.assert_array_equal(old['checkpoint_trace']['field_depths'],result['checkpoint_trace']['field_depths'])
        assert_same_state(reference=reference,candidate=candidate)


@pytest.mark.parametrize('method',['traditional','enhanced'])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('feature',['callback','trace'])
def test_automatic_graph_keeps_callback_and_trace_windows(method,device,feature):
    require_device(device=device)
    reference=selection_case(method=method,device=device)
    candidate=selection_case(method=method,device=device)
    calls=[]
    def on_sync(steps,time_s):
        calls.append((steps,time_s))
    options=dict(on_sync=on_sync) if feature=='callback' else dict(collect_free_outflow_diagnostics=True)
    with pytest.raises(ValueError,match='per-sync'):
        candidate.run_device_loop(duration=.07,use_cuda_graph=True,**options)
    assert candidate.simulation_time==0.
    old=reference.run_device_loop(duration=.07,sync_interval=2,use_cuda_graph=False,**options)
    expected_calls=calls.copy();calls.clear()
    result=candidate.run_device_loop(duration=.07,sync_interval=2,**options)
    assert result['execution']=='queued_windows'
    assert result['sync_count']==old['sync_count']
    assert calls==expected_calls
    if feature=='trace':assert len(result['free_outflow_limiter_trace'])==result['sync_count']
    assert_same_state(reference=reference,candidate=candidate)


@pytest.mark.parametrize('method',['traditional','enhanced'])
def test_automatic_graph_handles_unavailable_conditional_support(method,monkeypatch):
    require_device(device='cuda:0')
    calls=[]
    def unsupported():
        calls.append(wp.get_device().alias)
        return False
    monkeypatch.setattr(wp,'is_conditional_graph_supported',unsupported)
    candidate=selection_case(method=method,device='cuda:0')
    with pytest.raises(ValueError,match='conditional-graph support'):
        candidate.run_device_loop(duration=.07,use_cuda_graph=True)
    assert candidate.simulation_time==0.
    result=candidate.run_device_loop(duration=.07)
    assert result['execution']=='queued_windows'
    assert len(calls)==2
    candidate.run_device_loop(duration=.07,use_cuda_graph=False)
    assert len(calls)==2


@pytest.mark.parametrize("method", ["coarse", "traditional", "enhanced"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_progress_preserves_graph_selection_and_accepted_trajectory(method, device):
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    reference = selection_case(method=method, device=device)
    observed = selection_case(method=method, device=device)
    reports = []

    def record_progress(steps, simulation_time, execution):
        reports.append((steps, simulation_time, execution))

    options = dict(duration=.23, sync_interval=2, dt_history_capacity=1000)
    expected = reference.run_device_loop(**options)
    actual = observed.run_device_loop(**options, on_progress=record_progress, progress_interval_s=1.e-6)
    assert actual["execution"] == expected["execution"]
    assert actual["graph_capture_count"] == expected["graph_capture_count"]
    assert reports[-1][:2] == (actual["steps"], observed.simulation_time)
    assert all(report[2] == actual["execution"] for report in reports)
    assert all(right[0] > left[0] for left, right in zip(reports, reports[1:]))
    np.testing.assert_array_equal(actual["dt_history"], expected["dt_history"])
    np.testing.assert_array_equal(observed.get_water_depth(), reference.get_water_depth())
    np.testing.assert_array_equal(observed.get_qx(), reference.get_qx())
    np.testing.assert_array_equal(observed.get_qy(), reference.get_qy())
    np.testing.assert_array_equal(observed.get_max_water_depth(), reference.get_max_water_depth())
    if method == "enhanced":
        for name, values in reference.get_component_state().items():
            np.testing.assert_array_equal(observed.get_component_state()[name], values)
    for name, value in reference.cumulative_accounting().items():
        assert observed.cumulative_accounting()[name] == pytest.approx(value, rel=1.e-12, abs=1.e-12)
    if actual["execution"] == "conditional_cuda":
        assert actual["sync_count"] > 1
        assert actual["graph_capture_count"] == 1


def test_progress_callback_flushes_accounting_on_exception():
    model = selection_case(method="traditional", device="cpu")

    def stop_progress(steps, simulation_time, execution):
        raise RuntimeError("progress consumer failed")

    with pytest.raises(RuntimeError, match="progress consumer failed"):
        model.run_device_loop(duration=.23, on_progress=stop_progress, progress_interval_s=.01)
    assert model.cumulative_accounting()["rain"] > 0.


@pytest.mark.parametrize("interval", [0., -1., np.nan, np.inf])
def test_progress_interval_rejects_invalid_values(interval):
    model = selection_case(method="coarse", device="cpu")

    def record_progress(steps, simulation_time, execution):
        pass

    with pytest.raises(ValueError, match="progress_interval_s"):
        model.run_device_loop(duration=.1, on_progress=record_progress, progress_interval_s=interval)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_provisional_progress_keeps_nested_graph_and_component_state(device):
    from tests.test_enhanced_provisional_cfl import source_model
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    reference = source_model(device=device, method="exponential", momentum_integration="analytic")
    observed = source_model(device=device, method="exponential", momentum_integration="analytic")
    reports = []

    def record_progress(steps, simulation_time, execution):
        reports.append((steps, simulation_time, execution))

    options = dict(duration=5., sync_interval=2, dt_history_capacity=1000)
    expected = reference.run_device_loop(**options)
    actual = observed.run_device_loop(**options, on_progress=record_progress, progress_interval_s=1.e-6)
    assert actual["execution"] == expected["execution"]
    assert actual["graph_capture_count"] == expected["graph_capture_count"]
    np.testing.assert_array_equal(actual["dt_history"], expected["dt_history"])
    for name, values in reference.get_component_state().items():
        np.testing.assert_array_equal(observed.get_component_state()[name], values)
    assert reports[-1][:2] == (actual["steps"], observed.simulation_time)
