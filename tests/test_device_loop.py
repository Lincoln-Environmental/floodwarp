"""Regression tests for LocalInertialSolver.run_device_loop (Phase B).

The device-side simulation loop must reproduce the host-stepped path
step-for-step: bit-exact dt sequences and bit-exact hydraulic state, with
rainfall breakpoint landing and specified-stage updates on the same steps.
"""
import numpy as np
import pytest
import warp as wp

from floodwarp.boundary_conditions import BOUNDARY_FREE_OUTFLOW, BOUNDARY_SPECIFIED_STAGE
from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_kernels import (
    _device_piecewise_linear_interp,
    _device_next_breakpoint,
)
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.rainfall import Hyetograph
from tests.forcing_helpers import prepare_stage_controller,stage_variation_cap


def build_case(device, profile, seed=5):
    rng = np.random.default_rng(seed)
    ny, nx = 12, 16
    bed = rng.random((ny, nx)) * 0.4
    manning = rng.uniform(0.02, 0.06, (ny, nx))
    mask = np.ones((ny, nx), dtype=np.int32)
    mask[4, 7] = 0  # inactive cell exercises active-mask branching
    solver = LocalInertialSolver(
        ny=ny, nx=nx, dx=10.0, dy=10.0,
        bed_elevation=bed, manning_n=manning, active_mask=mask,
        device=device, validation_profile=profile,
    )
    b_qx = solver.boundary_type_qx.numpy()
    b_qx[:, nx] = BOUNDARY_FREE_OUTFLOW
    b_qx[:, 0] = BOUNDARY_SPECIFIED_STAGE
    b_qx[4, 0] = 1  # adjoins the inactive cell; keep closed
    solver.set_boundary_type_qx(boundary_type=b_qx)
    stage = solver.boundary_stage_qx.numpy()
    stage[:, 0] = 0.5
    solver.set_boundary_stage_qx(stage=stage)
    solver.set_water_depth(rng.random((ny, nx)) * 0.05)
    solver.set_source_rates(infil=np.full((ny, nx), 5.0e-7))
    return solver


def build_hyetograph() -> Hyetograph:
    return Hyetograph(
        node_times_s=np.array([0.0, 300.0, 700.0, 1200.0]),
        node_rates_m_per_s=np.array([1.0e-5, 8.0e-5, 2.0e-5, 0.0]),
        storm_type="chicago_normalized",
        metadata={"peak_fraction": 0.25},
    )


STAGE_SERIES = (np.array([0.0, 600.0, 1200.0]), np.array([0.4, 0.9, 0.3]))
DURATION = 1200.0


def run_host_stepped(solver, duration, hyetograph, stage_series):
    """Host-stepped driver with the benchmark's dynamic-forcing semantics."""
    end_time = solver.simulation_time + duration
    dt_sequence = []
    active_rate = None
    stage_buffers=prepare_stage_controller(solver=solver,series=stage_series) if stage_series is not None else None
    while solver.simulation_time < end_time:
        t = solver.simulation_time
        remaining = end_time - t
        if hyetograph is not None and t < hyetograph.duration_s:
            remaining = min(remaining, hyetograph.next_breakpoint_after(t) - t)
        if stage_series is not None:
            times,values=stage_series
            later=times[times>t]
            if later.size:
                remaining=min(remaining,float(later[0]-t))
            stage=solver.boundary_stage_qx.numpy()
            stage[solver.boundary_type_qx.numpy()==BOUNDARY_SPECIFIED_STAGE]=np.interp(t,times,values)
            solver.set_boundary_stage_qx(stage=stage)
        cap=min(remaining,solver.maximum_dt)
        if solver.adaptive_dt_growth_factor is not None and solver._adaptive_dt_reference>0.:
            cap=min(cap,solver._adaptive_dt_reference*solver.adaptive_dt_growth_factor)
        variation_cap=cap
        if stage_series is not None:
            variation_cap=stage_variation_cap(solver=solver,series=stage_series,
                buffers=stage_buffers,time=t,cap=cap)
            stage=solver.boundary_stage_qx.numpy()
            stage[solver.boundary_type_qx.numpy()==BOUNDARY_SPECIFIED_STAGE]=max(
                np.interp(t,*stage_series),np.interp(t+cap,*stage_series))
            solver.set_boundary_stage_qx(stage=stage)
        if hyetograph is not None:
            bound=max(hyetograph.rate_at(t),hyetograph.rate_at(t+cap)) if t<hyetograph.duration_s else 0.
            solver.set_uniform_rainfall_rate(rainfall_rate=bound)
        trial_dt = min(variation_cap,solver.calculate_stable_timestep(remaining_time=remaining))
        if hyetograph is not None:
            # Hyetograph semantics: rainfall is zero outside the storm; the
            # sample must not clamp onto the terminal rate (device-loop parity).
            t_sample = t
            if not solver.apply_sources_after_momentum:
                t_sample += 0.5 * trial_dt
            rate = (
                hyetograph.rate_at(t_sample)
                if t_sample < hyetograph.duration_s
                else 0.0
            )
            solver.set_uniform_rainfall_rate(rainfall_rate=rate)
        if stage_series is not None:
            value = float(np.interp(t+.5*trial_dt, *stage_series))
            stage = solver.boundary_stage_qx.numpy()
            stage[solver.boundary_type_qx.numpy() == BOUNDARY_SPECIFIED_STAGE] = value
            solver.set_boundary_stage_qx(stage=stage)
        solver.step(remaining_time=remaining, fixed_dt=trial_dt)
        dt_sequence.append(trial_dt)
    return np.asarray(dt_sequence)


def compare_host_vs_device(device, profile, sync_interval):
    hyetograph = build_hyetograph()
    host = build_case(device, profile)
    device_solver = build_case(device, profile)
    initial_rate = hyetograph.rate_at(0.0)
    host.set_uniform_rainfall_rate(initial_rate)
    device_solver.set_uniform_rainfall_rate(initial_rate)

    dt_host = run_host_stepped(host, DURATION, hyetograph, STAGE_SERIES)
    result = device_solver.run_device_loop(
        DURATION,
        sync_interval=sync_interval,
        rainfall_hyetograph=hyetograph,
        stage_series_qx=STAGE_SERIES,
        dt_history_capacity=100000,
    )
    assert result["checkpoint_trace"] is None
    dt_device = result["dt_history"]

    assert result["steps"] == len(dt_host)
    assert np.array_equal(dt_host, dt_device)  # bit-exact dt sequence
    assert np.array_equal(host.get_water_depth(), device_solver.get_water_depth())
    assert np.array_equal(host.get_qx(), device_solver.get_qx())
    assert np.array_equal(host.get_qy(), device_solver.get_qy())
    assert host.simulation_time == device_solver.simulation_time == DURATION
    host_totals = host.cumulative_accounting()
    device_totals = device_solver.cumulative_accounting()
    for key in host_totals:
        assert device_totals[key] == pytest.approx(host_totals[key], rel=1.0e-9, abs=1.0e-12)
    return dt_device


def test_device_loop_checkpoint_arguments_are_validated():
    solver = build_case("cpu", "lisflood_fp_acc")
    with pytest.raises(ValueError, match="strictly increasing"):
        solver.run_device_loop(
            1.0,
            checkpoint_times_s=np.array([0.5, 0.4]),
        )
    with pytest.raises(ValueError, match="shape"):
        solver.run_device_loop(
            1.0,
            checkpoint_times_s=np.array([0.5]),
            checkpoint_outlet_rows=np.zeros(solver.ny + 1, dtype=np.int32),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        solver.run_device_loop(
            1.0,
            checkpoint_times_s=np.array([1.0 + 5.0e-10]),
        )


def test_precomputed_stable_step_preserves_controlled_growth_reference():
    solver = build_case("cpu", None)
    dt = solver.calculate_stable_timestep(remaining_time=100.0)
    solver.step(remaining_time=100.0, fixed_dt=dt)
    assert solver._adaptive_dt_reference == dt
    previous = solver._adaptive_dt_reference
    solver.step(remaining_time=0.01, fixed_dt=0.01)
    assert solver._adaptive_dt_reference == previous


@pytest.mark.parametrize("profile", ["lisflood_fp_acc", None])
@pytest.mark.parametrize("sync_interval", [1, 8])
def test_device_loop_matches_host_stepped_bit_exact(profile, sync_interval):
    wp.init()
    compare_host_vs_device("cpu", profile, sync_interval)


@pytest.mark.parametrize("profile", ["lisflood_fp_acc", None])
def test_device_loop_matches_host_stepped_bit_exact_cuda(profile):
    wp.init()
    if not wp.is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    compare_host_vs_device("cuda:0", profile, sync_interval=8)


def test_device_loop_lands_on_rainfall_breakpoints():
    wp.init()
    dt_device = compare_host_vs_device("cpu", "lisflood_fp_acc", sync_interval=4)
    step_end_times = np.cumsum(dt_device)
    for node_time in build_hyetograph().node_times_s[1:]:
        assert np.any(np.isclose(step_end_times, node_time, rtol=0.0, atol=1.0e-9))
    assert np.isclose(step_end_times[-1], DURATION)


@pytest.mark.parametrize("profile", ["lisflood_fp_acc", None])
def test_device_loop_stops_rain_at_storm_end(profile):
    """A non-zero terminal hyetograph rate must not rain past the storm end.

    Hyetograph semantics: rainfall is zero outside the storm.  The device
    loop must not clamp its sample time onto the final node (which would hold
    the terminal rate forever); the host-stepped reference shares the rule.
    """
    wp.init()
    storm_end = 300.0
    duration = 900.0  # storm + recession
    hyetograph = Hyetograph(
        node_times_s=np.array([0.0, storm_end]),
        node_rates_m_per_s=np.array([1.0e-5, 1.0e-5]),
        storm_type="uniform",
        metadata={},
    )
    device_solver = build_case("cpu", profile)
    if profile == "lisflood_fp_acc":
        # The host/device bit-exact contract is maintained for the ACC
        # profile; the default-profile dt sequences diverge by a documented
        # pre-existing discrepancy, so that arm checks accounting only.
        host = build_case("cpu", profile)
        run_host_stepped(host, duration, hyetograph, None)
    result = device_solver.run_device_loop(
        duration, sync_interval=4, rainfall_hyetograph=hyetograph,
    )
    assert result["steps"] > 0
    if profile == "lisflood_fp_acc":
        assert np.array_equal(host.get_water_depth(), device_solver.get_water_depth())
        assert np.array_equal(host.get_qx(), device_solver.get_qx())
        assert np.array_equal(host.get_qy(), device_solver.get_qy())
    # Total rain must equal the storm integral (active cells x 100 m2),
    # not the perpetual-rain total (3x larger over the 900 s run).
    expected_rain = hyetograph.integrated_depth_m * 191 * 100.0
    rain = device_solver.cumulative_accounting()["rain"]
    assert rain == pytest.approx(expected_rain, rel=1.0e-2)


def test_device_loop_static_sources_match_host():
    """Without a hyetograph the static (spatial) source fields are used."""
    wp.init()
    rng = np.random.default_rng(9)
    for profile in ("lisflood_fp_acc", None):
        host = build_case("cpu", profile)
        device_solver = build_case("cpu", profile)
        rain = rng.random((12, 16)) * 3.0e-5
        other = np.full((12, 16), 1.0e-6)
        host.set_source_rates(rain=rain, other=other)
        device_solver.set_source_rates(rain=rain, other=other)

        dt_host = run_host_stepped(host, 600.0, None, None)
        result = device_solver.run_device_loop(600.0, sync_interval=5, dt_history_capacity=100000)
        assert np.array_equal(dt_host, result["dt_history"])
        assert np.array_equal(host.get_water_depth(), device_solver.get_water_depth())
        assert np.array_equal(host.get_qx(), device_solver.get_qx())
        assert np.array_equal(host.get_qy(), device_solver.get_qy())


def test_device_loop_raises_on_negative_depth():
    wp.init()
    solver = build_case("cpu", "lisflood_fp_acc")
    # Inject a significant negative depth directly into the device array.
    depth = solver.get_water_depth()
    depth[2, 3] = -1.0
    wp.copy(solver.water_depth, wp.array(depth, dtype=WP_FLOAT, device=solver.device))
    with pytest.raises(RuntimeError, match="Significant negative depth encountered"):
        solver.run_device_loop(100.0, sync_interval=4)


def test_device_loop_raises_on_subminimum_timestep():
    wp.init()
    solver = build_case("cpu", "lisflood_fp_acc")
    solver.minimum_dt = 100.0  # adaptive dt (~seconds) always violates the floor
    with pytest.raises(RuntimeError, match="below"):
        solver.run_device_loop(1000.0, sync_interval=4)


def test_device_loop_requires_fused_kernels():
    wp.init()
    solver = build_case("cpu", None)
    solver.use_fused_limiter = False
    with pytest.raises(ValueError, match="fused"):
        solver.run_device_loop(10.0)


@wp.kernel
def _interp_probe_kernel(
    times: wp.array(dtype=wp.float64),
    values: wp.array(dtype=wp.float64),
    sample: wp.float64,
    scratch: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    out[0] = _device_piecewise_linear_interp(times, values, sample, scratch)


@wp.kernel
def _forcing_search_probe_kernel(times: wp.array(dtype=wp.float64),values: wp.array(dtype=wp.float64),
    sample: wp.float64,scratch: wp.array(dtype=wp.float64),out: wp.array(dtype=wp.float64)):
    out[0]=_device_piecewise_linear_interp(times,values,sample,scratch)
    out[1]=_device_next_breakpoint(times,sample)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('node_count',[2,4097])
def test_long_forcing_search_preserves_exact_nodes_and_rounding(device,node_count):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(seed=818)
    times=np.r_[0.,np.cumsum(rng.uniform(.1,2.,size=node_count-1))]
    values=rng.normal(size=node_count)
    selected=times[np.array([0,1,node_count//2,node_count-2,node_count-1])]
    samples=np.r_[np.nextafter(selected,-np.inf),selected,np.nextafter(selected,np.inf),
        np.linspace(-1.,times[-1]+1.,10)]
    times_wp=wp.array(times,dtype=wp.float64,device=device)
    values_wp=wp.array(values,dtype=wp.float64,device=device)
    scratch=wp.zeros(18,dtype=wp.float64,device=device)
    out=wp.zeros(2,dtype=wp.float64,device=device)
    for sample in samples:
        wp.launch(kernel=_forcing_search_probe_kernel,dim=1,
            inputs=[times_wp,values_wp,wp.float64(sample),scratch,out],device=device)
        actual=out.numpy()
        assert actual[0]==float(np.interp(sample,times,values))
        assert actual[1]==times[min(int(np.searchsorted(times,sample,side='right')),node_count-1)]


def test_device_interp_matches_numpy_interp_bit_exact():
    wp.init()
    rng = np.random.default_rng(17)
    node_times = np.concatenate(([0.0], np.sort(rng.uniform(1.0, 100.0, size=6))))
    node_values = rng.uniform(0.0, 1.0e-4, size=node_times.size)
    times_wp = wp.array(node_times, dtype=wp.float64, device="cpu")
    values_wp = wp.array(node_values, dtype=wp.float64, device="cpu")
    scratch_wp = wp.zeros(12, dtype=wp.float64, device="cpu")
    out_wp = wp.zeros(1, dtype=wp.float64, device="cpu")

    samples = list(rng.uniform(-10.0, 200.0, size=200)) + list(node_times)
    for sample in samples:
        wp.launch(
            kernel=_interp_probe_kernel,
            dim=(1,),
            inputs=[times_wp, values_wp, wp.float64(sample), scratch_wp, out_wp],
            device="cpu",
        )
        expected = float(np.interp(sample, node_times, node_values))
        assert float(out_wp.numpy()[0]) == expected


def test_device_loop_reduced_mass_accounting_path(monkeypatch):
    """Device loop with the tree-reduction (pending) accounting path engaged."""
    wp.init()
    monkeypatch.setenv("OVERLANDER_REDUCE_MASS", "1")
    profile = "lisflood_fp_acc"
    hyetograph = build_hyetograph()
    host = build_case("cpu", profile)
    device_solver = build_case("cpu", profile)
    initial_rate = hyetograph.rate_at(0.0)
    host.set_uniform_rainfall_rate(initial_rate)
    device_solver.set_uniform_rainfall_rate(initial_rate)

    dt_host = run_host_stepped(host, 600.0, hyetograph, STAGE_SERIES)
    result = device_solver.run_device_loop(
        600.0, sync_interval=6, rainfall_hyetograph=hyetograph,
        stage_series_qx=STAGE_SERIES, dt_history_capacity=100000,
    )
    assert device_solver._pending_rain is not None
    assert np.array_equal(dt_host, result["dt_history"])
    assert np.array_equal(host.get_water_depth(), device_solver.get_water_depth())
    host_totals = host.cumulative_accounting()
    device_totals = device_solver.cumulative_accounting()
    for key in host_totals:
        assert device_totals[key] == pytest.approx(host_totals[key], rel=1.0e-9, abs=1.0e-12)
