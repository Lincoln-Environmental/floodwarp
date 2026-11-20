import numpy as np
import pytest
import warp as wp

from floodwarp.fv2_reference import reference_step
from floodwarp.fv2_solver import FV2Solver
from floodwarp.solver_factory import create_overland_solver


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("limiter", ("minmod", "mc"))
def test_fv2_oracle_and_restart(device, limiter, tmp_path):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(seed=4312)
    shape = (8, 15)
    z = rng.uniform(-0.2, 0.2, shape)
    h = rng.uniform(0.1, 1.1, shape)
    h[3:5, 4:8] = 0.0
    q = np.stack((h, 0.5 * h, -0.3 * h), axis=-1)
    rain = rng.uniform(0.0, 0.02, shape)
    n = rng.uniform(0.01, 0.04, shape)
    args = dict(
        solver_method="fv2",
        ny=shape[0],
        nx=shape[1],
        dx=1.0,
        dy=2.0,
        bed_elevation=z,
        manning_n=n,
        active_mask=np.ones(shape, bool),
        limiter=limiter,
        explicit_minimum_dt=0.0,
        explicit_maximum_dt=0.01,
        device=device,
    )
    s = create_overland_solver(**args)
    s.set_water_depth(depth=h)
    s.set_momentum(hu=q[..., 1], hv=q[..., 2])
    s.set_source_rates(rain=rain, infil=0.004)
    s.set_exponential_infiltration(f0=0.02, fd=0.05, conductivity=0.001)
    s.run_device_loop(duration=0.01)
    oracle, soil = reference_step(
        q=q,
        z=z,
        soil=np.zeros(shape),
        rain_start=rain,
        rain_end=rain,
        infil=0.004,
        f0=0.02,
        fd=0.05,
        conductivity=0.001,
        manning=n,
        dt=0.01,
        dx=1.0,
        dy=2.0,
        gravity=9.81,
        dry=0.0001,
        limiter=limiter,
    )
    np.testing.assert_allclose(s.get_conservative_state(), oracle, rtol=2.0e-13, atol=2.0e-14)
    np.testing.assert_allclose(s.cumulative_infiltration_m(), soil, rtol=2.0e-13, atol=2.0e-14)
    p = tmp_path / "fv2.npz"
    s.save_checkpoint(path=p)
    other = create_overland_solver(execution_mode="sparse", **args)
    other.load_checkpoint(path=p)
    diag = other.track_reconstruction(depth_edges=(0.001, 0.01, 0.1, 1.0))
    for solver in (s, other):
        solver.run_device_loop(duration=0.2)
        assert solver.get_water_depth().min() >= 0.0
        assert abs(solver.cumulative_accounting()["residual"]) < 2.0e-12
    np.testing.assert_array_equal(s.get_conservative_state(), other.get_conservative_state())
    np.testing.assert_array_equal(s.get_max_water_depth(), other.get_max_water_depth())
    assert np.asarray(diag.summary()["bands"])[:, :, 1].sum() > 0.0
    for a in ("qx", "qy"):
        np.testing.assert_array_equal(
            s.get_cumulative_face_volumes()[a], other.get_cumulative_face_volumes()[a]
        )
    with pytest.raises(ValueError, match="configuration"):
        create_overland_solver(
            **dict(args, limiter="mc" if limiter == "minmod" else "minmod")
        ).load_checkpoint(path=p)


def test_fv2_large_datum_shoreline_and_isolated_barrier():
    y, x = np.indices((19, 32))
    bed = 2.0e6 + 0.25 * np.sin(0.4 * x) + 0.38 * np.cos(0.25 * y)
    bed[:, 16] = 2.0e6 + 4.0
    active = np.ones_like(x, bool)
    active[5:8, 10:12] = False
    h = np.where(active & (x < 16), np.maximum(0.0, 2.0e6 + 0.1 - bed), 0.0)
    s = FV2Solver(
        ny=19, nx=32, dx=1.0, dy=1.0, bed_elevation=bed, manning_n=0.03, active_mask=active, device="cpu"
    )
    s.set_water_depth(depth=h)
    s.run_device_loop(duration=0.5)
    np.testing.assert_allclose(s.get_water_depth(), h, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(s.get_conservative_state()[..., 1:], 0.0, atol=2.0e-15)
    assert np.count_nonzero(s.get_water_depth()[:, 16:]) == 0


def test_fv2_exhaustion_and_structure_rejection():
    args = dict(
        ny=4, nx=6, dx=10.0, dy=10.0, bed_elevation=0.0, manning_n=0.03, active_mask=np.ones((4, 6), bool)
    )
    with pytest.raises(ValueError, match="culvert"):
        FV2Solver(structures=[object()], **args)
    s = FV2Solver(**args)
    s.set_water_depth(depth=0.001)
    s.set_source_rates(rain=0.001, infil=0.1)
    s.run_device_loop(duration=0.1)
    np.testing.assert_array_equal(s.get_water_depth(), 0.0)
    assert abs(s.cumulative_accounting()["residual"]) < 1.0e-14


def test_fv2_varying_stage_uses_both_stage_times_and_checks_identity(tmp_path):
    opts = dict(
        ny=3,
        nx=10,
        dy=4.0,
        dx=4.0,
        bed_elevation=0.0,
        active_mask=np.ones((3, 10), bool),
        manning_n=0.0,
        device="cpu",
        maximum_dt=0.04,
    )

    def create(*, varying, cap):
        s = FV2Solver(**dict(opts, maximum_dt=cap))
        s.set_water_depth(depth=1.0)
        types = np.zeros((3, 11), np.int32)
        types[:, 0] = 3
        s.set_boundary_type_qx(boundary_type=types)
        s.set_boundary_stage_qx(stage=np.ones_like(types, float))
        if varying:
            s.set_stage_series(axis="x", series=([0.0, 0.07, 0.2], [1.0, 1.03, 1.1]))
        return s

    fine = create(varying=True, cap=0.000625)
    fine.run_device_loop(duration=0.2)
    errors = []
    for cap in (0.04, 0.02, 0.01):
        s = create(varying=True, cap=cap)
        s.run_device_loop(duration=0.2)
        errors.append(np.linalg.norm(s.get_conservative_state() - fine.get_conservative_state()))
        assert abs(s.cumulative_accounting()["residual"]) < 1.0e-12
    assert errors[-1] < 0.14 * errors[0]
    path = tmp_path / "stage.npz"
    s.save_checkpoint(path=path)
    with pytest.raises(ValueError, match="configuration"):
        create(varying=False, cap=0.01).load_checkpoint(path=path)


def test_failed_fv2_trial_keeps_last_accepted_physical_state():
    s = FV2Solver(
        ny=4,
        nx=4,
        dx=1.0,
        dy=1.0,
        bed_elevation=0.0,
        manning_n=0.03,
        active_mask=np.ones((4, 4), bool),
        device="cpu",
        minimum_dt=0.2,
        maximum_dt=1.0,
    )
    s.set_water_depth(depth=3.0)
    before = s.get_conservative_state()
    with pytest.raises(ArithmeticError):
        s.run_device_loop(duration=0.5)
    np.testing.assert_array_equal(s.get_conservative_state(), before)
    assert s.time == s.steps == 0


def test_fv2_direct_model_graph_stage_restart(tmp_path):
    from floodwarp.inputs import boundary_faces, terrain_from_array
    from floodwarp.model import advance, create_model, load_checkpoint, save_checkpoint

    terrain = terrain_from_array(
        elevation=np.zeros((3, 6)), dx=3.0, dy=4.0, row_order="south_to_north", vertical_datum="local"
    )
    bx, by = boundary_faces(terrain=terrain)
    bx[:, 0] = 3
    options = dict(
        terrain=terrain,
        solver_method="fv2",
        initial_depth=0.5,
        stage_datum="local",
        boundary_qx=bx,
        boundary_qy=by,
        stage_series_qx=([0.0, 0.15, 0.3], [0.5, 0.55, 0.6]),
        explicit_maximum_dt=0.05,
        device="cpu",
    )
    test = create_model(**options)
    advance(model=test, end_time=0.15)
    checkpoint = tmp_path / "direct.npz"
    save_checkpoint(model=test, path=checkpoint)
    restart = create_model(**options)
    load_checkpoint(model=restart, path=checkpoint)
    for model in (test, restart):
        report = advance(model=model, end_time=0.3)
        assert report["steps"] >= 3
    np.testing.assert_array_equal(
        test.solver.get_conservative_state(), restart.solver.get_conservative_state()
    )
    assert test.solver.cumulative_accounting()["boundary_inflow"] > 0.0


def test_fv2_retry_has_one_accepted_step_on_host_and_device():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    options = dict(
        ny=4,
        nx=5,
        dx=1.0,
        dy=1.0,
        bed_elevation=0.0,
        manning_n=0.0,
        active_mask=np.ones((4, 5), dtype=bool),
        maximum_dt=1.0,
        device=device,
    )
    # The first trial exceeds reconstructed CFL. Observe the accepted dt and run
    # to that exact time on a fresh host-stepped solver with a single-step budget.
    baseline = FV2Solver(**options)
    baseline.set_water_depth(depth=2.0)
    dt = baseline.step(fixed_dt=1.0)
    assert 0.0 < dt < 1.0
    host = FV2Solver(**options)
    host.set_water_depth(depth=2.0)
    stats = host.run_device_loop(duration=dt, max_steps=1, use_cuda_graph=False)
    assert stats["steps"] == stats["host_state_syncs"] == 1
    np.testing.assert_array_equal(host.get_conservative_state(), baseline.get_conservative_state())

    rejected = FV2Solver(**options)
    rejected.set_water_depth(depth=2.0)
    with pytest.raises(ArithmeticError, match="step budget"):
        rejected.run_device_loop(duration=1.0, max_steps=1, use_cuda_graph=False)
    assert rejected.steps == 1 and rejected.time == dt


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("mode", ("dense", "sparse"))
def test_stage_reservoir_on_mask_has_perimeter_inflow(axis, mode):
    def run(*, border):
        shape = (6 + 2 * border, 8 + 2 * border)
        mask = np.zeros(shape, dtype=bool)
        inner = np.s_[border : border + 6, border : border + 8]
        mask[inner] = True
        solver = FV2Solver(
            ny=shape[0],
            nx=shape[1],
            dx=2.0,
            dy=3.0,
            active_mask=mask,
            bed_elevation=0.0,
            manning_n=0.03,
            maximum_dt=0.02,
            device="cpu",
            execution_mode=mode,
        )
        faces = list(shape)
        faces[1 - axis] += 1
        codes = np.zeros(faces, dtype=np.int32)
        if axis == 0:
            codes[border : border + 6, border] = 3
            solver.set_boundary_type_qx(boundary_type=codes)
            solver.set_boundary_stage_qx(stage=np.full(faces, 0.5))
        else:
            codes[border, border : border + 8] = 3
            solver.set_boundary_type_qy(boundary_type=codes)
            solver.set_boundary_stage_qy(stage=np.full(faces, 0.5))
        solver.run_device_loop(duration=0.5)
        ledger = solver.cumulative_accounting()
        assert ledger["boundary_inflow"] > 0.0
        assert abs(ledger["residual"]) < 1.0e-12
        return solver.get_conservative_state()[inner], ledger["boundary_inflow"]

    reference, perimeter = run(border=0)
    masked, internal = run(border=1)
    np.testing.assert_array_equal(masked, reference)
    np.testing.assert_allclose(internal, perimeter, rtol=1.0e-14)


@pytest.mark.parametrize("mode", ("dense", "sparse"))
def test_thin_fast_films_on_steep_bed_with_rapid_sources(mode):
    rng = np.random.default_rng(seed=20261120)
    shape = (13, 23)
    iy, ix = np.indices(shape)
    mask = np.ones(shape, bool)
    mask[3:7, 9:12] = False
    bed = -0.18 * ix + 0.12 * np.sin(iy * 0.7) + 0.015 * rng.standard_normal(size=shape)
    depth = np.where(mask, np.where(ix < 5, 0.45, rng.uniform(0.0, 0.0006, shape)), 0.0)
    depth[8:11, 12:17] = 0.0
    wet = depth > 0.0001
    momentum = np.where(wet, depth * np.where(ix < 5, 2.0, -4.0), 0.0)
    devices = ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)
    reference = None
    for device in devices:
        solver = FV2Solver(
            ny=shape[0],
            nx=shape[1],
            dx=0.8,
            dy=1.3,
            active_mask=mask,
            bed_elevation=bed,
            manning_n=0.015 + 0.002 * iy,
            maximum_dt=0.008,
            execution_mode=mode,
            device=device,
        )
        solver.set_water_depth(depth=depth)
        solver.set_momentum(hu=momentum, hv=0.4 * momentum)
        solver.set_rainfall_hyetograph(times_s=np.arange(5) * 0.08, rates_m_s=[0.0, 0.003, 0.0, 0.002, 0.0])
        solver.set_rainfall_spatial_multiplier(multiplier=0.5 + 0.5 * (ix % 3))
        solver.set_exponential_infiltration(f0=0.004, fd=0.008, conductivity=0.0007)
        for t in np.arange(1, 5) * 0.08:
            solver.run_device_loop(duration=t - solver.time)
            state = solver.get_conservative_state()
            assert state[..., 0].min() >= 0.0 and np.isfinite(state).all()
            assert np.all(state[..., 1:][state[..., 0] <= solver.dry_depth] == 0.0)
            assert abs(solver.cumulative_accounting()["residual"]) < 1.0e-11
        if reference is not None:
            np.testing.assert_allclose(state, reference, atol=3.0e-13, rtol=5.0e-13)
        reference = state
