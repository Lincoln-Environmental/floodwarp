import numpy as np
import pytest
import warp as wp

from floodwarp.full_swe_kernels import research_flux as warp_flux
from floodwarp.full_swe_reference import research_flux as numpy_flux
from floodwarp.full_swe_solver import FullSWESolver


@wp.kernel
def evaluate_faces(left: wp.array(dtype=wp.vec3d), right: wp.array(dtype=wp.vec3d),
                   output: wp.array(dtype=wp.vec4d), gravity: wp.float64, dry: wp.float64, mode: int):
    i = wp.tid()
    output[i] = warp_flux(left=left[i], right=right[i], g=gravity, dry=dry, mode=mode)


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
@pytest.mark.parametrize('scheme,mode', (('roe', 4), ('hllc', 5)))
def test_research_face_fluxes(device, scheme, mode):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    left = np.array([[0., 0., 0.], [1.e-10, 2.e-10, 0.], [1., -14., 7.], [1., 11., 2.],
                     [1., 0., 2.], [1., .8, 2.], [1., 0., 0.], [.5, 2., -.1]])
    right = np.array([[0., 0., 0.], [.1, .1, -.5], [1., 14., -7.], [1., -13., -2.],
                      [1., 0., -1.], [1., .8, -1.], [.25, .8, 0.], [.5, 2., -.1]])
    expected = numpy_flux(left=left, right=right, gravity=9.81, dry_depth=.0001, flux_scheme=scheme)
    output = wp.empty(shape=len(left), dtype=wp.vec4d, device=device)
    wp.launch(kernel=evaluate_faces, dim=len(left), inputs=[wp.array(left, dtype=wp.vec3d, device=device),
        wp.array(right, dtype=wp.vec3d, device=device), output, 9.81, .0001, mode], device=device)
    result = output.numpy()
    np.testing.assert_allclose(actual=result[:, :3], desired=expected, rtol=1.e-13, atol=1.e-13)
    np.testing.assert_array_equal(actual=result[:2, 3], desired=[0., 1.])
    np.testing.assert_allclose(actual=result[4, :3], desired=[0., 4.905, 0.], atol=1.e-14)
    np.testing.assert_allclose(actual=result[5, :3], desired=[.8, 5.545, 1.6], atol=1.e-14)
    np.testing.assert_allclose(actual=result[-1, :3], desired=[2., 9.22625, -.4], atol=1.e-14)
    if scheme == 'roe':
        assert result[2, 3] == 2
        assert result[6, 3] == 3


@pytest.mark.parametrize('scheme', ('roe', 'hllc'))
@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_research_lake_shoreline_and_checkpoint(tmp_path, scheme, device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    y, x = np.indices((12, 16))
    bed = .25 * np.sin(x / 3.) + .3 * np.cos(y / 2.)
    initial = np.maximum(0., .1 - bed)
    active = np.ones_like(bed, dtype=bool)
    active[4:7, 3:5] = False
    options = dict(ny=12, nx=16, dx=1., dy=1., bed_elevation=bed, manning_n=0.,
                   active_mask=active, maximum_dt=.03125, flux_scheme=scheme, device=device)
    sparse = FullSWESolver(execution_mode='sparse', **options)
    sparse.set_water_depth(depth=initial)
    stats = sparse.run_device_loop(duration=1., track_flux=True)
    assert stats['flux_outcomes']['faces'] > 0
    path = tmp_path / 'state.npz'
    sparse.save_checkpoint(path=path)
    dense = FullSWESolver(execution_mode='dense', **options)
    dense.load_checkpoint(path=path)
    if scheme == 'hllc':
        archived = sparse.capture_state()
        del archived['hllc_revision']
        with pytest.raises(ValueError, match='superseded experimental HLLC'):
            dense.restore_state(state=archived)
    for solver in (dense, sparse):
        solver.run_device_loop(duration=1.)
        state = solver.get_conservative_state()
        np.testing.assert_allclose(actual=state[..., 0], desired=np.where(active, initial, 0.), rtol=0., atol=1.e-15)
        np.testing.assert_allclose(actual=state[..., 1:], desired=0., rtol=0., atol=1.e-15)
        assert abs(solver.cumulative_accounting()['residual']) < 1.e-12
    np.testing.assert_array_equal(actual=dense.get_conservative_state(), desired=sparse.get_conservative_state())
    options['flux_scheme'] = 'contact_hll'
    other = FullSWESolver(execution_mode='dense', **options)
    with pytest.raises(ValueError, match='configuration'):
        other.load_checkpoint(path=path)


def physical_flux(*, state, g):
    h, m, n = state
    return np.array([m, m*m/h + g*h*h/2., m*n/h])


def outer_bounds(*, left, right, g):
    hl, hr = left[0], right[0]
    ul, ur = left[1]/hl, right[1]/hr
    cl, cr = np.sqrt(g*np.array([hl, hr]))
    head = min(ul-cl, ur-cr)
    tail = max(ul+cl, ur+cr)
    cm = (cl+cr)/2. + (ul-ur)/4.
    if cm > 0.:
        um = (ul+ur)/2. + cl-cr
        head, tail = min(head, um-cm), max(tail, um+cm)
    return head, tail


def random_states(*, rng, count):
    h = 10.**rng.uniform(low=-3., high=1., size=count)
    velocity = rng.uniform(low=-12., high=12., size=(count, 2))
    return np.column_stack((h, h[:, None]*velocity))


def test_hllc_three_rankine_hugoniot_jumps():
    rng = np.random.default_rng(seed=67123)
    left = random_states(rng=rng, count=500)
    right = random_states(rng=rng, count=500)
    left[0], right[0] = (2., 0., 0.), (.4, 0., 0.)
    g = 9.81
    flux = numpy_flux(left=left, right=right, gravity=g, dry_depth=1.e-4, flux_scheme='hllc')
    for a, b, actual in zip(left, right, flux, strict=True):
        lower, upper = outer_bounds(left=a, right=b, g=g)
        fl, fr = physical_flux(state=a, g=g), physical_flux(state=b, g=g)
        if lower >= 0. or upper <= 0.:
            expected = fl if lower >= 0. else fr
        else:
            hll = (upper*b - lower*a - (fr-fl))/(upper-lower)
            assert hll[0] > 0.
            la, rb = hll.copy(), hll.copy()
            la[2], rb[2] = hll[0]*a[2]/a[0], hll[0]*b[2]/b[0]
            fa = fl + lower*(la-a)
            fb = fr + upper*(rb-b)
            middle = fa[0]/hll[0]
            assert lower <= middle <= upper
            np.testing.assert_allclose(actual=fb-fa, desired=middle*(rb-la), atol=3.e-12)
            expected = fa if middle >= 0. else fb
        np.testing.assert_allclose(actual=actual, desired=expected, rtol=2.e-12, atol=3.e-12)


@pytest.mark.parametrize('scheme', ('roe', 'hllc'))
def test_random_faces_orientation_and_consistency(scheme):
    rng = np.random.default_rng(seed=19411)
    left, right = random_states(rng=rng, count=1000), random_states(rng=rng, count=1000)
    normal = np.array([1., -1., 1.])
    flux = numpy_flux(left=left, right=right, gravity=9.81, dry_depth=1.e-4, flux_scheme=scheme)
    reverse = numpy_flux(left=right*normal, right=left*normal, gravity=9.81, dry_depth=1.e-4, flux_scheme=scheme)
    np.testing.assert_allclose(actual=reverse, desired=-normal*flux, rtol=2.e-12, atol=2.e-12)
    same = numpy_flux(left=left, right=left, gravity=9.81, dry_depth=1.e-4, flux_scheme=scheme)
    np.testing.assert_allclose(actual=same, desired=[physical_flux(state=q, g=9.81) for q in left], atol=1.e-13)


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_roe_characteristic_flux_and_entropy_partition(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(seed=944801)
    left, right = random_states(rng=rng, count=600), random_states(rng=rng, count=600)
    out = wp.empty(shape=len(left), dtype=wp.vec4d, device=device)
    wp.launch(kernel=evaluate_faces, dim=len(left), inputs=[wp.array(left, dtype=wp.vec3d, device=device),
              wp.array(right, dtype=wp.vec3d, device=device), out, 9.81, .0001, 4], device=device)
    result = out.numpy()
    np.testing.assert_allclose(actual=result[:, :3], desired=numpy_flux(left=left, right=right,
        gravity=9.81, dry_depth=.0001, flux_scheme='roe'), rtol=2.e-12, atol=1.e-12)
    for a, b, actual in zip(left, right, result, strict=True):
        g = 9.81
        fl, fr = physical_flux(state=a, g=g), physical_flux(state=b, g=g)
        weights = np.sqrt([a[0], b[0]])
        u, v = np.average(np.stack((a[1:]/a[0], b[1:]/b[0])), axis=0, weights=weights)
        c = np.sqrt(g*(a[0]+b[0])/2.)
        speeds = np.array([u-c, u, u+c])
        vectors = np.array([[1., 0., 1.], [u-c, 0., u+c], [v, 1., v]])
        strengths = np.linalg.solve(vectors, b-a)
        waves = vectors * strengths
        np.testing.assert_allclose(actual=waves @ speeds, desired=fr-fl, atol=2.e-12)
        mid = a + waves[:, 0]
        lower, upper = outer_bounds(left=a, right=b, g=g)
        valid = mid[0] > .0001 and speeds[0] >= lower and speeds[2] <= upper
        factors = np.minimum(speeds, 0.)
        split = False
        if valid:
            um, cm = mid[1]/mid[0], np.sqrt(g*mid[0])
            for k, lo, hi in ((0, a[1]/a[0]-np.sqrt(g*a[0]), um-cm),
                              (2, um+cm, b[1]/b[0]+np.sqrt(g*b[0]))):
                if lo < 0. < hi:
                    valid &= lower <= lo <= speeds[k] <= hi <= upper
                    if valid:
                        # Fractions of a split wave conserve its strength and flux jump.
                        fractions = np.linalg.solve([[1., 1.], [lo, hi]], [1., speeds[k]])
                        factors[k] = fractions[0]*lo
                        split = True
        if valid:
            expected = fl + waves @ factors
            assert actual[3] == (3 if split else 0)
        else:
            assert actual[3] == 2
            expected = (upper*fl-lower*fr+lower*upper*(b-a))/(upper-lower)
            if lower >= 0. or upper <= 0.:
                expected = fl if lower >= 0. else fr
        np.testing.assert_allclose(actual=actual[:3], desired=expected, rtol=2.e-12, atol=2.e-12)
    assert set(result[:, 3]) == {0., 2., 3.}


@pytest.mark.parametrize('scheme', ('roe', 'hllc'))
def test_random_closed_basin_positivity_and_axis_rotation(scheme):
    rng = np.random.default_rng(seed=3567)
    states = random_states(rng=rng, count=15*24).reshape(15, 24, 3)
    states[::3, ::2] = 0.
    bed = rng.uniform(low=-.1, high=.1, size=states.shape[:2])
    outputs = []
    for elevation, q in ((bed, states), (bed.T, states.transpose(1, 0, 2)[..., [0, 2, 1]])):
        solver = FullSWESolver(ny=elevation.shape[0], nx=elevation.shape[1], dx=1., dy=1.,
                               bed_elevation=elevation, manning_n=0., active_mask=np.ones_like(elevation, dtype=bool),
                               device='cpu', flux_scheme=scheme, execution_mode='dense')
        solver.set_water_depth(depth=q[..., 0])
        solver.set_momentum(hu=q[..., 1], hv=q[..., 2])
        solver.run_device_loop(duration=.2, track_flux=True)
        final = solver.get_conservative_state()
        assert np.isfinite(final).all() and final[..., 0].min() >= 0.
        assert final[..., 0].sum() == pytest.approx(q[..., 0].sum(), abs=1.e-10, rel=0.)
        outputs.append(final)
    np.testing.assert_allclose(actual=outputs[0], desired=outputs[1].transpose(1, 0, 2)[..., [0, 2, 1]],
                               atol=2.e-12, rtol=2.e-12)


def test_stationary_shock_roe_has_no_fallback():
    g = 9.81
    h = np.geomspace(start=.005, stop=5., num=40)
    froude = np.linspace(start=1.1, stop=5., num=40)
    downstream = h*(np.sqrt(1.+8.*froude**2)-1.)/2.
    discharge = h*froude*np.sqrt(g*h)
    left = np.column_stack((h, discharge, h*.35))
    right = np.column_stack((downstream, discharge, downstream*.35))
    expected = [physical_flux(state=q, g=g) for q in left]
    np.testing.assert_allclose(actual=[physical_flux(state=q, g=g) for q in right], desired=expected, rtol=2.e-14)
    output = wp.empty(shape=len(left), dtype=wp.vec4d, device='cpu')
    wp.launch(kernel=evaluate_faces, dim=len(left), inputs=[wp.array(left, dtype=wp.vec3d, device='cpu'),
        wp.array(right, dtype=wp.vec3d, device='cpu'), output, g, .0001, 4], device='cpu')
    values = output.numpy()
    np.testing.assert_array_equal(actual=values[:, 3], desired=0.)
    np.testing.assert_allclose(actual=values[:, :3], desired=expected, rtol=2.e-14)


def test_interior_transport_matches_limited_face_ledger():
    from validation.fv1_riemann_transport import InteriorTransport

    shape = (5, 9)
    solver = FullSWESolver(ny=5, nx=9, dx=2., dy=3., bed_elevation=np.zeros(shape), manning_n=0.,
                           active_mask=np.ones(shape, dtype=bool), device='cpu', flux_scheme='roe', execution_mode='dense')
    depth = np.tile([0., .00001, .003, .03, .3, 2., .2, .01, 0.], (5, 1))
    solver.set_water_depth(depth=depth)
    dt = .01
    probe = InteriorTransport(solver=solver, edges=(.001, .01, .1, 1.), wet=.01, channel_depth=.1, channel_discharge=.05)
    stats = solver.run_device_loop(duration=dt, track_flux=True)
    assert stats['steps'] == 1
    values = np.array(probe.results()['groups']['all'])
    total = values.sum(axis=(0, 1))
    x = solver._fields['flux_x'].numpy().reshape(5, 10, 3)[:, 1:-1]
    y = solver._fields['flux_y'].numpy().reshape(6, 9, 3)[1:-1]
    expected_volume = dt*(3.*np.abs(x[..., 0]).sum() + 2.*np.abs(y[..., 0]).sum())
    expected_impulse = dt*(3.*np.linalg.norm(x[..., 1:], axis=-1).sum()+2.*np.linalg.norm(y[..., 1:], axis=-1).sum())
    np.testing.assert_allclose(actual=total[1:], desired=[expected_volume, expected_impulse], rtol=1.e-14)
    assert total[0] == 5*8 + 4*7
    assert values[:, 1, 0].sum() > 0
