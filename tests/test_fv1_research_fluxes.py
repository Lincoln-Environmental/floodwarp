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
