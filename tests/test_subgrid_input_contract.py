"""Physical raster alignment, input ownership, and GIS-to-hydraulics contracts."""
import numpy as np
import pytest

from floodwarp.subgrid_geometry import build_subgrid_geometry, storage_from_stage
from floodwarp.local_inertial_solver import LocalInertialSolver


@pytest.mark.parametrize('shape', [(1, 1), (1, 4), (4, 1), (3, 4)])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed', 'sorted'])
def test_native_input_unchanged_and_geometry_matches_original(shape, layout):
    ny, nx = shape
    factor = 3
    rng = np.random.default_rng(204)
    original = rng.uniform(0, 3, (ny*factor, nx*factor))
    if layout == 'strided':
        backing = np.zeros((ny*factor*2, nx*factor*2))
        backing[::2, ::2] = original
        native = backing[::2, ::2]
    elif layout == 'reversed':
        native = original[::-1, ::-1]
    elif layout == 'sorted':
        native = np.sort(original, axis=None).reshape(original.shape)
    else:
        native = original
    before = native.copy()
    expected_z = np.sort(before.reshape(ny, factor, nx, factor).transpose(0, 2, 1, 3).reshape(ny, nx, -1), axis=2)
    for repeat in range(2):
        geometry = build_subgrid_geometry(ny=ny, nx=nx, dx=3., dy=3.,
            subgrid_bed_elevation=native, subgrid_resolution_m=1.)
        np.testing.assert_array_equal(native, before)
        np.testing.assert_array_equal(geometry.cell_z, expected_z)
        assert not np.shares_memory(geometry.cell_z, native)
        eta = np.full(shape, 1.5)
        expected_storage = np.maximum(1.5-before, 0).reshape(ny, factor, nx, factor).mean(axis=(1, 3))
        np.testing.assert_allclose(storage_from_stage(geometry=geometry, stage=eta), expected_storage, atol=1e-15)
        np.testing.assert_array_equal(geometry.face_z_x[:, 0], before[:, 0].reshape(ny, factor))
        np.testing.assert_array_equal(geometry.face_z_y[0], before[0].reshape(nx, factor))


@pytest.mark.parametrize('dx,dy,resolution,native_shape', [
    (10.,10.,1.,(10,15)), (10.,10.,2.,(20,30)),
    (10.,10.,3.,(6,9)), (10.,5.,1.,(20,30)),
    (float('nan'),10.,1.,(20,30)), (float('inf'),10.,1.,(20,30)),
    (0.,0.,1.,(20,30)), (-1.,-1.,1.,(20,30)),
    (10.,float('nan'),1.,(20,30)), (10.,float('inf'),1.,(20,30)),
    (10.,0.,1.,(20,30)), (10.,10.,float('nan'),(20,30)),
    (10.,10.,float('inf'),(20,30)), (10.,10.,0.,(20,30)),
    (10.,10.,-1.,(20,30)), (10.,10.,1.,(20,15)),
])
def test_invalid_physical_metadata_rejected(dx, dy, resolution, native_shape):
    with pytest.raises(ValueError):
        build_subgrid_geometry(ny=2, nx=3, dx=dx, dy=dy,
            subgrid_bed_elevation=np.zeros(native_shape), subgrid_resolution_m=resolution)


@pytest.mark.parametrize('factor,resolution', [(1,10.), (2,5.), (5,2.), (10,1.)])
def test_rectangular_raster_with_square_cells_accepted(factor, resolution):
    geometry = build_subgrid_geometry(ny=2, nx=3, dx=10., dy=10.,
        subgrid_bed_elevation=np.zeros((2*factor,3*factor)), subgrid_resolution_m=resolution)
    assert geometry.factor == factor


def test_solver_rejects_nonsquare_subgrid_cells():
    with pytest.raises(ValueError, match='square coarse cells'):
        LocalInertialSolver(ny=2, nx=3, dx=10., dy=5., bed_elevation=np.zeros((2,3)),
            manning_n=np.full((2,3), .03), active_mask=np.ones((2,3), dtype=np.int32),
            hydraulic_geometry_mode='traditional', subgrid_bed_elevation=np.zeros((20,30)),
            subgrid_resolution_m=1., device='cpu')
