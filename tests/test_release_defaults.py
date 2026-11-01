"""Public formulation defaults, sampling endpoints and checkpoint identity."""
import numpy as np
import pytest

from floodwarp import FullSWESolver, LocalInertialSolver, create_overland_solver
from floodwarp.inputs import terrain_from_array
from floodwarp.model import advance, create_model, load_checkpoint, save_checkpoint
from floodwarp.sampling import sample_targets


@pytest.mark.parametrize(('duration', 'interval', 'expected'), [
    (2., 5., [2.]), (22., 5., [5., 10., 15., 20., 22.]),
    (23., 5., [5., 10., 15., 20., 23.]), (20., 5., [5., 10., 15., 20.])])
def test_sample_endpoints(duration, interval, expected):
    np.testing.assert_array_equal(actual=sample_targets(duration=duration, interval=interval), desired=expected)


@pytest.mark.parametrize(('duration', 'interval'), [(0., 1.), (1., 0.), (-1., 2.), (np.inf, 1.), (1., np.nan)])
def test_invalid_sample_times(duration, interval):
    with pytest.raises(ValueError, match='Positive finite'):
        sample_targets(duration=duration, interval=interval)


def test_standalone_replay_keeps_identical_sampling():
    from examples.orton_bradley.run_model import sample_targets as standalone_targets
    for duration in (2., 22., 23., 20.):
        np.testing.assert_array_equal(actual=sample_targets(duration=duration, interval=5.),
            desired=standalone_targets(duration=duration, interval=5.))


def test_default_factories_and_cross_formulation_restart(tmp_path):
    terrain = terrain_from_array(elevation=np.zeros((3, 4)), dx=2., dy=2.,
        row_order='south_to_north', vertical_datum='local')
    options = dict(ny=3, nx=4, dx=2., dy=2., bed_elevation=terrain.elevation,
        manning_n=np.full((3, 4), .03), active_mask=np.ones((3, 4), dtype=np.int32))
    assert isinstance(create_overland_solver(**options), FullSWESolver)
    assert isinstance(create_overland_solver(solver_method='li', **options), LocalInertialSolver)
    default = create_model(terrain=terrain, rainfall=.001)
    assert isinstance(default.solver, FullSWESolver)
    assert default.solver.execution_diagnostics()['execution'] == 'sparse_uniform'
    dense = create_model(terrain=terrain, execution_mode='dense')
    assert dense.solver.execution_diagnostics()['execution'] == 'dense_uniform'
    li = create_model(terrain=terrain, solver_method='li', rainfall=.001)
    advance(model=li, end_time=.1)
    save_checkpoint(model=li, path=tmp_path/'li.npz')
    before = default.solver.get_water_depth().copy()
    with pytest.raises(ValueError, match='provenance'):
        load_checkpoint(model=default, path=tmp_path/'li.npz')
    np.testing.assert_array_equal(actual=default.solver.get_water_depth(), desired=before)
    advance(model=default, end_time=.1)
    save_checkpoint(model=default, path=tmp_path/'fv1.npz')
    with pytest.raises(ValueError, match='provenance'):
        load_checkpoint(model=li, path=tmp_path/'fv1.npz')
