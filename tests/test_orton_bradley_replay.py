"""Standalone replay samples the exact end time and never implies auto-resume."""

import json

import numpy as np
import pytest

pytest.importorskip('rasterio')
pytest.importorskip('matplotlib')
from rasterio.transform import Affine

from examples.orton_bradley.reproduce import write_raster
from examples.orton_bradley.run_model import run, sample_targets
from floodwarp.model import model_time


def write_replay_inputs(*, directory, shape, rain):
    inputs = directory / 'inputs'
    inputs.mkdir(parents=True)
    active = np.ones(shape=shape, dtype=bool)
    write_raster(path=directory / 'outputs/orton_bradley_dem_5m.tif',
        values=np.zeros(shape=shape), active=active,
        transform=Affine(5., 0., 1574940., 0., -5., 5167275.), dtype='float64',
        nodata=-9999., description='Replay timing control',
        provenance=dict(elevation_attribution='Synthetic test terrain'))
    fields = np.zeros(shape=(6, *shape), dtype='<f8')
    fields[0:2] = 1.
    fields[3] = 1.
    with (inputs / 'inputs_f64.bin').open('wb') as stream:
        np.array([*shape, 2], dtype='<i4').tofile(file=stream)
        np.array([0., 1.], dtype='<f8').tofile(file=stream)
        np.array([rain, rain], dtype='<f8').tofile(file=stream)
        fields.tofile(file=stream)
    np.savetxt(fname=inputs / 'roughness.asc', X=np.full(shape=shape, fill_value=.03),
        header='header\nheader\nheader\nheader\nheader\nheader', comments='')


@pytest.mark.parametrize('duration,interval', ((.03, .06), (.12, .06), (.13, .06), (.15, .1)))
def test_targets_include_final_time_once(duration, interval):
    targets = sample_targets(duration=duration, interval=interval)
    assert targets[-1] == duration
    assert np.count_nonzero(targets == duration) == 1
    assert np.all(targets <= duration)
    assert np.all(np.diff(np.r_[0., targets]) > 0.)


@pytest.mark.parametrize('duration,interval', ((0., .1), (.1, 0.), (np.inf, .1), (.1, np.nan)))
def test_invalid_sampling_controls(duration, interval):
    with pytest.raises(ValueError, match='Positive finite'):
        sample_targets(duration=duration, interval=interval)


@pytest.mark.parametrize('method', ('fv1', 'li'))
@pytest.mark.parametrize('duration', (.03, .13))
def test_replay_reaches_exact_end_and_saves_final_state(method, duration, tmp_path):
    write_replay_inputs(directory=tmp_path, shape=(4, 5), rain=.001)
    output = tmp_path / 'replay'
    options = dict(directory=tmp_path, output=output, method=method, device='cpu',
        duration=duration, sample_interval=.06, maximum_dt=.02)
    model = run(**options)
    assert model_time(model=model) == duration
    assert (output / 'restart.npz').is_file()
    assert len(json.loads((output / 'ledger.json').read_text())) == len(sample_targets(duration=duration, interval=.06))
    with np.load(file=output / 'fields.npz', allow_pickle=False) as fields:
        np.testing.assert_array_equal(actual=fields['depth'], desired=model.solver.get_water_depth())
        np.testing.assert_allclose(actual=fields['depth'], desired=.001 * duration, rtol=1.e-13, atol=1.e-16)
    with pytest.raises(ValueError, match='starts fresh'):
        run(**options)
