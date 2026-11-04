"""Output/recovery and geometric controls; no full-domain hydraulic run."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rasterio')
pytest.importorskip('matplotlib')
from examples.orton_bradley import export_visualisation as export
from examples.orton_bradley.render_visualisation import grid_geometry, water_geometry, encode_video


def test_lossless_frame_and_duplicate_guard(tmp_path):
    active = np.array([[True, False], [True, True]])
    depth = np.array([[.123456789012345, 0.], [0., 3.25]])
    export.write_frame(output=tmp_path, seconds=300., depth=depth, active=active)
    path = export.frame_path(output=tmp_path, seconds=300.)
    timestamp = path.stat().st_mtime_ns
    export.write_frame(output=tmp_path, seconds=300., depth=depth.copy(), active=active)
    assert path.stat().st_mtime_ns == timestamp
    depth[0, 0] += .01
    with pytest.raises(ValueError, match='differs on replay'):
        export.write_frame(output=tmp_path, seconds=300., depth=depth, active=active)
    with np.load(file=path, allow_pickle=False) as saved:
        assert saved['depth'][0, 0] == .123456789012345
        assert saved['time_s'] == 300.


def test_native_orientation_and_no_water_over_dry_vertices():
    dem = np.array([[10., 12., 14.], [20., 22., 24.]])
    active = np.ones(shape=dem.shape, dtype=bool)
    points, triangles = grid_geometry(dem=dem, valid=active, grid=dict(dx=5., dy=5.))
    np.testing.assert_array_equal(actual=points[0], desired=[2.5, 2.5, 10.])
    np.testing.assert_array_equal(actual=points[3], desired=[2.5, 7.5, 20.])
    depth = np.array([[.2, .3, 0.], [.4, .5, 0.]])
    elevated, faces = water_geometry(points=points, triangles=triangles, depth=depth,
        active=active, wet_threshold=.01)
    assert len(faces) == 2
    assert np.all(depth.ravel()[faces] >= .01)
    np.testing.assert_array_equal(actual=elevated[:, 2], desired=dem.ravel() + depth.ravel())
    active[0, 0] = False
    _, faces = water_geometry(points=points, triangles=triangles, depth=depth,
        active=active, wet_threshold=.01)
    assert not np.any(faces == 0)


def test_interval_diagnostics_include_boundary_inflow():
    previous = dict(time_s=60., boundary_outflow=100., boundary_inflow=20., rain=80.)
    current = dict(time_s=120., boundary_outflow=250., boundary_inflow=50., rain=200.,
                   infiltration=100., storage_m3=30., residual_m3=1.e-10)
    row = export.diagnostic_row(current=current, previous=previous)
    assert row['discharge_m3_s'] == 2.
    assert row['rainfall_m3_s'] == 2.


def fake_model(*, source, device, maximum_dt):
    model = SimpleNamespace(time=0., terrain=SimpleNamespace(active=np.ones((2, 3), dtype=bool)))
    model.solver = SimpleNamespace(get_water_depth=ModelDepth(model=model))
    return model


class ModelDepth:
    def __init__(self, *, model):
        self.model = model

    def __call__(self):
        return np.full(shape=(2, 3), fill_value=self.model.time * 1.e-5)


def fake_time(*, model):
    return model.time


def fake_diagnostics(*, model):
    return dict(time_s=model.time, storage_m3=model.time, residual_m3=0.,
                rain=2. * model.time, infiltration=model.time, boundary_outflow=0., boundary_inflow=0.)


def fake_save(*, model, path):
    export.atomic_arrays(path=path, arrays=dict(time_s=model.time))


def fake_load(*, model, path):
    with np.load(file=path, allow_pickle=False) as saved:
        model.time = float(saved['time_s'])


def fake_advance(*, model, end_time):
    model.time = end_time


def test_resume_reconciles_orphan_frames_without_duplicate_diagnostics(tmp_path, monkeypatch):
    for name, function in (('build_model', fake_model), ('model_time', fake_time),
        ('diagnostics', fake_diagnostics), ('save_checkpoint', fake_save),
        ('load_checkpoint', fake_load), ('advance', fake_advance)):
        monkeypatch.setattr(export, name, function)
    identity = dict(duration_s=3600., frame_interval_s=300., diagnostic_interval_s=60.,
                    checkpoint_interval_s=3600., maximum_dt_s=1.)
    options = dict(source=tmp_path, output=tmp_path, identity=identity, device='cpu',
                   maximum_idle_utilisation=10)
    export.run_export(**options, stop_time=600.)
    # Model was killed after a frame write, before its next committed restart.
    export.write_frame(output=tmp_path, seconds=900.,
        depth=np.full(shape=(2, 3), fill_value=900. * 1.e-5), active=np.ones((2, 3), dtype=bool))
    orphan = export.frame_path(output=tmp_path, seconds=900.)
    stamp = orphan.stat().st_mtime_ns
    result = export.run_export(**options, stop_time=1200.)
    assert result['time_s'] == 1200.
    assert orphan.stat().st_mtime_ns == stamp
    rows = json.loads((tmp_path / 'diagnostics/ledger.json').read_text())
    assert [row['time_s'] for row in rows] == list(np.arange(60., 1201., 60.))
    assert len(list((tmp_path / 'frames').glob('*.npz'))) == 5
    # A damaged latest checkpoint falls back to the preceding intact bundle.
    (tmp_path / 'checkpoints/restart-001200.npz').write_bytes(b'interrupted checkpoint')
    (tmp_path / 'checkpoints/restart-001800.json').write_text('{incomplete marker')
    export.run_export(**options, stop_time=1200.)
    assert len(json.loads((tmp_path / 'diagnostics/ledger.json').read_text())) == 20


def test_busy_device_refuses_before_model_construction(tmp_path, monkeypatch):
    def busy_status(*, device, maximum_idle_utilisation):
        return dict(free=False, compute_processes='other-job')
    monkeypatch.setattr(export, 'device_status', busy_status)
    with pytest.raises(RuntimeError, match='Device busy; no model started'):
        export.run_export(source=tmp_path, output=tmp_path, identity={}, device='cuda:0',
                          maximum_idle_utilisation=10, stop_time=None)
    assert not list(tmp_path.iterdir())


def test_video_encoding_and_complete_decode(tmp_path):
    pytest.importorskip('imageio_ffmpeg')
    from PIL import Image
    images = tmp_path / 'images'
    images.mkdir()
    for index in range(4):
        Image.new(mode='RGB', size=(96, 64), color=(20 * index, 80, 150)).save(fp=images / f'{index:06d}.png')
    result = encode_video(image_directory=images, path=tmp_path / 'smoke.mp4', width=96,
        height=64, fps=2, frame_count=4, crf=20)
    assert result['duration_s'] == 2.
    assert result['frames'] == 4
    assert result['bytes'] > 0
