"""Display conversion controls: no hydraulic stepping or reference mutation."""
import gzip
import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

import numpy as np
import pytest

pytest.importorskip('matplotlib')
pytest.importorskip('rasterio')
from examples.orton_bradley.build_interactive import depth_payload, package_frame, terrain_payload, write_binary
from examples.orton_bradley.reproduce import file_hash
from examples.orton_bradley.serve_interactive import make_handler


def test_wet_mask_uses_native_values_before_display_rounding():
    threshold = .01
    depth = np.array([[threshold, np.nextafter(threshold, 0.), 6.331889819999019]], dtype=np.float64)
    active = np.ones(shape=depth.shape, dtype=bool)
    payload, error, wet_cells = depth_payload(depth=depth, active=active, wet_threshold=threshold)
    display = np.frombuffer(buffer=payload, dtype='<f4', count=depth.size)
    wet = np.frombuffer(buffer=payload, dtype=np.uint8, offset=depth.size * 4)
    assert display[0] == display[1]
    np.testing.assert_array_equal(actual=wet, desired=[1, 0, 1])
    assert wet_cells == 2
    assert error == np.max(np.abs(display.astype(np.float64) - depth.ravel()))
    assert error < 1.e-6


def test_browser_geometry_retains_south_first_north_and_height():
    dem = np.array([[10., 12., 14.], [20., 22., 24.]])
    valid = np.ones(shape=dem.shape, dtype=bool)
    active = valid.copy()
    active[0, 0] = False
    payload, error = terrain_payload(dem=dem, valid=valid, active=active, grid=dict(dx=5., dy=5.))
    xyz = np.frombuffer(buffer=payload, dtype='<f4', count=dem.size * 3).reshape(-1, 3)
    np.testing.assert_array_equal(actual=xyz[0], desired=[-5., 10., 2.5])
    np.testing.assert_array_equal(actual=xyz[3], desired=[-5., 20., -2.5])
    np.testing.assert_array_equal(actual=xyz[:, 1], desired=dem.ravel())
    np.testing.assert_array_equal(actual=np.frombuffer(buffer=payload, dtype=np.uint8,
        offset=dem.size * 16), desired=active.ravel().astype(np.uint8))
    assert error == 0.


def test_compressed_payload_is_deterministic_and_verified(tmp_path):
    payload = np.arange(100, dtype='<f4').tobytes()
    first = write_binary(path=tmp_path / 'first.bin.gz', payload=payload, compression_level=4)
    second = write_binary(path=tmp_path / 'second.bin.gz', payload=payload, compression_level=4)
    assert first['sha256'] == second['sha256']
    assert gzip.decompress(data=(tmp_path / 'first.bin.gz').read_bytes()) == payload
    assert first['raw_bytes'] == len(payload)


def test_resume_preserves_complete_browser_frames_and_rejects_corruption(tmp_path):
    source = tmp_path / 'depth-000300.npz'
    np.savez_compressed(file=source, time_s=300., depth=np.full(shape=(2, 3), fill_value=.025))
    output = tmp_path / 'browser'
    options = dict(source_path=source, expected=dict(time_s=300., sha256=file_hash(path=source)),
        active=np.ones(shape=(2, 3), dtype=bool), output=output, wet_threshold=.01, compression_level=4)
    first = package_frame(**options)
    binary = output / first['path']
    stamp = binary.stat().st_mtime_ns
    second = package_frame(**options)
    assert first == second
    assert binary.stat().st_mtime_ns == stamp
    binary.write_bytes(b'damaged browser display')
    with pytest.raises(ValueError, match='Existing browser frame differs'):
        package_frame(**options)


def test_static_server_delivers_data_without_a_solver_endpoint(tmp_path):
    (tmp_path / 'index.html').write_text('<!doctype html><title>Viewer</title>')
    (tmp_path / 'viewer.js').write_text('export const ready=true;')
    payload = gzip.compress(data=b'actual output bytes', mtime=0)
    (tmp_path / 'depth.bin.gz').write_bytes(payload)
    handler = make_handler(directory=tmp_path)
    server = ThreadingHTTPServer(server_address=('127.0.0.1', 0), RequestHandlerClass=handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        with urlopen(url=base + '/viewer.js', timeout=5) as response:
            assert 'javascript' in response.headers['Content-Type']
            assert response.headers['X-Content-Type-Options'] == 'nosniff'
        with urlopen(url=base + '/depth.bin.gz', timeout=5) as response:
            assert response.read() == payload
            assert response.headers['Content-Encoding'] is None
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_pycharm_launcher_finds_dataset_from_another_working_directory(tmp_path):
    """Exercise direct file execution and the real HTTP response, as in an IDE."""
    import selectors
    import subprocess
    import sys
    from pathlib import Path
    from examples.orton_bradley import serve_interactive

    checkout = tmp_path / 'checkout'
    script = checkout / 'examples/orton_bradley/serve_interactive.py'
    script.parent.mkdir(parents=True)
    # Change only the editable port to request a private ephemeral test port.
    source = Path(serve_interactive.__file__).read_text()
    script.write_text(source.replace("    port = 8766", "    port = 0"))
    import shutil
    shutil.copyfile(src=Path(serve_interactive.__file__).with_name('remote_rendering.py'),
                    dst=script.with_name('remote_rendering.py'))
    data = checkout / 'outputs/orton_bradley_interactive/data'
    data.mkdir(parents=True)
    (data / 'manifest.json').write_text(json.dumps(dict(site='Orton Bradley path regression')))
    with subprocess.Popen(args=[sys.executable, str(script)], cwd=script.parent,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(fileobj=process.stdout, events=selectors.EVENT_READ)
                assert selector.select(timeout=10), 'Launcher did not start'
            line = process.stdout.readline()
            assert line.startswith('FloodWarp 3D viewer:'), process.stderr.read()
            base = line.rsplit(' ', 1)[-1].strip()
            with urlopen(url=base + '/data/manifest.json', timeout=5) as response:
                assert json.load(fp=response) == dict(site='Orton Bradley path regression')
        finally:
            process.terminate()
            process.communicate(timeout=5)


@pytest.mark.parametrize('query', [dict(frame=['865']), dict(frame=['.5']), dict(width=['99999']),
    dict(distance=['nan']), dict(elevation=['0']), dict(frame=['1','2']), dict(path=['/tmp'])])
def test_remote_view_rejects_invalid_or_unbounded_requests(query):
    from examples.orton_bradley.remote_rendering import parse_view
    with pytest.raises(ValueError):
        parse_view(query=query, frame_count=865, maximum_width=1600, maximum_height=1200)


def test_remote_view_preserves_requested_native_frame_and_camera():
    from examples.orton_bradley.remote_rendering import parse_view
    view=parse_view(query=dict(frame=['294'], azimuth=['480'], opacity=['.9']),
        frame_count=865, maximum_width=1600, maximum_height=1200)
    assert view['frame']==294 and view['azimuth']==120. and view['opacity']==.9


def test_aerial_crop_preserves_model_grid_and_south_first_orientation(tmp_path):
    from rasterio.enums import ColorInterp, Resampling
    from rasterio.transform import Affine
    import rasterio
    from examples.orton_bradley.prepare_aerial_imagery import prepare

    dataset=tmp_path/'dataset'
    (dataset/'metadata').mkdir(parents=True)
    transform=Affine(5.,0.,1574940.,0.,-5.,5167275.)
    grid=dict(transform=list(transform),epsg=2193)
    (dataset/'metadata/dataset.json').write_text(json.dumps(dict(grid=grid)))
    np.savez_compressed(file=dataset/'metadata/terrain.npz',active=np.ones((2,3),dtype=bool))
    rgb=np.array([[[10,20,30],[40,50,60],[70,80,90]],[[100,110,120],[130,140,150],[160,170,180]]],dtype=np.uint8)
    alpha=np.full(shape=(2,3),fill_value=255,dtype=np.uint8)
    source=tmp_path/'tile.tiff'
    with rasterio.open(source,'w',driver='GTiff',width=3,height=2,count=4,dtype='uint8',crs='EPSG:2193',transform=transform) as target:
        target.write(np.moveaxis(rgb,source=-1,destination=0),indexes=[1,2,3])
        target.write(alpha,indexes=4)
        target.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
    collection=tmp_path/'collection.json'
    collection.write_text(json.dumps(dict(title='Synthetic imagery control',license='CC-BY-4.0',providers=[])))
    (tmp_path/'tile.json').write_text(json.dumps(dict(assets=dict(visual=dict(href='./tile.tiff')),properties={})))
    output=tmp_path/'browser'
    report=prepare(dataset=dataset,output=output,collection_url=collection.as_uri(),tiles=['tile'],resampling=Resampling.average)
    with np.load(file=output/'data/aerial.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(actual=saved['rgb'],desired=rgb[::-1])
        assert saved['covered'].all()
    assert report['active_domain_coverage_fraction']==1.
    assert report['grid']==grid
