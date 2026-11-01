"""Local HTTP controls; physical browser execution is recorded separately."""
from dataclasses import asdict
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

pytest.importorskip('pyogrio')
pytest.importorskip('matplotlib')
from floodwarp.ui import make_server
from floodwarp.ui_runner import RunConfig


def request_json(*, url, data=None, token=None, host=None):
    headers = {}
    if token is not None:
        headers['X-FloodWarp-Token'] = token
    if host is not None:
        headers['Host'] = host
    payload = None if data is None else json.dumps(data).encode()
    request = Request(url=url, data=payload, headers=headers)
    with urlopen(url=request, timeout=30.) as response:
        return json.load(fp=response)


def test_local_requests_and_duplicate_start(tmp_path, monkeypatch):
    import floodwarp.ui as ui
    import floodwarp
    assets = Path(floodwarp.__file__).parent/'ui_assets/example'
    server = make_server(port=0, initial_directory=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'

    class RunningProcess:
        def poll(self):
            return None

    def launch_process(*, args, stdout, stderr):
        return RunningProcess()

    monkeypatch.setattr(target=ui.subprocess, name='Popen', value=launch_process)
    try:
        boot = request_json(url=url+'/bootstrap')
        assert boot['defaults']['manning_n'] == .03
        with pytest.raises(HTTPError):
            request_json(url=url+'/browse', data=dict(path=str(tmp_path)))
        with pytest.raises(HTTPError):
            request_json(url=url+'/bootstrap', host='untrusted.example')
        config = RunConfig(dem=str(assets/'dem.tif'), rainfall_csv=str(assets/'rainfall.csv'),
            output_parent=str(tmp_path), vertical_datum='local', device='cpu')
        request_json(url=url+'/validate', data=asdict(config), token=boot['token'])
        first = request_json(url=url+'/run', data=asdict(config), token=boot['token'])
        assert first['state'] == 'running'
        with pytest.raises(HTTPError):
            request_json(url=url+'/run', data=asdict(config), token=boot['token'])
        assert len(list(tmp_path.glob('floodwarp-*'))) == 1
        assert request_json(url=url+'/status')['output'] == first['output']
        assert request_json(url=url+'/bootstrap')['validated'] is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.)
