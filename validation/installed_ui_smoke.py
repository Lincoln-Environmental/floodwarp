"""Installed-wheel UI launch and real worker run, outside the source checkout."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

import floodwarp
from floodwarp.ui_runner import RunConfig


def get_json(*, url):
    with urlopen(url=url, timeout=5.) as response:
        return json.load(fp=response)


def post_json(*, url, value, token):
    request = Request(url=url, data=json.dumps(value).encode(),
        headers={'Content-Type': 'application/json', 'X-FloodWarp-Token': token})
    return get_json(url=request)


def verify_interface(*, url, duration, interval, timeout_s):
    assets = Path(floodwarp.__file__).parent/'ui_assets/example'
    with tempfile.TemporaryDirectory(prefix='floodwarp-installed-ui-') as directory:
        output = Path(directory)
        with (output/'server.log').open('w') as log:
            process = subprocess.Popen(args=[sys.executable, '-m', 'floodwarp.ui'], cwd=output,
                stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+timeout_s
            while True:
                if process.poll() is not None:
                    raise RuntimeError((output/'server.log').read_text())
                try:
                    bootstrap = get_json(url=url+'/bootstrap')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('UI did not launch')
                    time.sleep(.1)
            config = RunConfig(dem=str(assets/'dem.tif'), rainfall_csv=str(assets/'rainfall.csv'),
                output_parent=str(output), vertical_datum='local', device='cpu', duration_s=duration,
                interval_s=interval, domain_mode='raster', domain=str(assets/'domain.tif'))
            post_json(url=url+'/validate', value=asdict(config), token=bootstrap['token'])
            post_json(url=url+'/run', value=asdict(config), token=bootstrap['token'])
            while True:
                status = get_json(url=url+'/status')
                if status['state'] in ('completed', 'failed'):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('UI worker did not finish')
                time.sleep(.1)
            assert status['state'] == 'completed', status
            assert status['time_s'] == duration
            assert (Path(status['output'])/'maximum_depth.tif').is_file()
            print(json.dumps(dict(launch='python -m floodwarp.ui', module=floodwarp.__file__, status=status)))
        finally:
            process.terminate()
            process.wait(timeout=10.)


if __name__ == "__main__":
    url = 'http://127.0.0.1:8765'
    duration = 73.
    interval = 30.
    timeout_s = 60.
    verify_interface(url=url, duration=duration, interval=interval, timeout_s=timeout_s)
