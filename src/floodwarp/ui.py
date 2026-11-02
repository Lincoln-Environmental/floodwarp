"""Single-user localhost interface. Launch with python -m floodwarp.ui."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import importlib.util
import json
import mimetypes
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

from floodwarp.ui_runner import (RunConfig, available_devices, create_run_directory,
    prepare_run, terrain_preview, write_json)
from floodwarp.ui_fields import boundary_template
from floodwarp.ui_maps import map_overlay, boundary_features
from floodwarp.ui_presets import find_orton_archive, prepare_orton_bradley, preset_catalog


class LocalApplication:
    def __init__(self, *, initial_directory):
        self.token = secrets.token_urlsafe(32)
        self.initial_directory = initial_directory
        self.default_output_parent = initial_directory/'outputs' if (initial_directory/'outputs').is_dir() else initial_directory
        self.lock = threading.RLock()
        self.process = None
        self.output = None
        self.started = None
        self.validated = None
        self.default_config = None
        self.template = None

    def validate(self, *, config):
        with self.lock:
            return self.validate_locked(config=config)

    def validate_locked(self, *, config):
        prepared = prepare_run(config=config)
        result = dict(summary=prepared.summary,
            terrain_png=base64.b64encode(terrain_preview(prepared=prepared)).decode('ascii'))
        result['terrain_map'] = map_overlay(terrain=prepared.terrain, values=prepared.terrain.elevation, label='Terrain elevation (m)')
        result['boundary_geojson'] = boundary_features(terrain=prepared.terrain, qx=prepared.qx, qy=prepared.qy)
        self.template = boundary_template(terrain=prepared.terrain)
        self.default_config = asdict(config)
        self.validated = dict(config=asdict(config), result=result)
        return result

    def load_orton(self, *, directory, output_parent):
        with self.lock:
            source, parent = Path(directory).expanduser().resolve(), Path(output_parent).expanduser().resolve()
            if not parent.is_dir():
                raise ValueError('Choose an existing output parent directory for the prepared input set')
            # An explicit reload verifies checksums even when the form was
            # previously loaded. Existing prepared fields are not decoded again.
            result = prepare_orton_bradley(directory=source, output_parent=parent)
            self.default_config = result['config']
            return result

    def start(self, *, config):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError('A run is already active; wait for it to finish')
            if self.validated is None or self.validated['config'] != asdict(config):
                self.validate(config=config)
            inputs = self.validated['result']['summary']['inputs']
            self.output = create_run_directory(parent=Path(config.output_parent).expanduser())
            self.started = time.monotonic()
            write_json(path=self.output/'request.json', value=dict(config=asdict(config), inputs=inputs))
            write_json(path=self.output/'status.json', value=dict(state='running', time_s=0.,
                end_time_s=config.duration_s, elapsed_s=0., output=str(self.output), error=None))
            try:
                with (self.output/'run.log').open('wb') as log:
                    self.process = subprocess.Popen(args=[sys.executable, '-m', 'floodwarp.ui_worker',
                        str(self.output/'request.json')], stdout=log, stderr=subprocess.STDOUT)
            except Exception as error:
                write_json(path=self.output/'status.json', value=dict(state='failed', error=str(error),
                    output=str(self.output), time_s=0., end_time_s=config.duration_s, elapsed_s=0.))
                raise
            return self.status()

    def status(self):
        if self.output is None:
            return dict(state='idle')
        status = json.loads((self.output/'status.json').read_text())
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                status['elapsed_s'] = time.monotonic() - self.started
            elif status['state'] == 'running':
                status.update(state='failed', error=f'Worker exited with code {code}; inspect run.log')
                write_json(path=self.output/'status.json', value=status)
        # An interrupted atomic checkpoint write may leave a temporary NPZ.
        # Offer only complete, named products rather than every matching suffix.
        products = ('maximum_depth.tif', 'final_depth.tif', 'water_balance.csv',
            'boundary_flow.csv', 'checkpoint.npz', 'run_config.json',
            'diagnostics.json', 'maximum_depth.png', 'run.log', 'map.json', 'maximum_map.png', 'final_map.png', 'terrain_map.png')
        status['files'] = [name for name in products if (self.output/name).is_file()]
        return status


def directory_listing(*, path):
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        directory = directory.parent
    entries = []
    for child in directory.iterdir():
        if not child.name.startswith('.') and (child.is_dir() or child.suffix.lower() in ('.tif', '.tiff', '.gpkg', '.geojson', '.csv')):
            entries.append(dict(name=child.name, path=str(child), directory=child.is_dir()))
    return dict(path=str(directory), parent=str(directory.parent), entries=sorted(entries, key=entry_key))


def entry_key(entry):
    return (not entry['directory'], entry['name'].lower())


class LocalHandler(BaseHTTPRequestHandler):
    def respond(self, *, status=200, data, content_type='application/json', download=None):
        body = json.dumps(data).encode() if content_type == 'application/json' else data
        self.send_response(code=status)
        self.send_header(keyword='Content-Type', value=content_type)
        self.send_header(keyword='Content-Length', value=str(len(body)))
        self.send_header(keyword='Cache-Control', value='no-store')
        self.send_header(keyword='X-Content-Type-Options', value='nosniff')
        self.send_header(keyword='Content-Security-Policy', value="default-src 'self'; img-src 'self' data: https://tile.openstreetmap.org; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
        if download:
            self.send_header(keyword='Content-Disposition', value=f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, *, path):
        self.send_response(code=200)
        self.send_header(keyword='Content-Type', value=mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header(keyword='Content-Length', value=str(path.stat().st_size))
        self.send_header(keyword='Cache-Control', value='no-store')
        self.send_header(keyword='X-Content-Type-Options', value='nosniff')
        if path.suffix != '.png':
            self.send_header(keyword='Content-Disposition', value=f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open('rb') as stream:
            shutil.copyfileobj(fsrc=stream, fdst=self.wfile)

    def check_local(self):
        expected = f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host') != expected:
            raise ValueError('Use the printed 127.0.0.1 URL')
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + expected:
            raise ValueError('Cross-origin requests are disabled')

    def do_GET(self):
        try:
            self.check_local()
            parsed = urlsplit(self.path)
            app = self.server.app
            assets = Path(__file__).with_name('ui_assets')
            if parsed.path in ('/', '/app.js', '/style.css', '/map.js', '/leaflet.js', '/leaflet.css'):
                name = 'index.html' if parsed.path == '/' else parsed.path[1:]
                self.respond(data=(assets/name).read_bytes(), content_type=mimetypes.guess_type(name)[0])
            elif parsed.path == '/bootstrap':
                defaults = asdict(RunConfig(dem='', rainfall_csv='', output_parent=str(app.default_output_parent), vertical_datum='local'))
                ordinary_defaults = dict(defaults)
                if app.default_config is not None:
                    defaults = app.default_config
                example = assets/'example'
                self.respond(data=dict(token=app.token, directory=str(app.initial_directory),
                    defaults=defaults, ordinary_defaults=ordinary_defaults, devices=available_devices(), has_configuration=app.default_config is not None,
                    orton_directory=find_orton_archive(directory=app.initial_directory), orton_catalog=preset_catalog(),
                    validated=None if app.validated is None else app.validated['result'],
                    example=dict(dem=str(example/'dem.tif'), domain=str(example/'domain.tif'),
                        rainfall_csv=str(example/'rainfall.csv'), multiplier=str(example/'multiplier.tif'))))
            elif parsed.path == '/boundary-template':
                if app.template is None:
                    raise ValueError('Validate the terrain/domain before downloading its face template')
                self.respond(data=app.template.encode(), content_type='text/csv', download='boundary_faces.csv')
            elif parsed.path == '/status':
                self.respond(data=app.status())
            elif parsed.path == '/example.csv':
                self.respond(data=(assets/'rainfall.csv').read_bytes(), content_type='text/csv', download='rainfall.csv')
            elif parsed.path == '/result':
                name = parse_qs(parsed.query).get('name', [''])[0]
                files = app.status().get('files', [])
                if name not in files or Path(name).name != name:
                    raise ValueError('Result unavailable')
                self.send_file(path=app.output/name)
            else:
                self.respond(status=404, data=dict(error='Not found'))
        except (ValueError, OSError) as error:
            self.respond(status=400, data=dict(error=str(error)))

    def do_POST(self):
        try:
            self.check_local()
            if not secrets.compare_digest(self.headers.get('X-FloodWarp-Token', ''), self.server.app.token):
                raise ValueError('Invalid local request token; reload the page')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 65536:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(length))
            if self.path == '/browse':
                result = directory_listing(path=Path(request['path']))
            elif self.path == '/preset/orton':
                result = self.server.app.load_orton(directory=request['directory'], output_parent=request['output_parent'])
            elif self.path == '/validate':
                result = self.server.app.validate(config=RunConfig(**request))
            elif self.path == '/run':
                result = self.server.app.start(config=RunConfig(**request))
            else:
                self.respond(status=404, data=dict(error='Not found'))
                return
            self.respond(data=result)
        except Exception as error:
            self.respond(status=400, data=dict(error=f'{type(error).__name__}: {error}'))


def make_server(*, port, initial_directory):
    for module in ('rasterio', 'matplotlib', 'pyogrio', 'shapely', 'pyproj'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError('Install the optional interface: python -m pip install "floodwarp[ui]"')
    server = ThreadingHTTPServer(server_address=('127.0.0.1', port), RequestHandlerClass=LocalHandler)
    server.app = LocalApplication(initial_directory=initial_directory)
    return server


def serve(*, port, initial_directory):
    server = make_server(port=port, initial_directory=initial_directory)
    print(f'FloodWarp local interface: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    port = 8765
    initial_directory = Path.cwd()
    serve(port=port, initial_directory=initial_directory)
