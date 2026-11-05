"""Serve the prepared, static viewer locally; no hydraulic execution endpoints."""
from pathlib import Path
from concurrent.futures import CancelledError
import json
from urllib.parse import parse_qs, urlsplit
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def make_handler(*, directory, remote_renderer=None):
    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            parsed = urlsplit(url=self.path)
            if parsed.path != '/api/scene':
                return super().do_GET()
            try:
                if remote_renderer is None:
                    raise RuntimeError('Server rendering is not enabled; restart serve_interactive.py')
                result = remote_renderer.request(query=parse_qs(qs=parsed.query, keep_blank_values=True))
                self.send_response(code=200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(result['image'])))
                self.send_header('X-FloodWarp-Time', str(result['time_s']))
                self.send_header('X-FloodWarp-Frame', str(result['frame']))
                self.send_header('X-FloodWarp-Wet-Triangles', str(result['wet_triangles']))
                self.end_headers()
                self.wfile.write(result['image'])
            except (BrokenPipeError, ConnectionResetError):
                pass  # A newer camera/time request superseded this browser request.
            except Exception as error:
                status = 400 if isinstance(error, (ValueError, TypeError)) else 409 if isinstance(error, CancelledError) else 503
                payload = json.dumps(dict(error=str(error) or 'View superseded; request the latest view')).encode()
                try:
                    self.send_response(code=status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def log_message(self, format, *args):
            if str(args[1]) not in ('200', '304'):
                super().log_message(format, *args)
    return ViewerHandler


def serve(*, directory, host, port, native_dataset=None, maximum_width=1600, maximum_height=1200):
    if not (directory / 'data/manifest.json').is_file():
        raise FileNotFoundError(
            f'Interactive dataset not found at {directory / "data/manifest.json"}. '
            'Build it from the project root with: python -m examples.orton_bradley.build_interactive')
    remote_renderer = None
    if native_dataset is not None:
        if __package__:
            from .remote_rendering import RemoteRenderer
        else:
            from remote_rendering import RemoteRenderer
        remote_renderer = RemoteRenderer(dataset=native_dataset, directory=directory,
            maximum_width=maximum_width, maximum_height=maximum_height)
    handler = make_handler(directory=directory, remote_renderer=remote_renderer)
    with ThreadingHTTPServer(server_address=(host, port), RequestHandlerClass=handler) as server:
        print(f'FloodWarp 3D viewer: http://{host}:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        finally:
            if remote_renderer is not None:
                remote_renderer.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    directory = project_root / 'outputs/orton_bradley_interactive'
    native_dataset = project_root / 'outputs/orton_bradley_visualisation'
    maximum_width = 1600
    maximum_height = 1200
    host = '127.0.0.1'
    port = 8766
    serve(directory=directory, host=host, port=port, native_dataset=native_dataset,
          maximum_width=maximum_width, maximum_height=maximum_height)
