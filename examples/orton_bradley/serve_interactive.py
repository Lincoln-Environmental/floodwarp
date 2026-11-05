"""Serve the prepared, static viewer locally; no hydraulic execution endpoints."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def make_handler(*, directory):
    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def log_message(self, format, *args):
            if str(args[1]) not in ('200', '304'):
                super().log_message(format, *args)
    return ViewerHandler


def serve(*, directory, host, port):
    if not (directory / 'data/manifest.json').is_file():
        raise FileNotFoundError('Build the interactive dataset first: python -m examples.orton_bradley.build_interactive')
    handler = make_handler(directory=directory)
    with ThreadingHTTPServer(server_address=(host, port), RequestHandlerClass=handler) as server:
        print(f'FloodWarp 3D viewer: http://{host}:{server.server_port}', flush=True)
        server.serve_forever()


if __name__ == "__main__":
    directory = Path('outputs/orton_bradley_interactive').resolve()
    host = '127.0.0.1'
    port = 8766
    serve(directory=directory, host=host, port=port)
