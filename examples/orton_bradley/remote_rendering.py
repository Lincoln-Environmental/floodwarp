"""Read-only CPU rendering of saved results for browsers without WebGL."""
from concurrent.futures import CancelledError, Future
from io import BytesIO
import json
import hashlib
import math
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread


def file_hash(*, path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_mesh(*, points, triangles):
    import numpy as np
    import pyvista as pv
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(points, faces=faces)


def parse_view(*, query, frame_count, maximum_width, maximum_height):
    defaults = dict(frame=0, width=1200, height=800, azimuth=120., elevation=35.,
                    distance=10500., east=0., north=0., opacity=.9, aerial=1)
    if set(query) - set(defaults):
        raise ValueError('Unknown render parameter')
    values = {}
    for name, default in defaults.items():
        supplied = query.get(name, [str(default)])
        if len(supplied) != 1:
            raise ValueError('Render parameters must occur once')
        value = float(supplied[0])
        if not math.isfinite(value):
            raise ValueError('Render parameters must be finite')
        if name in ('frame', 'width', 'height', 'aerial'):
            if value != int(value):
                raise ValueError('Frame and image dimensions must be integers')
            value = int(value)
        values[name] = value
    bounds = dict(frame=(0, frame_count - 1), width=(160, maximum_width),
        height=(160, maximum_height), elevation=(5., 89.5), distance=(30., 25000.),
        east=(-20000., 20000.), north=(-20000., 20000.), opacity=(.15, 1.), aerial=(0,1))
    for name, (minimum, maximum) in bounds.items():
        if not minimum <= values[name] <= maximum:
            raise ValueError('Render parameter out of range: ' + name)
    values['azimuth'] %= 360.
    return values


def create_render_state(*, dataset, directory, width, height):
    # Set before importing VTK. The renderer never uses the remote X display.
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkEGLRenderWindow'
    os.environ['ALLOW_PLOTTING'] = 'true'
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    import pyvista as pv
    from matplotlib import use
    use(backend='Agg')
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    manifest = json.loads((directory / 'data/manifest.json').read_text())
    validation = json.loads((dataset / 'diagnostics/validation.json').read_text())
    if validation['status'] != 'passed' or validation['final_time_s'] != 259200.:
        raise ValueError('A completed validated event is required')
    configuration = manifest['configuration']
    if file_hash(path=dataset / 'metadata/terrain.npz') != configuration['source_terrain_sha256']:
        raise ValueError('Native terrain differs from the display dataset')
    with np.load(file=dataset / 'metadata/terrain.npz', allow_pickle=False) as saved:
        dem, valid, active = saved['dem'], saved['valid_dem'], saved['active']
    ny, nx = dem.shape
    x, y = np.meshgrid((np.arange(nx) + .5) * manifest['grid']['dx'],
                       (np.arange(ny) + .5) * manifest['grid']['dy'])
    points = np.column_stack((x.ravel(), y.ravel(), dem.ravel()))
    cells = np.arange(ny * nx).reshape(ny, nx)
    a, b = cells[:-1, :-1].ravel(), cells[:-1, 1:].ravel()
    c, d = cells[1:, :-1].ravel(), cells[1:, 1:].ravel()
    triangles = np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    triangles = triangles[np.all(valid.ravel()[triangles], axis=1)]
    terrain = make_mesh(points=points, triangles=triangles)
    palette = LinearSegmentedColormap.from_list(name='terrain',
        colors=['#354d3b', '#71805b', '#a5a179', '#b5aa89', '#cec7b2'])
    colours = LightSource(azdeg=315., altdeg=45.).shade(data=np.where(valid, dem, np.median(dem[valid])),
        cmap=palette, vert_exag=1., dx=manifest['grid']['dx'], dy=manifest['grid']['dy'], blend_mode='soft')
    terrain['rgb'] = (colours[..., :3].reshape(-1, 3) * 255).astype(np.uint8)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(color='#0b1720')
    plotter.add_mesh(mesh=terrain, scalars='rgb', rgb=True, lighting=False, show_scalar_bar=False)
    # Sorting transparency against terrain is important for shallow water.
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.)
    selected = points[active.ravel()]
    lower, upper = selected.min(axis=0), selected.max(axis=0)
    centre = (lower + upper) / 2.
    centre[2] = lower[2] + .2 * (upper[2] - lower[2])
    return dict(plotter=plotter, points=points, triangles=triangles, active=active,
        manifest=manifest, centre=centre, frame=None, opacity=None, triangle_count=0,
        terrain=terrain, base_colours=terrain['rgb'].copy(), aerial_colours=None, aerial_mode=None,
        aerial_path=directory / 'data/aerial.npz')


def render_image(*, state, dataset, view):
    import numpy as np
    from PIL import Image
    from matplotlib.colors import LinearSegmentedColormap

    plotter = state['plotter']
    record = state['manifest']['frames'][view['frame']]
    if view['aerial'] and state['aerial_colours'] is None and state['aerial_path'].is_file():
        with np.load(file=state['aerial_path'],allow_pickle=False) as imagery:
            colours=state['base_colours'].copy()
            covered=imagery['covered'].ravel()
            colours[covered]=imagery['rgb'].reshape(-1,3)[covered]
            state['aerial_colours']=colours
    if state['aerial_mode'] != view['aerial']:
        selected=state['aerial_colours'] if view['aerial'] and state['aerial_colours'] is not None else state['base_colours']
        state['terrain']['rgb']=selected
        state['aerial_mode']=view['aerial']
    if state['frame'] != view['frame']:
        source = dataset / 'frames' / f'depth-{int(record["time_s"]):06d}.npz'
        if file_hash(path=source) != record['source_sha256']:
            raise ValueError('Native result changed since validation')
        with np.load(file=source, allow_pickle=False) as saved:
            state['depth'] = saved['depth']
    if state['frame'] != view['frame'] or state['opacity'] != view['opacity']:
        configuration = state['manifest']['configuration']
        wet = state['active'].ravel() & (state['depth'].ravel() >= configuration['wet_threshold_m'])
        faces = state['triangles'][np.all(wet[state['triangles']], axis=1)]
        elevated = state['points'].copy()
        elevated[:, 2] += state['depth'].ravel()
        plotter.remove_actor(actor='water', render=False)
        if len(faces):
            water = make_mesh(points=elevated, triangles=faces)
            water['depth_m'] = state['depth'].ravel()
            plotter.add_mesh(mesh=water, scalars='depth_m', cmap=LinearSegmentedColormap.from_list(name='water',
                    colors=['#58dfeb', '#2bade0', '#1677c6', '#093b8c']),
                clim=(configuration['wet_threshold_m'], configuration['depth_maximum_m']),
                log_scale=True, opacity=view['opacity'], lighting=False,
                show_scalar_bar=False, name='water', render=False)
        state.update(frame=view['frame'], opacity=view['opacity'], triangle_count=len(faces))
    centre = state['centre'] + np.array([view['east'], view['north'], 0.])
    azimuth, elevation = np.deg2rad(view['azimuth']), np.deg2rad(view['elevation'])
    offset = view['distance'] * np.array([np.cos(azimuth) * np.cos(elevation),
        np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
    plotter.window_size = (view['width'], view['height'])
    plotter.camera_position = [(centre + offset).tolist(), centre.tolist(), [0., 0., 1.]]
    plotter.camera.view_angle = 38.
    plotter.reset_camera_clipping_range()
    plotter.render()
    pixels = plotter.screenshot(return_img=True)
    buffer = BytesIO()
    Image.fromarray(obj=pixels).save(fp=buffer, format='PNG', compress_level=3)
    return dict(image=buffer.getvalue(), time_s=record['time_s'],
                frame=view['frame'], wet_triangles=state['triangle_count'])


class RemoteRenderer:
    """Keep the EGL context on one owner thread; queue only the newest view."""
    def __init__(self, *, dataset: Path, directory: Path, maximum_width, maximum_height):
        self.dataset = dataset
        self.directory = directory
        self.maximum_width = maximum_width
        self.maximum_height = maximum_height
        self.jobs = Queue()
        self.lock = Lock()
        self.thread = None

    def request(self, *, query):
        manifest = json.loads((self.directory / 'data/manifest.json').read_text())
        view = parse_view(query=query, frame_count=len(manifest['frames']),
            maximum_width=self.maximum_width, maximum_height=self.maximum_height)
        future = Future()
        with self.lock:
            while True:
                try:
                    pending = self.jobs.get_nowait()
                    if pending is not None:
                        pending[1].cancel()
                except Empty:
                    break
            self.jobs.put(item=(view, future))
            if self.thread is None:
                self.thread = Thread(target=self._run, name='floodwarp-renderer', daemon=True)
                self.thread.start()
        return future.result(timeout=90.)

    def _run(self):
        state = None
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                view, future = job
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if state is None:
                        state = create_render_state(dataset=self.dataset, directory=self.directory,
                            width=view['width'], height=view['height'])
                    future.set_result(render_image(state=state, dataset=self.dataset, view=view))
                except Exception as error:
                    future.set_exception(exception=error)
        finally:
            if state is not None:
                state['plotter'].close()

    def close(self):
        with self.lock:
            while True:
                try:
                    pending = self.jobs.get_nowait()
                    if pending is not None:
                        pending[1].cancel()
                except Empty:
                    break
            self.jobs.put(item=None)
        if self.thread is not None:
            self.thread.join(timeout=10.)
