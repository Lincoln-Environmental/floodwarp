"""True 3D DEM/water meshes with synchronised scientific annotations.

VTK/PyVista renders geometry; Matplotlib supplies the conference layout.
There is no temporal interpolation of hydraulic fields or vertical exaggeration.
"""
from pathlib import Path
import json
import os
import subprocess
import time

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LightSource, LogNorm, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from PIL import Image

from examples.orton_bradley.export_visualisation import atomic_json, frame_path
from examples.orton_bradley.reproduce import file_hash


def water_colormap():
    """Keep shallow water distinguishable from hillshade without altering depth."""
    return LinearSegmentedColormap.from_list(name='water_depth',
        colors=['#58dfeb', '#2bade0', '#1677c6', '#093b8c'])


def grid_geometry(*, dem, valid, grid):
    """Native cell-centre triangulation, south first, in local metre coordinates."""
    ny, nx = dem.shape
    x, y = np.meshgrid((np.arange(nx) + .5) * grid['dx'],
                       (np.arange(ny) + .5) * grid['dy'])
    points = np.column_stack((x.ravel(), y.ravel(), dem.ravel()))
    corners = np.arange(ny * nx).reshape(ny, nx)
    a, b = corners[:-1, :-1].ravel(), corners[:-1, 1:].ravel()
    c, d = corners[1:, :-1].ravel(), corners[1:, 1:].ravel()
    triangles = np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    triangles = triangles[np.all(valid.ravel()[triangles], axis=1)]
    return points, triangles


def water_geometry(*, points, triangles, depth, active, wet_threshold):
    """Only fully wet triangles: never span a dry cell or the inactive mask."""
    wet = active.ravel() & (depth.ravel() >= wet_threshold)
    faces = triangles[np.all(wet[triangles], axis=1)]
    elevated = points.copy()
    elevated[:, 2] += depth.ravel()
    return elevated, faces


def polydata(*, points, triangles):
    import pyvista as pv
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(points, faces=faces)


def create_scene(*, dem, valid, grid, width, height, background):
    from matplotlib import use
    use(backend='Agg')
    import pyvista as pv
    points, triangles = grid_geometry(dem=dem, valid=valid, grid=grid)
    terrain = polydata(points=points, triangles=triangles)
    shaded_dem = np.where(valid, dem, np.nanmedian(dem[valid]))
    terrain_colours = LinearSegmentedColormap.from_list(name='terrain',
        colors=['#354d3b', '#71805b', '#a5a179', '#b5aa89', '#cec7b2'])
    colours = LightSource(azdeg=315., altdeg=45.).shade(data=shaded_dem,
        cmap=terrain_colours, vert_exag=1., dx=grid['dx'], dy=grid['dy'], blend_mode='soft')
    terrain['terrain_rgb'] = (colours[..., :3].reshape(-1, 3) * 255).astype(np.uint8)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(color=background)
    plotter.add_mesh(mesh=terrain, scalars='terrain_rgb', rgb=True, lighting=False,
                     show_scalar_bar=False, name='terrain')
    # Cardinal reference is tied to local x=east/y=north/z=up, not screen rotation.
    plotter.add_axes(xlabel='E', ylabel='N', zlabel='Up', color='#b9c8d2')
    return plotter, points, triangles


def update_scene(*, plotter, points, triangles, depth, active, wet_threshold,
                 depth_maximum, fraction, camera_start, camera_end, opacity):
    elevated, faces = water_geometry(points=points, triangles=triangles, depth=depth,
        active=active, wet_threshold=wet_threshold)
    plotter.remove_actor(actor='water', render=False)
    if len(faces):
        water = polydata(points=elevated, triangles=faces)
        water['depth_m'] = depth.ravel()
        plotter.add_mesh(mesh=water, scalars='depth_m', cmap=water_colormap(),
            clim=(wet_threshold, depth_maximum), log_scale=True,
            opacity=opacity, lighting=False, show_scalar_bar=False,
            name='water', render=False)
    valid_points = points[active.ravel()]
    low, high = np.min(valid_points, axis=0), np.max(valid_points, axis=0)
    centre = (low + high) / 2.
    centre[2] = low[2] + .2 * (high[2] - low[2])
    distance = float(np.linalg.norm(high[:2] - low[:2])) * 1.12
    angle = np.deg2rad(camera_start + (camera_end - camera_start) * fraction)
    elevation = np.deg2rad(43. - 5. * fraction)
    position = centre + distance * np.array([np.cos(angle) * np.cos(elevation),
        np.sin(angle) * np.cos(elevation), np.sin(elevation)])
    plotter.camera_position = [position.tolist(), centre.tolist(), [0., 0., 1.]]
    plotter.camera.view_angle = 38.
    plotter.reset_camera_clipping_range()
    plotter.render()
    return plotter.screenshot(return_img=True)


def create_layout(*, width, height, rows, wet_threshold, depth_maximum, title_note):
    figure = Figure(figsize=(19.2, 19.2 * height / width), dpi=width / 19.2, facecolor='#0c1720')
    canvas = FigureCanvasAgg(figure)
    figure.text(.045, .95, 'ORTON BRADLEY', fontsize=25, color='white', weight='bold')
    figure.text(.045, .915, 'Full-domain 5 m FV1  |  Uncalibrated 100-year design rainfall demonstration',
                fontsize=12, color='#a6c8d8')
    figure.text(.045, .885, title_note, fontsize=10, color='#a6c8d8')
    image_axis = figure.add_axes(rect=(.015, .16, .73, .70))
    image_axis.set_axis_off()
    image_artist = image_axis.imshow(np.zeros(shape=(height, width, 3), dtype=np.uint8))
    hours = np.array([row['time_s'] for row in rows]) / 3600.
    rain = np.array([row['rainfall_mm_h'] for row in rows])
    q = np.array([row['discharge_m3_s'] for row in rows])
    cursors = []
    for rectangle, values, label, colour in (
        ((.77, .58, .20, .23), rain, 'Domain-mean rainfall (mm/h)', '#7acbe5'),
        ((.77, .27, .20, .23), q, 'Net coastal discharge (m³/s)', '#55b8ed')):
        axis = figure.add_axes(rect=rectangle, facecolor='#122431')
        axis.plot(hours, values, color=colour, linewidth=1.6)
        axis.fill_between(hours, values, color=colour, alpha=.14)
        axis.set_xlim(left=0., right=72.)
        axis.set_ylim(bottom=0., top=max(1., float(values.max()) * 1.14))
        axis.set_title(label=label, color='white', fontsize=10, loc='left', pad=12.)
        axis.set_xlabel(xlabel='Simulation time (hours)', color='#b9c8d2', fontsize=9)
        axis.tick_params(colors='#b9c8d2', labelsize=8)
        axis.grid(visible=True, alpha=.1, color='white')
        for spine in axis.spines.values():
            spine.set_color('#36515f')
        cursors.append(axis.axvline(x=0., color='#ffcf78', linewidth=1.5))
    colour_axis = figure.add_axes(rect=(.09, .14, .51, .016))
    figure.colorbar(mappable=ScalarMappable(
        norm=LogNorm(vmin=wet_threshold, vmax=depth_maximum), cmap=water_colormap()),
        cax=colour_axis, orientation='horizontal', label='Instantaneous depth (m; logarithmic colour scale)')
    colour_axis.tick_params(colors='#b9c8d2', labelsize=8)
    colour_axis.xaxis.label.set_color('#b9c8d2')
    colour_axis.xaxis.label.set_size(9)
    clock = figure.text(.77, .885, '', color='#ffcf78', fontsize=16, weight='bold')
    figure.text(.045, .045, 'NZTM2000 / EPSG:2193  •  NZVD2016  •  Vertical scale 1:1  •  Not an observed flood',
                fontsize=10, color='#b9c8d2')
    figure.text(.045, .022, 'Patrick Durney / Lincoln Agritech Ltd  |  Elevation: LINZ, ECan / CCC, Aerial Surveys, Landpro, AAM NZ · CC BY 4.0',
                fontsize=8, color='#7d9cac')
    return figure, canvas, image_artist, cursors, clock


def render_image(*, scene, layout, depth, active, seconds, fraction, wet_threshold,
                 depth_maximum, camera_start, camera_end, opacity):
    plotter, points, triangles = scene
    figure, canvas, image_artist, cursors, clock = layout
    screenshot = update_scene(plotter=plotter, points=points, triangles=triangles,
        depth=depth, active=active, wet_threshold=wet_threshold, depth_maximum=depth_maximum,
        fraction=fraction, camera_start=camera_start, camera_end=camera_end, opacity=opacity)
    image_artist.set_data(A=screenshot)
    for cursor in cursors:
        cursor.set_xdata(x=[seconds / 3600., seconds / 3600.])
    clock.set_text(s=f'{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d} elapsed')
    canvas.draw()
    return Image.fromarray(obj=np.asarray(canvas.buffer_rgba())[..., :3].copy())


def save_image(*, image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.png')
    image.save(fp=temporary)
    temporary.replace(target=path)


def encode_video(*, image_directory, path, width, height, fps, frame_count, crf):
    import imageio_ffmpeg
    path.parent.mkdir(parents=True, exist_ok=True)
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    temporary = path.with_name(path.stem + '.partial.mp4')
    subprocess.run(args=[executable, '-y', '-v', 'error', '-framerate', str(fps), '-i',
        str(image_directory / '%06d.png'), '-frames:v', str(frame_count), '-vf',
        f'scale={width}:{height}:flags=lanczos', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], check=True)
    # Decode the entire encoded stream; fail on corrupt frames, then inspect dimensions/count.
    subprocess.run(args=[executable, '-v', 'error', '-xerror', '-i', str(temporary),
                         '-f', 'null', '-'], check=True, capture_output=True)
    probe = imageio_ffmpeg.read_frames(path=str(temporary))
    metadata = next(probe)
    count = sum(1 for _ in probe)
    if metadata['size'] != (width, height) or count != frame_count or abs(metadata['fps'] - fps) > 1.e-6:
        raise ValueError('Encoded video size, rate or frame count differs')
    temporary.replace(target=path)
    return dict(path=str(path), width=width, height=height, fps=fps, frames=count,
                duration_s=count / fps, bytes=path.stat().st_size, sha256=file_hash(path=path))


def load_diagnostics(*, dataset, source, preflight, active, grid):
    if preflight:
        record = json.loads((source / 'results/fv1_record.json').read_text())
        rows = []
        previous = dict(seconds=0., rain_m3=0., outflow_m3=0., inflow_m3=0.)
        for row in record['ledger']:
            dt = row['seconds'] - previous['seconds']
            if dt > 0.:
                rows.append(dict(time_s=row['seconds'],
                    discharge_m3_s=(row['outflow_m3'] - previous['outflow_m3'] - row['inflow_m3'] + previous['inflow_m3']) / dt,
                    rainfall_m3_s=(row['rain_m3'] - previous['rain_m3']) / dt))
            previous = row
    else:
        rows = json.loads((dataset / 'diagnostics/ledger.json').read_text())
    area = float(active.sum()) * grid['dx'] * grid['dy']
    return [dict(row, rainfall_mm_h=row['rainfall_m3_s'] / area * 3600000.) for row in rows]


def render(*, dataset, source, output, preflight, width, height, seconds, fps, wet_threshold,
           depth_maximum, camera_start, camera_end, opacity, maximum_frames):
    identity = json.loads((dataset / 'metadata/dataset.json').read_text())
    checksums = {}
    if not preflight:
        checksums = {row['path']: row['sha256'] for row in json.loads(
            (dataset / 'diagnostics/frame_manifest.json').read_text())}
        validation = json.loads((dataset / 'diagnostics/validation.json').read_text())
        if validation['status'] != 'passed' or validation['final_time_s'] != 259200.:
            raise ValueError('Full event must pass numerical validation before conference rendering')
    with np.load(file=dataset / 'metadata/terrain.npz', allow_pickle=False) as saved:
        dem, valid, active = saved['dem'], saved['valid_dem'], saved['active']
    rows = load_diagnostics(dataset=dataset, source=source, preflight=preflight,
                           active=active, grid=identity['grid'])
    output.mkdir(parents=True, exist_ok=True)
    count = int(round(seconds * fps))
    render_identity = dict(frame_manifest_sha256=None if preflight else file_hash(
        path=dataset / 'diagnostics/frame_manifest.json'), renderer_sha256=file_hash(path=Path(__file__)), dataset=identity,
        preflight=preflight, width=width, height=height, duration_s=seconds, fps=fps,
        wet_threshold_m=wet_threshold, depth_maximum_m=depth_maximum,
        camera_start=camera_start, camera_end=camera_end, water_opacity=opacity,
        representation='Native centre triangulation; only triangles with three wet active vertices. '
        'Water vertices exactly DEM+h, no vertical exaggeration; five-minute fields held until next sample.',
        final_fields_sha256=file_hash(path=source / 'results/fv1_fields.npz') if preflight else
        file_hash(path=dataset / 'metadata/native_final.npz'))
    render_manifest = output / 'render_identity.json'
    if render_manifest.exists() and json.loads(render_manifest.read_text()) != render_identity and any((output / 'images').glob('*.png')):
        raise ValueError('Render settings changed; preserve this render and select a separate destination')
    atomic_json(path=render_manifest, value=render_identity)
    scene = create_scene(dem=dem, valid=valid, grid=identity['grid'],
        width=int(width * .73), height=int(height * .70), background='#0c1720')
    note = ('PIPELINE CHECK · retained 72-hour final field only · camera motion, no event animation' if preflight else
            f'Five-minute hydraulic samples · no temporal interpolation · water shown from {wet_threshold:g} m')
    layout = create_layout(width=width, height=height, rows=rows,
        wet_threshold=wet_threshold, depth_maximum=depth_maximum, title_note=note)
    last_time, depth = None, None
    started = time.perf_counter()
    limit = count if maximum_frames is None else min(count, maximum_frames)
    try:
        for index in range(limit):
            path = output / 'images' / f'{index:06d}.png'
            if path.exists():
                with Image.open(fp=path) as saved:
                    saved.verify()
                continue
            fraction = index / max(1, count - 1)
            sample_time = 259200. if preflight else np.floor(fraction * 259200. / 300.) * 300.
            if sample_time != last_time:
                field = source / 'results/fv1_fields.npz' if preflight else frame_path(output=dataset, seconds=sample_time)
                if not preflight and file_hash(path=field) != checksums[str(field.relative_to(dataset))]:
                    raise ValueError('Depth frame changed since numerical validation')
                with np.load(file=field, allow_pickle=False) as saved:
                    depth = saved['depth']
                last_time = sample_time
            image = render_image(scene=scene, layout=layout, depth=depth, active=active,
                seconds=sample_time, fraction=fraction, wet_threshold=wet_threshold,
                depth_maximum=depth_maximum, camera_start=camera_start, camera_end=camera_end,
                opacity=opacity)
            save_image(image=image, path=path)
            if index % fps == 0:
                print(f'render {index + 1}/{count}; {time.perf_counter() - started:.1f} s', flush=True)
    finally:
        scene[0].close()
    if limit < count:
        return dict(status='partial render', completed_frames=limit, expected_frames=count)
    with Image.open(fp=output / 'images/000000.png') as first_image, Image.open(
            fp=output / 'images' / f'{count - 1:06d}.png') as last_image:
        crop = (int(width * .015), int(height * .14), int(width * .745), int(height * .84))
        first_scene = np.asarray(first_image.crop(box=crop))
        last_scene = np.asarray(last_image.crop(box=crop))
    changed_scene_pixels = int(np.count_nonzero(np.any(first_scene != last_scene, axis=-1)))
    if changed_scene_pixels == 0:
        raise ArithmeticError('Scene did not update between first and last frames')
    videos = []
    for name, video_width, video_height, crf in (
        ('orton_bradley_pipeline_check.mp4' if preflight else 'orton_bradley_fv1_3d.mp4', width, height, 18),
        ('orton_bradley_pipeline_preview.mp4' if preflight else 'orton_bradley_fv1_preview.mp4', 960, 540, 24)):
        videos.append(encode_video(image_directory=output / 'images', path=output / name,
            width=video_width, height=video_height, fps=fps, frame_count=count, crf=crf))
    stills = []
    # Frames selected by actual time: development, global sampled inundation peak, recession.
    if not preflight:
        peaks = []
        frame_files = sorted((dataset / 'frames').glob('depth-*.npz'))
        for path in frame_files:
            with np.load(file=path, allow_pickle=False) as saved:
                peaks.append(int(np.count_nonzero((saved['depth'] >= wet_threshold) & active)))
        peak_time = float(np.argmax(peaks)) * 300.
        for label, target in (('development', peak_time * .65), ('peak_inundation', peak_time), ('recession', 259200.)):
            index = int(np.ceil(target / 259200. * (count - 1)))
            with Image.open(fp=output / 'images' / f'{index:06d}.png') as image:
                still_path = output / 'stills' / (label + '.png')
                save_image(image=image, path=still_path)
                stills.append(dict(path=str(still_path), time_s=float(np.floor(
                    index / max(1, count - 1) * 259200. / 300.) * 300.)))
    report = dict(status='pipeline check' if preflight else 'complete', videos=videos, stills=stills, changed_scene_pixels=changed_scene_pixels,
        render_seconds=time.perf_counter() - started, geometry='Native 5 m terrain and water meshes',
        limitations=render_identity['representation'])
    atomic_json(path=output / 'render_report.json', value=report)
    return report


if __name__ == "__main__":
    dataset = Path('outputs/orton_bradley_visualisation')
    source = Path('outputs/orton_bradley_release_staging')
    action = os.environ.get('FLOODWARP_VIS_RENDER', 'preflight')
    if action not in ('preflight', 'conference'):
        raise ValueError('Render action must be preflight or conference')
    preflight = action == 'preflight'
    render_output = dataset / 'renders' / action
    width = 960 if preflight else 1920
    height = 540 if preflight else 1080
    seconds = 2. if preflight else 72.
    fps = 30
    wet_threshold = .01
    depth_maximum = 7.
    camera_start = 115.
    camera_end = 145.
    opacity = .88
    maximum_frames = None  # Bound a smoke render; rerun to resume PNGs.
    software_rendering = True  # CPU Mesa: leave the occupied CUDA device alone.
    if software_rendering:
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkEGLRenderWindow'
        os.environ['ALLOW_PLOTTING'] = 'true'
        os.environ.pop('DISPLAY', None)
    print(json.dumps(render(dataset=dataset, source=source, output=render_output, preflight=preflight,
        width=width, height=height, seconds=seconds, fps=fps, wet_threshold=wet_threshold,
        depth_maximum=depth_maximum, camera_start=camera_start, camera_end=camera_end,
        opacity=opacity, maximum_frames=maximum_frames), indent=2))
