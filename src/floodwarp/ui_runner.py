"""Reusable local rainfall-run preparation and execution; no UI framework imports."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from floodwarp import __version__, FullSWESolver, create_overland_solver
from floodwarp.inputs import (boundary_faces, read_active_mask, read_aligned_field,
                             read_terrain, with_active_mask, write_raster)
from floodwarp.model import advance, create_model, diagnostics, model_time, save_checkpoint
from floodwarp.sampling import sample_targets


@dataclass(frozen=True)
class RunConfig:
    dem: str
    rainfall_csv: str
    output_parent: str
    vertical_datum: str
    duration_s: float = 120.
    domain_mode: str = 'all'
    domain: str = ''
    distribution: str = 'uniform'
    multiplier: str = ''
    manning_n: float = .03
    infiltration_mm_h: float = 0.
    device: str = 'auto'
    boundary: str = 'free'
    interval_s: float = 60.


@dataclass
class PreparedRun:
    config: RunConfig
    terrain: object
    times_s: np.ndarray
    rates_m_s: np.ndarray
    multiplier: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    summary: dict


def file_identity(*, path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter_read(stream=stream):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def iter_read(*, stream):
    while block := stream.read(1024 * 1024):
        yield block


def read_hyetograph(*, path, duration_s):
    """Linear nodes; a final zero node defines event end, then zero recession."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['time_h', 'rainfall_mm_h']:
            raise ValueError('Rainfall CSV header must be exactly time_h,rainfall_mm_h; discharge is not rainfall')
        try:
            rows = [[float(row['time_h']), float(row['rainfall_mm_h'])] for row in reader]
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError('Rainfall CSV must contain two numeric columns without missing values') from error
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise ValueError('Rainfall needs at least two piecewise-linear samples')
    times, rates = values[:, 0] * 3600., values[:, 1] / 3.6e6
    if (not np.isfinite(values).all() or times[0] != 0. or
            np.any(np.diff(times) <= 0.) or np.any(rates < 0.)):
        raise ValueError('Rainfall needs finite increasing times starting at 0 and finite non-negative intensities')
    if times[-1] > duration_s:
        raise ValueError('Simulation duration must cover the entire rainfall event; increase duration or edit the CSV')
    if rates[-1] != 0.:
        raise ValueError('End the rainfall event with an explicit zero-intensity node; recession then stays dry')
    if times[-1] < duration_s:
        times = np.r_[times, duration_s]
        rates = np.r_[rates, 0.]
    return times, rates


def polygon_mask(*, path, terrain):
    import pyogrio
    from pyogrio.raw import read
    from affine import Affine
    from pyproj import CRS
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    from shapely import from_wkb
    from shapely.geometry import mapping

    if path.suffix.lower() not in ('.gpkg', '.geojson'):
        raise ValueError('Polygon domains support .gpkg (one layer) or .geojson with a declared CRS')
    declared_crs = None
    if path.suffix.lower() == '.geojson':
        declaration = json.loads(path.read_text()).get('crs')
        if not declaration:
            raise ValueError('GeoJSON requires an explicit crs member; use a GeoPackage to preserve declared CRS')
        if declaration.get('type') != 'name' or not declaration.get('properties', {}).get('name'):
            raise ValueError('GeoJSON crs must declare a valid named CRS; export to GeoPackage if needed')
        declared_crs = CRS.from_user_input(declaration['properties']['name'])
    if len(pyogrio.list_layers(path)) != 1:
        raise ValueError('Domain must contain one polygon layer; export the desired layer to a separate file')
    # Pyogrio declares its path argument positional-only.
    metadata, _, geometry_wkb, _ = read(path, columns=[], force_2d=True)
    if not metadata['crs']:
        raise ValueError('Polygon domain has no declared CRS; assign the correct CRS before running')
    source_crs = declared_crs if declared_crs is not None else CRS.from_user_input(metadata['crs'])
    geometries = []
    for wkb in geometry_wkb:
        polygon = from_wkb(geometry=wkb)
        if polygon is None or polygon.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Domain features must be nonempty Polygon or MultiPolygon geometries')
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError('Domain contains invalid or empty polygons; repair them explicitly in GIS')
        geometries.append(transform_geom(src_crs=source_crs.to_wkt(),
            dst_crs=terrain.metadata['crs'], geom=mapping(ob=polygon)))
    if not geometries:
        raise ValueError('Polygon domain is empty')
    north_first = rasterize(shapes=geometries, out_shape=terrain.active.shape,
        transform=Affine(*terrain.metadata['transform'][:6]), fill=0, default_value=1,
        all_touched=False, dtype='uint8')
    return north_first[::-1].astype(bool)


def boundary_preset(*, terrain, preset):
    """FREE opens exterior-connected domain edges; enclosed inactive holes stay closed."""
    from scipy.ndimage import binary_fill_holes
    qx, qy = boundary_faces(terrain=terrain)
    if preset == 'closed':
        return qx, qy
    if preset != 'free':
        raise ValueError('Boundary must be free or closed')
    active = terrain.active
    exterior = ~binary_fill_holes(input=active)
    qx[:, 0][active[:, 0]] = 2
    qx[:, -1][active[:, -1]] = 2
    qy[0][active[0]] = 2
    qy[-1][active[-1]] = 2
    qx[:, 1:-1][(active[:, :-1] & exterior[:, 1:]) | (exterior[:, :-1] & active[:, 1:])] = 2
    qy[1:-1][(active[:-1] & exterior[1:]) | (exterior[:-1] & active[1:])] = 2
    return qx, qy


def available_devices():
    import warp as wp
    return [str(device) for device in wp.get_devices()]


def resolve_device(*, requested):
    devices = available_devices()
    if requested == 'auto':
        return next((device for device in devices if device.startswith('cuda:')), 'cpu')
    if requested not in devices:
        raise ValueError(f'Device {requested!r} unavailable; choose auto or one of {devices}')
    return requested


def numerical_defaults():
    native = inspect.signature(obj=FullSWESolver).parameters
    factory = inspect.signature(obj=create_overland_solver).parameters
    return dict(cfl=native['cfl'].default, dry_depth_m=native['dry_depth'].default,
        minimum_dt_s=factory['explicit_minimum_dt'].default,
        maximum_dt_s=factory['explicit_maximum_dt'].default,
        flux_scheme=native['flux_scheme'].default,
        stage_boundary_momentum=native['stage_boundary_momentum'].default)


def prepare_run(*, config):
    sample_targets(duration=config.duration_s, interval=config.interval_s)
    if not np.isfinite([config.manning_n, config.infiltration_mm_h]).all() or config.manning_n <= 0 or config.infiltration_mm_h < 0:
        raise ValueError('Manning n must be positive; infiltration must be non-negative and finite')
    if not config.vertical_datum.strip():
        raise ValueError('Supply the DEM vertical datum, or explicitly use local for an unrecorded local datum')
    dem_path, rain_path = Path(config.dem).expanduser(), Path(config.rainfall_csv).expanduser()
    terrain = read_terrain(path=dem_path, vertical_datum=config.vertical_datum)
    identities = dict(dem=file_identity(path=dem_path), rainfall=file_identity(path=rain_path))
    if config.domain_mode != 'all':
        domain_path = Path(config.domain).expanduser()
        if config.domain_mode == 'raster':
            # Mask nodata is inactive; active field coverage is checked after intersection.
            terrain = read_active_mask(path=domain_path, terrain=terrain)
        elif config.domain_mode == 'polygon':
            mask = polygon_mask(path=domain_path, terrain=terrain)
            terrain = with_active_mask(terrain=terrain, active=mask & terrain.active)
        else:
            raise ValueError('Choose all valid DEM cells, raster mask or polygon domain')
        identities['domain'] = file_identity(path=domain_path)
    if not terrain.active.any():
        raise ValueError('Domain is empty after intersection with valid DEM cells')
    if config.distribution == 'uniform':
        multiplier = np.where(terrain.active, 1., 0.)
    elif config.distribution == 'map':
        multiplier_path = Path(config.multiplier).expanduser()
        multiplier = read_aligned_field(path=multiplier_path, terrain=terrain)
        if np.any(multiplier[terrain.active] < 0.):
            raise ValueError('Rainfall multipliers must be finite and non-negative over active cells; zero is valid')
        multiplier = np.where(terrain.active, multiplier, 0.)
        identities['multiplier'] = file_identity(path=multiplier_path)
    else:
        raise ValueError('Choose uniform rainfall or a spatial scaling map')
    times, rates = read_hyetograph(path=rain_path, duration_s=config.duration_s)
    qx, qy = boundary_preset(terrain=terrain, preset=config.boundary)
    parent = Path(config.output_parent).expanduser().resolve()
    if not parent.is_dir():
        raise ValueError('Output parent must be an existing directory; create it or select another directory')
    summary = dict(inputs=identities, solver='fv1', precision='float64', execution='sparse',
        numerical_defaults=numerical_defaults(),
        device=resolve_device(requested=config.device), boundary=config.boundary,
        open_faces=int(np.count_nonzero(qx == 2) + np.count_nonzero(qy == 2)),
        boundary_rule='FREE opens exterior-connected inactive interfaces and active raster edges; enclosed holes closed',
        manning_n=config.manning_n, infiltration_mm_h=config.infiltration_mm_h,
        duration_s=config.duration_s, interval_s=config.interval_s, output_parent=str(parent),
        active_cells=int(terrain.active.sum()), shape=list(terrain.active.shape),
        cell_size_m=[terrain.dx, terrain.dy], crs=terrain.metadata['crs'],
        transform=terrain.metadata['transform'], vertical_datum=config.vertical_datum,
        domain_rule='Cell centre inside polygon; intersect with valid DEM cells; no resampling',
        multiplier_range=[float(multiplier[terrain.active].min()), float(multiplier[terrain.active].max())],
        rainfall='time_h,rainfall_mm_h; linear interpolation; final node must be zero; zero recession',
        rainfall_times_s=times.tolist(), rainfall_rates_m_s=rates.tolist())
    return PreparedRun(config=config, terrain=terrain, times_s=times, rates_m_s=rates,
        multiplier=multiplier, qx=qx, qy=qy, summary=summary)


def create_run_model(*, prepared):
    config = prepared.config
    model = create_model(terrain=prepared.terrain, solver_method='fv1',
        manning_n=config.manning_n, infiltration=config.infiltration_mm_h / 3.6e6,
        rainfall_series=(prepared.times_s, prepared.rates_m_s), boundary_qx=prepared.qx,
        boundary_qy=prepared.qy, device=prepared.summary['device'])
    model.solver.set_rainfall_spatial_multiplier(multiplier=prepared.multiplier)
    model.identity['fields']['ui_rainfall_multiplier'] = hashlib.sha256(prepared.multiplier.tobytes()).hexdigest()
    return model


def write_json(*, path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def create_run_directory(*, parent):
    return Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('floodwarp-%Y%m%dT%H%M%SZ-'), dir=parent))


def source_identity():
    stamp = Path(__file__).with_name('_build_info.json')
    if stamp.exists():
        return json.loads(stamp.read_text())
    root = Path(__file__).resolve().parents[2]
    if (root/'.git').exists() and (root/'pyproject.toml').exists():
        revision = subprocess.run(args=['git', '-C', str(root), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(args=['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=revision, source='checkout', tracked_changes=bool(dirty))
    return dict(commit=None, source='unidentified source; use a stamped release distribution for revision provenance')


def execute_run(*, config, output, expected_inputs=None):
    """Headless execution path used by the worker. Checkpoint each completed sample."""
    started = time.monotonic()
    status = dict(state='running', time_s=0., end_time_s=config.duration_s,
        elapsed_s=0., output=str(output), error=None)
    output.mkdir(parents=True, exist_ok=True)
    write_json(path=output/'status.json', value=status)
    try:
        prepared = prepare_run(config=config)
        if expected_inputs is not None and prepared.summary['inputs'] != expected_inputs:
            raise ValueError('Inputs changed after validation; validate again before running')
        model = create_run_model(prepared=prepared)
        resolved = dict(config=asdict(config), resolved=prepared.summary,
            numerical=model.solver.numerical_configuration(), version=__version__,
            source=source_identity(), checkpoint='checkpoint.npz', use_cuda_graph=None,
            graph_policy='Existing solver default; no UI override', actual_final_time_s=None)
        write_json(path=output/'run_config.json', value=resolved)
        save_checkpoint(model=model, path=output/'checkpoint.npz')
        rows = []
        previous = dict(time_s=0., boundary_outflow=0., boundary_inflow=0.)
        with (output/'water_balance.csv').open('w', newline='') as balance, (output/'boundary_flow.csv').open('w', newline='') as flow:
            balance_writer = csv.DictWriter(balance, fieldnames=['time_s', 'storage_m3', 'residual_m3',
                'rain', 'infiltration', 'boundary_inflow', 'boundary_outflow'], extrasaction='ignore')
            flow_writer = csv.DictWriter(flow, fieldnames=['time_s', 'interval_s',
                'aggregate_boundary_outflow_m3_s', 'aggregate_boundary_inflow_m3_s'])
            balance_writer.writeheader()
            flow_writer.writeheader()
            for target in sample_targets(duration=config.duration_s, interval=config.interval_s):
                advance(model=model, end_time=float(target))
                row = diagnostics(model=model)
                dt = row['time_s'] - previous['time_s']
                balance_writer.writerow(row)
                flow_writer.writerow(dict(time_s=row['time_s'], interval_s=dt,
                    aggregate_boundary_outflow_m3_s=(row['boundary_outflow']-previous['boundary_outflow'])/dt,
                    aggregate_boundary_inflow_m3_s=(row['boundary_inflow']-previous['boundary_inflow'])/dt))
                balance.flush()
                flow.flush()
                rows.append(row)
                previous = row
                save_checkpoint(model=model, path=output/'checkpoint.npz')
                status.update(time_s=row['time_s'], elapsed_s=time.monotonic()-started)
                write_json(path=output/'status.json', value=status)
        write_raster(path=output/'maximum_depth.tif', terrain=prepared.terrain,
            values=model.solver.get_max_water_depth(), quantity='maximum_water_depth')
        write_raster(path=output/'final_depth.tif', terrain=prepared.terrain,
            values=model.solver.get_water_depth(), quantity='final_water_depth')
        (output/'maximum_depth.png').write_bytes(depth_preview(terrain=prepared.terrain,
            values=model.solver.get_max_water_depth()))
        write_json(path=output/'diagnostics.json', value=rows)
        resolved['actual_final_time_s'] = model_time(model=model)
        write_json(path=output/'run_config.json', value=resolved)
        status.update(state='completed', time_s=model_time(model=model), elapsed_s=time.monotonic()-started)
        write_json(path=output/'status.json', value=status)
        return resolved
    except Exception as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}', elapsed_s=time.monotonic()-started)
        write_json(path=output/'status.json', value=status)
        raise


def preview_figure(*, terrain, values, label):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(5, 3), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    # Bound preview memory; the solver grid and outputs remain unchanged.
    stride = max(1, int(np.ceil(max(terrain.active.shape) / 500)))
    image = axes.imshow(np.where(terrain.active[::stride, ::stride], values[::stride, ::stride], np.nan),
        origin='lower', extent=(0, terrain.active.shape[1]*terrain.dx, 0, terrain.active.shape[0]*terrain.dy))
    axes.set(xlabel='East from grid origin (m)', ylabel='North from grid origin (m)')
    figure.colorbar(mappable=image, ax=axes, label=label)
    buffer = io.BytesIO()
    figure.savefig(fname=buffer, format='png', dpi=100)
    return buffer.getvalue()


def depth_preview(*, terrain, values):
    return preview_figure(terrain=terrain, values=values, label='Maximum water depth (m)')


def terrain_preview(*, prepared):
    return preview_figure(terrain=prepared.terrain, values=prepared.terrain.elevation, label='Active terrain elevation (m)')
