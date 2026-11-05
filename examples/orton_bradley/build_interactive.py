"""Prepare a static browser display of the validated FP64 event; never runs FV1."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil

import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap

from examples.orton_bradley.export_visualisation import atomic_json
from examples.orton_bradley.reproduce import file_hash


def write_binary(*, path, payload, compression_level):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream,
                           compresslevel=compression_level, mtime=0) as compressed:
            compressed.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target=path)
    return dict(path=str(path), raw_bytes=len(payload), bytes=path.stat().st_size,
        raw_sha256=hashlib.sha256(payload).hexdigest(), sha256=file_hash(path=path))


def terrain_payload(*, dem, valid, active, grid):
    ny, nx = dem.shape
    x, north = np.meshgrid((np.arange(nx) + .5) * grid['dx'],
                          (np.arange(ny) + .5) * grid['dy'])
    positions = np.column_stack((x.ravel() - nx * grid['dx'] / 2.,
        np.where(valid, dem, 0.).ravel(), ny * grid['dy'] / 2. - north.ravel())).astype('<f4')
    cmap = LinearSegmentedColormap.from_list(name='terrain',
        colors=['#354d3b', '#71805b', '#a5a179', '#b5aa89', '#cec7b2'])
    filled = np.where(valid, dem, np.median(dem[valid]))
    colours = LightSource(azdeg=315., altdeg=45.).shade(data=filled, cmap=cmap,
        vert_exag=1., dx=grid['dx'], dy=grid['dy'], blend_mode='soft')
    rgb = np.round(colours[..., :3] * 255.).astype(np.uint8)
    payload = (positions.tobytes() + rgb.tobytes() + valid.astype(np.uint8).tobytes() +
               active.astype(np.uint8).tobytes())
    maximum_error = float(np.max(np.abs(positions[:, 1][valid.ravel()].astype(np.float64) - dem[valid])))
    return payload, maximum_error


def depth_payload(*, depth, active, wet_threshold):
    if depth.shape != active.shape or not np.isfinite(depth).all() or np.any(depth < 0.) or np.any(depth[~active] != 0.):
        raise ValueError('Invalid source depth')
    display = depth.astype('<f4')
    # Classify the native FP64 values, not their rounded browser display values.
    wet = ((depth >= wet_threshold) & active).astype(np.uint8)
    error = float(np.max(np.abs(display.astype(np.float64) - depth)))
    return display.tobytes() + wet.tobytes(), error, int(wet.sum())


def relative_record(*, record, output):
    return dict(record, path=str(Path(record['path']).relative_to(output)))


def package_frame(*, source_path, expected, active, output, wet_threshold, compression_level):
    if file_hash(path=source_path) != expected['sha256']:
        raise ValueError('Native frame changed since validation: ' + str(source_path))
    marker = output / 'data/frames' / (source_path.stem + '.json')
    binary = marker.with_suffix('.bin.gz')
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if (recorded['source_sha256'] == expected['sha256'] and binary.exists() and
                file_hash(path=binary) == recorded['sha256']):
            return recorded
        raise ValueError('Existing browser frame differs; preserve it and use a separate output root')
    with np.load(file=source_path, allow_pickle=False) as saved:
        seconds = float(saved['time_s'])
        if seconds != expected['time_s']:
            raise ValueError('Native frame timestamp differs')
        depth = saved['depth']
        payload, error, wet_cells = depth_payload(depth=depth, active=active, wet_threshold=wet_threshold)
        record = relative_record(record=write_binary(path=binary, payload=payload,
            compression_level=compression_level), output=output)
        record.update(time_s=seconds, source_sha256=expected['sha256'],
            maximum_display_depth_error_m=error, wet_cells=wet_cells, native_maximum_depth_m=float(depth.max()))
    atomic_json(path=marker, value=record)
    return record


def copy_application(*, application, output):
    for source in application.rglob('*'):
        if source.is_file():
            destination = output / source.relative_to(application)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=source, dst=destination)


def build(*, dataset, application, output, wet_threshold, depth_maximum, compression_level):
    validation_path = dataset / 'diagnostics/validation.json'
    validation = json.loads(validation_path.read_text())
    identity = json.loads((dataset / 'metadata/dataset.json').read_text())
    if validation['status'] != 'passed' or validation['final_time_s'] != 259200.:
        raise ValueError('A completed, validated full-domain event is required')
    expected = json.loads((dataset / 'diagnostics/frame_manifest.json').read_text())
    if [row['time_s'] for row in expected] != list(np.arange(0., 259201., 300.)):
        raise ValueError('Missing or duplicate source frames')
    configuration = dict(schema=1, dataset=str(dataset.resolve()),
        source_identity_sha256=file_hash(path=dataset / 'metadata/dataset.json'),
        source_validation_sha256=file_hash(path=validation_path),
        source_frame_manifest_sha256=file_hash(path=dataset / 'diagnostics/frame_manifest.json'),
        source_terrain_sha256=file_hash(path=dataset / 'metadata/terrain.npz'),
        wet_threshold_m=wet_threshold, depth_maximum_m=depth_maximum,
        builder_sha256=file_hash(path=Path(__file__)), compression_level=compression_level,
        browser_precision='Float32 display; original FP64 depths retained unchanged',
        wet_classification='Native FP64 h >= threshold and active; no rounded-threshold classification')
    configuration_path = output / 'data/build_identity.json'
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise ValueError('Browser dataset configuration differs; select a separate output')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use an empty or previously identified browser output directory')
        atomic_json(path=configuration_path, value=configuration)
    with np.load(file=dataset / 'metadata/terrain.npz', allow_pickle=False) as saved:
        dem, active, valid = saved['dem'], saved['active'], saved['valid_dem']
    payload, elevation_error = terrain_payload(dem=dem, valid=valid, active=active, grid=identity['grid'])
    terrain_record = relative_record(record=write_binary(path=output / 'data/terrain.bin.gz',
        payload=payload, compression_level=compression_level), output=output)
    records = []
    for index, row in enumerate(expected):
        records.append(package_frame(source_path=dataset / row['path'], expected=row, active=active,
            output=output, wet_threshold=wet_threshold, compression_level=compression_level))
        if index % 60 == 0:
            print(f'Browser frames {index + 1}/{len(expected)}', flush=True)
    rows = json.loads((dataset / 'diagnostics/ledger.json').read_text())
    if [row['time_s'] for row in rows] != list(np.arange(60., 259201., 60.)):
        raise ValueError('Missing minute diagnostics')
    area = float(active.sum()) * identity['grid']['dx'] * identity['grid']['dy']
    series = dict(time_s=[0.] + [row['time_s'] for row in rows],
        discharge_m3_s=[0.] + [row['discharge_m3_s'] for row in rows],
        rainfall_mm_h=[0.] + [row['rainfall_m3_s'] / area * 3600000. for row in rows],
        rainfall_interval='Mean over the preceding 60 seconds; zero placeholder at t=0',
        discharge_interval='Net coastal export per preceding 60 seconds; zero placeholder at t=0')
    manifest = dict(configuration=configuration, grid=identity['grid'], array_row_order='south_to_north',
        terrain=terrain_record, frames=records, diagnostics=series, total_frames=len(records),
        initial_frame=int(round(series['time_s'][int(np.argmax(series['discharge_m3_s']))] / 300.)),
        peak_inundation_frame=int(np.argmax([row['wet_cells'] for row in records])),
        elevation_range_m=[float(dem[valid].min()), float(dem[valid].max())],
        maximum_display_depth_error_m=max(row['maximum_display_depth_error_m'] for row in records),
        maximum_display_dem_error_m=elevation_error, interpretation=identity['interpretation'],
        attribution=identity['provenance'], software_attribution=identity['software_attribution'],
        elevation_license=identity['elevation_license'], crs='EPSG:2193', vertical_datum='NZVD2016',
        representation='Native 5 m cell-centre triangles, all three vertices wet, z=DEM+h; vertical scale 1:1; no temporal interpolation')
    copy_application(application=application, output=output)
    atomic_json(path=output / 'data/manifest.json', value=manifest)
    report = dict(status='complete', frames=len(records),
        maximum_display_depth_error_m=manifest['maximum_display_depth_error_m'],
        maximum_display_dem_error_m=elevation_error,
        compressed_frame_bytes=sum(row['bytes'] for row in records),
        terrain_bytes=terrain_record['bytes'], manifest_sha256=file_hash(path=output / 'data/manifest.json'),
        application_sha256={str(path.relative_to(application)):file_hash(path=path)
            for path in application.rglob('*') if path.is_file()})
    atomic_json(path=output / 'build_report.json', value=report)
    return report


if __name__ == "__main__":
    dataset = Path('outputs/orton_bradley_visualisation')
    application = Path('examples/orton_bradley/viewer')
    output = Path('outputs/orton_bradley_interactive')
    wet_threshold = .01
    depth_maximum = 7.
    compression_level = 4
    report = build(dataset=dataset, application=application, output=output,
        wet_threshold=wet_threshold, depth_maximum=depth_maximum, compression_level=compression_level)
    print(json.dumps(report, indent=2))
