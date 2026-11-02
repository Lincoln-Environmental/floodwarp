"""Faithful conversion of the authorised ordinary Orton Bradley release inputs."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from floodwarp.inputs import read_terrain, boundary_faces
from floodwarp.ui_runner import RunConfig, file_identity


def preset_catalog():
    return json.loads((Path(__file__).with_name('ui_assets')/'orton_bradley.json').read_text())


def find_orton_archive(*, directory):
    for candidate in (directory/'outputs/release/orton_bradley_v0.1.0', directory/'orton_bradley_v0.1.0', directory):
        if (candidate/'inputs/inputs_f64.bin').is_file() and (candidate/'outputs/orton_bradley_dem_5m.tif').is_file():
            return str(candidate.resolve())
    return ''


def read_orton_packet(*, path):
    with path.open('rb') as stream:
        shape_and_count = np.fromfile(file=stream, dtype='<i4', count=3)
        if shape_and_count.size != 3 or np.any(shape_and_count <= 0):
            raise ValueError('Invalid Orton Bradley packet header')
        ny, nx, count = (int(value) for value in shape_and_count)
        expected_size = 12 + 16*count + 6*ny*nx*8
        if path.stat().st_size != expected_size:
            raise ValueError('Orton Bradley packet dimensions/size do not agree')
        times = np.fromfile(file=stream, dtype='<f8', count=count)
        rates = np.fromfile(file=stream, dtype='<f8', count=count)
        fields = np.fromfile(file=stream, dtype='<f8').reshape(6, ny, nx)[:, ::-1].copy()
    return times, rates, fields


def write_field(*, path, terrain, values, units, quantity):
    import rasterio
    from affine import Affine
    field = np.where(terrain.active, values, -9999.)[::-1].astype(np.float64)
    with rasterio.open(path, 'w', driver='GTiff', width=field.shape[1], height=field.shape[0],
            count=1, dtype='float64', crs=terrain.metadata['crs'],
            transform=Affine(*terrain.metadata['transform'][:6]), nodata=-9999., compress='deflate') as target:
        target.write(field, 1)
        target.set_band_unit(1, units)
        target.update_tags(units=units, quantity=quantity, vertical_datum=terrain.metadata['vertical_datum'])


def convert_orton_inputs(*, directory, output, expected_digests):
    """Inputs are read-only; publish a complete derived setup in a separate directory."""
    source = {}
    for name, expected in expected_digests.items():
        identity = file_identity(path=directory/name)
        if identity['sha256'] != expected:
            raise ValueError(f'Orton Bradley source checksum mismatch: {name}; use the original v0.1.0 archive')
        source[name] = identity
    if output.exists():
        manifest_path = output/'preset_manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Preset output already exists without a manifest; select a different output parent')
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_inputs'] != source:
            raise ValueError('Existing preset comes from different source paths; select a different output parent')
        for name, expected in manifest['converted_sha256'].items():
            if file_identity(path=output/name)['sha256'] != expected:
                raise ValueError(f'Prepared preset file changed: {name}; preserve it and choose a new output parent')
        return manifest
    terrain = read_terrain(path=directory/'outputs/orton_bradley_dem_5m.tif', vertical_datum='NZVD2016')
    times, rates, fields = read_orton_packet(path=directory/'inputs/inputs_f64.bin')
    if not np.array_equal(fields[0].astype(bool), terrain.active):
        raise ValueError('Orton Bradley packet mask does not match the DEM')
    roughness = np.loadtxt(fname=directory/'inputs/roughness.asc', skiprows=6)[::-1].copy()
    if roughness.shape != terrain.active.shape:
        raise ValueError('Orton Bradley roughness shape does not match the DEM')
    qx, qy = boundary_faces(terrain=terrain)
    ports = np.r_[fields[5, 1:], np.zeros_like(fields[5, :1])].astype(bool)
    rows, columns = np.nonzero(ports)
    if np.any(qy[rows+1, columns] != 1):
        raise ValueError('Archived coastal contacts must be active/domain interfaces')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name+'-preparing-', dir=output.parent))
    try:
        shutil.copyfile(src=directory/'outputs/orton_bradley_dem_5m.tif', dst=temporary/'dem.tif')
        for name, values, units in (('roughness', roughness, 's/m^(1/3)'), ('multiplier', fields[1], '1'),
                ('infiltration_f0', fields[2], 'm/s'), ('infiltration_fd', fields[3], 'm'), ('infiltration_k', fields[4], 'm/s')):
            write_field(path=temporary/(name+'.tif'), terrain=terrain, values=values, units=units, quantity=name)
        # SI decimal round-trip retains the exact original FP64 forcing nodes.
        np.savetxt(fname=temporary/'rainfall.csv', X=np.column_stack((times, rates)), delimiter=',',
            header='time_s,rainfall_m_s', comments='', fmt='%.17g')
        with (temporary/'boundaries.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['axis','row','col','type','stage_m'])
            for row, col in zip(rows, columns, strict=True):
                writer.writerow(['qy', int(row+1), int(col), 'stage', 0.])
        for name in ('provenance.json', 'DATA_LICENSE.md'):
            if (directory/name).is_file():
                shutil.copyfile(src=directory/name, dst=temporary/name)
        converted = {path.name: file_identity(path=path)['sha256'] for path in temporary.iterdir() if path.is_file()}
        manifest = dict(name='Orton Bradley Park — complete 5 m model', source_inputs=source,
            converted_sha256=converted, active_cells=int(terrain.active.sum()), shape=list(terrain.active.shape),
            coastal_stage_faces=int(rows.size), stage_m=0., stage_datum='NZVD2016',
            rainfall='Original FP64 SI nodes; native finite support ends at 48 hours; zero recession to 72 hours',
            numerical_profile='orton_bradley_matched', note='Matched numerical demonstration; uncalibrated, not observed-event validation')
        (temporary/'preset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.rename(target=output)
    finally:
        if temporary.exists():
            shutil.rmtree(path=temporary)
    return manifest


def prepare_orton_bradley(*, directory, output_parent):
    catalog = preset_catalog()
    output = output_parent/'orton-bradley-ui-inputs'
    manifest = convert_orton_inputs(directory=directory, output=output, expected_digests=catalog['source_sha256'])
    config = RunConfig(dem=str(output/'dem.tif'), rainfall_csv=str(output/'rainfall.csv'),
        output_parent=str(output_parent), vertical_datum='NZVD2016', duration_s=259200., interval_s=60.,
        roughness_mode='map', roughness_map=str(output/'roughness.tif'),
        distribution='map', multiplier=str(output/'multiplier.tif'),
        infiltration_mode='exponential', infiltration_map_units='m_s',
        infiltration_f0_map=str(output/'infiltration_f0.tif'), infiltration_fd_map=str(output/'infiltration_fd.tif'),
        infiltration_k_map=str(output/'infiltration_k.tif'), boundary='custom', boundary_csv=str(output/'boundaries.csv'),
        boundary_stage_datum='NZVD2016', rainfall_units='seconds_m_s', rainfall_end='finite_support',
        numerical_profile='orton_bradley_matched', preset_manifest=str(output/'preset_manifest.json'), checkpoint_interval_s=3600.)
    return dict(config=asdict(config), manifest=manifest)
