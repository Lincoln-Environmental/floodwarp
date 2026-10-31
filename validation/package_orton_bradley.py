"""Package previously completed records and authoritative staged model arrays.

Inputs must be staged explicitly; this script contains no private input paths
and never imports another model repository or reruns hydraulics.
"""

from pathlib import Path
import json
import shutil
import tarfile

import numpy as np

from examples.orton_bradley.reproduce import reproduce, file_hash, write_json


def public_provenance(*, value):
    if isinstance(value, dict):
        return {key: public_provenance(value=item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_provenance(value=item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def package(*, staging, dataset, archive, benchmark_revision, extraction_revision, dpi):
    dataset.mkdir(parents=True, exist_ok=True)
    for name in ('inputs', 'results'):
        shutil.copytree(src=staging / name, dst=dataset / name, dirs_exist_ok=True)
    metadata_path = staging / 'original_model_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    grid = metadata['grid']
    bounds = grid['bounds']
    transform = [grid['dx'], 0., bounds[0], 0., -grid['dy'], bounds[3], 0., 0., 1.]
    if (grid['nrow'], grid['ncol'], grid['dx'], grid['dy'], grid['epsg']) != (1123, 843, 5., 5., 2193):
        raise ValueError('Authoritative model grid differs from expected domain')
    with np.load(file=dataset / 'inputs/comparison_inputs.npz', allow_pickle=False) as saved:
        active = saved['active'].astype(bool)
        if active.shape != (grid['nrow'], grid['ncol']):
            raise ValueError('Metadata and numerical grid differ')
    crop = json.loads(Path('examples/purau/provenance.json').read_text())
    hardware = json.loads((dataset / 'results/hardware.json').read_text())
    benchmark = json.loads((dataset / 'results/fv1_record.json').read_text())
    with (dataset / 'inputs/inputs_f64.bin').open('rb') as stream:
        ny, nx, count = np.fromfile(file=stream, dtype='<i4', count=3)
        times = np.fromfile(file=stream, dtype='<f8', count=int(count))
        rates = np.fromfile(file=stream, dtype='<f8', count=int(count))
        source_fields = np.fromfile(file=stream, dtype='<f8').reshape(6, int(ny), int(nx))
    np.testing.assert_array_equal(actual=source_fields[0].astype(bool), desired=active)
    roughness = np.loadtxt(fname=dataset / 'inputs/roughness.asc', skiprows=6)
    parameter_ranges = {name: [float(values[active].min()), float(values[active].max())] for name, values in
        (('manning_n', roughness), ('rainfall_multiplier', source_fields[1]), ('f0_m_s', source_fields[2]),
         ('fd_m', source_fields[3]), ('conductivity_m_s', source_fields[4]))}
    provenance = dict(site='Orton Bradley Park, Banks Peninsula, Canterbury, New Zealand',
        historical_dataset_identifier='purau_event_72h / purau_point_1575902_5166615_catchment_model',
        software_copyright='Patrick Durney and Lincoln Agritech Ltd', software_license='AGPL-3.0-only',
        elevation_license='CC-BY-4.0', elevation_license_url='https://creativecommons.org/licenses/by/4.0/',
        elevation_attribution=crop['attribution'], elevation_sources=crop['source_surveys'],
        vertical_datum_authority='NZVD2016 identity supplied by the model owner for the authorised v0.1 release; no vertical conversion applied.',
        grid=dict(shape=[grid['nrow'], grid['ncol']], dx=grid['dx'], dy=grid['dy'], epsg=grid['epsg'],
            transform=transform, bounds=bounds, vertical_datum='NZVD2016', units='metres', row_order='north_to_south',
            active_cells=int(active.sum()), active_area_km2=float(active.sum() * grid['dx'] * grid['dy'] / 1.e6),
            raster_area_km2=grid['nrow'] * grid['ncol'] * grid['dx'] * grid['dy'] / 1.e6,
            catchment_scope='Complete established hydraulic model raster and physical mask; not a claim of complete geographical catchment coverage.'),
        source_metadata_sha256=file_hash(path=metadata_path), original_content_hashes=metadata['content_hashes'],
        processing_history=['Original multi-survey LINZ elevation sources and historical hydraulic conditioning retained.',
            'The original conditioned model DEM and benchmark comparison packet agree exactly at every active cell.',
            'No resampling, interpolation, vertical shift or DEM modification during release export.',
            'Only the separately supplied physical mask defines nodata; legitimate low elevations remain active.',
            'Retained south-first solver outputs are explicitly reversed to the original north-first raster orientation.',
            'GeoTIFF FP64 values are losslessly compressed with DEFLATE; conversion error verified per product.'],
        terrain_provenance=public_provenance(value=metadata['terrain_provenance']),
        rainfall_provenance=public_provenance(value=metadata['rainfall_provenance']),
        soil_provenance=public_provenance(value=metadata['infiltration_provenance']),
        roughness_provenance=public_provenance(value=metadata['roughness_provenance']),
        benchmark_revision_anchor=benchmark_revision, extraction_revision=extraction_revision,
        source_revision_note='Exact runtime module and external binary hashes are preserved in each immutable completed record; the anchor identifies retained completion evidence.',
        benchmark_numerical=benchmark['signature']['numerical'], parameter_ranges=parameter_ranges,
        infiltration_equation='f(F) = k + (f0 - k) exp(-F/fd), limited by available water; F is cumulative infiltration depth',
        rain_node_count=int(count), rain_support_s=[float(times[0]), float(times[-1])],
        rainfall_rate_range_m_s=[float(rates.min()), float(rates.max())],
        hardware=hardware, published_source_files={})
    for folder in ('inputs', 'results'):
        for path in sorted((dataset / folder).iterdir()):
            provenance['published_source_files'][path.relative_to(dataset).as_posix()] = file_hash(path=path)
    for name in ('reproduce.py', 'run_model.py', 'README.md', 'DATA_LICENSE.md'):
        source = Path('examples/orton_bradley') / name
        if source.exists():
            shutil.copy2(src=source, dst=dataset / name)
    for name in ('LICENSE', 'COPYRIGHT', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(src=Path(name), dst=dataset / name)
    write_json(path=dataset / 'provenance.json', value=provenance)
    reproduce(directory=dataset, output=dataset / 'outputs', figures=dataset / 'figures', dpi=dpi,
              depth_minimum=.001, thresholds=(.01, .10), wet_union_threshold=.001)
    shutil.copy2(src=dataset / 'outputs/COMPARISON.md', dst=dataset / 'COMPARISON.md')
    checksum_lines = []
    for path in sorted(dataset.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            checksum_lines.append(file_hash(path=path) + '  ' + path.relative_to(dataset).as_posix())
    (dataset / 'SHA256SUMS').write_text('\n'.join(checksum_lines) + '\n')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(name=archive, mode='w:gz') as stream:
        stream.add(name=dataset, arcname=dataset.name)
    print('dataset archive:', archive, file_hash(path=archive))


if __name__ == "__main__":
    staging = Path('outputs/orton_bradley_release_staging')
    dataset = Path('outputs/release/orton_bradley_v0.1.0')
    archive = Path('outputs/release/orton_bradley_v0.1.0.tar.gz')
    benchmark_revision = '183c884'
    extraction_revision = 'ba503b8bde51d3f212199e81efc7038d9fffecc8'
    dpi = 180
    package(staging=staging, dataset=dataset, archive=archive, benchmark_revision=benchmark_revision,
            extraction_revision=extraction_revision, dpi=dpi)
