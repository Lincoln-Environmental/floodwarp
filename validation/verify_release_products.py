"""Independent checks of release rasters/CSV against original FP64 records."""

from pathlib import Path
import csv
import json

import numpy as np
import rasterio


def verify(*, directory, output):
    with np.load(file=directory / 'inputs/comparison_inputs.npz', allow_pickle=False) as saved:
        active, bed = saved['active'].astype(bool), saved['bed'].copy()
    fields = {}
    for method in ('fv1', 'reference', 'li'):
        with np.load(file=directory / 'results' / (method + '_fields.npz'), allow_pickle=False) as saved:
            fields[method] = saved['maximum'][::-1].copy()
    expected = dict(orton_bradley_dem_5m=bed,
        orton_bradley_floodwarp_fv1_max_depth_5m=fields['fv1'],
        orton_bradley_lisflood_fv1_max_depth_5m=fields['reference'],
        orton_bradley_floodwarp_li_max_depth_5m=fields['li'],
        orton_bradley_fv1_minus_lisflood_max_depth_5m=fields['fv1'] - fields['reference'],
        orton_bradley_li_minus_lisflood_max_depth_5m=fields['li'] - fields['reference'])
    for threshold, suffix in ((.01, '001m'), (.1, '010m')):
        expected['orton_bradley_fv1_max_extent_' + suffix] = (fields['fv1'] >= threshold).astype(np.uint8)
        expected['orton_bradley_lisflood_max_extent_' + suffix] = (fields['reference'] >= threshold).astype(np.uint8)
    results = {}
    for name, values in expected.items():
        with rasterio.open(directory / 'outputs' / (name + '.tif')) as raster:
            assert raster.shape == (1123, 843)
            assert raster.crs.to_epsg() == 2193
            assert tuple(raster.bounds) == (1574940., 5161660., 1579155., 5167275.)
            assert tuple(raster.transform) == (5., 0., 1574940., 0., -5., 5167275., 0., 0., 1.)
            assert raster.tags()['vertical_datum'] == 'NZVD2016'
            np.testing.assert_array_equal(actual=raster.read_masks(1).astype(bool), desired=active)
            np.testing.assert_array_equal(actual=raster.read(1)[active], desired=values[active])
            results[name] = dict(maximum_active_error=0., crs=raster.crs.to_string(), dtype=raster.dtypes[0], nodata=raster.nodata)
    with (directory / 'outputs/orton_bradley_hydrographs_72h.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4320
    for method, label in (('fv1', 'floodwarp_fv1'), ('reference', 'lisflood_fv1'), ('li', 'floodwarp_li')):
        record = json.loads((directory / 'results' / (method + '_record.json')).read_text())
        ledger = {row['seconds']: row for row in record['ledger']}
        previous = 0.
        for index, row in enumerate(rows):
            end = (index + 1) * 60.
            assert float(row['interval_start_s']) == end - 60.
            assert float(row['interval_end_s']) == end
            cumulative = ledger[end]['outflow_m3'] - ledger[end]['inflow_m3']
            assert float(row[label + '_cumulative_net_export_m3']) == cumulative
            assert float(row[label + '_mean_net_outlet_m3_s']) == (cumulative - previous) / 60.
            previous = cumulative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(rasters=results, csv_rows=len(rows),
        all_csv_values_exact_against_cumulative_records=True, original_active_cells=int(active.sum()),
        maximum_maps_compared_to_retained_envelopes=True), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('outputs/release/orton_bradley_v0.1.0')
    output = Path('validation/release_v0.1.0/product_verification.json')
    verify(directory=directory, output=output)
