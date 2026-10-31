"""Release data semantics: nodata, precision, and true minute hydrographs."""

from pathlib import Path
import tomllib

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('matplotlib')
from rasterio.transform import Affine

from examples.orton_bradley.reproduce import write_raster, hydrographs, extent_statistics


def test_public_license_and_version_metadata():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    assert project['license'] == 'AGPL-3.0-only'
    assert project['version'] == '0.1.0'
    assert {author['name'] for author in project['authors']} == {'Patrick Durney', 'Lincoln Agritech Ltd'}
    license_text = (root / 'LICENSE').read_text()
    assert 'GNU AFFERO GENERAL PUBLIC LICENSE' in license_text
    assert '13. Remote Network Interaction' in license_text
    assert len(license_text) > 34000


def test_raster_keeps_active_zero_and_low_elevations(tmp_path):
    values = np.array([[-.02, 0., 2.], [3., 4., 5.]], dtype=np.float64)
    active = np.array([[True, True, False], [True, False, True]])
    path = tmp_path / 'terrain.tif'
    transform = Affine(5., 0., 1574940., 0., -5., 5167275.)
    result = write_raster(path=path, values=values, active=active, transform=transform,
        dtype='float64', nodata=-9999., description='precision and mask control',
        provenance=dict(elevation_attribution='Control dataset'))
    assert result['maximum_conversion_error'] == 0.
    with rasterio.open(path) as saved:
        assert saved.tags()['vertical_datum'] == 'NZVD2016'
        assert saved.crs.to_epsg() == 2193
        assert saved.transform == transform
        assert saved.read(1)[0, 0] == -.02
        assert saved.read(1)[0, 1] == 0.
        assert saved.read(1)[0, 2] == -9999.
        np.testing.assert_array_equal(actual=saved.read_masks(1).astype(bool), desired=active)


def test_hydrograph_uses_exact_minutes_with_extra_initial_diagnostic(tmp_path):
    times = np.r_[1., np.arange(60., 259201., 60.)]
    ledger = [dict(seconds=float(time), outflow_m3=float(time**2), inflow_m3=0.) for time in times]
    records = {name: dict(ledger=ledger) for name in ('fv1', 'reference', 'li')}
    times, series = hydrographs(records=records, path=tmp_path / 'hydrographs.csv')
    assert times.size == 4320
    assert series['reference']['discharge'][0] == 60.
    assert series['reference']['discharge'][1] == 180.
    assert np.sum(series['reference']['discharge'] * 60.) == 259200.**2


def test_maximum_extent_classification_area():
    candidate = np.array([[.02, .2], [0., .2]])
    reference = np.array([[.02, 0.], [.2, .2]])
    active = np.array([[True, True], [True, False]])
    metrics = extent_statistics(candidate=candidate, reference=reference, active=active, threshold=.01, cell_area=25.)
    assert metrics['iou'] == 1. / 3.
    assert metrics['false_positive_area_m2'] == 25.
    assert metrics['false_negative_area_m2'] == 25.
