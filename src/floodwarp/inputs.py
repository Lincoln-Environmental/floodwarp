"""Direct SI inputs. Solver arrays always run west-to-east, south-to-north."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Terrain:
    elevation: np.ndarray
    active: np.ndarray
    dx: float
    dy: float
    metadata: dict


def terrain_from_array(*, elevation, dx, dy, row_order, vertical_datum,
                       active=None, nodata=None, crs=None, units="m"):
    bed = np.asarray(elevation, dtype=np.float64).copy()
    if bed.ndim != 2 or min(bed.shape) < 1:
        raise ValueError("Terrain must be a nonempty 2D array")
    if units != "m" or not vertical_datum:
        raise ValueError("Supply metre elevations and an explicit vertical datum (or 'local')")
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("dx and dy must be finite positive metres")
    if row_order not in ("south_to_north", "north_to_south"):
        raise ValueError("Explicit row_order must be south_to_north or north_to_south")
    valid = np.isfinite(bed)
    if nodata is not None:
        valid &= bed != nodata
    mask = valid.copy() if active is None else np.asarray(active)
    if mask.shape != bed.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("Active mask must match terrain and contain only 0/1")
    mask = mask.astype(bool)
    if np.any(mask & ~valid) or not mask.any():
        raise ValueError("Active domain must be nonempty and exclude all nodata elevations")
    if row_order == "north_to_south":
        bed, mask = bed[::-1].copy(), mask[::-1].copy()
    return Terrain(elevation=bed, active=mask, dx=float(dx), dy=float(dy),
        metadata=dict(crs=crs, vertical_datum=vertical_datum, units=units,
                      nodata=nodata, source_row_order=row_order, row_order="south_to_north"))


def read_terrain(*, path: Path, vertical_datum, elevation_units="m", dx=None, dy=None):
    import rasterio
    with rasterio.open(path) as dataset:
        transform = dataset.transform
        if dataset.count != 1 or transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise ValueError("Require a single-band, unrotated north-up raster")
        if dataset.crs is None or not dataset.crs.is_projected or dataset.crs.linear_units != "metre":
            raise ValueError("Raster CRS must be projected in metres; reproject explicitly before loading")
        if dx is not None and not np.isclose(dx, transform.a) or dy is not None and not np.isclose(dy, -transform.e):
            raise ValueError("Explicit resolution conflicts with raster metadata")
        tagged_unit = dataset.units[0] or dataset.tags().get('units')
        if tagged_unit and tagged_unit not in ('m', 'metre', 'meter'):
            raise ValueError("Raster elevation unit is not metres")
        tagged_datum = dataset.tags().get('vertical_datum')
        if tagged_datum and tagged_datum != vertical_datum:
            raise ValueError("Vertical datum conflicts with raster metadata")
        terrain = terrain_from_array(elevation=dataset.read(1), active=dataset.read_masks(1) > 0,
            dx=transform.a, dy=-transform.e, row_order="north_to_south",
            vertical_datum=vertical_datum, units=elevation_units, nodata=dataset.nodata,
            crs=dataset.crs.to_wkt())
        terrain.metadata.update(transform=list(transform), tags=dataset.tags(), source=str(path))
        return terrain


def read_aligned_field(*, path: Path, terrain: Terrain):
    import rasterio
    if "transform" not in terrain.metadata:
        raise ValueError("Raster fields require georeferenced terrain for alignment validation")
    with rasterio.open(path) as dataset:
        if (dataset.count != 1 or dataset.shape != terrain.elevation.shape or
            list(dataset.transform) != terrain.metadata['transform'] or
            dataset.crs is None or dataset.crs.to_wkt() != terrain.metadata['crs']):
            raise ValueError("Raster geometry/CRS must exactly match terrain")
        values = dataset.read(1)[::-1].copy()
        valid = dataset.read_masks(1)[::-1] > 0
        if np.any(terrain.active & (~valid | ~np.isfinite(values))):
            raise ValueError("Raster field has nodata within the active domain")
        return values


def write_raster(*, path: Path, terrain: Terrain, values, quantity, nodata=-9999.):
    import rasterio
    from affine import Affine
    field = np.asarray(values)
    if field.shape != terrain.elevation.shape or 'transform' not in terrain.metadata:
        raise ValueError("Output shape must match georeferenced terrain")
    with rasterio.open(path, 'w', driver='GTiff', width=field.shape[1], height=field.shape[0],
                       count=1, dtype='float64', crs=terrain.metadata['crs'],
                       transform=Affine(*terrain.metadata['transform'][:6]), nodata=nodata,
                       compress='deflate') as dataset:
        dataset.write(np.where(terrain.active, field, nodata)[::-1], 1)
        dataset.update_tags(**terrain.metadata.get('tags', {}))
        dataset.update_tags(vertical_datum=terrain.metadata['vertical_datum'], units='m', quantity=quantity)
        dataset.set_band_unit(1, 'm')


def field_array(*, values, terrain: Terrain, name, nonnegative=True):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 0 and array.shape != terrain.elevation.shape:
        raise ValueError(f"{name} must be scalar or match terrain; implicit broadcasting is ambiguous")
    array = np.broadcast_to(array, terrain.elevation.shape).copy()
    if not np.isfinite(array[terrain.active]).all() or nonnegative and np.any(array[terrain.active] < 0):
        raise ValueError(f"{name} contains invalid active values")
    return np.where(terrain.active, array, 0.)


def boundary_faces(*, terrain: Terrain):
    """Closed exterior and mask-hole walls; internal faces remain code 0."""
    active = terrain.active
    padded_x = np.pad(active, ((0, 0), (1, 1)))
    padded_y = np.pad(active, ((1, 1), (0, 0)))
    return ((padded_x[:, :-1] != padded_x[:, 1:]).astype(np.int32),
            (padded_y[:-1] != padded_y[1:]).astype(np.int32))


def validate_boundaries(*, terrain, qx, qy):
    defaults = boundary_faces(terrain=terrain)
    result = []
    for name, values, exterior in zip(('qx', 'qy'), (qx, qy), defaults, strict=True):
        values = np.asarray(values)
        if values.shape != exterior.shape or not np.isin(values, (0, 1, 2, 3)).all():
            raise ValueError(f"{name} requires face-shaped codes 0=internal, 1=closed, 2=free, 3=stage")
        if np.any((exterior == 1) & (values == 0)) or np.any((exterior == 0) & (values != 0)):
            raise ValueError("Every domain interface must be explicit; openings must border active/exterior cells")
        result.append(values.astype(np.int32))
    # LI uses directional codes for openings along inactive mask cells.
    x, y = result
    a = terrain.active
    for code, left_code, right_code in ((2, 7, 6), (3, 10, 5)):
        inner = x[:, 1:-1]
        inner[(inner == code) & a[:, :-1] & ~a[:, 1:]] = left_code
        inner[(inner == code) & ~a[:, :-1] & a[:, 1:]] = right_code
    for code, south_code, north_code in ((2, 9, 8), (3, 4, 11)):
        inner = y[1:-1]
        inner[(inner == code) & a[:-1] & ~a[1:]] = south_code
        inner[(inner == code) & ~a[:-1] & a[1:]] = north_code
    return x, y


def with_active_mask(*, terrain: Terrain, active):
    """Restrict the computational domain while preserving terrain metadata."""
    from dataclasses import replace
    mask = np.asarray(active)
    if mask.shape != terrain.active.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("Domain mask must match terrain and contain only 0/1")
    mask = mask.astype(bool)
    if not mask.any() or np.any(mask & ~terrain.active):
        raise ValueError("Domain mask must be nonempty and cannot activate terrain nodata")
    return replace(terrain, active=mask.copy(), metadata=dict(terrain.metadata))


def read_active_mask(*, path: Path, terrain: Terrain):
    """Load an aligned 0/1 raster; mask nodata is excluded from the domain."""
    import rasterio
    with rasterio.open(path) as dataset:
        if (dataset.count != 1 or dataset.shape != terrain.elevation.shape or
            list(dataset.transform) != terrain.metadata.get('transform') or
            dataset.crs is None or dataset.crs.to_wkt() != terrain.metadata.get('crs')):
            raise ValueError("Domain mask geometry/CRS must exactly match terrain")
        values = dataset.read(1)[::-1]
        valid = dataset.read_masks(1)[::-1] > 0
        if not np.isin(values[valid], (0, 1)).all():
            raise ValueError("Domain mask raster must contain only 0/1 or nodata")
        return with_active_mask(terrain=terrain, active=terrain.active & valid & (values == 1))
