"""Validated spatial coefficients and explicitly selected FV1 domain faces."""
import csv
import io
from pathlib import Path

import numpy as np

from floodwarp.inputs import boundary_faces, read_aligned_field, validate_boundaries


def aligned_coefficient(*, path, terrain, name, positive=False):
    try:
        values = read_aligned_field(path=path, terrain=terrain)
    except ValueError as error:
        raise ValueError(f'{name}: {error}. Export onto the exact DEM CRS, transform, shape and cell size; cover every active cell.') from error
    active_values = values[terrain.active]
    if np.any(active_values <= 0. if positive else active_values < 0.):
        comparison = 'positive' if positive else 'non-negative'
        raise ValueError(f'{name} must be finite and {comparison} on every active cell')
    return np.where(terrain.active, values, 0.)


def rate_factor(*, units):
    if units == 'mm_h':
        return 1. / 3.6e6
    if units == 'm_s':
        return 1.
    raise ValueError('Infiltration raster units must be mm_h or m_s')


def active_range(*, values, terrain):
    selected = np.broadcast_to(values, terrain.active.shape)[terrain.active]
    return [float(selected.min()), float(selected.max())]


def read_boundary_csv(*, path, terrain, stage_datum):
    """Unlisted interfaces stay closed; indices refer to the south-first model grid."""
    qx, qy = boundary_faces(terrain=terrain)
    sx, sy = np.zeros_like(qx, dtype=np.float64), np.zeros_like(qy, dtype=np.float64)
    seen = set()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['axis', 'row', 'col', 'type', 'stage_m']:
            raise ValueError('Boundary CSV header must be axis,row,col,type,stage_m')
        for line, row in enumerate(reader, start=2):
            try:
                if set(row) != set(reader.fieldnames):
                    raise ValueError('Unexpected columns; use decimal points for stage elevations')
                axis = row['axis'].strip().lower()
                index = (int(row['row']), int(row['col']))
                name = row['type'].strip().lower()
                if axis not in ('qx', 'qy') or name not in ('closed', 'free', 'stage'):
                    raise ValueError('Choose qx/qy and closed/free/stage')
                array, stage = (qx, sx) if axis == 'qx' else (qy, sy)
                if min(index) < 0 or index[0] >= array.shape[0] or index[1] >= array.shape[1]:
                    raise ValueError('Face index is outside the model grid')
                key = (axis, *index)
                if key in seen:
                    raise ValueError('Duplicate face; provide exactly one rule for each selected face')
                seen.add(key)
                code = {'closed': 1, 'free': 2, 'stage': 3}[name]
                if array[index] != 1:
                    raise ValueError('Selected face is not an active/domain interface')
                array[index] = code
                if name == 'stage':
                    value = float(row['stage_m'])
                    if not np.isfinite(value):
                        raise ValueError('Stage must be finite absolute elevation in metres')
                    stage[index] = value
                elif row['stage_m'].strip():
                    raise ValueError('stage_m must be empty for closed/free faces')
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise ValueError(f'Boundary CSV line {line}: {error}') from error
    if (np.any(qx == 3) or np.any(qy == 3)) and stage_datum != terrain.metadata['vertical_datum']:
        raise ValueError('Boundary stage datum must explicitly match the DEM vertical datum')
    validate_boundaries(terrain=terrain, qx=qx, qy=qy)
    return qx, qy, sx, sy


def boundary_template(*, terrain):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['axis', 'row', 'col', 'type', 'stage_m'])
    for axis, faces in zip(('qx', 'qy'), boundary_faces(terrain=terrain), strict=True):
        for row, col in np.argwhere(faces == 1):
            writer.writerow([axis, int(row), int(col), 'closed', ''])
    return stream.getvalue()


def numerical_options(*, profile):
    if profile == 'ordinary':
        return {}
    if profile == 'orton_bradley_matched':
        return dict(explicit_minimum_dt=1.e-6, explicit_maximum_dt=1., gravity=9.80665,
            dry_depth=.0001, cfl=.4, stage_boundary_momentum='discharge_extrapolation')
    raise ValueError('Choose ordinary or orton_bradley_matched numerical profile')
