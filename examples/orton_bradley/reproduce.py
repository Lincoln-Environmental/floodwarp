"""Regenerate the complete Orton Bradley release GIS, figures and comparisons.

Standalone: only NumPy, Rasterio and Matplotlib are needed. No solver runs,
private directories, external executable or other repository is consulted.
"""

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import rasterio
from rasterio.transform import Affine
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LightSource, LogNorm, SymLogNorm, ListedColormap, BoundaryNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter


def file_hash(*, path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(*, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def payload_hash(*, value):
    copy = dict(value)
    copy.pop('payload_sha256', None)
    return hashlib.sha256(json.dumps(copy, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_sources(*, directory, provenance):
    for relative, expected in provenance['published_source_files'].items():
        if file_hash(path=directory / relative) != expected:
            raise ValueError('Source checksum differs: ' + relative)


def load_data(*, directory, provenance):
    verify_sources(directory=directory, provenance=provenance)
    with np.load(file=directory / 'inputs/comparison_inputs.npz', allow_pickle=False) as saved:
        inputs = {name: saved[name].copy() for name in saved.files}
    active, bed = inputs['active'].astype(bool), inputs['bed']
    if active.shape != (1123, 843) or np.count_nonzero(active) != 559985:
        raise ValueError('Unexpected complete model domain')
    grid = provenance['grid']
    transform = Affine(*grid['transform'][:6])
    with (directory / 'inputs/terrain.asc').open() as stream:
        header = {key.lower(): float(value) for key, value in (stream.readline().split() for _ in range(6))}
    expected = Affine(header['cellsize'], 0., header['xllcorner'], 0., -header['cellsize'],
                      header['yllcorner'] + header['nrows'] * header['cellsize'])
    if transform != expected or grid['epsg'] != 2193 or grid['vertical_datum'] != 'NZVD2016':
        raise ValueError('Georeferencing/datum identity differs')
    with rasterio.open(directory / 'inputs/original_model_dem.tif') as original:
        if original.transform != transform or original.crs.to_epsg() != 2193 or original.shape != active.shape:
            raise ValueError('Original DEM grid differs')
        np.testing.assert_array_equal(actual=original.read(1)[active].astype(np.float64), desired=bed[active])
        if not np.all(original.read_masks(1)[active] != 0):
            raise ValueError('Active model cells are nodata in the original DEM')
    with rasterio.open(directory / 'inputs/original_active_mask.tif') as original_mask:
        if original_mask.transform != transform or original_mask.crs.to_epsg() != 2193:
            raise ValueError('Original active-mask grid differs')
        np.testing.assert_array_equal(actual=original_mask.read(1).astype(bool), desired=active)
    records, fields = {}, {}
    for method in ('fv1', 'reference', 'li'):
        record = json.loads((directory / 'results' / (method + '_record.json')).read_text())
        field_path = directory / 'results' / (method + '_fields.npz')
        if record['payload_sha256'] != payload_hash(value=record) or record['fields_sha256'] != file_hash(path=field_path):
            raise ValueError('Completed record/result checksum mismatch: ' + method)
        if not record['physical_completion_verified'] or record['time_s'] != 259200.:
            raise ValueError('Incomplete 72-hour result')
        with np.load(file=field_path, allow_pickle=False) as saved:
            # Retained solver fields are south-first. This declared orientation
            # conversion places them on the north-first original DEM grid.
            fields[method] = {name: saved[name][::-1].copy() for name in ('depth', 'maximum')}
        for values in fields[method].values():
            if values.shape != active.shape or not np.isfinite(values[active]).all() or np.any(values[active] < 0.):
                raise ValueError('Invalid completed depth field')
        records[method] = record
    return inputs, records, fields, transform


def write_raster(*, path, values, active, transform, dtype, nodata, description, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = np.asarray(values, dtype=dtype)
    error = float(np.max(np.abs(converted[active].astype(np.float64) - values[active])))
    output = np.where(active, converted, nodata).astype(dtype)
    with rasterio.open(path, 'w', driver='GTiff', width=active.shape[1], height=active.shape[0],
        count=1, dtype=dtype, crs='EPSG:2193', transform=transform, nodata=nodata,
        tiled=True, compress='DEFLATE', predictor=3 if np.issubdtype(np.dtype(dtype), np.floating) else 2) as target:
        target.write(output, 1)
        target.write_mask(active.astype(np.uint8) * 255)
        target.set_band_description(1, description)
        target.set_band_unit(1, 'class' if dtype == 'uint8' else 'm')
        target.update_tags(AREA_OR_POINT='Area', vertical_datum='NZVD2016',
            ELEVATION_UNITS='m', units='class' if dtype == 'uint8' else 'm',
            SITE='Orton Bradley Park, Banks Peninsula, Canterbury, New Zealand',
            SOFTWARE='FloodWarp v0.1.0', SOURCE_PROVENANCE='provenance.json',
            DESCRIPTION=description, ATTRIBUTION=provenance['elevation_attribution'],
            DATA_LICENSE='CC-BY-4.0 for attributed elevation data; see DATA_LICENSE.md',
            MAXIMUM_CONVERSION_ERROR_M=str(error))
    with rasterio.open(path) as check:
        assert check.transform == transform and check.crs.to_epsg() == 2193
        np.testing.assert_array_equal(actual=check.read_masks(1).astype(bool), desired=active)
        np.testing.assert_array_equal(actual=check.read(1)[active], desired=converted[active])
    return dict(sha256=file_hash(path=path), dtype=dtype, nodata=nodata, maximum_conversion_error=error)


def boundaries(*, inputs, transform):
    active = inputs['active'].astype(bool)
    ports = inputs['ports'].astype(bool)
    rows, columns = np.nonzero(ports)
    if rows.size != 212 or np.any(rows + 1 >= active.shape[0]) or not np.all(active[rows + 1, columns]) or np.any(active[rows, columns]):
        raise ValueError('Coastal north-contact ownership differs')
    donor = np.zeros_like(active)
    donor[rows + 1, columns] = True
    padded_x = np.pad(array=active, pad_width=((0, 0), (1, 1)))
    padded_y = np.pad(array=active, pad_width=((1, 1), (0, 0)))
    qx = (padded_x[:, :-1] ^ padded_x[:, 1:]).astype(np.int32)
    qy = (padded_y[:-1] ^ padded_y[1:]).astype(np.int32)
    qy[rows + 1, columns] = 3
    features = []
    for index, (row, column) in enumerate(zip(rows, columns)):
        west, north = transform * (int(column), int(row) + 1)
        east, _ = transform * (int(column) + 1, int(row) + 1)
        features.append(dict(type='Feature', properties=dict(id=index, kind='prescribed_stage', stage_m=0.,
            vertical_datum='NZVD2016', physical_donor_row_north_first=int(row) + 1, column=int(column)),
            geometry=dict(type='LineString', coordinates=[[west, north], [east, north]])))
    return donor, qx, qy, dict(type='FeatureCollection',
        crs=dict(type='name', properties=dict(name='EPSG:2193')), features=features)


def extent_statistics(*, candidate, reference, active, threshold, cell_area):
    a, b = (candidate >= threshold) & active, (reference >= threshold) & active
    union, intersection = a | b, a & b
    return dict(threshold_m=threshold, iou=float(intersection.sum() / union.sum()) if union.any() else 1.,
        candidate_area_m2=float(a.sum() * cell_area), reference_area_m2=float(b.sum() * cell_area),
        false_positive_area_m2=float(np.count_nonzero(a & ~b) * cell_area),
        false_negative_area_m2=float(np.count_nonzero(b & ~a) * cell_area))


def minute_ledger(*, record):
    # The external solver also retained an initial 1-second diagnostic. The
    # cumulative samples at exact minute ends define the required 60 s means.
    rows = [row for row in record['ledger'] if row['seconds'] > 0. and row['seconds'] % 60. == 0.]
    times = np.asarray([row['seconds'] for row in rows])
    if not np.array_equal(times, np.arange(60., 259200. + 1., 60.)):
        raise ValueError('Missing exact minute-end cumulative diagnostics')
    return rows


def hydrographs(*, records, path):
    common = None
    series = {}
    for method, record in records.items():
        ledger = minute_ledger(record=record)
        times = np.asarray([row['seconds'] for row in ledger])
        if times.size != 4320 or not np.array_equal(times, np.arange(60., 259200. + 1., 60.)):
            raise ValueError('Hydrograph is not the complete common 60-second record')
        if common is not None:
            np.testing.assert_array_equal(actual=times, desired=common)
        common = times
        volume = np.asarray([row['outflow_m3'] - row['inflow_m3'] for row in ledger])
        discharge = np.diff(np.r_[0., volume]) / 60.
        series[method] = dict(volume=volume, discharge=discharge)
    names = ['interval_start_s', 'interval_end_s', 'interval_mid_h']
    labels = dict(fv1='floodwarp_fv1', reference='lisflood_fv1', li='floodwarp_li')
    for method in records:
        names.extend([labels[method] + '_mean_net_outlet_m3_s', labels[method] + '_cumulative_net_export_m3'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index, time in enumerate(common):
            values = [time - 60., time, (time - 30.) / 3600.]
            for method in records:
                values.extend([series[method]['discharge'][index], series[method]['volume'][index]])
            writer.writerow(values)
    return common, series


def compare(*, records, fields, active, thresholds, cell_area, wet_union_threshold):
    reference = fields['reference']
    ref_ledger = records['reference']['ledger'][-1]
    ref_volume = np.asarray([row['outflow_m3'] - row['inflow_m3'] for row in minute_ledger(record=records['reference'])])
    ref_q = np.diff(np.r_[0., ref_volume]) / 60.
    result = {}
    for method, record in records.items():
        candidate = fields[method]
        error = candidate['depth'][active] - reference['depth'][active]
        wet = active & ((candidate['depth'] >= wet_union_threshold) | (reference['depth'] >= wet_union_threshold))
        envelope = candidate['maximum'][active] - reference['maximum'][active]
        volume = np.asarray([row['outflow_m3'] - row['inflow_m3'] for row in minute_ledger(record=record)])
        q = np.diff(np.r_[0., volume]) / 60.
        peak = int(np.argmax(q))
        ref_peak = int(np.argmax(ref_q))
        ledger = record['ledger'][-1]
        rmse = float(np.sqrt(np.mean((q - ref_q)**2)))
        result[method] = dict(solver=record['solver'], final_depth_rmse_m=float(np.sqrt(np.mean(error**2))),
            final_wet_union_rmse_m=float(np.sqrt(np.mean((candidate['depth'][wet] - reference['depth'][wet])**2))) if wet.any() else 0.,
            final_wet_union_threshold_m=wet_union_threshold, maximum_depth_envelope_rmse_m=float(np.sqrt(np.mean(envelope**2))),
            maximum_depth_envelope_maximum_absolute_error_m=float(np.max(np.abs(envelope))),
            maximum_depth_signed_difference_min_m=float(envelope.min()), maximum_depth_signed_difference_max_m=float(envelope.max()),
            maximum_depth_absolute_error_median_m=float(np.median(np.abs(envelope))),
            maximum_depth_absolute_error_p95_m=float(np.quantile(np.abs(envelope), .95)),
            maximum_depth_fraction_within_010m=float(np.mean(np.abs(envelope) < .1)),
            maximum_depth_m=float(candidate['maximum'][active].max()),
            maximum_event_extent={str(t): extent_statistics(candidate=candidate['maximum'], reference=reference['maximum'],
                active=active, threshold=t, cell_area=cell_area) for t in thresholds},
            hydrograph_rmse_m3_s=rmse, hydrograph_nrmse=rmse / float(ref_q.max()),
            peak_discharge_m3_s=float(q[peak]), peak_interval_start_h=peak / 60., peak_interval_end_h=(peak + 1) / 60.,
            peak_discharge_error_m3_s=float(q[peak] - ref_q[ref_peak]),
            peak_discharge_relative_error=float(q[peak] / ref_q[ref_peak] - 1.),
            peak_interval_end_timing_error_s=float((peak - ref_peak) * 60.),
            cumulative_net_export_m3=float(volume[-1]), cumulative_net_export_difference_m3=float(volume[-1] - ref_volume[-1]),
            infiltration_m3=float(ledger['infiltration_m3']), infiltration_difference_m3=float(ledger['infiltration_m3'] - ref_ledger['infiltration_m3']),
            final_storage_m3=float(ledger['storage_m3']), rainfall_m3=float(ledger['rain_m3']),
            final_water_balance_residual_m3=float(ledger['residual_m3']),
            maximum_absolute_water_balance_residual_m3=max(abs(row['residual_m3']) for row in record['ledger']),
            hydraulic_seconds=record['hydraulic_seconds'], end_to_end_seconds=record['end_to_end_seconds'],
            accepted_timesteps=record['steps'], dt_min_s=record['dt_min_s'], dt_mean_s=record['dt_mean_s'], dt_max_s=record['dt_max_s'])
    return result


def map_axis(*, axis, active, extent, transform, ports, title, decorations):
    axis.set_title(label=title, fontsize=12, weight='semibold')
    axis.contour(active.astype(float), levels=[.5], extent=extent, origin='upper', colors='#303e44', linewidths=.35)
    rows, columns = np.nonzero(ports)
    axis.scatter(transform.c + (columns + .5) * transform.a,
                 transform.f + (rows + 1.) * transform.e, s=3., c='#db4b26', marker='s', linewidths=0., zorder=5)
    axis.set_xlabel(xlabel='NZTM2000 easting [m]')
    axis.set_ylabel(ylabel='NZTM2000 northing [m]')
    axis.set_aspect(aspect='equal')
    axis.set_xlim(left=extent[0], right=extent[1])
    axis.set_ylim(bottom=extent[2], top=extent[3])
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(b=False)
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axis.tick_params(axis='x', rotation=25, labelsize=8)
    axis.tick_params(axis='y', labelsize=8)
    if decorations:
        axis.annotate(text='N', xy=(.93, .95), xytext=(.93, .85), xycoords='axes fraction',
                      arrowprops=dict(facecolor='#162d37', width=2, headwidth=9), ha='center', va='center', weight='bold')
        x, y = extent[0] + .07 * (extent[1] - extent[0]), extent[2] + .06 * (extent[3] - extent[2])
        axis.plot([x, x + 1000.], [y, y], color='#152b35', linewidth=3)
        axis.text(x=x + 500., y=y + 90., s='1 km', ha='center', fontsize=9)


def make_figure(*, size):
    figure = Figure(figsize=size, layout='constrained', facecolor='white')
    FigureCanvasAgg(figure=figure)
    return figure


def map_figures(*, inputs, fields, transform, statistics, output, dpi, depth_minimum):
    active, bed = inputs['active'].astype(bool), inputs['bed']
    extent = (transform.c, transform.c + transform.a * active.shape[1],
              transform.f + transform.e * active.shape[0], transform.f)
    # Outside-domain values are used only for background hillshade and remain
    # masked in every quantitative colour layer.
    shade = LightSource(azdeg=315., altdeg=40.).hillshade(elevation=bed, dx=transform.a, dy=-transform.e)
    terrain = np.ma.masked_where(~active, bed)
    figure = make_figure(size=(9., 11.))
    axis = figure.add_subplot(111)
    image = axis.imshow(terrain, extent=extent, origin='upper', cmap='terrain', interpolation='nearest')
    axis.imshow(np.ma.masked_where(~active, shade), extent=extent, origin='upper', cmap='gray', alpha=.27)
    map_axis(axis=axis, active=active, extent=extent, transform=transform, ports=inputs['ports'],
        title='Orton Bradley Park — 5 m model terrain', decorations=True)
    figure.colorbar(mappable=image, ax=axis, label='Elevation [m NZVD2016]', shrink=.76)
    axis.legend(handles=[Line2D([0], [0], color='#303e44', lw=.8, label='Active hydraulic boundary'),
                         Line2D([0], [0], marker='s', color='none', markerfacecolor='#db4b26', label='Prescribed coastal outlets')], loc='upper left', fontsize=8)
    figure.supxlabel(t='EPSG:2193 · NZVD2016 · LINZ elevation data CC BY 4.0 · conditioned model terrain', fontsize=9)
    figure.savefig(fname=output / 'orton_bradley_dem.png', dpi=dpi)
    maximum = max(float(fields[name]['maximum'][active].max()) for name in ('fv1', 'reference'))
    norm = LogNorm(vmin=depth_minimum, vmax=maximum)
    figure = make_figure(size=(9., 11.))
    axis = figure.add_subplot(111)
    axis.imshow(np.ma.masked_where(~active, shade), extent=extent, cmap='gray', origin='upper', vmin=0., vmax=1., alpha=.35)
    image = axis.imshow(np.ma.masked_where(~active | (fields['fv1']['maximum'] < depth_minimum), fields['fv1']['maximum']),
        extent=extent, origin='upper', cmap='viridis', norm=norm)
    map_axis(axis=axis, active=active, extent=extent, transform=transform, ports=inputs['ports'],
        title='Orton Bradley Park — FV1 maximum event depth', decorations=True)
    figure.colorbar(mappable=image, ax=axis, label='Maximum depth [m] · logarithmic scale', shrink=.76)
    figure.supxlabel(t='Complete 72-hour event · values below 0.001 m transparent · no water removed from results', fontsize=9)
    figure.savefig(fname=output / 'orton_bradley_fv1_max_depth.png', dpi=dpi)
    error = fields['fv1']['maximum'] - fields['reference']['maximum']
    limit = float(np.max(np.abs(error[active])))
    difference_norm = Normalize(vmin=-limit, vmax=limit)
    palette = ListedColormap(['#f2f3f2', '#278578', '#df7b31', '#7970ad'])
    for threshold in (.01, .10):
        figure = make_figure(size=(12., 14.))
        axes = figure.subplots(nrows=2, ncols=2)
        for axis, method, label in ((axes[0, 0], 'fv1', 'FloodWarp FV1'), (axes[0, 1], 'reference', 'External LISFLOOD-FP FV1')):
            image = axis.imshow(np.ma.masked_where(~active | (fields[method]['maximum'] < depth_minimum), fields[method]['maximum']),
                                extent=extent, origin='upper', cmap='viridis', norm=norm)
            map_axis(axis=axis, active=active, extent=extent, transform=transform, ports=inputs['ports'], title=label + ' · maximum depth', decorations=False)
        figure.colorbar(mappable=image, ax=list(axes[0]), label='Maximum depth [m] · common logarithmic scale', shrink=.8)
        image = axes[1, 0].imshow(np.ma.masked_where(~active, error), extent=extent, origin='upper', cmap='RdBu_r', norm=difference_norm)
        map_axis(axis=axes[1, 0], active=active, extent=extent, transform=transform, ports=inputs['ports'],
                 title='Signed envelope difference · FloodWarp − LISFLOOD', decorations=False)
        figure.colorbar(mappable=image, ax=axes[1, 0], label='Δ maximum depth [m] · linear; red positive', shrink=.75)
        a, b = fields['fv1']['maximum'] >= threshold, fields['reference']['maximum'] >= threshold
        classes = np.where(a & b, 1, np.where(a & ~b, 2, np.where(~a & b, 3, 0)))
        axes[1, 1].imshow(np.ma.masked_where(~active, classes), extent=extent, origin='upper', cmap=palette, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], palette.N))
        map_axis(axis=axes[1, 1], active=active, extent=extent, transform=transform, ports=inputs['ports'],
                 title=f'Maximum inundation ≥ {threshold:.2f} m', decorations=False)
        axes[1, 1].legend(handles=[Patch(color=palette(index), label=label) for index, label in enumerate(
            ('Both dry', 'Both inundated', 'FloodWarp only', 'LISFLOOD only'))], loc='lower right', fontsize=8)
        metrics = statistics['fv1']
        iou = metrics['maximum_event_extent'][str(threshold)]['iou']
        figure.suptitle(t='Orton Bradley Park — complete 5 m domain · 72-hour maximum envelopes', fontsize=15)
        figure.supxlabel(t=f'Δhₘₐₓ = maxₜ h FloodWarp − maxₜ h LISFLOOD; range [{error[active].min():.6f}, {error[active].max():.6f}] m\n'
            f'Envelope RMSE {metrics["maximum_depth_envelope_rmse_m"]:.6g} m · max |Δ| {limit:.6g} m · maximum-event IoU {iou:.6f}\n'
            'Full error range shown, no clipping · linear signed difference scale · EPSG:2193 / NZVD2016', fontsize=9)
        suffix = '001m' if threshold == .01 else '010m'
        figure.savefig(fname=output / ('orton_bradley_fv1_comparison_' + suffix + '.png'), dpi=dpi)
    figure = make_figure(size=(17., 8.))
    axes = figure.subplots(nrows=1, ncols=3)
    li_max = float(fields['li']['maximum'][active].max())
    for axis, method, label in ((axes[0], 'li', 'FloodWarp LI'), (axes[1], 'reference', 'External LISFLOOD-FP FV1')):
        image = axis.imshow(np.ma.masked_where(~active | (fields[method]['maximum'] < depth_minimum), fields[method]['maximum']),
            extent=extent, origin='upper', cmap='viridis', norm=LogNorm(vmin=depth_minimum, vmax=max(maximum, li_max)))
        map_axis(axis=axis, active=active, extent=extent, transform=transform, ports=inputs['ports'],
                 title=label + ' · maximum depth', decorations=False)
    figure.colorbar(mappable=image, ax=list(axes[:2]), label='Maximum depth [m] · shared log scale', shrink=.8)
    delta = fields['li']['maximum'] - fields['reference']['maximum']
    limit = float(np.max(np.abs(delta[active])))
    image = axes[2].imshow(np.ma.masked_where(~active, delta), extent=extent, origin='upper', cmap='RdBu_r',
        norm=Normalize(vmin=-limit, vmax=limit))
    map_axis(axis=axes[2], active=active, extent=extent, transform=transform, ports=inputs['ports'], title='LI − external FV1 · linear difference', decorations=False)
    figure.colorbar(mappable=image, ax=axes[2], label='Δ maximum depth [m] · linear, full range', shrink=.8)
    figure.suptitle(t='Orton Bradley Park — LI and external FV1 event maxima')
    figure.supxlabel(t=f'Active-cell |Δ|: median {statistics["li"]["maximum_depth_absolute_error_median_m"]:.6g} m · '
        f'95th percentile {statistics["li"]["maximum_depth_absolute_error_p95_m"]:.6g} m · '
        f'{100. * statistics["li"]["maximum_depth_fraction_within_010m"]:.2f}% below 0.10 m\n'
        f'Envelope RMSE {statistics["li"]["maximum_depth_envelope_rmse_m"]:.6g} m · max |Δ| {limit:.6g} m · '
        'localized channel maxima in the retained LI run require investigation; no error clipping', fontsize=9)
    figure.savefig(fname=output / 'orton_bradley_li_comparison.png', dpi=dpi)


def hydrograph_figure(*, times, series, statistics, output, dpi):
    figure = make_figure(size=(12., 11.))
    axes = figure.subplots(nrows=3, ncols=1)
    labels = dict(fv1='FloodWarp FV1', reference='LISFLOOD-FP FV1', li='FloodWarp LI')
    colours = dict(fv1='#116f97', reference='#d0662f', li='#756094')
    hours = (times - 30.) / 3600.
    for method in ('reference', 'fv1', 'li'):
        q = series[method]['discharge']
        for axis in axes[:2]:
            axis.plot(hours, q, label=labels[method], color=colours[method], linewidth=1.6 if method != 'li' else 1.,
                      linestyle=':' if method == 'reference' else '--' if method == 'li' else '-',
                      zorder=4 if method == 'reference' else 3 if method == 'fv1' else 2)
        axes[2].plot(hours, series[method]['volume'] / 1.e6, label=labels[method], color=colours[method])
    peak_hour = statistics['reference']['peak_interval_end_h']
    axes[0].set_xlim(left=0., right=72.)
    axes[0].set_title(label='Full 72-hour event · actual 60-second interval-mean net outlet discharge')
    axes[1].set_xlim(left=peak_hour - 2., right=peak_hour + 2.)
    axes[1].set_title(label='Peak detail · values placed at interval midpoints')
    for axis in axes[:2]:
        axis.set_ylabel(ylabel='Discharge [m³/s]')
    axes[2].set_xlim(left=0., right=72.)
    axes[2].set_ylabel(ylabel='Cumulative net export [million m³]')
    for axis in axes:
        axis.set_xlabel(xlabel='Time from simulation start [h]')
        axis.grid(visible=True, alpha=.2)
        axis.legend(loc='upper right', fontsize=9)
    figure.suptitle(t='Orton Bradley Park — full-domain outlet hydrographs', fontsize=16)
    figure.supxlabel(t=' · '.join(labels[method] + f': peak {statistics[method]["peak_discharge_m3_s"]:.3f} m³/s, interval ending {statistics[method]["peak_interval_end_h"]:.3f} h'
        for method in ('reference', 'fv1', 'li')), fontsize=9)
    figure.savefig(fname=output / 'orton_bradley_hydrographs_72h.png', dpi=dpi)


def report_markdown(*, statistics, provenance, path):
    lines = ['# Orton Bradley Park — complete 5 m, 72-hour comparison', '',
        'Maximum inundation is calculated directly from each pointwise maximum-depth envelope. '
        'Δh_max is FloodWarp FV1 maximum minus LISFLOOD-FP FV1 maximum; it is not a maximum instantaneous difference.', '',
        f'Grid: 1123 × 843 at 5 m; {provenance["grid"]["active_cells"]:,} physical cells; '
        f'active area {provenance["grid"]["active_area_km2"]:.6f} km²; raster footprint {provenance["grid"]["raster_area_km2"]:.6f} km². '
        'The model domain is not asserted to contain the entire geographical catchment.', '',
        '| Metric | LISFLOOD-FP FV1 | FloodWarp FV1 | FloodWarp LI |', '|---|---:|---:|---:|']
    metrics = [('Hydraulic runtime [s]', 'hydraulic_seconds'), ('End-to-end component runtime [s]', 'end_to_end_seconds'),
        ('Accepted timesteps', 'accepted_timesteps'), ('Peak 60 s mean discharge [m³/s]', 'peak_discharge_m3_s'),
        ('Peak interval ending [h]', 'peak_interval_end_h'), ('Final-depth RMSE [m]', 'final_depth_rmse_m'),
        ('Final wet-union RMSE [m]', 'final_wet_union_rmse_m'), ('Maximum-envelope RMSE [m]', 'maximum_depth_envelope_rmse_m'),
        ('Maximum-envelope max absolute error [m]', 'maximum_depth_envelope_maximum_absolute_error_m'),
        ('Median maximum-envelope absolute error [m]', 'maximum_depth_absolute_error_median_m'),
        ('95th percentile maximum-envelope absolute error [m]', 'maximum_depth_absolute_error_p95_m'),
        ('Active-cell fraction with envelope error below 0.10 m', 'maximum_depth_fraction_within_010m'),
        ('Hydrograph RMSE [m³/s]', 'hydrograph_rmse_m3_s'), ('Hydrograph NRMSE / reference peak', 'hydrograph_nrmse'),
        ('Peak-discharge relative error', 'peak_discharge_relative_error'), ('Peak timing error [s]', 'peak_interval_end_timing_error_s'),
        ('Cumulative net export [m³]', 'cumulative_net_export_m3'), ('Export difference [m³]', 'cumulative_net_export_difference_m3'),
        ('Infiltration [m³]', 'infiltration_m3'), ('Infiltration difference [m³]', 'infiltration_difference_m3'),
        ('Final mass residual [m³]', 'final_water_balance_residual_m3')]
    for label, key in metrics:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{statistics[method][key]:.9g}' for method in ('reference', 'fv1', 'li')) + ' |')
    lines.extend(['', '| Maximum-event threshold | Method | IoU | False-positive area [m²] | False-negative area [m²] |', '|---|---|---:|---:|---:|'])
    for threshold in (.01, .1):
        for method in ('fv1', 'li'):
            m = statistics[method]['maximum_event_extent'][str(threshold)]
            lines.append(f'| {threshold:g} m | {statistics[method]["solver"]} | {m["iou"]:.9f} | {m["false_positive_area_m2"]:.0f} | {m["false_negative_area_m2"]:.0f} |')
    lines.extend(['', 'Final-depth wet union uses 0.001 m. RMSEs use active physical cells only; outside-domain nodata is excluded. '
        'GeoTIFFs use FP64 and lossless compression; active values have zero conversion error. Active dry cells remain numeric zero.', '',
        'The reference is an independent adapted external LISFLOOD-FP FV1 run, with the retained source/binary hashes in each record. '
        'The matched FloodWarp FV1 benchmark uses contact HLL, FP64, CFL 0.4, maximum dt 1 s, gravity 9.80665 m/s², dry momentum threshold '
        '0.0001 m and explicitly discharge-extrapolated prescribed-stage ghosts. Public FV1 defaults use velocity extrapolation. '
        'The benchmark does not qualify every closure or flood scenario.', '',
        'Rain is the established spatially scaled HIRDS v4 100-year historical design event over 48 hours, followed by recession to 72 hours. '
        'Fine input arrays preserve the original PWL forcing, spatial multipliers, Manning field, exponential infiltration f0/fd/k and 212 '
        'north-facing stage-0 coastal contacts. Soil and roughness mappings are modelling assumptions, not site calibration.', '',
        'Timing records describe one completed matched run per solver on the GPU named in provenance.json. End-to-end component timers include '
        'diagnostic/checkpoint work and exclude some manifest/checksum publication and shared input loading; they are not independent fresh wall-clock repetitions. '
        'No full simulations were rerun to create this release.', '',
        'Elevation credit: ' + provenance['elevation_attribution'], '',
        'Software: Copyright Patrick Durney and Lincoln Agritech Ltd; AGPL-3.0-only. Elevation data retain CC BY 4.0. '
        'Full source hashes, original conditioning provenance, numerical identities and hardware are in provenance.json and the checksummed records.', ''])
    path.write_text('\n'.join(lines))


def reproduce(*, directory, output, figures, dpi, depth_minimum, thresholds, wet_union_threshold):
    provenance = json.loads((directory / 'provenance.json').read_text())
    inputs, records, fields, transform = load_data(directory=directory, provenance=provenance)
    active = inputs['active'].astype(bool)
    donor, qx, qy, geometry = boundaries(inputs=inputs, transform=transform)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    layers = dict(orton_bradley_dem_5m=inputs['bed'],
        orton_bradley_floodwarp_fv1_max_depth_5m=fields['fv1']['maximum'],
        orton_bradley_lisflood_fv1_max_depth_5m=fields['reference']['maximum'],
        orton_bradley_fv1_minus_lisflood_max_depth_5m=fields['fv1']['maximum'] - fields['reference']['maximum'],
        orton_bradley_floodwarp_li_max_depth_5m=fields['li']['maximum'],
        orton_bradley_li_minus_lisflood_max_depth_5m=fields['li']['maximum'] - fields['reference']['maximum'])
    exports = {}
    for name, values in layers.items():
        exports[name + '.tif'] = write_raster(path=output / (name + '.tif'), values=values, active=active,
            transform=transform, dtype='float64', nodata=-9999., description=name, provenance=provenance)
    classes = dict(orton_bradley_active_mask_5m=active.astype(np.uint8), orton_bradley_coastal_outlet_cells_5m=donor.astype(np.uint8))
    for threshold in thresholds:
        suffix = '001m' if threshold == .01 else '010m'
        a, b = fields['fv1']['maximum'] >= threshold, fields['reference']['maximum'] >= threshold
        classes['orton_bradley_fv1_max_extent_' + suffix] = a.astype(np.uint8)
        classes['orton_bradley_lisflood_max_extent_' + suffix] = b.astype(np.uint8)
        classes['orton_bradley_max_extent_class_' + suffix] = np.where(a & b, 1, np.where(a & ~b, 2, np.where(~a & b, 3, 0))).astype(np.uint8)
    for name, values in classes.items():
        exports[name + '.tif'] = write_raster(path=output / (name + '.tif'), values=values, active=active,
            transform=transform, dtype='uint8', nodata=255, description=name, provenance=provenance)
    write_json(path=output / 'orton_bradley_coastal_boundaries_2193.geojson', value=geometry)
    np.savez_compressed(file=output / 'orton_bradley_boundary_faces.npz', qx_north_first=qx, qy_north_first=qy)
    times, series = hydrographs(records=records, path=output / 'orton_bradley_hydrographs_72h.csv')
    statistics = compare(records=records, fields=fields, active=active, thresholds=thresholds,
                         cell_area=abs(transform.a * transform.e), wet_union_threshold=wet_union_threshold)
    write_json(path=output / 'comparison.json', value=dict(metrics=statistics, hardware=provenance['hardware'],
        signed_difference='maximum_depth_floodwarp_fv1 - maximum_depth_lisflood_fv1', extent_classes=dict(both_dry=0, both_wet=1, floodwarp_only=2, lisflood_only=3)))
    report_markdown(statistics=statistics, provenance=provenance, path=output / 'COMPARISON.md')
    map_figures(inputs=inputs, fields=fields, transform=transform, statistics=statistics, output=figures,
                dpi=dpi, depth_minimum=depth_minimum)
    hydrograph_figure(times=times, series=series, statistics=statistics, output=figures, dpi=dpi)
    write_json(path=output / 'raster_manifest.json', value=exports)
    samples = []
    locations = np.argwhere(active)
    for index in (0, locations.shape[0] // 2, locations.shape[0] - 1):
        row, column = locations[index]
        x, y = transform * (int(column) + .5, int(row) + .5)
        samples.append(dict(row_north_first=int(row), column=int(column), easting=x, northing=y, elevation_m=float(inputs['bed'][row, column])))
    write_json(path=output / 'verification.json', value=dict(shape=list(active.shape), active_cells=int(active.sum()),
        bounds=provenance['grid']['bounds'], crs='EPSG:2193', vertical_datum='NZVD2016',
        authoritative_active_elevation_maximum_error_m=0., samples=samples,
        source_row_order='north_to_south', retained_solver_row_order='south_to_north',
        resampling='none', vertical_adjustment_m=0., dem_grid_unchanged=True,
        coastal_faces=len(geometry['features']), reproduction_script_sha256=file_hash(path=Path(__file__))))


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    output = directory / 'outputs'
    figures = directory / 'figures'
    dpi = 180
    depth_minimum = .001
    thresholds = (.01, .10)
    wet_union_threshold = .001
    reproduce(directory=directory, output=output, figures=figures, dpi=dpi,
              depth_minimum=depth_minimum,
              thresholds=thresholds, wet_union_threshold=wet_union_threshold)
