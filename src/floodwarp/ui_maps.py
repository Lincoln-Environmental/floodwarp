"""Bounded map previews. Display reprojection never changes hydraulic inputs."""
import base64
import io

import numpy as np


def map_overlay(*, terrain, values, label, water=False, maximum_size=768):
    from affine import Affine
    from matplotlib import colormaps
    from PIL import Image
    from rasterio.transform import array_bounds, from_bounds
    from rasterio.warp import transform_bounds, reproject, Resampling

    shape = terrain.active.shape
    source_transform = Affine(*terrain.metadata['transform'][:6])
    bounds = array_bounds(height=shape[0], width=shape[1], transform=source_transform)
    west, south, east, north = transform_bounds(src_crs=terrain.metadata['crs'], dst_crs='EPSG:3857',
        left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3], densify_pts=21)
    if not np.isfinite([west, south, east, north]).all() or east <= west or north <= south:
        raise ValueError('This DEM cannot be displayed on the Web Mercator basemap')
    aspect = (east-west)/(north-south)
    width = max(1, int(min(maximum_size, maximum_size*aspect)))
    height = max(1, int(min(maximum_size, maximum_size/aspect)))
    target_transform = from_bounds(west=west, south=south, east=east, north=north, width=width, height=height)
    display = np.full((height, width), np.nan)
    source = np.where(terrain.active, values, np.nan)[::-1].copy()
    reproject(source=source, destination=display, src_transform=source_transform,
        src_crs=terrain.metadata['crs'], src_nodata=np.nan, dst_transform=target_transform,
        dst_crs='EPSG:3857', dst_nodata=np.nan, resampling=Resampling.nearest)
    finite = np.isfinite(display)
    active_values = np.asarray(values)[terrain.active]
    lower = 0. if water else float(active_values.min())
    upper = float(active_values.max())
    scale = max(upper-lower, np.finfo(np.float64).eps)
    normalized = np.clip((np.nan_to_num(display, nan=lower)-lower)/scale, 0., 1.)
    rgba = colormaps['Blues' if water else 'terrain'](normalized, bytes=True)
    rgba[..., 3] = np.where(finite & ((display > 0.) if water else True), 220, 0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(obj=rgba).save(fp=stream, format='PNG')
    lon_west, lat_south, lon_east, lat_north = transform_bounds(src_crs='EPSG:3857', dst_crs='EPSG:4326',
        left=west, bottom=south, right=east, top=north)
    return dict(png=base64.b64encode(stream.getvalue()).decode('ascii'),
        bounds=[[lat_south, lon_west], [lat_north, lon_east]], label=label, range=[lower, upper],
        display_crs='EPSG:3857', shape=[height, width], resampling='nearest; display only')


def boundary_features(*, terrain, qx, qy, maximum_faces=20000):
    from affine import Affine
    from pyproj import Transformer
    transform = Affine(*terrain.metadata['transform'][:6])
    converter = Transformer.from_crs(crs_from=terrain.metadata['crs'], crs_to='EPSG:4326', always_xy=True)
    rows = terrain.active.shape[0]
    features = []
    count = int(np.count_nonzero(qx >= 2)+np.count_nonzero(qy >= 2))
    for axis, codes in (('qx', qx), ('qy', qy)):
        for row, col in np.argwhere(codes >= 2):
            if len(features) >= maximum_faces:
                break
            if axis == 'qx':
                pixels = [(float(col), float(rows-row)), (float(col), float(rows-row-1))]
            else:
                pixels = [(float(col), float(rows-row)), (float(col+1), float(rows-row))]
            coordinates = []
            for pixel in pixels:
                east, north = transform @ pixel
                longitude, latitude = converter.transform(xx=east, yy=north)
                coordinates.append([longitude, latitude])
            features.append(dict(type='Feature', properties=dict(axis=axis, row=int(row), col=int(col),
                closure='stage' if codes[row, col] == 3 else 'free'),
                geometry=dict(type='LineString', coordinates=coordinates)))
    return dict(type='FeatureCollection', features=features, total_faces=count, truncated=count>len(features))


def save_result_maps(*, output, model, boundary_qx, boundary_qy):
    import json
    metadata = {}
    for key, values in (('terrain', model.terrain.elevation),
            ('maximum', model.solver.get_max_water_depth()), ('final', model.solver.get_water_depth())):
        label = 'Terrain elevation (m)' if key == 'terrain' else key.title()+' water depth (m)'
        overlay = map_overlay(terrain=model.terrain, values=values, label=label, water=key!='terrain')
        (output/(key+'_map.png')).write_bytes(base64.b64decode(overlay.pop('png')))
        overlay['file'] = key+'_map.png'
        metadata[key] = overlay
    metadata['boundaries'] = boundary_features(terrain=model.terrain, qx=boundary_qx, qy=boundary_qy)
    (output/'map.json').write_text(json.dumps(metadata, indent=2)+'\n')
