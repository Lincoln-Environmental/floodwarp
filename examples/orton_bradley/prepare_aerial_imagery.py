"""Crop open LINZ aerial COGs onto the existing DEM grid for display only."""
from pathlib import Path
import gzip
import hashlib
import json
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from PIL import Image


def read_metadata(*, url):
    with urlopen(url=url, timeout=30) as response:
        payload=response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def atomic_bytes(*, path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_bytes(payload)
    temporary.replace(target=path)


def prepare(*, dataset, output, collection_url, tiles, resampling):
    identity=json.loads((dataset/'metadata/dataset.json').read_text())
    with np.load(file=dataset/'metadata/terrain.npz',allow_pickle=False) as source:
        active=source['active']
    collection, collection_hash=read_metadata(url=collection_url)
    if collection['license']!='CC-BY-4.0':
        raise ValueError('Unexpected imagery licence')
    transform=Affine(*identity['grid']['transform'][:6])
    ny,nx=active.shape
    rgb=np.zeros(shape=(ny,nx,3),dtype=np.uint8)
    covered=np.zeros(shape=active.shape,dtype=bool)
    sources=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT=30):
        for tile in tiles:
            item_url=urljoin(collection_url,tile+'.json')
            item,item_hash=read_metadata(url=item_url)
            asset=item['assets']['visual']
            raster_url=urljoin(item_url,asset['href'])
            with rasterio.open(raster_url) as source:
                if source.crs.to_epsg()!=identity['grid']['epsg']:
                    raise ValueError('Aerial CRS differs from the model')
                with WarpedVRT(src_dataset=source,crs=source.crs,transform=transform,
                    width=nx,height=ny,resampling=resampling,add_alpha=source.count==3) as warped:
                    rgba=warped.read()
                valid=rgba[3]>0
                rgb[valid]=np.moveaxis(rgba[:3],source=0,destination=-1)[valid]
                covered|=valid
                sources.append(dict(item_url=item_url,item_sha256=item_hash,asset_url=raster_url,
                    published_asset_checksum=asset.get('file:checksum'),
                    source_pixel_size_m=[source.transform.a,-source.transform.e],
                    start_datetime=item['properties'].get('start_datetime'),end_datetime=item['properties'].get('end_datetime')))
            print('Read aerial source',tile,flush=True)
    south_rgb=rgb[::-1].copy()
    south_covered=covered[::-1].copy()
    coverage=float(np.count_nonzero(south_covered & active)/np.count_nonzero(active))
    payload=south_rgb.tobytes()+south_covered.astype(np.uint8).tobytes()
    resource=output/'data/aerial.bin.gz'
    atomic_bytes(path=resource,payload=gzip.compress(data=payload,compresslevel=6,mtime=0))
    native=output/'data/aerial.npz'
    native.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(file=native,rgb=south_rgb,covered=south_covered)
    Image.fromarray(obj=rgb).save(fp=output/'data/aerial_preview.png')
    metadata=dict(title=collection['title'],license=collection['license'],collection_url=collection_url,
        collection_sha256=collection_hash,providers=collection['providers'],sources=sources,
        attribution='Aerial imagery: Canterbury 2023; Environment Canterbury / Landpro; sourced from LINZ Data Service, CC BY 4.0.',
        grid=identity['grid'],array_row_order='south_to_north',
        image_resampling=resampling.name,active_domain_coverage_fraction=coverage,
        valid_raster_coverage_fraction=float(covered.mean()),
        source_note='Cloud-optimised source subsets/overviews read over HTTPS; published whole-asset checksums are recorded, not claimed revalidated from partial reads.',
        resource=dict(path='data/aerial.bin.gz',raw_bytes=len(payload),raw_sha256=hashlib.sha256(payload).hexdigest()))
    atomic_bytes(path=output/'data/aerial.json',payload=(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n').encode())
    print(json.dumps(dict(status='complete',active_domain_coverage_fraction=coverage,source_tiles=len(sources),compressed_bytes=resource.stat().st_size),indent=2))
    return metadata


if __name__ == "__main__":
    project_root=Path(__file__).resolve().parents[2]
    dataset=project_root/'outputs/orton_bradley_visualisation'
    output=project_root/'outputs/orton_bradley_interactive'
    collection_url='https://nz-imagery.s3.ap-southeast-2.amazonaws.com/canterbury/canterbury_2023_0.3m/rgb/2193/collection.json'
    tiles=['BX24_5000_0705','BX24_5000_0706','BX24_5000_0707',
           'BX24_5000_0805','BX24_5000_0806','BX24_5000_0807']
    resampling=Resampling.average
    prepare(dataset=dataset,output=output,collection_url=collection_url,tiles=tiles,resampling=resampling)
