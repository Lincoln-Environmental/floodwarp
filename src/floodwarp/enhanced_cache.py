"""Content-bound, non-pickle reusable enhanced hydraulic geometry."""
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from importlib.metadata import version

import numpy as np
import scipy

from floodwarp.enhanced_geometry import DynamicGeometry


def geometry_cache_key(*, native_identity, face_identity, parameters, encoding, tolerance):
    """Bind every physical/table input and the preprocessing implementation."""
    directory=Path(__file__).parent
    source_hashes={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ('enhanced_geometry.py','enhanced_reduction.py','enhanced_cache.py',
                     'enhanced_preprocessing.py','enhanced_native_compiled.py','enhanced_native_sparse.py','enhanced_topology.py')}
    contract=dict(schema=1,native_identity=native_identity,face_identity=face_identity,
        parameters=parameters,encoding=encoding,tolerance=tolerance,sources=source_hashes,
        numpy_version=np.__version__,scipy_version=scipy.__version__,numba_version=version('numba'))
    return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()


def array_digest(*, array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def cache_json_value(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Non-serializable geometry metadata: {type(value).__name__}')


def save_geometry_cache(*, geometry, directory: Path, key: str):
    """Publish only a complete archive. Source datasets are never modified."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays={name:value for name,value in vars(geometry).items() if isinstance(value,np.ndarray)}
    metadata={name:value for name,value in vars(geometry).items() if name not in arrays}
    manifest=dict(key=key,metadata=metadata,sha256={name:array_digest(array=value) for name,value in arrays.items()})
    arrays['manifest']=np.asarray(json.dumps(manifest,default=cache_json_value,sort_keys=True))
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,prefix=key+'.',suffix='.partial',delete=False) as stream:
            temporary=Path(stream.name)
            np.savez(stream,**arrays)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,directory/(key+'.npz'))
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def load_geometry_cache(*, directory: Path, key: str):
    """Missing entry returns None; incompatible/corrupt entries fail explicitly."""
    path=Path(directory)/(key+'.npz')
    if not path.exists():return None
    started=time.perf_counter()
    try:
        with np.load(path,allow_pickle=False) as archive:
            manifest=json.loads(str(archive['manifest']))
            if manifest['key']!=key:raise ValueError('geometry cache identity mismatch')
            if set(archive.files)!=(set(manifest['sha256'])|{'manifest'}):
                raise ValueError('geometry cache array manifest mismatch')
            values=dict(manifest['metadata'])
            for name,digest in manifest['sha256'].items():
                array=archive[name]
                if array_digest(array=array)!=digest:raise ValueError(f'geometry cache checksum mismatch: {name}')
                values[name]=array
            if set(values)!={f.name for f in fields(DynamicGeometry)}:
                raise ValueError('geometry cache schema mismatch')
            values['shape']=tuple(values['shape'])
            geometry=DynamicGeometry(**values)
    except (KeyError,OSError,TypeError) as error:
        raise ValueError(f'Invalid enhanced geometry cache: {path}') from error
    return geometry,time.perf_counter()-started
