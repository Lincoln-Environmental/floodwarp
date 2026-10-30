"""Complete standalone LI/SGS disk state; serialization extracted from baseline."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import warp as wp
from floodwarp.enhanced_cache import save_geometry_cache, load_geometry_cache
from floodwarp.surface_restart import json_value


def durable_checkpoint(*, surface, path, signature, report):
    """Serialize the existing complete capture contract and immutable SGS geometry."""
    state=surface.capture_state()
    arrays={name:field.numpy() for name,field in state['device'].items() if field is not None}
    host=state['host'];host['_conditioned_bed_array']=None
    geometry=host.pop('enhanced_geometry',None)
    geometry_key=None
    if geometry is not None:
        geometry_key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        directory=path.parent/'geometry'
        if not (directory/(geometry_key+'.npz')).exists():
            save_geometry_cache(geometry=geometry,directory=directory,key=geometry_key)
    metadata=dict(signature=signature,host=host,initial_volume=surface.initial_volume,
        none_fields=[name for name,field in state['device'].items() if field is None],
        cumulative_report=report,geometry_key=geometry_key)
    temporary=path.with_suffix('.partial.npz')
    with temporary.open('wb') as stream:
        np.savez(file=stream,metadata=json.dumps(metadata,default=json_value),**arrays)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(target=path)
    with np.load(file=path,allow_pickle=False) as saved:
        for name,values in arrays.items():
            np.testing.assert_array_equal(actual=saved[name],desired=values,err_msg=name)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        time_s=surface.simulation_time,bytes=path.stat().st_size,device_arrays=len(arrays),
        geometry_key=geometry_key,round_trip='all saved device arrays exact')


def restore_complete(*, surface, path, signature):
    # Allocate the existing global device-loop scratch before restoring every field.
    surface.run_device_loop(duration=surface.minimum_dt,use_cuda_graph=None)
    expected=surface.capture_state()
    with np.load(file=path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata']))
        if metadata['signature']!=signature:
            raise ValueError('Checkpoint source/input/configuration mismatch')
        device={name:None for name in metadata['none_fields']}
        for name in saved.files:
            if name=='metadata':continue
            values=saved[name]
            template=expected['device'].get(name)
            device[name]=wp.array(values,dtype=template.dtype,device=surface.device) if template is not None else wp.array(values,device=surface.device)
        if np.any(saved['water_depth']<0) or not np.isfinite(saved['water_depth']).all():
            raise ValueError('Invalid checkpoint depths')
    host=metadata['host'];host['_conditioned_bed_array']=surface.bed_elevation
    if host['_infiltration_exclusion_host'] is not None:
        host['_infiltration_exclusion_host']=np.asarray(host['_infiltration_exclusion_host'],dtype=np.int32)
    if metadata['geometry_key'] is not None:
        geometry,seconds=load_geometry_cache(directory=path.parent/'geometry',key=metadata['geometry_key'])
        host['enhanced_geometry']=geometry
    surface.initial_volume=metadata['initial_volume']
    surface.restore_state(state=dict(host=host,device=device))
    return metadata['cumulative_report']
