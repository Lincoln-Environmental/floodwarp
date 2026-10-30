"""Conservative interfaces between component states and the coarse model API.

Native raster work here is initialization/output work, never a hydraulic step.
The scalar coarse stage is storage-equivalent; it does not replace independent
component stages or suffice for native-depth rendering/checkpoint restoration.
"""
import numpy as np

from floodwarp.enhanced_geometry import storage_curve,storage_stage,component_depth


def coarse_storage_to_components(*, geometry, native_bed, storage_depth):
    """Initialize the underdetermined component volumes at one common cell stage.

    This allocation is used only when the caller supplies coarse initial data.
    It retains exactly the requested active-cell volume, without transferring
    water between components during subsequent hydraulic evolution.
    """
    depth=np.asarray(storage_depth,dtype=float)
    if depth.shape!=geometry.shape or not np.all(np.isfinite(depth)) or np.any(depth<0):
        raise ValueError('finite nonnegative coarse storage depths required')
    bed=np.asarray(native_bed,dtype=float)
    if bed.shape!=geometry.pixel_owner.shape:raise ValueError('native bed must match ownership raster')
    result=np.zeros(len(geometry.storage_offsets)-1)
    f=geometry.factor;pixel_area=geometry.resolution**2;cell_area=(f*geometry.resolution)**2
    for cell in range(depth.size):
        start,stop=geometry.cell_component_offsets[cell:cell+2]
        if start==stop:continue
        j,i=divmod(cell,geometry.shape[1]);patch=bed[j*f:(j+1)*f,i*f:(i+1)*f]
        owners=geometry.pixel_owner[j*f:(j+1)*f,i*f:(i+1)*f]
        target=depth.flat[cell]*cell_area
        stage,_=storage_stage(volume=target,curve=storage_curve(patch=patch,pixel_area=pixel_area))
        for state in range(start,stop):result[state]=np.maximum(stage-patch[owners==state],0.).sum()*pixel_area
        # Restore the prescribed total after the hypsometry's rounding error.
        largest=start+int(np.argmax(result[start:stop]));result[largest]+=target-result[start:stop].sum()
        if result[largest]<0:raise RuntimeError('initial component allocation produced negative storage')
    return result


def coarse_component_diagnostics(*, geometry, native_bed, volume):
    """Independent-state output reference, including actual local depth/wet area."""
    field=component_depth(geometry=geometry,volume=volume,bed=native_bed)
    ny,nx=geometry.shape;f=geometry.factor
    patches=field.reshape(ny,f,nx,f).transpose(0,2,1,3).reshape(ny,nx,-1)
    wet=np.count_nonzero(patches>0.,axis=2)
    storage=patches.mean(axis=2)
    mean_wet=np.divide(patches.sum(axis=2),wet,out=np.zeros_like(storage),where=wet>0)
    stage=np.zeros(geometry.shape)
    for cell in range(ny*nx):
        if geometry.cell_component_offsets[cell]==geometry.cell_component_offsets[cell+1]:continue
        j,i=divmod(cell,nx);patch=native_bed[j*f:(j+1)*f,i*f:(i+1)*f]
        stage.flat[cell]=storage_stage(volume=storage.flat[cell]*(f*geometry.resolution)**2,
            curve=storage_curve(patch=patch,pixel_area=geometry.resolution**2))[0]
    return dict(storage_depth=storage,water_surface_elevation=stage,
        maximum_local_depth=patches.max(axis=2),wet_fraction=wet/f**2,mean_wet_depth=mean_wet)


def coarse_face_connections(*, geometry):
    """Map each inter-cell/ghost group to its signed Cartesian coarse face.

    Intra-cell connections deliberately have no coarse face. Q is m3/s;
    coarse staggered q is sum(sign*Q)/full face width in m2/s.
    """
    ny,nx=geometry.shape;x_count=ny*(nx+1);count=x_count+(ny+1)*nx
    face=np.full(len(geometry.endpoints),-1,np.int32);sign=np.zeros(len(face),np.int32)
    members=[[] for _ in range(count)]
    for link,(a,b) in enumerate(geometry.endpoints):
        ca=int(geometry.state_cell[a]);j,i=divmod(ca,nx)
        if b>=0:
            cb=int(geometry.state_cell[b]);bj,bi=divmod(cb,nx)
            if ca==cb:continue
            if j==bj and abs(i-bi)==1:
                face[link]=j*(nx+1)+max(i,bi);sign[link]=1 if bi>i else -1
            elif i==bi and abs(j-bj)==1:
                face[link]=x_count+max(j,bj)*nx+i;sign[link]=1 if bj>j else -1
            else:raise ValueError('component link endpoints are not cardinal neighbouring cells')
        else:
            side=geometry.external_side[link]
            if side in (0,1):face[link]=j*(nx+1)+i+int(side==1)
            elif side in (2,3):face[link]=x_count+(j+int(side==3))*nx+i
            else:raise ValueError('external component link needs a cardinal contact side')
            sign[link]=-1 if side in (0,2) else 1
        members[face[link]].append(link)
    offsets=np.r_[0,np.cumsum([len(m) for m in members])].astype(np.int32)
    return face,sign,offsets,np.asarray([link for group in members for link in group],np.int32)


def structure_component_endpoints(*, geometry, native_bed, records):
    """Locate each engineered opening in one persistent storage owner.

    Existing records locate endpoints only to a coarse cell. For those inputs
    the declared allocation rule is its lowest native pixel (row-major tie).
    This represents an opening at the cell's low point, not access to all its
    pools. Optional inlet/outlet_native_row/col give the actual native location
    in solver row order. No hydraulic exchange occurs in this mapping.
    """
    bed=np.asarray(native_bed);factor=geometry.factor;result=[]
    for record in records:
        owners=[]
        for prefix in ('inlet','outlet'):
            row=int(record[prefix+'_row']);col=int(record[prefix+'_col'])
            row_key=prefix+'_native_row';col_key=prefix+'_native_col'
            if (row_key in record)!=(col_key in record):
                raise ValueError('Both native row and column are required for a structure endpoint')
            if row_key in record:
                native_row=int(record[row_key]);native_col=int(record[col_key])
                if native_row!=record[row_key] or native_col!=record[col_key]:
                    raise ValueError('Structure native endpoint indices must be integers')
                if not (row*factor<=native_row<(row+1)*factor and col*factor<=native_col<(col+1)*factor):
                    raise ValueError('Structure native endpoint must lie inside its declared coarse cell')
            else:
                patch=bed[row*factor:(row+1)*factor,col*factor:(col+1)*factor]
                j,i=np.unravel_index(np.argmin(patch),patch.shape)
                native_row=row*factor+j;native_col=col*factor+i
            owner=int(geometry.pixel_owner[native_row,native_col])
            if owner<0:raise ValueError('Structure endpoint lies in inactive terrain')
            owners.append(owner)
        result.append(owners)
    return np.asarray(result,dtype=np.int32).reshape(-1,2)
