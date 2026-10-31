"""Time-centred stage forcing with a local hydraulic-head variation bound."""
import warp as wp

from floodwarp.config import BED_FLOAT, WP_FLOAT
from floodwarp.local_inertial_kernels import (
    DL_TIME, DL_DT, DL_MAX_DT, DL_STAGE_QX, DL_STAGE_QY, DL_DONE,
    _device_piecewise_linear_interp,
)
from floodwarp.subgrid_hypsometry import subgrid_stage_from_storage,subgrid_wet_fraction_from_storage

# Accuracy controls, not a timestep floor. A 2.5% head increment limits the
# change of quadratic hydrostatic forcing to about 5%; midpoint quadrature
# then resolves a dry-to-wet linear ramp in at least forty intervals.
# Sub-millimetre forcing is regularised in metres, independently of datum.
HEAD_CHANGE_FRACTION = wp.constant(0.025)
HEAD_CHANGE_ATOL_M = wp.constant(0.0001)


@wp.func
def change_bound(times: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float64),
                 time: wp.float64, current: wp.float64, bed: wp.float64,
                 inside_depth: wp.float64, inside_rate: wp.float64, cap: wp.float64) -> wp.float64:
    if time >= times[times.shape[0]-1]:return cap
    lo=int(0);hi=times.shape[0]-1
    while lo+1<hi:
        mid=(lo+hi)//2
        if times[mid]<=time:lo=mid
        else:hi=mid
    slope=(values[lo+1]-values[lo])/(times[lo+1]-times[lo])
    recent=lo
    while recent>0 and values[recent+1]==values[recent]:recent-=1
    if values[recent+1]==values[recent]:return cap
    head=wp.max(wp.float64(0.),current-bed)
    first=wp.max(wp.float64(0.),values[recent]-bed)
    last=wp.max(wp.float64(0.),values[recent+1]-bed)
    if wp.abs(last-first)<=wp.float64(HEAD_CHANGE_ATOL_M):return cap
    scale=wp.max(wp.max(head,inside_depth),wp.abs(last-first))
    increment=wp.max(wp.float64(HEAD_CHANGE_ATOL_M),wp.float64(HEAD_CHANGE_FRACTION)*scale)
    limit=cap
    if slope!=wp.float64(0.):limit=wp.min(limit,increment/wp.abs(slope))
    # The pressure difference contains BOTH exterior and interior heads.
    # A pulse can leave a moving wave after the external stage becomes flat.
    if inside_rate>wp.float64(0.):limit=wp.min(limit,increment/inside_rate)
    # Do not subcycle a boundary which is still hydraulically below its sill.
    if current<bed and slope>wp.float64(0.):limit=wp.max(limit,(bed-current)/slope)
    return wp.min(cap,limit)


@wp.kernel
def forecast(control: wp.array(dtype=wp.float64),
             times_x: wp.array(dtype=wp.float64), values_x: wp.array(dtype=wp.float64), use_x: int,
             times_y: wp.array(dtype=wp.float64), values_y: wp.array(dtype=wp.float64), use_y: int):
    """Upper exterior head for CFL; accepted fluxes use the later midpoint."""
    if control[DL_DONE]!=wp.float64(0.):return
    time=control[DL_TIME]+control[DL_MAX_DT]
    if use_x==1:control[DL_STAGE_QX]=wp.max(control[DL_STAGE_QX],_device_piecewise_linear_interp(times_x,values_x,time,control))
    if use_y==1:control[DL_STAGE_QY]=wp.max(control[DL_STAGE_QY],_device_piecewise_linear_interp(times_y,values_y,time,control))


@wp.kernel
def limit_faces(indices: wp.array(dtype=int), nx: int, x_count: int,
    depth: wp.array2d(dtype=WP_FLOAT), bed: wp.array2d(dtype=BED_FLOAT),
    face_bed_x: wp.array2d(dtype=BED_FLOAT),face_bed_y: wp.array2d(dtype=BED_FLOAT),traditional: int,
    qx: wp.array2d(dtype=WP_FLOAT),qy: wp.array2d(dtype=WP_FLOAT),dx: WP_FLOAT,dy: WP_FLOAT,
    cell_z: wp.array3d(dtype=BED_FLOAT),breakpoints: wp.array3d(dtype=BED_FLOAT),cell_count: int,
    bx: wp.array2d(dtype=int),by: wp.array2d(dtype=int),
    times_x: wp.array(dtype=wp.float64),values_x: wp.array(dtype=wp.float64),use_x: int,
    times_y: wp.array(dtype=wp.float64),values_y: wp.array(dtype=wp.float64),use_y: int,
    control: wp.array(dtype=wp.float64),minimum: wp.array(dtype=WP_FLOAT)):
    if control[DL_DONE]!=wp.float64(0.):return
    face=indices[wp.tid()];limit=control[DL_MAX_DT]
    row=int(0);column=int(0);axis=int(0)
    if face<x_count and use_x==1:
        j=face//(nx+1);i=face%(nx+1);cell=i
        if i==nx or bx[j,i]==10:cell=i-1
        sill=wp.float64(bed[j,cell])
        if traditional==1:sill=wp.float64(face_bed_x[j,i])
        row=j;column=cell
    elif face>=x_count and use_y==1:
        face-=x_count;j=face//nx;i=face%nx;cell=j
        if j==depth.shape[0] or by[j,i]==4:cell=j-1
        sill=wp.float64(bed[cell,i])
        if traditional==1:sill=wp.float64(face_bed_y[j,i])
        row=cell;column=i;axis=1
    inside=wp.float64(depth[row,column])
    rate=wp.abs(wp.float64((qx[row,column]-qx[row,column+1])/dx+(qy[row,column]-qy[row+1,column])/dy))
    if traditional==1:
        level=subgrid_stage_from_storage(depth[row,column],cell_z,breakpoints,row,column,cell_count)
        inside=wp.max(wp.float64(0.),wp.float64(level)-sill)
        fraction=subgrid_wet_fraction_from_storage(depth[row,column],cell_z,breakpoints,row,column,cell_count)
        rate/=wp.max(wp.float64(fraction),wp.float64(1.)/wp.float64(cell_count))
    if axis==0:limit=change_bound(times_x,values_x,control[DL_TIME],control[DL_STAGE_QX],sill,inside,rate,limit)
    else:limit=change_bound(times_y,values_y,control[DL_TIME],control[DL_STAGE_QY],sill,inside,rate,limit)
    # A remaining fragment to a series node is an output/forcing cap, not
    # a hydraulic restriction. Do not feed that cap back as an uncapped
    # stable dt, which would falsely trigger the minimum-dt failure guard.
    if limit<control[DL_MAX_DT]:
        wp.atomic_min(minimum,0,WP_FLOAT(limit))
        wp.atomic_min(control,DL_DT,limit)


@wp.kernel
def limit_connections(faces: wp.array(dtype=int),endpoints: wp.array2d(dtype=int),
    sill: wp.array(dtype=wp.float64),eta: wp.array(dtype=wp.float64),nx: int,x_count: int,
    offsets: wp.array(dtype=int),adjacent: wp.array(dtype=int),q: wp.array(dtype=WP_FLOAT),area: wp.array(dtype=wp.float64),
    bx: wp.array2d(dtype=int),by: wp.array2d(dtype=int),
    times_x: wp.array(dtype=wp.float64),values_x: wp.array(dtype=wp.float64),use_x: int,
    times_y: wp.array(dtype=wp.float64),values_y: wp.array(dtype=wp.float64),use_y: int,
    control: wp.array(dtype=wp.float64),minimum: wp.array(dtype=WP_FLOAT)):
    if control[DL_DONE]!=wp.float64(0.):return
    f=wp.tid()
    if endpoints[f,1]>=0:return
    face=faces[f];inside=wp.max(wp.float64(0.),eta[endpoints[f,0]]-sill[f]);limit=control[DL_MAX_DT]
    cell=endpoints[f,0];net=wp.float64(0.)
    for index in range(offsets[cell],offsets[cell+1]):
        entry=adjacent[index];sign=wp.float64(1.)
        if entry<0:sign=wp.float64(-1.)
        net+=sign*wp.float64(q[wp.abs(entry)-1])
    rate=wp.float64(0.)
    if area[cell]>wp.float64(0.):rate=wp.abs(net)/area[cell]
    if face>=0 and face<x_count and use_x==1:
        j=face//(nx+1);i=face%(nx+1);code=bx[j,i]
        if code==3 or code==5 or code==10:
            limit=change_bound(times_x,values_x,control[DL_TIME],control[DL_STAGE_QX],sill[f],inside,rate,limit)
    elif face>=x_count and use_y==1:
        face-=x_count;j=face//nx;i=face%nx;code=by[j,i]
        if code==3 or code==4 or code==11:
            limit=change_bound(times_y,values_y,control[DL_TIME],control[DL_STAGE_QY],sill[f],inside,rate,limit)
    if limit<control[DL_MAX_DT]:
        wp.atomic_min(minimum,0,WP_FLOAT(limit))
        wp.atomic_min(control,DL_DT,limit)
