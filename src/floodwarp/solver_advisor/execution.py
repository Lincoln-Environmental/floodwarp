"""Opt-in Hancock schedule choice once per configured forecast or explicit forcing change."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
import warp as wp

from floodwarp.hancock_tiles import TILE_WIDTH,TILE_HEIGHT


def coverage(*, seeds):
    ny,nx=seeds.shape
    pad=np.pad(seeds,((0,-ny%TILE_HEIGHT),(0,-nx%TILE_WIDTH)))
    tiles=pad.reshape(pad.shape[0]//TILE_HEIGHT,TILE_HEIGHT,-1,TILE_WIDTH).any(axis=(1,3))
    halo=maximum_filter(tiles,size=3,mode="constant",cval=False)
    return float(halo.repeat(TILE_HEIGHT,axis=0).repeat(TILE_WIDTH,axis=1)[:ny,:nx].mean())


def select(*, shape, scheduled, active_fraction, device_name, free_bytes, policy):
    ny,nx=shape
    required=4*(4*(nx//TILE_WIDTH+1)*(ny//TILE_HEIGHT+1)+1)
    if device_name!=policy["device_name"] or free_bytes<required:
        return "dense"
    if nx*ny<policy["smallest_measured_sparse_win_cells"] or not active_fraction:
        return "dense"
    return "tiles" if scheduled<policy["dense_work_fraction"] else "dense"


def configure_hancock_execution(*, solver, execution_mode, distributed_rain_expected=True, policy=None):
    """Apply dense/sparse/tiles or the frozen GPU rule after setting initial state and rain maps."""
    if solver.method_name!="fv1_hancock":
        raise ValueError("Hancock schedule selection requires a Hancock solver")
    if execution_mode in ("dense","tiles","sparse"):
        solver.configure_execution(mode=execution_mode)
        return dict(requested=execution_mode,selected=execution_mode)
    if execution_mode!="auto":
        raise ValueError("Choose dense, tiles, sparse or auto")
    if policy is None:
        policy=json.loads(Path(__file__).with_name("execution_policy.json").read_text())
    device=wp.get_device(solver.device)
    active=solver._active
    shape=active.shape
    state=solver._fields["q"].numpy().reshape(*shape,3)
    rain=solver._fields["rain"].numpy().reshape(shape)!=0
    if distributed_rain_expected:
        rain |= solver._fields["multiplier"].numpy().reshape(shape)!=0
    bx=solver._fields["types_x"].numpy().reshape(shape[0],shape[1]+1)
    by=solver._fields["types_y"].numpy().reshape(shape[0]+1,shape[1])
    wet=state[...,0]>0
    seeds=(state!=0).any(axis=2)|rain|(bx[:,:-1]==3)|(bx[:,1:]==3)|(by[:-1]==3)|(by[1:]==3)
    work=coverage(seeds=seeds & active)
    mode=select(shape=shape,scheduled=work,active_fraction=float(active.mean()),device_name=device.name,
                free_bytes=device.free_memory if device.is_cuda else 0,policy=policy)
    solver.configure_execution(mode=mode)
    return dict(requested="auto",selected=mode,wet_fraction=float(wet.mean()),rain_fraction=float((rain&active).mean()),
                active_fraction=float(active.mean()),scheduled_estimate=work,device=device.name)

