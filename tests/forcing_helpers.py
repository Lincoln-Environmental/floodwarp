"""Host orchestration of the shared stage variation controller for parity tests."""
import numpy as np
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_kernels import DL_TIME,DL_MAX_DT,DL_STAGE_QX


def prepare_stage_controller(*,solver,series):
    solver._prepare_stage_variation_faces(use_x=1,use_y=0)
    times=wp.array(series[0],dtype=wp.float64,device=solver.device)
    values=wp.array(series[1],dtype=wp.float64,device=solver.device)
    control=wp.zeros(18,dtype=wp.float64,device=solver.device)
    minimum=wp.zeros(1,dtype=WP_FLOAT,device=solver.device)
    return times,values,control,minimum


def stage_variation_cap(*,solver,series,buffers,time,cap):
    times,values,control,minimum=buffers
    state=np.zeros(18)
    state[DL_TIME]=time;state[DL_MAX_DT]=cap
    state[DL_STAGE_QX]=np.interp(time,*series)
    control.assign(state);minimum.fill_(WP_FLOAT(cap))
    solver._limit_stage_variation(control=control,minimum=minimum,
        times_x=times,values_x=values,use_x=1,times_y=times,values_y=values,use_y=0)
    return float(minimum.numpy()[0])
