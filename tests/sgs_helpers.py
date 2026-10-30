"""Read-only instrumentation must preserve results and identify the exact bound."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver



def disconnected_source_model(*,device):
    model=LocalInertialSolver(ny=1,nx=3,dx=2.,dy=2.,bed_elevation=np.zeros((1,3)),
        active_mask=np.array([[1,0,1]],np.int32),manning_n=np.full((1,3),.03),
        subgrid_bed_elevation=np.zeros((2,6)),subgrid_resolution_m=1.,
        hydraulic_geometry_mode='enhanced',device=device,maximum_dt=.1,adaptive_dt_growth_factor=None,
        track_max_water_depth=True)
    model.set_source_rates(rain=np.array([[.01,0.,.01]]))
    return model
