"""Record accepted dt only on a dedicated numerical replay."""

from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import warp as wp

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.fv2_solver import FV2Solver


@wp.kernel
def record_step(ctrl:wp.array(dtype=wp.float64),values:wp.array(dtype=wp.float64),start:int,multistage:int):
    if ctrl[7]==wp.float64(0.0) and (multistage==0 or ctrl[2]<=ctrl[17]):
        i=int(ctrl[3])-1-start
        if i>=0 and i<values.shape[0]:
            values[i]=ctrl[2]


@contextmanager
def history(*, solver, capacity):
    if isinstance(solver,FullSWESolver):
        data=wp.zeros(capacity,dtype=wp.float64,device=solver.device)
        wp.load_module(module=record_step.module,device=solver.device)
        original=solver._enqueue_step
        start=solver.steps
        multi=int(isinstance(solver,FV2Solver))
        def trace():
            original()
            wp.launch(kernel=record_step,dim=1,inputs=[solver.ctrl,data,start,multi],device=solver.device)
        solver._graph=None
        try:
            with patch.object(solver,"_enqueue_step",trace):
                yield data.numpy
        finally:
            solver._graph=None
    else:
        pieces=[]
        original=solver.run_device_loop
        def timed(**kwargs):
            result=original(**kwargs,dt_history_capacity=capacity)
            pieces.append(result["dt_history"])
            return result
        def array():
            return np.concatenate(pieces)
        with patch.object(solver,"run_device_loop",timed):
            yield array


def checked(*, times, count, duration):
    if len(times)!=count or (times<=0).any():
        raise ArithmeticError("Incomplete accepted-step record")
    np.testing.assert_allclose(times.sum(),duration,rtol=3e-11,atol=1e-10)
