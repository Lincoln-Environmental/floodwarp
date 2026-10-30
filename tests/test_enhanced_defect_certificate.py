"""Check the dissipative ODE certificate against independently evaluated probes."""
import numpy as np
import pytest
import warp as wp
from floodwarp.enhanced_defect import frozen_certificates
from floodwarp.enhanced_momentum import frozen_discharge


@wp.kernel
def evaluate(values: wp.array2d(dtype=wp.float64),result: wp.array2d(dtype=wp.float64)):
    i=wp.tid()
    q0=values[i,0];a0=values[i,1];b0=values[i,2];a1=values[i,3];b1=values[i,4];dt=values[i,5]
    q1=frozen_discharge(q0,a0,b0,dt)
    actual=frozen_discharge(q0,(a0+a1)*wp.float64(.5),(b0+b1)*wp.float64(.5),dt)
    midpoint=frozen_discharge(q0,(a0+a1)*wp.float64(.5),(b0+b1)*wp.float64(.5),dt*wp.float64(.5))
    bounds=frozen_certificates(q0,q1,a0,b0,a1,b1,dt)
    result[i,0]=wp.abs(actual-q1);result[i,1]=bounds[0]
    result[i,2]=wp.abs(midpoint-bounds[1]);result[i,3]=bounds[2]


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_certificate_bounds_pressure_drag_and_reversal(device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(20261028)
    count=4096
    q=rng.uniform(-1.,1.,count)*np.power(10.,rng.uniform(-6.,3.,count))
    a0=rng.uniform(-1.,1.,count)*np.power(10.,rng.uniform(-6.,3.,count))
    a1=rng.uniform(-1.,1.,count)*np.power(10.,rng.uniform(-6.,3.,count))
    b0=np.power(10.,rng.uniform(-6.,9.,count));b1=np.power(10.,rng.uniform(-6.,9.,count))
    dt=np.power(10.,rng.uniform(-8.,1.,count))
    values=np.column_stack((q,a0,b0,a1,b1,dt))
    out=wp.zeros((count,4),dtype=wp.float64,device=device)
    wp.launch(kernel=evaluate,dim=count,inputs=[wp.array(values,dtype=wp.float64,device=device),out],device=device)
    result=out.numpy()
    assert np.isfinite(result).all()
    assert np.all(result[:,0]<=result[:,1]*(1.+1.e-12)+1.e-12)
    assert np.all(result[:,2]<=result[:,3]*(1.+1.e-12)+1.e-12)
