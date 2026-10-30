"""Fast inactive paths must match every connection-level reference result."""
import numpy as np
import pytest
import warp as wp

from floodwarp import enhanced_accuracy as reference
from floodwarp import enhanced_embedded_defect as optimized
from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_kernels import DL_SIZE,DL_DT


def arguments(*,device,count,seed,bounded):
    rng=np.random.default_rng(seed)
    q0=rng.uniform(-1.,1.,count)*10.**rng.uniform(-6.,2.,count)
    q=q0+rng.uniform(-1.,1.,count)*10.**rng.uniform(-8.,1.,count)
    initial_d=10.**rng.uniform(-3.,4.,count);d=initial_d*10.**rng.uniform(-1.,1.,count)
    r0=10.**rng.uniform(-6.,9.,count);r=r0*10.**rng.uniform(-2.,2.,count)
    p0=rng.uniform(-1.,1.,count);p=p0+rng.uniform(-1.,1.,count)*.01
    inactive=np.arange(count)%8==0;newly_wet=np.arange(count)%8==1;newly_dry=np.arange(count)%8==2
    d[inactive]=0.;initial_d[inactive]=0.;q[inactive]=0.
    initial_d[newly_wet]=0.;d[newly_dry]=0.;q[newly_dry]=0.
    proposed=q.copy();proposed[np.arange(count)%9==0]*=2.
    kind=rng.choice([0,2,3,6,9],size=count).astype(np.int32)
    endpoints=np.column_stack((2*np.arange(count),2*np.arange(count)+1)).astype(np.int32)
    volume=10.**rng.uniform(-8.,2.,2*count);area=np.ones(2*count)
    control=np.zeros(DL_SIZE);control[DL_DT]=.05
    args=[wp.array(control,dtype=wp.float64,device=device)]
    for array,dtype in ((q,WP_FLOAT),(q0,WP_FLOAT),(proposed,WP_FLOAT),(kind,int),(endpoints,int),
        (np.arange(2*count+1,dtype=np.int32),int),(area,wp.float64),(volume,WP_FLOAT),(area,wp.float64),
        (d,wp.float64),(r,wp.float64),(p,wp.float64),(initial_d,wp.float64),(r0,wp.float64),(p0,wp.float64)):
        args.append(wp.array(array,dtype=dtype,device=device))
    return args


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('bounded',[0,1])
def test_independent_reference_connection_results(device,bounded):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    count=4096
    shared=arguments(device=device,count=count,seed=20261028,bounded=bounded)
    results=[]
    for kernel in (reference.embedded_connections,optimized.connections):
        mid=wp.zeros(count,dtype=WP_FLOAT,device=device);radius=wp.zeros(count,dtype=wp.float64,device=device)
        counts=wp.zeros((count,2),dtype=int,device=device);state=wp.zeros(18,dtype=wp.float64,device=device)
        wp.launch(kernel=kernel,dim=count,inputs=[*shared,mid,radius,counts,state,9.80665,.001,.001,.05,.0001,bounded],device=device)
        results.append((mid.numpy(),radius.numpy(),counts.numpy(),state.numpy()))
    for original,fast in zip(results[0],results[1]):np.testing.assert_array_equal(original,fast)
