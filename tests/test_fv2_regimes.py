"""Independent shock, dry cutoff and source-path checks for the FV2 optimisations."""
import numpy as np
import pytest
import warp as wp
from floodwarp.fv2_kernels import cut_trace, primitive_contact
from floodwarp.full_swe_kernels import hll,hll_speeds
from floodwarp.fv2_reference import traces
from validation.fv2_jump_review import trial


@wp.kernel
def compare_faces(q:wp.array(dtype=wp.vec3d),s:wp.array(dtype=wp.vec4d),rise:wp.array(dtype=wp.float64),
                  values:wp.array(dtype=wp.vec4d),expected:wp.array(dtype=wp.vec4d)):
    from_index=2*wp.tid()
    dry=wp.float64(.0001)
    axis=wp.tid()%2
    states=wp.mat33d()
    prim=wp.mat33d()
    for side in range(2):
        k=from_index+side
        sign=wp.float64(1-2*side)
        inv=wp.float64(0.0)
        if q[k][0]>dry:
            inv=wp.float64(1.)/q[k][0]
        prim[side]=cut_trace(q[k],inv,s[k],sign,rise[wp.tid()]*sign,axis,dry)
        # Independent conserved linear face construction from cell momenta.
        dh=wp.float64(.5)*sign*(s[k][0]-s[k][1])
        h=q[k][0]+dh
        hcut=wp.max(wp.float64(0.),h-wp.max(wp.float64(0.),sign*rise[wp.tid()]))
        states[side,0]=hcut
        for j in range(2):
            comp=(axis+j)%2
            if h>dry and q[k][0]>dry:
                states[side,j+1]=hcut*(q[k][comp+1]+wp.float64(.5)*sign*(q[k][0]-dh)*s[k][comp+2])/q[k][0]
    g=wp.float64(9.81)
    f=hll(states[0],states[1],g,dry,1)
    bounds=hll_speeds(states[0],states[1],g,dry,1)
    expected[wp.tid()]=wp.vec4d(f[0],f[1],f[2],wp.max(wp.abs(bounds[0]),wp.abs(bounds[1])))
    values[wp.tid()]=primitive_contact(prim[0],prim[1],g,dry)


@pytest.mark.parametrize("device",("cpu","cuda:0"))
def test_interior_face_and_cfl_against_conservative_construction(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    rng=np.random.default_rng(seed=6251)
    n=2048
    h=10**rng.uniform(-7,2,n*2)
    h[::7]=0.
    q=np.column_stack((h,h[:,None]*rng.uniform(-30,30,(2*n,2))))
    slopes=rng.uniform(-1,1,(2*n,4))
    slopes[:,0]=slopes[:,1]+rng.uniform(-1.9,1.9,2*n)*h
    rises=rng.uniform(-1,1,n)*np.maximum(h[::2],h[1::2])
    arrays=[wp.array(v,dtype=t,device=device) for v,t in ((q,wp.vec3d),(slopes,wp.vec4d),(rises,wp.float64))]
    actual=wp.empty(n,dtype=wp.vec4d,device=device)
    expected=wp.empty_like(actual)
    wp.launch(kernel=compare_faces,dim=n,inputs=[*arrays,actual,expected],device=device)
    np.testing.assert_allclose(actual.numpy(),expected.numpy(),rtol=5e-13,atol=2e-11)


def test_local_limiting_in_compression_and_expansion():
    h=np.broadcast_to([.2,.4,.7,.95,1.],(4,5)).copy()
    q=np.stack((h,np.full_like(h,1.2),np.zeros_like(h)),axis=-1)
    for direction in (1,-1):
        states=q[:,::direction].copy()
        states[...,1]*=direction
        kw=dict(q=states,z=np.zeros_like(h),dry=.0001,axis=1)
        adaptive=traces(limiter="adaptive",**kw)[0]
        minmod=traces(limiter="minmod",**kw)[0]
        mc=traces(limiter="mc",**kw)[0]
        np.testing.assert_allclose(adaptive[:,2],minmod[:,2])
        assert np.abs(mc[:,2]-minmod[:,2]).max()>1e-3
        states[...,1]*=-1
        np.testing.assert_array_equal(traces(limiter="adaptive",**kw)[0],traces(limiter="mc",**kw)[0])


def test_transcritical_discharge_refines():
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    common=dict(method="adaptive",axis="x",case="transcritical_jump",jump=22.5,cap_ratio=1/160,
                duration=20.,interval=1.,device="cuda:0")
    coarse=trial(dx=5.,**common)
    fine=trial(dx=2.5,**common)
    assert fine["q_rmse"] < coarse["q_rmse"] < .01
    assert abs(fine["balance"]["residual"]) < 1e-7
