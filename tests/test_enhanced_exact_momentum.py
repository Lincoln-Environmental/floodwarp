"""Analytical frozen momentum against an independent ODE integrator."""
import math
import numpy as np
import pytest
import warp as wp
from scipy.integrate import solve_ivp
from floodwarp.enhanced_momentum import frozen_discharge


@wp.kernel
def evaluate(inputs: wp.array2d(dtype=wp.float64),result: wp.array(dtype=wp.float64)):
    i=wp.tid();result[i]=frozen_discharge(inputs[i,0],inputs[i,1],inputs[i,2],inputs[i,3])


def compute(*,values,device):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    data=wp.array(np.asarray(values),dtype=wp.float64,device=device)
    result=wp.zeros(len(values),dtype=wp.float64,device=device)
    wp.launch(kernel=evaluate,dim=len(values),inputs=[data,result],device=device)
    return result.numpy()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_independent_continuous_ode_reference(device):
    values=[];expected=[]
    for q in (-3.,-.1,0.,.1,3.):
        for a in (-2.,0.,2.):
            for b in (0.,.1,2.):
                for dt in (.05,.3,2.):
                    def equation(time,state):return a-b*state*abs(state)
                    ref=solve_ivp(fun=equation,t_span=(0.,dt),y0=[q],method='DOP853',rtol=3e-13,atol=2e-14)
                    assert ref.success
                    values.append((q,a,b,dt));expected.append(ref.y[0,-1])
    np.testing.assert_allclose(compute(values=values,device=device),expected,rtol=2e-11,atol=2e-12)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_zero_terms_reversal_equilibrium_and_stiff_acceleration(device):
    values=[(2.,0.,3.,.5),(2.,3.,0.,.5),(-1.,1.,1.,math.pi/4),(-1.,1.,1.,math.pi/4+.5),
        (-1.,1.,1.,.3),(0.,1e6,1e4,10.),(1e12,1.,1.,100.),(0.,1e-10,1e8,1e4),
        (0.,0.,0.,1.),(2.,3.,4.,0.),(0.,1.,1.,1e-12)]
    expected=[2/4,3.5,0.,math.tanh(.5),math.tan(-math.pi/4+.3),10.,1.,1e-9,0.,2.,1e-12]
    result=compute(values=values,device=device)
    np.testing.assert_allclose(result,expected,rtol=3e-12,atol=3e-14)
    inverse=[(-q,-a,b,dt) for q,a,b,dt in values]
    np.testing.assert_allclose(compute(values=inverse,device=device),-result,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_frozen_flow_semigroup_and_nonnegative_dissipation(device):
    rng=np.random.default_rng(610)
    q=rng.uniform(-10.,10.,256);a=rng.uniform(-5.,5.,256);b=10.**rng.uniform(-5.,5.,256)
    dt=10.**rng.uniform(-5.,3.,256)
    full=compute(values=np.column_stack((q,a,b,dt)),device=device)
    half=compute(values=np.column_stack((q,a,b,dt/2)),device=device)
    half=compute(values=np.column_stack((half,a,b,dt/2)),device=device)
    np.testing.assert_allclose(full,half,rtol=2e-12,atol=2e-12)
    unforced=compute(values=np.column_stack((q,np.zeros_like(a),b,dt)),device=device)
    assert np.all(abs(unforced)<=abs(q))
    assert np.all(b*abs(full)**3>=0.)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_production_frozen_pressure_from_rest_and_donor_conservation(device):
    from tests.test_enhanced_production_state import create,STATE_RTOL,STATE_ATOL,VOLUME_ATOL
    from floodwarp.config import WP_FLOAT
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    model=create(bed=np.zeros((5,10)),factor=5,device=device,method='none',momentum_integration='analytic')
    model.set_water_depth(depth=np.array([[.002,.001]]))
    model._component_coefficients()
    d=model._ec_d.numpy();r=model._ec_r.numpy();head=model._ec_forcing.numpy();dt=.1
    a=float(model.gravity)*head/d;b=float(model.gravity)*r/d
    expected=np.sqrt(abs(a)/b)*np.tanh(np.sqrt(abs(a)*b)*dt)*np.sign(a)
    model.step(fixed_dt=dt)
    np.testing.assert_allclose(model._ec_proposed.numpy(),expected,rtol=STATE_RTOL,atol=STATE_ATOL)
    state=model.get_component_state()
    assert state['volume_m3'].min()>=0.
    assert abs(state['volume_m3'].sum(dtype=np.float64)-model.initial_volume)<VOLUME_ATOL
    # Exact frozen integration bounds acceleration by the Manning equilibrium.
    assert np.all(abs(model._ec_proposed.numpy())<=np.sqrt(abs(a)/b)*(1+STATE_RTOL))


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_analytic_momentum_restart_and_lts_zero_level(device):
    from tests.test_enhanced_spatial_lts import make_lts
    from tests.test_enhanced_production_state import STATE_RTOL,STATE_ATOL
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    host=make_lts(device=device,method='none');lts=make_lts(device=device,method='none')
    for model in (host,lts):model._exact_momentum=1
    initial=lts.capture_state();lts._exact_momentum=0;lts.restore_state(state=initial)
    assert lts._exact_momentum==1
    while host.simulation_time<.1-1e-12:host.step(remaining_time=.1-host.simulation_time)
    lts.run_spatially_adaptive_loop(duration=.1,max_levels=0,wet_max_level=0,use_device_graph=False)
    for key,value in host.get_component_state().items():
        np.testing.assert_allclose(value,lts.get_component_state()[key],rtol=STATE_RTOL,atol=STATE_ATOL)
