"""A completed output boundary must not zero the next source/CFL forecast."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.rainfall import Hyetograph
from floodwarp.config import NP_FLOAT,PRECISION_MODE


@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    return request.param


def source_model(*, device, mode):
    if PRECISION_MODE=='mixed' and mode!='coarse':
        pytest.skip('Traditional/enhanced SGS do not support conditioned mixed precision')
    options={}
    if mode!='coarse':options.update(hydraulic_geometry_mode=mode,
        subgrid_bed_elevation=np.zeros((4,8)),subgrid_resolution_m=.5)
    return LocalInertialSolver(ny=2,nx=4,dx=1.,dy=1.,bed_elevation=np.zeros((2,4)),
        manning_n=np.full((2,4),.035),active_mask=np.ones((2,4),np.int32),device=device,
        maximum_dt=60.,minimum_dt=.001,adaptive_dt_growth_factor=None,infiltration_method='none',**options)


@pytest.mark.parametrize('mode',['coarse','traditional','enhanced'])
@pytest.mark.parametrize('graph',[False,None])
def test_output_checkpoint_does_not_change_next_cfl_forecast(device,mode,graph):
    rain=Hyetograph(node_times_s=np.array([0.,30.,60.]),
        node_rates_m_per_s=np.array([0.,0.,72./3.6e6]),storm_type='rising',metadata={})
    traced=source_model(device=device,mode=mode)
    plain=source_model(device=device,mode=mode)
    actual=traced.run_device_loop(duration=60.,rainfall_hyetograph=rain,use_cuda_graph=graph,
        checkpoint_times_s=[30.],field_checkpoint_indices=[0],dt_history_capacity=10000)
    expected=plain.run_device_loop(duration=60.,rainfall_hyetograph=rain,use_cuda_graph=graph,
        dt_history_capacity=10000)
    assert actual['steps']==expected['steps']
    np.testing.assert_allclose(actual['dt_history'],expected['dt_history'],atol=1e-9,rtol=1e-10)
    times=np.cumsum(actual['dt_history'])
    assert np.all(actual['dt_history']>0.) and np.count_nonzero(np.isclose(times,30.,atol=1e-9,rtol=0.))==1
    index=int(np.argmin(np.abs(times-30.)))+1
    # Forecast after t=30 includes at most 30 s of the 72 mm/h endpoint.
    if mode!='enhanced':
        bound=float(traced.cfl_number)/np.sqrt(float(traced.gravity)*30.*72./3.6e6)
        assert actual['dt_history'][index]<=bound*(1+max(1e-10,8.*np.finfo(NP_FLOAT).eps))
    np.testing.assert_allclose(traced.get_water_depth(),plain.get_water_depth(),atol=1e-5,rtol=1e-4)
    ledger=traced.cumulative_accounting()
    storage=float(traced.get_water_depth().sum(dtype=np.float64))
    supplied=8.*.5*30.*72./3.6e6
    # FP32 state/source products incur bounded state rounding. Keep the
    # original FP64 bound and scale the additional allowance by water volume.
    volume_tolerance=1e-10+16.*np.finfo(NP_FLOAT).eps*supplied
    assert abs(ledger['rain']-ledger['infiltration']-storage)<volume_tolerance
    assert ledger['rain']==pytest.approx(supplied,abs=volume_tolerance)
