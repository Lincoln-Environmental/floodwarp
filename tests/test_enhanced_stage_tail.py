"""Unbounded stage lookup compared with unchanged finite native reductions."""
import numpy as np
import pytest

from floodwarp.enhanced_geometry import build_component_geometry,table_value
from floodwarp.enhanced_solver import EnhancedSGSSolver
from tests.test_enhanced_production_state import device,STATE_RTOL,STATE_ATOL,VOLUME_ATOL


def terrain(*,kind):
    if kind=='flat':return np.zeros((3,6)),.03,None
    bed=np.full((3,6),1.);bed[0,:]=np.array([0.,.1,.15,.1,.05,0.]);bed[2,:]=.05
    if kind=='connected':bed[1,:]=.08
    roughness=.03+np.indices(bed.shape)[1]*.001 if kind=='spatial' else .03
    mask=np.array([[1,0]],np.int32) if kind=='mask' else None
    return bed,roughness,mask


def geometry(*,bed,roughness,mask,formulation,maximum,unbounded):
    return build_component_geometry(bed_m=bed,factor=3,resolution_m=1.,manning_n=roughness,
        active_mask=mask,maximum_stage_m=maximum,unbounded_stage=unbounded,open_sides=(0,1,2,3),
        formulation=formulation,interpolation_tolerance=.0004,
        maximum_ports_per_cell=512,maximum_rows_per_port=32768)


@pytest.mark.parametrize('kind',['flat','connected','parallel','spatial','mask'])
@pytest.mark.parametrize('formulation',['matrix_lumped','transport_star'])
def test_tail_preserves_finite_tables_and_matches_direct_native_reductions(kind,formulation):
    bed,n,mask=terrain(kind=kind);start=float(bed.max()+2.)
    bounded=geometry(bed=bed,roughness=n,mask=mask,formulation=formulation,maximum=start,unbounded=False)
    tail=geometry(bed=bed,roughness=n,mask=mask,formulation=formulation,maximum=start,unbounded=True)
    for name in ('rows','offsets','storage_rows','storage_offsets','pixel_owner','halves','sill','endpoints'):
        np.testing.assert_array_equal(getattr(tail,name),getattr(bounded,name))
    assert np.isinf(tail.maximum_stage)
    assert np.all(tail.tail_rows[:,1:]>0.)
    for stage in (start*1.01,start*2.,start*1000.):
        direct=geometry(bed=bed,roughness=n,mask=mask,formulation=formulation,maximum=stage,unbounded=False)
        np.testing.assert_array_equal(direct.connection_keys,tail.connection_keys)
        x=1./(stage-tail.tail_reference)
        for f,sill in enumerate(tail.sill):
            expected=np.zeros(2);actual=np.zeros(2)
            for side in (0,1):
                p=direct.halves[f,side]
                if p<0:continue
                expected+=table_value(stage=stage,sill=sill,rows=direct.rows[direct.offsets[p]:direct.offsets[p+1]])
                p=tail.tail_halves[f,side]
                rows=tail.tail_rows[tail.tail_offsets[p]:tail.tail_offsets[p+1]]
                normalized=np.array([np.interp(x,rows[:,0],rows[:,1]),np.interp(x,rows[:,0],rows[:,2])])
                actual+=normalized/np.array([stage-sill,(stage-sill)**(10/3)])
            np.testing.assert_allclose(actual,expected,rtol=.00012,atol=0.)


def build_solver(*,device,encoding,maximum=None):
    bed,n,mask=terrain(kind='parallel')
    return EnhancedSGSSolver(ny=1,nx=2,dx=3.,dy=3.,bed_elevation=np.zeros((1,2)),
        manning_n=np.full((1,2),n),active_mask=np.ones((1,2),np.int32),device=device,
        subgrid_bed_elevation=bed,subgrid_resolution_m=1.,maximum_stage_m=maximum,
        coefficient_encoding=encoding,coefficient_tolerance=.005,track_max_water_depth=True,
        minimum_dt=1e-8,maximum_dt=.005)


@pytest.mark.parametrize('encoding',['uncompressed','scalar'])
def test_default_stage_tail_runs_sources_and_momentum_above_core_range(device,encoding):
    model=build_solver(device=device,encoding=encoding)
    assert model._enhanced_geometry.tail_start==4.
    stage=np.array([[8.,6.]]);model.set_water_surface_elevation(stage=stage)
    initial=model.get_component_state()['volume_m3'].sum(dtype=np.float64)
    model.set_source_rates(rain=np.array([[.002,.001]]),infil=np.zeros((1,2)))
    result=model.run_device_loop(duration=.02,sync_interval=7,use_cuda_graph=device!='cpu')
    state=model.get_component_state();ledger=model.cumulative_accounting()
    assert result['steps']==4
    assert state['stage_m'].min()>4.
    assert state['volume_m3'].min()>0.
    assert np.any(abs(state['discharge_m3_per_s'])>0.)
    assert state['volume_m3'].sum(dtype=np.float64)-initial==pytest.approx(ledger['rain'],abs=VOLUME_ATOL*100)
    saved=model.capture_state();expected={name:value.copy() for name,value in state.items()}
    model.step(fixed_dt=.005);model.restore_state(state=saved)
    for name,value in expected.items():np.testing.assert_allclose(model.get_component_state()[name],value,rtol=STATE_RTOL,atol=STATE_ATOL)


def test_high_stage_default_component_cpu_cuda_parity():
    import warp as wp
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    models=[build_solver(device=device,encoding='scalar') for device in ('cpu','cuda:0')]
    for model in models:
        model.set_water_surface_elevation(stage=np.array([[8.,6.]]))
        for _ in range(10):model.step(fixed_dt=.005)
    for name,value in models[0].get_component_state().items():
        np.testing.assert_allclose(models[1].get_component_state()[name],value,rtol=STATE_RTOL,atol=STATE_ATOL)
