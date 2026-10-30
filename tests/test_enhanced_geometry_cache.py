"""Parallel preparation/cache reuse cannot alter any hydraulic input or table."""
from pathlib import Path
import json
import numpy as np
import pytest

from floodwarp.enhanced_geometry import build_component_geometry,pack_coefficients
from floodwarp.enhanced_cache import geometry_cache_key,save_geometry_cache,load_geometry_cache
from tests.test_enhanced_production_state import create


def geometry_arguments(*, unbounded):
    bed=np.array([[0.,.3,0.,.4],[.1,.2,.1,.5],[0.,.4,0.,.3],[.1,.2,.1,.4]])
    return dict(bed_m=bed,factor=2,resolution_m=1.,manning_n=.03+np.arange(bed.size).reshape(bed.shape)*.001,
        maximum_stage_m=2.,open_sides=(0,1,2,3),formulation='transport_star',
        interpolation_tolerance=.02,maximum_ports_per_cell=512,maximum_rows_per_port=32768,
        unbounded_stage=unbounded)


@pytest.mark.parametrize('unbounded',[False,True])
def test_parallel_component_preparation_is_bitwise_identical(unbounded):
    arguments=geometry_arguments(unbounded=unbounded)
    serial=build_component_geometry(**arguments,preprocessing_workers=1)
    parallel=build_component_geometry(**arguments,preprocessing_workers=2)
    assert not parallel.diagnostics['connected_domain_identity']
    for name,value in vars(serial).items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(parallel,name),err_msg=name)
    assert serial.diagnostics==parallel.diagnostics
    a=pack_coefficients(geometry=serial,relative_tolerance=.005)
    b=pack_coefficients(geometry=parallel,relative_tolerance=.005,preprocessing_workers=2)
    for name,value in vars(a).items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(b,name),err_msg=name)


def test_cache_roundtrip_and_corruption_are_explicit(tmp_path):
    geometry=pack_coefficients(geometry=build_component_geometry(**geometry_arguments(unbounded=True)),relative_tolerance=.005)
    key='test_geometry'
    assert load_geometry_cache(directory=tmp_path,key=key) is None
    save_geometry_cache(geometry=geometry,directory=tmp_path,key=key)
    result,elapsed=load_geometry_cache(directory=tmp_path,key=key)
    assert elapsed>=0.
    for name,value in vars(geometry).items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(result,name),err_msg=name)
        else:assert value==getattr(result,name)
    path=tmp_path/(key+'.npz')
    with np.load(path,allow_pickle=False) as archive:arrays={name:archive[name] for name in archive.files}
    arrays['sill']=arrays['sill'].copy();arrays['sill'][0]+=.001
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='checksum'):load_geometry_cache(directory=tmp_path,key=key)


@pytest.mark.parametrize('field',['native_identity','face_identity','parameters','encoding','tolerance'])
def test_cache_identity_binds_every_preprocessing_input(field):
    arguments=dict(native_identity='terrain_mask_resolution_manning_hash',face_identity='all_faces',
        parameters=dict(maximum_stage_m=2.,formulation='transport_star',opening_depth_m=1e-5,
            interpolation_tolerance=.02,contact_grouping='connected_runs',inertia_weighting='uniform',
            maximum_ports_per_cell=100,maximum_rows_per_port=1000,unbounded_stage=True),
        encoding='scalar',tolerance=.005)
    first=geometry_cache_key(**arguments)
    arguments[field]=dict(arguments[field],opening_depth_m=1e-4) if field=='parameters' else str(arguments[field])+'changed'
    assert first!=geometry_cache_key(**arguments)


def test_production_cache_reuses_without_reduction_and_rejects_renamed_entry(tmp_path,monkeypatch):
    import floodwarp.enhanced_solver as module
    bed=np.array([[0.,.3,0.,.4],[.1,.2,.1,.5],[0.,.4,0.,.3],[.1,.2,.1,.4]])
    original=create(bed=bed,factor=2,device='cpu',enhanced_geometry_cache_dir=tmp_path)
    assert not original._enhanced_preparation['cache_hit']
    assert original._enhanced_preparation['kind']=='cold_build'
    def unexpected(**arguments):raise AssertionError('Warm geometry cache must not traverse native graphs')
    monkeypatch.setattr(module,'build_component_geometry',unexpected)
    repeated=create(bed=bed,factor=2,device='cpu',enhanced_geometry_cache_dir=tmp_path)
    assert repeated._enhanced_preparation['cache_hit']
    assert repeated._enhanced_preparation['kind']=='validated_cache_load'
    assert repeated._enhanced_preparation['native_preprocessing_s']==0.
    assert repeated._enhanced_preparation['compression_s']==0.
    for model in (original,repeated):
        model.set_water_depth(depth=np.full((2,2),.05));model.step(fixed_dt=.01)
    np.testing.assert_array_equal(original.get_component_state()['volume_m3'],repeated.get_component_state()['volume_m3'])
    path=next(tmp_path.glob('*.npz'))
    renamed=tmp_path/'different_key.npz';renamed.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='identity'):load_geometry_cache(directory=tmp_path,key='different_key')


@pytest.mark.parametrize('backend',['enhanced_native_sparse.py','enhanced_native_compiled.py'])
def test_cache_identity_includes_backend_source_and_internal_numerical_policy(monkeypatch,backend):
    arguments=dict(native_identity='native',face_identity='faces',parameters={},encoding='scalar',tolerance=.005)
    expected=geometry_cache_key(**arguments)
    original_read=Path.read_bytes
    def changed_source(path):
        result=original_read(path)
        return result+b'\n# different numerical policy\n' if path.name==backend else result
    monkeypatch.setattr(Path,'read_bytes',changed_source)
    assert geometry_cache_key(**arguments)!=expected
