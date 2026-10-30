"""Preprocessing shortcuts retain the same transient hydraulic coefficients."""
import numpy as np
import pytest

from floodwarp import enhanced_geometry
from floodwarp.enhanced_reduction import reduce_cell,port_coefficients,transport_star


@pytest.mark.parametrize('spatial',[False,True])
@pytest.mark.parametrize('stage',[.05,.4,2.])
def test_omitting_steady_operator_keeps_every_transient_array(spatial,stage):
    bed=np.array([[0.,.1,.1],[.2,0.,.3],[.3,.3,0.]])
    roughness=np.full_like(bed,.03)+np.arange(bed.size).reshape(bed.shape)*.002 if spatial else .03
    arguments=dict(bed_m=bed,stage_m=stage,resolution_m=1.,manning_n=roughness,
        momentum_source_weights=np.maximum(stage-bed,0.)**(5/3))
    reference=reduce_cell(**arguments)
    result=reduce_cell(**arguments,compute_conductance=False)
    assert result.conductance is None
    assert reference.conductance is not None
    for name,value in vars(reference).items():
        if isinstance(value,np.ndarray) and name!='conductance':
            np.testing.assert_array_equal(value,getattr(result,name),err_msg=name)
    for original,fast in zip(port_coefficients(reduction=reference,formulation='matrix_lumped'),
                             port_coefficients(reduction=result,formulation='matrix_lumped'),strict=True):
        np.testing.assert_array_equal(original,fast)
    for original,fast in zip(transport_star(reduction=reference),transport_star(reduction=result),strict=True):
        np.testing.assert_array_equal(original,fast)


@pytest.mark.parametrize('flat',[False,True])
def test_complete_tables_identical_with_and_without_auxiliary_solve(monkeypatch,flat):
    bed=np.zeros((5,10))
    if not flat:
        bed[2,:]=.2;bed[1,3]=.3;bed[3,7]=.4
    arguments=dict(bed_m=bed,factor=5,resolution_m=1.,manning_n=.03,
        maximum_stage_m=2.,open_sides=(0,1,2,3),formulation='transport_star',
        interpolation_tolerance=.02,maximum_ports_per_cell=100,maximum_rows_per_port=32768,
        unbounded_stage=True)
    fast=enhanced_geometry.build_component_geometry(**arguments)
    def include_steady(**options):
        options['compute_conductance']=True
        return reduce_cell(**options)
    monkeypatch.setattr(enhanced_geometry,'reduce_cell',include_steady)
    reference=enhanced_geometry.build_component_geometry(**arguments)
    for name,value in vars(reference).items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(fast,name),err_msg=name)


def test_exact_flat_closure_never_builds_native_operator(monkeypatch):
    def unexpected(**options):
        raise AssertionError('Flat analytical coefficients should bypass native reductions')
    monkeypatch.setattr(enhanced_geometry,'reduce_cell',unexpected)
    geometry=enhanced_geometry.build_component_geometry(bed_m=np.zeros((5,10)),factor=5,
        resolution_m=1.,manning_n=.03,maximum_stage_m=2.,open_sides=(0,1,2,3),
        formulation='transport_star',interpolation_tolerance=.02,maximum_ports_per_cell=100,
        maximum_rows_per_port=32768,unbounded_stage=True)
    assert len(geometry.state_cell)==2
    assert np.all(geometry.rows[:,2:]>0)
