"""CFL-only lookup is the same inertia as the complete momentum lookup."""
import numpy as np
import pytest
import warp as wp

from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.enhanced_geometry import component_volumes


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('encoding',['scalar','uncompressed'])
@pytest.mark.parametrize('formulation',['matrix_lumped','transport_star'])
def test_inertia_only_matches_full_at_activation_and_infinite_head_tail(device,encoding,formulation):
    if device!='cpu' and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    bed=np.ones((5,10));bed[1,:]=0.;bed[3,:]=0.;bed[2,4:7]=.2
    model=LocalInertialSolver(ny=1,nx=2,dx=5.,dy=5.,
        bed_elevation=bed.reshape(1,5,2,5).mean(axis=(1,3)),
        manning_n=np.full((1,2),.03),active_mask=np.ones((1,2),np.int32),
        hydraulic_geometry_mode='enhanced',subgrid_bed_elevation=bed,subgrid_resolution_m=1.,
        device=device,coefficient_encoding=encoding,momentum_formulation=formulation)
    bx=model.boundary_type_qx.numpy();bx[:,0]=3;bx[:,-1]=2
    model.set_boundary_type_qx(boundary_type=bx)
    model.set_boundary_stage_qx(stage=np.full_like(bx,.35,dtype=float))
    for stage in (0.,float(model.dry_depth),.1,.3,8.):
        model.set_component_state(volume=component_volumes(geometry=model._enhanced_geometry,
            native_depth=np.maximum(stage-bed,0.)))
        model._component_coefficients()
        reference=model._ec_d.numpy()
        model._component_coefficients(inertia_only=True)
        np.testing.assert_array_equal(reference,model._ec_d.numpy())
        model._check_component_status()
