"""Factory contracts and import-only benchmark integration; CPU only."""

import importlib

import numpy as np
import pytest

from floodwarp.solver_factory import create_overland_solver


@pytest.mark.parametrize('method',['explicit','fv1'])
def test_all_factory_methods_construct_with_retained_defaults(method):
    bed = np.zeros((2,3))
    options = dict(solver_method=method,ny=2,nx=3,dx=2.,dy=2.,bed_elevation=bed,
        manning_n=np.full(bed.shape,.04),active_mask=np.ones_like(bed,bool),device='cpu')
    solver = create_overland_solver(**options)
    assert solver.minimum_dt == .001
    assert solver.maximum_dt == 60.
    if method != 'fv1':
        assert float(solver.gravity) == 9.80665
    if method == 'explicit':
        assert solver.hydraulic_geometry_mode == 'coarse'
        assert solver.adaptive_dt_growth_factor == 1.5
