"""Run with a freshly installed wheel from a directory outside the checkout."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import floodwarp
from floodwarp import FullSWESolver, LocalInertialSolver
from floodwarp.inputs import terrain_from_array
from floodwarp.model import advance, create_model, diagnostics


def verify_installation(*, expected_version, duration, rainfall):
    module = Path(floodwarp.__file__).resolve()
    assert 'site-packages' in module.parts, module
    assert importlib.metadata.version('floodwarp') == floodwarp.__version__ == expected_version
    terrain = terrain_from_array(elevation=np.zeros((3, 4)), dx=2., dy=2.,
        row_order='south_to_north', vertical_datum='local')
    default = create_model(terrain=terrain, rainfall=rainfall, device='cpu')
    li = create_model(terrain=terrain, solver_method='li', rainfall=rainfall, device='cpu')
    assert isinstance(default.solver, FullSWESolver)
    assert isinstance(li.solver, LocalInertialSolver)
    for model in (default, li):
        advance(model=model, end_time=duration)
        np.testing.assert_allclose(actual=model.solver.get_water_depth(), desired=rainfall*duration,
            rtol=1.e-12, atol=1.e-14)
        assert abs(diagnostics(model=model)['residual_m3']) < 1.e-12
    assert 'matplotlib' not in sys.modules and 'pyogrio' not in sys.modules
    print(json.dumps(dict(version=expected_version, module=str(module), python=sys.version,
        fv1=True, li=True, ui_installed=importlib.util.find_spec('pyogrio') is not None)))


if __name__ == "__main__":
    expected_version = '0.1.2'
    duration = .2
    rainfall = .001
    verify_installation(expected_version=expected_version, duration=duration, rainfall=rainfall)
