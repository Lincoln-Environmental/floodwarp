# FloodWarp

Private **v0.1 extraction candidate** of Overlander: standalone local-inertial
(LI) and FP64 finite-volume shallow-water (FV1) hydraulics on CPU or CUDA.
No other HydroOne repository, prepared-model package or GIS preprocessing
framework is required. Numerical baseline:
`Lincoln-Environmental/Overlander@ba503b8bde51d3f212199e81efc7038d9fffecc8`.

Software redistribution permission is pending. This is not a qualified public
release. See [release status](docs/RELEASE_STATUS.md),
[numerical validation](validation/REPORT.md) and [licensing](LICENSE).

## Install

Use Python 3.14 and a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test,raster,plot]'
python -m pip check
python -m examples.run_synthetic
python -m examples.run_purau
python -m pytest -q
```

`pip install .` installs the CPU hydraulic core. `raster` adds GeoTIFF IO;
`plot` adds figures. Warp also supplies CUDA execution: use `device='cuda:0'`
with a compatible NVIDIA driver. A CUDA toolkit is not needed for CPU use.
No external LISFLOOD executable is installed or bundled.

## Direct inputs

```python
import numpy as np
from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model, advance, diagnostics

terrain = terrain_from_array(
    elevation=np.zeros((10, 10)), dx=5., dy=5.,
    row_order='south_to_north', vertical_datum='local',
)
model = create_model(terrain=terrain, solver_method='li', manning_n=.035,
                     rainfall=20./3.6e6)
advance(model=model, end_time=60.)
print(diagnostics(model=model))
```

Use `solver_method='fv1'` for full SWE. The ordinary LI physical profile remains
the default; FV1 retains `contact_hll` and `velocity_extrapolation`.
Both `hydraulic_geometry_mode='traditional'` and `'enhanced'` remain accessible
through the LI interface and emit **experimental SGS warnings**.

The [input guide](docs/INPUTS.md) covers masks, rainfall series, infiltration,
partial edge openings, stages, restart and raster metadata. Edit paths and
parameters directly in [run_direct.py](examples/run_direct.py).

The included [Purau example](examples/purau/README.md) is a 40×40-cell crop of
real, previously conditioned terrain with a 1 m underlying DEM. It produces
depth maps, interval hydrographs and water balances on CPU. It is an
illustration, not a calibrated flood prediction or a complete catchment model.
