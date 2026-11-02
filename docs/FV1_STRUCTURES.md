# FV1 circular culverts (development branch)

The FV1 solver now accepts **opt-in, disjoint circular culverts** using the implementation ported from Overlander's `dev` branch. The local-inertial hydraulic-structure implementation remains unchanged. This feature is available from the native Python interface and `create_overland_solver`/`create_model`; it is **not** yet exposed through the browser UI.

## Example

```python
import numpy as np

from floodwarp import create_overland_solver

structure = {
    "id": "crossing-1", "type": "culvert", "shape": "circular",
    "inlet_row": 0, "inlet_col": 0,
    "outlet_row": 0, "outlet_col": 2,
    "diameter_m": 1.0, "inlet_invert_m": 0.0, "outlet_invert_m": 0.0,
    "length_m": 10.0, "manning_n": 0.013,
    "entrance_loss_coefficient": 0.5, "exit_loss_coefficient": 1.0,
    "one_way": False, "max_discharge_m3_s": 100.0,
}
solver = create_overland_solver(
    solver_method="fv1", ny=1, nx=3, dx=5.0, dy=5.0,
    bed_elevation=np.zeros((1, 3)), manning_n=0.03,
    active_mask=np.array([[True, False, True]]),
    structures=[structure], device="cpu",
)
solver.set_water_depth(depth=np.array([[1.0, 0.0, 0.0]]))
solver.run_device_loop(duration=10.0, use_cuda_graph=False)
print(solver.get_structure_diagnostics())
```

## Numerical contract and limitations

- Only circular culverts with **distinct, active inlet and outlet cells** are supported. Every endpoint cell may occur in one link only, so networks or shared junctions are not supported.
- Exchange occurs after the conservative FV1 surface update. The link uses a quasi-steady Manning/entrance/exit energy-loss law, permits reverse flow unless `one_way=True`, and is limited by available donor water and pairwise head equalization.
- Donor momentum is withdrawn proportionately; the incoming volume has zero resolved horizontal jet momentum. Pipe inertia, pressurisation dynamics, internal pipe-network hydraulics, bridges, and weir overtopping are not resolved.
- Accepted-step maximum depth includes the composed surface-plus-culvert state. Internal exchange does not create net water; cumulative directional volumes, last discharge, maximum absolute discharge, head difference, and limiter counts are exposed by `get_structure_diagnostics()`.
- Structure geometry and parameters are included in the numerical configuration and checkpoint integrity checks. A checkpoint without the same structures cannot be restored into a structure-enabled solver. Runs with `structures=()` preserve the previous configuration and execution path.
- Endpoint inverts are **absolute elevations** in the DEM vertical datum; they must be provided explicitly and should come from surveyed or clearly qualified estimated data. Do not infer them from DEM levels without an explicit assumption.
- This is a development-only port. It needs independent end-to-end site replay and GPU validation in FloodWarp before being included in a tagged release.

Source: `Lincoln-Environmental/Overlander`, `dev`, `overland_flow/full_swe_culverts.py`, and the corresponding `test_fv1_culverts.py` regression suite.
