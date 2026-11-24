# Native terrain, complete events and GPU cost

All positions share the same native metre grid origin. Each 10 m elevation averages its four 5 m children. Each row reports the cost of its recorded method, execution policy and common event endpoint. The normalized-error Pareto tables use independently screened references for each hydraulic observable.

| Case | Grid (m) | Method | CFL factor | End (h) | Peak (m³/s) | Export (m³) | Hydraulic time (s) | Steps |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| incised-707-spatial-10240x10240 | 10 | adaptive | 1.0 | 16 | 66.399 | 825,018 | 364.24 | 95,614 |
| incised-707-spatial-10240x10240 | 5 | adaptive | 1.0 | 16 | 48.330 | 809,311 | 2451.83 | 155,174 |
| incised-707-spatial-10240x10240 | 10 | hancock_adaptive | 1.0 | 16 | 79.651 | 825,170 | 74.94 | 95,424 |
| incised-707-spatial-10240x10240 | 5 | hancock_adaptive | 1.0 | 16 | 48.286 | 809,296 | 464.12 | 155,194 |
| incised-707-spatial-10240x10240 | 10 | fv1 | 1.0 | 16 | 48.371 | 910,217 | 38.43 | 48,782 |
| incised-707-spatial-10240x10240 | 5 | fv1 | 1.0 | 16 | 49.186 | 910,143 | 298.66 | 99,493 |

0 of 36 assessed objective/configuration rows have an eligible numerical reference. Multiple objectives on the same simulation share a physical realisation. A frontier can be estimated only among eligible observations for the same case, common time window and reference.

[Eligibility by objective](reference_eligibility.csv) · [Measured windows and full water budgets](../native_results/measured_windows.csv) · [Peak, timing and shape comparisons](../native_results/hydrograph_pareto.csv) · [Depth and extent comparisons](../native_results/recession_depth_pareto.csv)

## Dense and tiled execution

[original-window, 3 matched grid/event comparisons](policy_cost_original_window.pdf)

Each upper-panel point represents a measured forecast window. Each lower-panel point represents a recorded timing interval (minutes: 15). Footprints are sampled at interval end. A ratio below one favours tiles. The same trajectory can contribute to both the initial window and the matched recession products.

[Policy run identities and timings](../native_results/policy_cost_events.csv) · [Individual timing intervals](../native_results/policy_cost_windows.csv)

## Terrain control elevations and connectivity

A 80 m window surrounds the prescribed opening on each of 21 distinct native beds. Repeated storms on a bed count once. A connection requires a continuous path along shared cell faces from west to east. A single water-surface elevation is prescribed across each window. Its height above the native spill controls which paths are available.

| Level above native spill (cm) | Disconnected at 1 m | At 5 m | At 10 m |
|---:|---:|---:|---:|
| 0 | 0 | 16 | 16 |
| 1 | 0 | 14 | 15 |
| 5 | 0 | 8 | 11 |
| 10 | 0 | 3 | 8 |
| 20 | 0 | 0 | 0 |

All grid comparisons start from native metre elevation. The control-stage test is static. Hydraulic peak and arrival errors need separate matched storm observations.

[Spill elevations by terrain](control_stage_sensitivity.csv) · [Map read-back checks](control_map_checks.csv) · [Connectivity figure](control_stage_sensitivity.pdf)

## Held-out Solver Advisor evaluation

The held-out native-terrain evaluation is pending.
