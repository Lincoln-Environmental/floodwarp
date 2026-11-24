# Native terrain, complete events and GPU cost

All positions share the same native metre grid origin. Each 10 m elevation averages its four 5 m children. Each row reports the cost of its recorded method, execution policy and common event endpoint. The normalized-error Pareto tables use independently screened references for each hydraulic observable.

The primary event rows use the current numerical source `7e935dd08d51`.

The internal gauge spans the middle column of x faces. Both peaks use interval-mean discharge from accepted face volumes. Positive net transport points downstream.

| Case | Grid (m) | Method | Execution | CFL factor | End (h) | Cadence (s) | Internal peak (m³/s) | Outlet peak (m³/s) | Net export (m³) | Primary hydraulic time (s) | Steps |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal-703-moderate-10240x10240 | 10 | FV1 | sparse | 1.0 | 32 | 300 | 86.612 | 131.717 | 4,655,733 | 183.92 | 85,542 |
| coastal-703-moderate-10240x10240 | 10 | Hancock | dense | 1.0 | 32 | 300 | 73.261 | 112.551 | 3,785,687 | 355.17 | 93,585 |
| coastal-703-moderate-10240x10240 | 10 | Hancock | tiles | 1.0 | 32 | 300 | 73.261 | 112.551 | 3,785,687 | 322.16 | 93,585 |
| coastal-703-moderate-10240x10240 | 10 | Hancock | auto → dense | 1.0 | 32 | 300 | 73.261 | 112.551 | 3,785,687 | 352.17 | 93,585 |
| floodplain-702-moderate-10240x10240 | 10 | FV1 | sparse | 1.0 | 28 | 300 | 139.438 | 141.054 | 5,818,422 | 138.20 | 74,331 |
| floodplain-702-moderate-10240x10240 | 10 | Hancock | dense | 1.0 | 28 | 300 | 111.792 | 123.695 | 4,695,050 | 302.68 | 79,979 |
| floodplain-702-moderate-10240x10240 | 10 | Hancock | tiles | 1.0 | 28 | 300 | 111.792 | 123.695 | 4,695,050 | 270.09 | 79,979 |
| floodplain-702-moderate-10240x10240 | 10 | Hancock | auto → dense | 1.0 | 28 | 300 | 111.792 | 123.695 | 4,695,050 | 298.86 | 79,979 |
| plain-701-spatial-10240x10240 | 10 | FV1 | sparse | 1.0 | 52 | 300 | 11.935 | 0.000 | 0 | 35.74 | 48,849 |
| plain-701-spatial-10240x10240 | 10 | Hancock | dense | 1.0 | 52 | 300 | 10.643 | 0.000 | 0 | 83.41 | 41,769 |
| plain-701-spatial-10240x10240 | 10 | Hancock | tiles | 1.0 | 52 | 300 | 10.643 | 0.000 | 0 | 33.69 | 41,769 |
| plain-701-spatial-10240x10240 | 10 | Hancock | auto → tiles | 1.0 | 52 | 300 | 10.643 | 0.000 | 0 | 33.68 | 41,769 |
| plain-701-spatial-10240x10240 | 10 | LI | native | 1.0 | 52 | 300 | 10.579 | 0.000 | 0 | 42.69 | 18,644 |
| plain-701-spatial-10240x10240 | 10 | MUSCL–Euler | dense | 1.0 | 52 | 300 | 10.300 | 0.000 | 0 | 90.75 | 41,095 |
| plain-701-spatial-10240x10240 | 10 | FV2 RK2 | dense | 1.0 | 52 | 300 | 10.643 | 0.000 | 0 | 159.34 | 42,003 |
| rolling-704-soil-5120x5120 | 5 | FV1 | sparse | 1.0 | 12 | 300 | 74.948 | 61.208 | 292,147 | 69.96 | 37,349 |
| rolling-704-soil-5120x5120 | 5 | Hancock | dense | 1.0 | 12 | 300 | 63.865 | 25.644 | 130,400 | 159.50 | 48,422 |
| rolling-704-soil-5120x5120 | 5 | Hancock | tiles | 1.0 | 12 | 300 | 63.865 | 25.644 | 130,400 | 140.04 | 48,422 |
| rolling-704-soil-5120x5120 | 5 | Hancock | auto → dense | 1.0 | 12 | 300 | 63.865 | 25.644 | 130,400 | 170.14 | 48,422 |
| fan-705-spatial-5120x5120 | 5 | FV1 | sparse | 1.0 | 12 | 300 | 14.151 | 10.590 | 108,524 | 32.62 | 36,855 |
| fan-705-spatial-5120x5120 | 5 | Hancock | dense | 1.0 | 12 | 300 | 13.657 | 10.125 | 112,186 | 53.59 | 35,833 |
| fan-705-spatial-5120x5120 | 5 | Hancock | tiles | 1.0 | 12 | 300 | 13.657 | 10.125 | 112,186 | 40.62 | 35,833 |
| fan-705-spatial-5120x5120 | 5 | Hancock | auto → tiles | 1.0 | 12 | 300 | 13.657 | 10.125 | 112,186 | 40.60 | 35,833 |

0 of 207 assessed objective/configuration rows have an eligible numerical reference. Multiple objectives on the same simulation share a physical realisation. A frontier can be estimated only among eligible observations for the same case, common time window and reference.

[Eligibility by objective](reference_eligibility.csv) · [Measured windows and full water budgets](../native_results/measured_windows.csv) · [Peak, timing and shape comparisons](../native_results/hydrograph_pareto.csv) · [Depth and extent comparisons](../native_results/recession_depth_pareto.csv)

## Dense and tiled execution

[complete-native-metre, 5 matched grid/event comparisons](policy_cost_complete_native_metre.pdf)
[original-window, 5 matched grid/event comparisons](policy_cost_original_window.pdf)

Each upper-panel point represents a measured forecast window. Each lower-panel point represents a recorded timing interval (minutes: 15). Footprints are sampled at interval end. A ratio below one favours tiles. The same trajectory can contribute to both the initial window and the matched recession products.

[Policy run identities and timings](../native_results/policy_cost_events.csv) · [Individual timing intervals](../native_results/policy_cost_windows.csv)

0 of 23 matched full-recession configurations have at least three independent hydraulic timings. Cost comparisons use their median. Other configurations retain their measured single-run costs where available. End-to-end timing includes setup and recorded continuation work.

[Timing identities, quartiles and range](../native_results/timing_distribution.csv)

## Terrain control elevations and connectivity

A 80 m window surrounds the prescribed opening on each of 25 distinct native beds. Repeated storms on a bed count once. A connection requires a continuous path along shared cell faces from west to east. A single water-surface elevation is prescribed across each window. Its height above the native spill controls which paths are available.

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
