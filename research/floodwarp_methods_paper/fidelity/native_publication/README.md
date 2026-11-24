# Native terrain, complete events and GPU cost

All positions share the same native metre grid origin. Each 10 m elevation averages its four 5 m children. Each row reports the cost of its recorded method, execution policy and common event endpoint. The normalized-error Pareto tables use independently screened references for each hydraulic observable.

The primary event rows use the current numerical source `7e935dd08d51`.

The internal gauge spans the middle column of x faces. Both peaks use interval-mean discharge from accepted face volumes. Positive net transport points downstream.

| Case | Grid (m) | Method | Execution | CFL factor | End (h) | Cadence (s) | Internal peak (m³/s) | Outlet peak (m³/s) | Net export (m³) | Primary hydraulic time (s) | Steps |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fan-711-spatial-10000x5000 | 10 | FV1 | sparse | 1.0 | 16 | 300 | 25.075 | 15.220 | 168,125 | 12.45 | 29,494 |
| fan-711-spatial-10000x5000 | 10 | Hancock | dense | 1.0 | 16 | 300 | 24.519 | 15.825 | 196,205 | 17.99 | 23,975 |
| fan-711-spatial-10000x5000 | 10 | Hancock | tiles | 1.0 | 16 | 300 | 24.519 | 15.825 | 196,205 | 13.28 | 23,975 |
| coastal-703-moderate-10240x10240 | 10 | FV1 | sparse | 1.0 | 32 | 300 | 86.612 | 131.717 | 4,655,733 | 183.92 | 85,542 |
| coastal-703-moderate-10240x10240 | 10 | Hancock | dense | 1.0 | 32 | 300 | 73.261 | 112.551 | 3,785,687 | 355.17 | 93,585 |
| coastal-703-moderate-10240x10240 | 10 | Hancock | tiles | 1.0 | 32 | 300 | 73.261 | 112.551 | 3,785,687 | 322.16 | 93,585 |
| coastal-703-moderate-10240x10240 | 10 | Hancock | auto → dense | 1.0 | 32 | 300 | 73.261 | 112.551 | 3,785,687 | 352.17 | 93,585 |
| floodplain-702-moderate-10240x10240 | 10 | FV1 | sparse | 1.0 | 28 | 300 | 139.438 | 141.054 | 5,818,422 | 138.20 | 74,331 |
| floodplain-702-moderate-10240x10240 | 10 | Hancock | dense | 1.0 | 28 | 300 | 111.792 | 123.695 | 4,695,050 | 302.68 | 79,979 |
| floodplain-702-moderate-10240x10240 | 10 | Hancock | tiles | 1.0 | 28 | 300 | 111.792 | 123.695 | 4,695,050 | 270.09 | 79,979 |
| floodplain-702-moderate-10240x10240 | 10 | Hancock | auto → dense | 1.0 | 28 | 300 | 111.792 | 123.695 | 4,695,050 | 298.86 | 79,979 |
| plain-709-spatial-1000x1000 | 1 | FV1 | sparse | 1.0 | 20 | 300 | 0.358 | 0.016 | 280 | 347.78 | 380,455 |
| plain-709-spatial-1000x1000 | 1 | Hancock | dense | 1.0 | 20 | 300 | 0.356 | 0.035 | 537 | 861.35 | 395,545 |
| plain-709-spatial-1000x1000 | 1 | Hancock | tiles | 1.0 | 20 | 300 | 0.356 | 0.035 | 537 | 425.32 | 395,545 |
| plain-709-spatial-1000x1000 | 1 | Hancock | auto → tiles | 1.0 | 20 | 300 | 0.356 | 0.035 | 537 | 425.50 | 395,545 |
| plain-709-spatial-1000x1000 | 1 | FV2 RK2 | dense | 1.0 | 20 | 300 | 0.356 | 0.035 | 538 | 1488.20 | 393,719 |
| plain-709-spatial-1000x1000 | 1 | FV2 RK2 | dense | 0.5 | 20 | 300 | 0.356 | 0.035 | 538 | 2973.35 | 787,273 |
| plain-709-spatial-1000x1000 | 5 | FV1 | sparse | 1.0 | 20 | 300 | 0.371 | 0.015 | 311 | 7.30 | 72,788 |
| plain-709-spatial-1000x1000 | 5 | Hancock | dense | 1.0 | 20 | 300 | 0.353 | 0.030 | 471 | 10.76 | 72,848 |
| plain-709-spatial-1000x1000 | 5 | Hancock | tiles | 1.0 | 20 | 300 | 0.353 | 0.030 | 471 | 11.74 | 72,848 |
| plain-709-spatial-1000x1000 | 5 | Hancock | auto → dense | 1.0 | 20 | 300 | 0.353 | 0.030 | 471 | 10.45 | 72,848 |
| plain-709-spatial-1000x1000 | 5 | FV2 RK2 | dense | 1.0 | 20 | 300 | 0.353 | 0.030 | 473 | 14.49 | 70,200 |
| plain-709-spatial-1000x1000 | 10 | FV1 | sparse | 1.0 | 20 | 300 | 0.393 | 0.014 | 344 | 2.29 | 31,722 |
| plain-709-spatial-1000x1000 | 10 | Hancock | dense | 1.0 | 20 | 300 | 0.349 | 0.024 | 392 | 2.93 | 30,913 |
| plain-709-spatial-1000x1000 | 10 | Hancock | tiles | 1.0 | 20 | 300 | 0.349 | 0.024 | 392 | 3.26 | 30,913 |
| plain-709-spatial-1000x1000 | 10 | Hancock | auto → dense | 1.0 | 20 | 300 | 0.349 | 0.024 | 392 | 2.84 | 30,913 |
| plain-709-spatial-1000x1000 | 10 | FV2 RK2 | dense | 1.0 | 20 | 300 | 0.349 | 0.024 | 394 | 3.61 | 29,207 |
| hill-706-back-10240x10240 | 10 | FV1 | sparse | 1.0 | 16 | 300 | 698.384 | 1174.700 | 13,384,371 | 130.11 | 63,168 |
| hill-706-back-10240x10240 | 10 | Hancock | dense | 1.0 | 16 | 300 | 729.513 | 1251.557 | 13,255,499 | 206.77 | 52,330 |
| hill-706-back-10240x10240 | 10 | Hancock | tiles | 1.0 | 16 | 300 | 729.513 | 1251.557 | 13,255,499 | 192.20 | 52,330 |
| hill-706-back-10240x10240 | 10 | Hancock | auto → dense | 1.0 | 16 | 300 | 729.513 | 1251.557 | 13,255,499 | 202.95 | 52,330 |
| plain-710-spatial-5000x5000 | 10 | FV1 | sparse | 1.0 | 60 | 300 | 4.640 | 0.000 | 0 | 11.72 | 57,951 |
| plain-710-spatial-5000x5000 | 10 | Hancock | dense | 1.0 | 60 | 300 | 4.186 | 0.000 | 0 | 17.92 | 37,647 |
| plain-710-spatial-5000x5000 | 10 | Hancock | tiles | 1.0 | 60 | 300 | 4.186 | 0.000 | 0 | 10.29 | 37,647 |
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

40 of 432 assessed objective/configuration rows have an eligible numerical reference. Multiple objectives on the same simulation share a physical realisation. A frontier can be estimated only among eligible observations for the same case, common time window and reference.

[Eligibility by objective](reference_eligibility.csv) · [Measured windows and full water budgets](../native_results/measured_windows.csv) · [Peak, timing and shape comparisons](../native_results/hydrograph_pareto.csv) · [Depth and extent comparisons](../native_results/recession_depth_pareto.csv)

## Measured budget decisions

Each decision fixes the event, reference, gauge or spatial support, cadence and physical endpoint. Tolerance compliance includes the measured reference sensitivity. An empty selection records a budget or tolerance for which no qualified choice was measured.

Lowest hydraulic cost meeting a tolerance [for discharge and water accounts](../native_results/least_cost_fidelity.csv) · [for depth and inundation](../native_results/least_cost_depth_extent.csv). Lowest qualified error within an FV1 runtime budget [for discharge and water accounts](../native_results/equal_cost_fidelity.csv) · [for depth and inundation](../native_results/equal_cost_depth_extent.csv).

## Dense and tiled execution

[complete-native-metre, 11 matched grid/event comparisons](policy_cost_complete_native_metre.pdf)
[original-window, 11 matched grid/event comparisons](policy_cost_original_window.pdf)

Each upper-panel point represents a measured forecast window. Each lower-panel point represents a recorded timing interval (minutes: 15). Footprints are sampled at interval end. A ratio below one favours tiles. The same trajectory can contribute to both the initial window and the matched recession products.

[Policy run identities and timings](../native_results/policy_cost_events.csv) · [Individual timing intervals](../native_results/policy_cost_windows.csv)

### Automatic policy over the complete event

| Case | Grid (m) | End (h) | Auto chose | Fastest explicit | Chosen explicit / fastest explicit | Actual auto / chosen explicit |
|---|---:|---:|---|---|---:|---:|
| coastal-703-moderate-10240x10240 | 10 | 32 | dense | tiles | 1.102 | 0.992 |
| fan-705-spatial-5120x5120 | 5 | 12 | tiles | tiles | 1.000 | 0.999 |
| floodplain-702-moderate-10240x10240 | 10 | 28 | dense | tiles | 1.121 | 0.987 |
| hill-706-back-10240x10240 | 10 | 16 | dense | tiles | 1.076 | 0.982 |
| plain-701-spatial-10240x10240 | 10 | 52 | tiles | tiles | 1.000 | 1.000 |
| plain-709-spatial-1000x1000 | 1 | 20 | tiles | tiles | 1.000 | 1.000 |
| plain-709-spatial-1000x1000 | 5 | 20 | dense | dense | 1.000 | 0.971 |
| plain-709-spatial-1000x1000 | 10 | 20 | dense | dense | 1.000 | 0.969 |
| rolling-704-soil-5120x5120 | 5 | 12 | dense | tiles | 1.139 | 1.067 |

The first ratio measures the cost of selecting a schedule using the two explicit runs. The second compares the measured automatic run with its selected explicit counterpart and includes timing variability. Each cost uses its available hydraulic timing median. [Run identities, sample sizes and timing ranges](execution_choice_cost.csv)

0 of 49 matched full-recession configurations have at least three independent hydraulic timings. Cost comparisons use their median. Other configurations retain their measured single-run costs where available. End-to-end timing includes setup and recorded continuation work.

[Timing identities, quartiles and range](../native_results/timing_distribution.csv)

### Characteristic-wave timestep during recession

At 20 h in plain-709-spatial-1000x1000, the fastest cell-centred signal lay at domain coordinates (876.5, 423.5) m. The local depth was 0.1587 m, the gravity-wave speed 1.2475 m/s and the largest axial flow velocity 7.5e-07 m/s. The cell estimate implied a 0.16031458 s CFL bound. The hydraulic kernel recorded 0.16031457 s across reconstructed faces and a 0.15229884 s proposal for the next step. The cell location and the global face minimum are independent measurements at one checkpoint. [Checkpoint, field fingerprints and estimates](../wave_clock/42545c16666c1c94109b354a/72000.json)

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
