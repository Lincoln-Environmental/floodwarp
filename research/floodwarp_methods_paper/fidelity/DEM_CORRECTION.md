# Incised catchment DEM correction

The earlier refinement copied a 10 m centre elevation into its four 5 m children. The limiter can reconstruct a nonzero slope between adjacent parent centres on the 10 m grid. On the child grid each pair contains a flat tread and a jump. The experiment changes the bed gradient seen by the flux and source operators. Its depths, storage and discharge describe a staircase sensitivity experiment. We withdraw its use for choosing practical grid resolution or ranking solver accuracy.

The corrected 10,240 × 10,240 m terrain uses 104,857,600 native metre elevations. Each 5 m elevation is the mean of 25 native cells. Each 10 m elevation is the mean of the four 5 m cells covering that footprint. Every 10 m elevation was independently checked against its 100 native cells. Maximum absolute difference was 4.55e-13 m. Only the short reach replay holds the original forcing support fixed to isolate the effect of bed construction. [DEM identities](incised_crest/mean_dem_inputs.json)

## Numerical mechanism

The plane test starts a 3 cm uniform sheet from rest on a 10% slope with Manning roughness 0.035. The infinite smooth plane has constant depth and obeys du/dt = gS − g n² u² / h^(4/3). Values below are mass-weighted interior velocities after two seconds. The smooth-plane solution is a reference for the smooth geometry. The staircase has a different bed source term.

| Method | Grid (m) | Bed spacing (m) | Velocity (m/s) | Smooth-plane exact (m/s) |
|---|---:|---:|---:|---:|
| fv2 | 10 | 10 | 0.8530 | 0.8531 |
| fv2 | 5 | 5 | 0.8530 | 0.8531 |
| fv2 | 5 | 10 | 0.0404 | 0.8531 |
| fv2 | 1 | 1 | 0.8530 | 0.8531 |
| fv2 | 1 | 10 | 0.0456 | 0.8531 |
| hancock | 10 | 10 | 0.8530 | 0.8531 |
| hancock | 5 | 5 | 0.8530 | 0.8531 |
| hancock | 5 | 10 | 0.0403 | 0.8531 |
| hancock | 1 | 1 | 0.8530 | 0.8531 |
| hancock | 1 | 10 | 0.0458 | 0.8531 |
| fv1 | 10 | 10 | 0.0287 | 0.8531 |
| fv1 | 5 | 5 | 0.0560 | 0.8531 |
| fv1 | 5 | 10 | 0.0323 | 0.8531 |
| fv1 | 1 | 1 | 0.2362 | 0.8531 |
| fv1 | 1 | 10 | 0.0401 | 0.8531 |

## Replay on the incised terrain

The same 400 m internal cross-section and first 60 seconds of the storm were replayed with FV2 on CPU using the actual initial water and momentum, identical forcing and a common timestep cap. The table uses a 400 m buffer between the measured section and each external reach boundary. The archive includes both buffer sizes. This short replay tests the cause of the initial discrepancy.

| Bed construction | Hydraulic grid (m) | Mean discharge (m³/s) |
|---|---:|---:|
| parent | 10 | 11.1149 |
| replicated | 5 | 1.0450 |
| linear | 5 | 11.1109 |
| sampled | 5 | 11.1084 |
| area_mean_10m | 10 | 11.1149 |
| area_mean_5m | 5 | 11.1084 |

[Matched reach controls](incised_crest/reach_replay.csv) · [Plane controls](incised_crest/plane_controls.csv)

## Historical eight-hour states

Direct native depth threshold counts reproduce every 1, 5 and 10 cm area. The 1 cm measure is a thin-water diagnostic. Accepted face volumes reproduce the outlet and internal hydrograph integrals. The closed faces have zero accepted cumulative transport. Bed, active mask, roughness, rain multiplier and infiltration match across every four-child block in the historical replicated-bed study.

| Comparison | Common wet-area criterion (cm) | Compared 40 m blocks | Depth RMSE (cm) |
|---|---:|---:|---:|
| FV2 grid refinement | 0 | 65536 | 1.531 |
| FV2 grid refinement | 5 | 1561 | 8.179 |
| FV2 grid refinement | 10 | 319 | 11.383 |
| Hancock versus FV2 | 0 | 65536 | 0.213 |
| Hancock versus FV2 | 5 | 1561 | 0.188 |
| Hancock versus FV2 | 10 | 319 | 0.229 |

Masks use the union across the three compared states at hour eight. A threshold of zero includes the full grid. Differences between forecasts quantify their disagreement. [Native checksums and areas](incised_crest/checkpoint_audit.csv) · [Common-support differences](incised_crest/checkpoint_spatial_differences.csv) · [Station hydrographs](incised_crest/checkpoint_station_histories.csv)

## Corrected complete storms

Whole-catchment comparisons start at time zero from 104,857,600 native metre cells. Bed elevation, initial water depth, initial x momentum, rainfall multiplier and infiltration capacity each have an independent native checksum. Each coarser input is its aligned block mean. At the prescribed initial condition, channel water deeper than 1 cm flows at 0.12 m/s. Shallower margins retain their water and start at rest. All methods use this same physical state. A continuation must match the new inputs and its hydraulic checkpoint identity. [Native input identities and block checks](mean_storms/terrain.json)

Read-back of the solver input rasters found different elevations among the four 5 m children in 1,048,576 of 1,048,576 parent footprints. Every 10 m elevation equalled the four-child mean exactly. [Input raster read-back](mean_storms/grid_audit.csv)

The free boundary produced measurable terminal-cell storage on the native-averaged grids. Independent analytical and reach controls move the boundary while holding the measurement location fixed. [Local depth differences and independent outlet checks](native_outlet/README.md)

The corrected whole-event comparison is pending. The original eight-hour staircase result cannot supply it.
