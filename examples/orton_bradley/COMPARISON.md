# Orton Bradley Park — complete 5 m, 72-hour comparison

**Research demonstration only — not an official flood-hazard assessment.
Independent site-specific validation is required before decision-making.**
See the [developer disclaimer](../../DISCLAIMER.md).

The Orton Bradley demonstration compares uncalibrated local-inertial and full
shallow-water formulations over steep terrain. Both conserve mass within their
recorded numerical tolerances, yet substantial differences occur in predicted
maximum inundation depths and extents. These results highlight limitations of
relying on reduced-complexity hydraulics in rapidly varying terrain, where
calibration against outlet discharge alone may conceal spatial inundation
errors. This comparison is not an outlet-calibration experiment and does not
isolate terrain or formulation as the sole cause of the differences.

Maximum inundation is calculated directly from each pointwise maximum-depth envelope. Δh_max is FloodWarp FV1 maximum minus LISFLOOD-FP FV1 maximum; it is not a maximum instantaneous difference.

Grid: 1123 × 843 at 5 m; 559,985 physical cells; active area 13.999625 km²; raster footprint 23.667225 km². The model domain is not asserted to contain the entire geographical catchment.

| Metric | LISFLOOD-FP FV1 | FloodWarp FV1 | FloodWarp LI |
|---|---:|---:|---:|
| Hydraulic runtime [s] | 1403.55942 | 1906.1062 | 1019.46035 |
| End-to-end component runtime [s] | 1486.86521 | 2007.81904 | 1126.94604 |
| Accepted timesteps | 816514 | 1570053 | 694389 |
| Peak 60 s mean discharge [m³/s] | 161.482999 | 161.172982 | 182.167233 |
| Peak interval ending [h] | 24.4833333 | 24.5 | 24.4166667 |
| Final-depth RMSE [m] | 0 | 0.000240504345 | 0.0036164457 |
| Final wet-union RMSE [m] | 0 | 0.00393011668 | 0.0587762715 |
| Maximum-envelope RMSE [m] | 0 | 0.00143480626 | 0.256379797 |
| Maximum-envelope max absolute error [m] | 0 | 0.119908348 | 6.2438127 |
| Median maximum-envelope absolute error [m] | 0 | 2.86556712e-06 | 0.00281185337 |
| 95th percentile maximum-envelope absolute error [m] | 0 | 0.000382579895 | 0.0419659687 |
| Active-cell fraction with envelope error below 0.10 m | 1 | 0.999996428 | 0.973397502 |
| Hydrograph RMSE [m³/s] | 0 | 0.0452825143 | 3.04098242 |
| Hydrograph NRMSE / reference peak | 0 | 0.000280416605 | 0.0188315949 |
| Peak-discharge relative error | 0 | -0.00191980932 | 0.128089237 |
| Peak timing error [s] | 0 | 60 | -240 |
| Cumulative net export [m³] | 1661083.53 | 1661096.28 | 1660247 |
| Export difference [m³] | 0 | 12.7501666 | -836.525034 |
| Infiltration [m³] | 2135565.07 | 2135548.33 | 2136464.15 |
| Infiltration difference [m³] | 0 | -16.7395771 | 899.084921 |
| Final mass residual [m³] | 2.28174031e-08 | 4.4801709e-09 | 3.11396434e-07 |

| Maximum-event threshold | Method | IoU | False-positive area [m²] | False-negative area [m²] |
|---|---|---:|---:|---:|
| 0.01 m | FloodWarp FV1 | 0.998711452 | 3375 | 3650 |
| 0.01 m | FloodWarp LI | 0.497595955 | 603800 | 2436900 |
| 0.1 m | FloodWarp FV1 | 0.995776095 | 1875 | 900 |
| 0.1 m | FloodWarp LI | 0.697169085 | 123800 | 112075 |

Final-depth wet union uses 0.001 m. RMSEs use active physical cells only; outside-domain nodata is excluded. GeoTIFFs use FP64 and lossless compression; active values have zero conversion error. Active dry cells remain numeric zero.

The reference is an independent adapted external LISFLOOD-FP FV1 run, with the retained source/binary hashes in each record. The matched FloodWarp FV1 benchmark uses contact HLL, FP64, CFL 0.4, maximum dt 1 s, gravity 9.80665 m/s², dry momentum threshold 0.0001 m and explicitly discharge-extrapolated prescribed-stage ghosts. Public FV1 defaults use velocity extrapolation. The benchmark does not qualify every closure or flood scenario.

Rain is the established spatially scaled HIRDS v4 100-year historical design event over 48 hours, followed by recession to 72 hours. Fine input arrays preserve the original PWL forcing, spatial multipliers, Manning field, exponential infiltration f0/fd/k and 212 north-facing stage-0 coastal contacts. Soil and roughness mappings are modelling assumptions, not site calibration.

Timing records describe one completed matched run per solver on the GPU named in provenance.json. End-to-end component timers include diagnostic/checkpoint work and exclude some manifest/checksum publication and shared input loading; they are not independent fresh wall-clock repetitions. No full simulations were rerun to create this release.

Elevation credit: Contains data sourced from the LINZ Data Service licensed for reuse under CC BY 4.0. Environment Canterbury; Christchurch City Council where applicable; surveys by Aerial Surveys, Landpro and AAM NZ; hosted/processed by Toitu Te Whenua Land Information New Zealand.

Software: Copyright Patrick Durney and Lincoln Agritech Ltd; AGPL-3.0-only. Elevation data retain CC BY 4.0. Full source hashes, original conditioning provenance, numerical identities and hardware are in provenance.json and the checksummed records.
