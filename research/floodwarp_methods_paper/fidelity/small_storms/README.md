# Recession in the 48 small six-hour rainfall events

All 48 events completed at 10 m on the same 120 × 80 m physical domain, with 5 m and 2.5 m refinements. 31 needed additional recession after hour 8. Each group uses its slowest passing configuration to determine the common end time.

| Common end (h) | Events |
|---:|---:|
| 8 | 17 |
| 12 | 21 |
| 16 | 2 |
| 20 | 2 |
| 24 | 2 |
| 40 | 1 |
| 48 | 1 |
| 68 | 1 |
| 76 | 1 |

Each 10 m forecast is compared with the 2.5 m adaptive FV2 outlet hydrograph for the same physical event. Its numerical sensitivity is the largest observed change against 5 m FV2, 2.5 m FV2 with a halved CFL setting and 2.5 m Hancock. The peak sensitivity limit is 5% and the width limit is 10%. Each observable must also pass the discharge and temporal sampling checks. Positive centroid bias indicates later passage. Negative peak bias indicates attenuation. For the signed quantities, the final two columns count comparisons whose bias exceeds the reference-sensitivity envelope. Width is the central 80% discharge duration.

| 10 m method | Observable | Qualified / 48 | Median signed (%) | Median absolute (%) | Below envelope | Above envelope |
|---|---|---:|---:|---:|---:|---:|
| FV1 contact-HLL | Peak bias | 24 | +0.08 | 2.06 | 10 | 11 |
| FV1 contact-HLL | Discharge volume | 10 | -1.02 | 1.02 | 9 | 0 |
| FV1 contact-HLL | Centroid delay | 13 | +1.92 | 2.62 | 5 | 7 |
| FV1 contact-HLL | Central 80% duration | 13 | +1.26 | 6.79 | 6 | 7 |
| FV1 contact-HLL | Redistributed volume | 12 | +2.75 | 2.75 | 0 | 12 |
| Hancock dense | Peak bias | 24 | -0.35 | 2.63 | 11 | 7 |
| Hancock dense | Discharge volume | 10 | -1.10 | 1.10 | 9 | 0 |
| Hancock dense | Centroid delay | 13 | +1.13 | 2.12 | 4 | 8 |
| Hancock dense | Central 80% duration | 13 | +1.06 | 5.32 | 6 | 7 |
| Hancock dense | Redistributed volume | 12 | +2.51 | 2.51 | 0 | 12 |
| FV2 RK2 | Peak bias | 24 | +0.00 | 2.00 | 10 | 5 |
| FV2 RK2 | Discharge volume | 10 | -1.01 | 1.01 | 9 | 0 |
| FV2 RK2 | Centroid delay | 13 | +1.09 | 2.04 | 4 | 8 |
| FV2 RK2 | Central 80% duration | 13 | +0.82 | 5.29 | 6 | 7 |
| FV2 RK2 | Redistributed volume | 12 | +2.56 | 2.56 | 0 | 12 |

## Paired hydraulic differences

Positive gain means a reduction in absolute error. The last columns count gains or losses exceeding the sum of the two reference-sensitivity estimates.

| Comparison | Metric | Matched events | Median gain (points) | Better beyond sensitivity | Worse beyond sensitivity |
|---|---|---:|---:|---:|---:|
| FV1 contact-HLL to Hancock dense | Peak bias | 24 | +0.17 | 9 | 2 |
| FV1 contact-HLL to Hancock dense | Redistributed volume | 12 | +1.14 | 6 | 0 |
| FV1 contact-HLL to Hancock dense | Central 80% duration | 13 | +0.20 | 4 | 0 |
| Hancock dense to FV2 RK2 | Peak bias | 24 | +0.14 | 4 | 0 |
| Hancock dense to FV2 RK2 | Redistributed volume | 12 | +0.04 | 0 | 0 |
| Hancock dense to FV2 RK2 | Central 80% duration | 13 | +0.13 | 1 | 0 |

## Measured runtime

| Method | Median runtime / FV1 | At or below 1.5× |
|---|---:|---:|
| Hancock dense | 1.358 | 30/48 |
| Hancock tiled | 2.208 | 0/48 |
| FV2 RK2 | 1.888 | 0/48 |

These grids contain 96 cells at the candidate resolution. The table supports event-scale numerical comparisons on this geometry. The million-cell operational cases have separate measurements.

[Paired outlet results and reference IDs](outlet_metrics.csv) · [Paired differences](paired_differences.csv) · [Matched horizons](recession_horizons.csv) · [Final two-hour transport and storage](recession_terminal.csv)

## Peak discharge by landform

Each value is median absolute peak bias over the storms that qualify for all three methods. The counts give the number of matched storms and distinct terrain seeds. Results for a single qualified storm describe that case only.

| Landform | Matched / total storms | Terrain seeds in matched storms | FV1 | Hancock dense | FV2 RK2 |
|---|---:|---:|---:|---:|---:|
| Coastal | 0/6 | 0 | unresolved | unresolved | unresolved |
| Fan | 4/6 | 3 | 1.64% | 2.18% | 1.57% |
| Floodplain | 1/6 | 1 | 23.87% | 6.92% | 8.11% |
| Hill | 5/6 | 3 | 0.49% | 0.40% | 0.07% |
| Incised | 5/6 | 3 | 0.65% | 0.05% | 0.11% |
| Mixed | 1/6 | 1 | 3.58% | 5.48% | 5.34% |
| Plain | 4/6 | 2 | 17.50% | 9.06% | 8.49% |
| Rolling | 4/6 | 3 | 10.24% | 10.57% | 9.93% |

[Signed distributions and relative runtime for every landform and objective](landform_fidelity.csv)

Dense and tiled Hancock reproduced the same interval transports and accepted step counts in all 48 paired forecasts. Final conservative depth and both momentum components, cumulative face volumes, soil history and maximum depth had identical FP64 byte hashes in the 31 extended forecasts with retained solver checkpoints. [Exact fields compared in each forecast](recession_execution_parity.csv)

## Depth and wave errors on the same forecast

The table counts how often a reference-qualified FV1 final-depth RMSE of at most 1 mm accompanied an outlet error above the specified tolerance. A comparison needs independent reference support for both quantities. Each row uses a different hydraulic observable.

| Observable | Jointly assessable | Depth within threshold | Outlet exceeds 5% | Outlet exceeds 10% | Outlet exceeds 20% |
|---|---:|---:|---:|---:|---:|
| Peak bias | 12 | 6 | 3 | 3 | 1 |
| Redistributed volume | 6 | 2 | 1 | 0 | 0 |
| Central 80% duration | 6 | 2 | 2 | 1 | 0 |
| Centroid delay | 7 | 3 | 2 | 1 | 0 |

[Final depth, maximum depth, connectivity and reference sensitivities](recession_depth.csv) · [Measured depth and extent frontiers](recession_depth_pareto.csv)

[Final depth against outlet peak and shape errors](../figures/final_depth_and_outlet_wave.pdf)
