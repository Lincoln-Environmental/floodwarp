# Full-event method and grid comparisons

The small tests use a common 240 × 160 m domain. All errors use area-mean depths on a common 10 m support and interval-mean discharge. A 1 m forecast compared against a 1 m FV2 reference tests formulation differences at that resolution. Channels narrower than the evaluation support can have greater local errors than these averages indicate.

## Landform comparisons

The entries are medians over the six rain events at 10 m. Depth and outlet columns are absolute differences from the finest available FV2 solution for the same physical case. Reference uncertainty varies by event.

| Landform | Hancock/FV1 time | FV1 depth (mm) | Hancock depth (mm) | FV1 outlet (m³/s) | Hancock outlet (m³/s) | Hancock 1 cm space–time IoU |
|---|---|---|---|---|---|---|
| Coastal | 1.58 | 0.54 | 0.54 | 0.0158 | 0.0064 | 0.90 |
| Fan | 1.68 | 8.05 | 8.83 | 0.0052 | 0.0039 | 0.55 |
| Floodplain | 1.72 | 29.29 | 3.58 | 0.0322 | 0.0086 | 0.85 |
| Hill | 1.87 | 3.15 | 3.69 | 0.0094 | 0.0101 | 0.67 |
| Incised | 1.89 | 4.03 | 5.59 | 0.0115 | 0.0033 | 0.55 |
| Mixed | 2.25 | 122.55 | 28.32 | 0.1001 | 0.0480 | 0.71 |
| Plain | 1.70 | 0.46 | 0.28 | 0.0032 | 0.0011 | 0.99 |
| Rolling | 1.68 | 6.80 | 7.13 | 0.0094 | 0.0077 | 0.67 |

## Resolution of the reference

For a depth comparison, successive refinements must differ by at most 1 mm RMSE or 2% relative L2. For discharge they must differ by at most 5% of the finer peak over the whole hydrograph, with an exact match when the reference has zero flow. These are engineering convergence screens. They do not bound a shared formulation or boundary error. The storm comparisons use FV2 grid refinement, plus independent formulations at 1 m on selected cases. The analytical controls have separate exact references.

Sampled terrain passed both screens on 32 of 48 cases.

Fixed terrain passed both screens on 1 of 3 cases.

For a pairwise advantage we additionally require the difference of the two forecast error norms to exceed twice the observed fine-grid refinement difference. The triangle inequality motivates this margin. A resolution difference remains an estimator of uncertainty. It cannot guarantee proximity to the continuum solution.

| Method | Pairs vs FV1 | Median runtime ratio | Method lower depth error | FV1 lower depth error | Method lower outlet error | FV1 lower outlet error |
|---|---|---|---|---|---|---|
| hancock | 48 | 1.72 | 7 | 5 | 23 | 0 |
| fv2 | 16 | 2.11 | 3 | 2 | 7 | 0 |
| li | 18 | 0.34 | 0 | 9 | 5 | 11 |

## Hydraulic grid and DEM representation

The moderate storm uses the same domain and end time at each spacing. Each entry gives hydraulic runtime in seconds, final-depth RMSE in millimetres and outlet hydrograph RMSE in m³/s. Runtime includes the production timestep policy. The final-depth RMSE can be small even where peak depth or connectivity differs. Both are also available in the full metric table.

| Sampled terrain | Method | 1 m | 5 m | 10 m |
|---|---|---|---|---|
| Plain | fv1 | 64.05 / 0.45 / 0.000898 | 5.94 / 0.166 / 0.00111 | 2.81 / 0.615 / 0.00189 |
| Plain | hancock | 107.02 / 0.00352 / 5.83e-06 | 9.20 / 0.121 / 0.000303 | 4.79 / 0.407 / 0.000758 |
| Plain | fv2 | 126.33 / ref | 11.45 / 0.108 / 0.000313 | 5.89 / 0.383 / 0.000782 |
| Plain | li | 22.57 / 33.1 / 0.0564 | 1.05 / 33.1 / 0.0564 | 0.45 / 33.1 / 0.0563 |
| Fan | fv1 | 14.14 / 3.4 / 0.00583 | 2.14 / 1.42 / 0.00255 | 1.31 / 6.95 / 0.00397 |
| Fan | hancock | 41.75 / 0.0214 / 0.000265 | 3.39 / 1.11 / 0.00257 | 2.21 / 7.86 / 0.00331 |
| Fan | fv2 | 49.41 / ref | 4.40 / 1.08 / 0.0028 | 2.80 / 7.31 / 0.00296 |
| Incised | fv1 | 19.04 / 0.00933 / 0.0247 | 2.28 / 0.723 / 0.0236 | 1.36 / 4.54 / 0.0233 |
| Incised | hancock | 85.40 / 0.0645 / 0.0143 | 4.23 / 1.39 / 0.0266 | 2.34 / 5.92 / 0.026 |
| Incised | fv2 | 99.69 / 0.0701 / 0.0145 | 5.89 / 1.17 / 0.0267 | 3.38 / 5.79 / 0.0262 |

| Fixed terrain | Method | 1 m | 5 m | 10 m |
|---|---|---|---|---|
| Plain | fv1 | 63.46 / 0.32 / 0.000466 | 5.95 / 0.73 / 0.00166 | 2.81 / 0.0555 / 0.00151 |
| Plain | hancock | 107.58 / 0.000731 / 6.9e-06 | 9.11 / 0.00279 / 9.16e-05 | 4.79 / 0.271 / 0.000632 |
| Plain | fv2 | 119.74 / ref | 11.11 / 0.00665 / 8.78e-05 | 5.89 / 0.296 / 0.000673 |
| Plain | li | 21.97 / 33.7 / 0.0568 | 1.04 / 33.7 / 0.0568 | 0.45 / 33.7 / 0.0568 |
| Fan | fv1 | 31.19 / 6.67 / 0.00719 | 3.02 / 2.95 / 0.0165 | 1.31 / 9.73 / 0.0231 |
| Fan | hancock | 53.76 / 0.542 / 0.00098 | 4.51 / 4.59 / 0.00625 | 2.21 / 8.86 / 0.0239 |
| Fan | fv2 | 59.90 / ref | 5.48 / 2.72 / 0.00468 | 2.80 / 9.37 / 0.0247 |
| Incised | fv1 | 50.20 / 37 / 0.00804 | 4.14 / 5.87 / 0.0149 | 1.36 / 7.66 / 0.0236 |
| Incised | hancock | 69.53 / 0.247 / 0.00148 | 5.99 / 3 / 0.00432 | 2.34 / 6.65 / 0.0306 |
| Incised | fv2 | 76.22 / ref | 7.09 / 2.18 / 0.00376 | 3.38 / 6.74 / 0.0308 |

## Independent analytical rainfall and recession

Over a frictional uniform sheet, exact depth is the time integral of rain plus initial depth. The reciprocal discharge follows the integral of Manning drag divided by the depth to the 7/3 power. A central section stays outside the boundary signal cone. The analytical comparison integrates interval transport per unit width across six hours of rain and two further hours. The kinematic conditions give both the SWE and LI the same governing interior solution.

| Method | Depth RMSE (m) | Section-discharge RMSE (m²/s) | Final transport error (m²) |
|---|---|---|---|
| fv1 | 1.17e-16 | 2.16e-06 | 0.0111 |
| hancock | 2.25e-16 | 9.43e-08 | -0.00048 |
| fv2 | 3.87e-16 | 9.43e-08 | -0.00048 |
| li | 2.49e-16 | 3.14e-06 | -0.016 |

Maxima, wet-area departures, arrival and recession, threshold-specific IoU, connected components, internal sections, outlet volume and peak timing are available for every grid pair. A missing threshold crossing has an empty arrival/recession RMSE. An empty inundation union has IoU one. Peak timing is the end of the largest output interval. [Full accuracy table](accuracy.csv) · [Reference differences](reference_uncertainty.csv) · [Pairwise comparisons](solver_tradeoffs.csv) · [Runtime and depth](accuracy_cost.png) · [Discharge at fixed and resampled terrain](resolution_discharge.png)
