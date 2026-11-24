# FloodWarp results to date

Paired sparse FV1 and adaptive Hancock forecasts are complete on 8 native 1,048,576-cell landforms. Each received six hours of rain. The verified endpoints range from 12 to 48 h. Taking the longest measured pair once per landform, Hancock's median hydraulic runtime is 1.92× FV1, with 3/8 within 1.5×. Execution policies here were selected from original-window measurements.

[Landform timings, storage, omitted export and accepted timesteps](NATIVE_RECESSIONS.md) · [Common-timestep GPU qualification](../../../validation/hancock_long_storm_review/DECISION.md) · [Hancock structures and verification](../../../docs/HANCOCK_STRUCTURES.md)

## Travelling flood pulse

At the section 200 m from the initial pulse centre, the analytical peak averaged over one second is 2.274 m³/s. The central 80% of discharge passes in 14.69 s. The table gives signed bias relative to this solution.

| Hydraulic grid | FV1 peak | Hancock peak | FV2 peak | FV1 width | Hancock width | FV2 width |
|---|---:|---:|---:|---:|---:|
| 1 m | -19.34% | -0.52% | -0.61% | +24.11% | +0.04% | +0.05% |
| 5 m | -47.97% | -10.08% | -10.71% | +92.88% | +4.07% | +5.54% |
| 10 m | -60.50% | -27.08% | -28.34% | +155.68% | +22.70% | +26.03% |

### Measured 5 m Hancock and 1 m FV1 costs

The following runs cover the same 1,600 × 40 m channel and 176 s of propagation, with the same Courant cap and 1 s output. Hydraulic times exclude compilation, graph capture and field extraction. The signed biases use the exact wave at 200 m. A single timing per configuration supports this channel comparison.

| Pulse | Method | Hydraulic time (s) | Peak bias | Central 80% width bias |
|---|---|---:|---:|---:|
| Double | FV1 sparse, 1 m | 1.170 | -32.86% | +13.78% |
| Double | Hancock dense, 5 m | 0.082 | -22.29% | +6.06% |
| Sharp | FV1 sparse, 1 m | 1.178 | unresolved | unresolved |
| Sharp | Hancock dense, 5 m | 0.082 | unresolved | unresolved |
| Smooth | FV1 sparse, 1 m | 1.150 | -19.34% | +24.11% |
| Smooth | Hancock dense, 5 m | 0.082 | -10.08% | +4.07% |

The sharp pulse spans about five one-second observations. The qualification threshold is six. [Analytical check at quarter-second intervals](results/sharp_analytical_cadence.csv)

[Measured physical and relative quantities](results/pulses.csv) · [Independent integrations of the reference](results/pulse_reference_checks.csv)

## Six-hour rainfall on the small domains

All 48 events have a common checked recession. 31 needed longer than eight hours. The longest reached 76 h. Each 10 m forecast is compared to its own 2.5 m adaptive FV2 numerical reference. A 5 m spatial control, a 2.5 m halved-CFL control and a 2.5 m Hancock control test its sensitivity.

| Median absolute outlet error | Matched eligible events | FV1 | Hancock dense | FV2 |
|---|---:|---:|---:|---:|
| Peak discharge | 24 | 2.06% | 2.63% | 2.00% |
| Central 80% duration | 13 | 6.79% | 5.32% | 5.29% |
| Volume redistribution | 12 | 2.75% | 2.51% | 2.56% |

[Results by landform, signed effects and uncertainty](small_storms/README.md)

## Operational reference controls

The following counts describe the original forecast windows. A separate matched-recession receipt records the common end used for hydraulic assessment.

| Investigation | Original windows completed | Verified matched end |
|---|---:|---|
| Rolling terrain, independently refined SWE | 4/4 | pending |
| Incised late crest, fixed 10 m bed | 4/5 | pending |

### Incised channel through hour 8

Each run subdivides the same 10 m bed and physical inputs. Outlet peaks are accepted face transport averaged over the reporting interval. Late export is the fraction of outward volume passing in the final 2 h of this window. Stored water can contribute to the later hydrograph.

| Method | Grid | CFL | Peak (m³/s) | Outlet volume (m³) | Late export | End storage (m³) | Hydraulic time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hancock tiled | 10 m | 0.2 | 48.010 | 745,566 | 18.81% | 106,932 | 113.3 |
| FV2 RK2 | 10 m | 0.4 | 47.500 | 768,761 | 18.11% | 97,935 | 182.7 |
| FV2 RK2 | 5 m | 0.4 | 33.536 | 386,724 | 53.15% | 502,761 | 1708.1 |
| Hancock tiled | 5 m | 0.4 | 34.293 | 401,852 | 51.84% | 491,800 | 516.0 |

The incised diagnostic was defined after observing a late crest. It remains separate from advisor training and test decisions. Each control needs the same recession coverage before its hydrograph can support a solver ranking.

[Incised one-minute and 15-minute hydrographs](figures/incised_native_cadence.pdf) · [Original terrain, calibration and advisor results](../CORRECTIVE_AUDIT.md)
