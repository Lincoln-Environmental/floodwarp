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
| Incised late crest, fixed 10 m bed | 3/5 | pending |

The incised diagnostic was defined after observing a late crest. It remains separate from advisor training and test decisions. Each control needs the same recession coverage before its hydrograph can support a solver ranking.

[Incised one-minute and 15-minute hydrographs](figures/incised_native_cadence.pdf) · [Original terrain, calibration and advisor results](../CORRECTIVE_AUDIT.md)
