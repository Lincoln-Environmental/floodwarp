# FloodWarp results to date

## Incised storm on metre-derived terrain

The 10 m bed averages four 5 m elevations. Each 5 m elevation averages 25 native metre cells. The six forecasts used a common rainfall field and native initial water. Peak discharge is a one-minute mean of accepted face volumes.

| Grid (m) | Method | End (h) | Peak (m³/s) | Outlet volume (m³) | Infiltration (m³) | Hydraulic time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 10 | FV2 RK2 | 16 | 66.40 | 825,018 | 444,115 | 364.2 |
| 5 | FV2 RK2 | 16 | 48.33 | 809,311 | 462,657 | 2451.8 |
| 10 | Hancock tiled | 16 | 79.65 | 825,170 | 443,988 | 74.9 |
| 5 | Hancock tiled | 16 | 48.29 | 809,296 | 462,684 | 464.1 |
| 10 | FV1 sparse | 16 | 48.37 | 910,217 | 362,294 | 38.4 |
| 5 | FV1 sparse | 16 | 49.19 | 910,143 | 362,416 | 298.7 |

Runoff can peak after rainfall ends. The two FV2 grids disagree about the largest one-minute peak. Repeated measurements at the outlet and upstream sections locate the water released during that interval. [Hydrographs](native_tail/README.md) · [Peak-minute budget](native_tail/CREST.md) · [Runoff partition](native_outlet/runoff_partition.csv)

In the original centre-sampled archive, paired sparse FV1 and adaptive Hancock forecasts are complete on 8 1,048,576-cell landforms. Each received six hours of rain. The verified endpoints range from 12 to 48 h. Taking the longest measured pair once per landform, Hancock's median hydraulic runtime is 1.92× FV1 (range 0.95–2.77×). Execution policies here were selected from original-window measurements.

[Landform timings, storage, omitted export and accepted timesteps](NATIVE_RECESSIONS.md) · [Common-timestep GPU qualification](../../../validation/hancock_long_storm_review/DECISION.md) · [Hancock structures and verification](../../../docs/HANCOCK_STRUCTURES.md)

The 5 m incised runs that replicated 10 m elevations are a staircase diagnostic. Practical terrain comparisons use bottom-up native cell averages. [Cause, verification and corrected runs](DEM_CORRECTION.md)

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

With quarter-second accepted transport at 200 m, 5 m Hancock took 0.149 s and 1 m FV1 took 1.348 s (9.03×). Their respective sharp-pulse peak biases were -54.09% and -58.60%. Refining Hancock to 1 m reduced its peak bias to -7.94%. [Quarter-second analytical comparisons](results/sharp_pulses.csv)

[Measured physical and relative quantities](results/pulses.csv) · [Independent integrations of the reference](results/pulse_reference_checks.csv)

## Six-hour rainfall on the small domains

All 48 events have a common checked recession. 31 needed longer than eight hours. The longest reached 76 h. Each 10 m forecast is compared to its own 2.5 m adaptive FV2 numerical reference. A 5 m spatial control, a 2.5 m halved-CFL control and a 2.5 m Hancock control test its sensitivity.

| Median absolute outlet error | Matched eligible events | FV1 | Hancock dense | FV2 |
|---|---:|---:|---:|---:|
| Peak discharge | 24 | 2.06% | 2.63% | 2.00% |
| Central 80% duration | 13 | 6.79% | 5.32% | 5.29% |
| Volume redistribution | 12 | 2.75% | 2.51% | 2.56% |

[Results by landform, signed effects and uncertainty](small_storms/README.md)

## Terrain, calibration and solver selection

With bottom-up area averaging, the prescribed 8 m opening lost its native four-neighbour connection in 0/15 at 1 m, 11/15 at 5 m, 11/15 at 10 m. The probe stage was 1 cm above the native spill elevation. [Aligned native and aggregated terrain controls](mean_windows/controls.csv)

Bounded FV1 Manning fits improved 64/105 reference-qualified transfer scores and worsened 41. Several scores come from different objectives and calibration bounds on the same terrain. [Calibration and transfer](../tables/09_calibration.md)

In the original independent million-cell advisor evaluation, 10/30 decisions had a measured configuration within hydraulic tolerance. Of those, 0 also met the terrain requirement. This test supplies no feasible operational cost comparison for the advisor. [Held-out decisions and abstentions](../tables/13_advisor_scale.md)

## Operational reference controls

The following counts describe the original forecast windows. A separate matched-recession receipt records the common end used for hydraulic assessment.

| Investigation | Original windows completed | Verified matched end |
|---|---:|---|
| Rolling terrain, independently refined SWE | 4/4 | pending |
| Incised late crest, fixed 10 m bed | 5/5 | pending |

### Incised channel through hour 8

These historical runs repeat each 10 m parent elevation on its children. They diagnose artificial terrain steps. Outlet peaks are accepted face transport averaged over the reporting interval. Late export is the fraction of outward volume passing in the final 2 h of this window. Stored water can contribute to the later hydrograph.

| Method | Grid | CFL | Peak (m³/s) | Outlet volume (m³) | Late export | End storage (m³) | Hydraulic time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hancock tiled | 10 m | 0.2 | 48.010 | 745,566 | 18.81% | 106,932 | 113.3 |
| FV2 RK2 | 10 m | 0.4 | 47.500 | 768,761 | 18.11% | 97,935 | 182.7 |
| FV2 RK2 | 5 m | 0.4 | 33.536 | 386,724 | 53.15% | 502,761 | 1708.1 |
| Hancock tiled | 5 m | 0.4 | 34.293 | 401,852 | 51.84% | 491,800 | 516.0 |
| FV2 RK2 | 5 m | 0.2 | 33.608 | 386,872 | 53.19% | 502,683 | 3414.2 |

At hour 8, outward boundary transport on the finer FV2 grid was 382,037 m³ lower. It held 404,827 m³ more water and had infiltrated 22,790 m³ less. At depths ≥5 cm the inundated area was 3.426 versus 0.039 km² (5 m versus 10 m grid). At depths ≥10 cm the corresponding areas were 0.520 versus 0.037 km². [Storage and inundation maps](incised_crest/initial_window_storage.pdf) · [Water balance and run identities](incised_crest/initial_window_storage.csv)

The incised diagnostic was defined after observing a late crest. It remains separate from advisor training and test decisions. Practical grid comparisons require the correctly aggregated DEMs and matched recession.

[Incised one-minute and 15-minute hydrographs](figures/incised_native_cadence.pdf) · [Original terrain, calibration and advisor results](../CORRECTIVE_AUDIT.md)
