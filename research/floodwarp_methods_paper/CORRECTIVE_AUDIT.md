# Corrective numerical and GPU audit

The steep FREE-outlet investigation compares audited revision `0afdd684d2d1` with repair `46afa3f280a5`. The original and extended 104.86 km² incised forecasts, observable-specific reference checks and measured costs are in [Repaired FREE outlet](#repaired-free-outlet). The earlier operational timings and advisor scores below describe their archived implementations. [Current native-grid qualification](fidelity/native_publication/README.md)

The starting development revision was `360b3a37b5389128e7d4146eee00b1c3ca1a66ee`. All numerical forecasts use FP64. Configuration, implementation, device, input hashes, immutable attempts and output hashes identify every measurement. The historical 96-cell controls and original validation archives retain their own identities.

## Defects and changes

| Finding | Cause and correction | Verification |
|---|---|---|
| Hancock was represented by tiled runtime | The equal-cost and manuscript code selected a single execution identifier. Both policies now appear separately and the retrospective feasible comparison carries the exact winning run ID. | Dense/tiled field hashes, budgets and timestep controls. |
| GPU scale inference used 96 cells | Grid size and physical domain were conflated. The prospective operational suite uses 250,000 through 4,194,304 cells. A separate 1 km² case varies hydraulic spacing at fixed physical extent. | Dimensions, integrated forcing and accepted horizon in each receipt. |
| Configuration and qualification were implicit in Pareto rows | Every point now identifies formulation, limiter/flux, policy, spacing and source experiment. Dominance uses a strict improvement on at least one axis within an event. Unresolved points remain in the eligibility table. | Tie, reference-self, CSV-flag and unmatched-domain regression tests. |
| Reference uncertainty and output cadence limited ranking | Selected cases have finer output intervals, successive same-domain refinements and an independent fine-grid full-SWE formulation. The more conservative of the two measured differences determines each observable’s screen. | Signed interval integration, finest-reference exclusions and native 15-minute products. |
| Coarsening could remove hydraulic controls | The GIS comparison now accepts the actual represented bed. Full-domain drainage descriptors are supplemented with metre-grid windows across a prescribed opening. Fixed-bed, sampled-bed and initial-map controls are distinct. | Independent evaluation of continuous window elevations and four-neighbour hydraulic connectivity. |
| Advisor omitted dense Hancock and operational domain sizes | Domain scale and execution mode enter the empirical neighbourhood. Validation groups choose the support width and interval expansion. Lost connectivity can require terrain refinement. | Disjoint realisation IDs, out-of-range refusal, frozen model digest, oracle and common-denominator comparisons. |
| Conditional success obscured abstentions | Reports now include feasible coverage, agreement across all feasible opportunities, false feasibility per decision and both cost and error stratification. | Individual held-out decisions, grouped tables and independent archive verification. |

The numerical adapter records imposed forcing, transport, state and timings. The separate profile worker measures instrumented kernels and peak allocations. [Forecast adapter](qualification/hydraulics.py) · [GPU profiling](qualification/profiled.py) · [Execution preflight](../../src/floodwarp/solver_advisor/execution.py)

The reference screens and matched comparisons link decisions to experiment IDs. The terrain and advisor modules join those IDs to the represented DEM and preserve independent partitions. [Reference metrics](qualification/observations.py) · [Pareto dominance](metrics.py) · [Cost comparisons](comparisons.py) · [Control geometry](qualification/terrain.py) · [Training evidence](qualification/advisor_data.py) · [Held-out decisions](qualification/decisions.py)

## Small-domain runtime correction

| Hancock execution | Events | Median / FV1 | Observed range / FV1 |
|---|---:|---:|---:|
| Dense | 48 | 1.352 | 1.247–2.521 |
| Tiles | 48 | 2.220 | 1.737–3.901 |
| Fastest measured | 48 | 1.352 | 1.247–2.521 |

For 96 cells, compacting the footprint costs extra launches and leaves little hydraulic work to skip. On large sparse rasters, each omitted tile avoids reconstruction, HLL flux, draining, update and accounting work. The separate common-cap measurements isolate this effect. Normal-policy storm times also include differences in accepted timesteps and rejected trials.

## Operational storm matrix

The 8 matched 1,024 × 1,024 comparisons covered 8 landform families. Across the original fixed-duration forecasts, the median runtime of the faster verified explicit Hancock schedule relative to FV1 was 1.69×. This retrospective choice gives an attainable measured cost for each event. The automatic runs used a policy frozen before these test observations.

| Original forecast window | Grid cells | Δx (m) | Hours | FV1 (s) | Hancock dense (s) | Tiles (s) | Auto (s) | Fastest explicit / FV1 | Hancock / FV1 steps | Fastest / FV1 per step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal-703-moderate-10240x10240 | 1048576 | 10 | 8 | 67.651 | 105.3 | 107.89 | 104.32 | 1.56 | 1.06 | 1.47 |
| fan-705-spatial-5120x5120 | 1048576 | 5 | 8 | 22.542 | 78.177 | 37.264 | 37.128 | 1.65 | 1.42 | 1.16 |
| floodplain-702-moderate-10240x10240 | 1048576 | 10 | 12 | 101 | 171.69 | 175.58 | 170.57 | 1.7 | 1.12 | 1.51 |
| hill-706-back-10240x10240 | 1048576 | 10 | 8 | 104.27 | 226.84 | 232.48 | 225.29 | 2.18 | 1.45 | 1.5 |
| incised-707-spatial-10240x10240 | 1048576 | 10 | 8 | 28.372 | 118.26 | 56.092 | 55.975 | 1.98 | 1.62 | 1.22 |
| mixed-708-spatial-10240x10240 | 1048576 | 10 | 8 | 30.451 | 124.16 | 51.032 | 50.9 | 1.68 | 1.47 | 1.14 |
| mixed-714-spatial-20480x10240 | 2097152 | 10 | 8 | 59.353 | 231.01 | 87.453 | 87.417 | 1.47 | 1.33 | 1.11 |
| plain-701-moderate-10240x10240-24h | 1048576 | 10 | 24 | 193.52 | 296.96 | 298.19 | 297 | 1.53 | 1.06 | 1.45 |
| plain-701-spatial-10240x10240 | 1048576 | 10 | 8 | 8.633 | 27.231 | 9.6535 | 9.6754 | 1.12 | 1.1 | 1.02 |
| plain-701-spatial-10240x10240-48h | 1048576 | 10 | 48 | 35.359 | 100.39 | 33.853 | 33.746 | 0.957 | 0.861 | 1.11 |
| plain-709-spatial-1000x1000 | 1000000 | 1 | 8 | 112.72 | 251.72 | 150.23 | 149.92 | 1.33 | 1.03 | 1.29 |
| plain-712-spatial-20480x10240 | 2097152 | 10 | 24 | 48.748 | 164.42 | 53.375 | 53.353 | 1.09 | 1.11 | 0.983 |
| plain-715-spatial-10240x10240 | 4194304 | 5 | 8 | 65.287 | 219.66 | 71.023 | 71.239 | 1.09 | 1.09 | 1 |
| plain-715-spatial-10240x10240 | 1048576 | 10 | 8 | 8.6163 | 26.982 | 9.5917 | 9.5797 | 1.11 | 1.1 | 1.02 |
| rolling-704-soil-5120x5120 | 1048576 | 5 | 8 | 70.34 | 156.65 | 144.93 | 156.03 | 2.06 | 1.28 | 1.61 |
| rolling-713-moderate-20480x10240 | 2097152 | 10 | 8 | 120.76 | 242.01 | 249.96 | 242.08 | 2 | 1.33 | 1.51 |

The archive contains 80 forecasts covering their prescribed observation windows across the scale and fixed-area designs and 8 supplementary recession controls. The largest measured primary grid has 4,194,304 cells. The memory observations are live device-pool allocations. Compilation, graph capture, hydraulic execution, extraction and receipt wall time are separate. For changing coastal stages the frame hydraulic timer surrounds the advance call. Prescribed boundary assignments between calls are included in end-to-end time.


The supplementary recessions use the same plain DEM. Spatial rain probes delayed export from the original footprint. Widespread rain also reaches the downstream cells. The last column gives the mean discharge in the final 15-minute interval. [Hydrographs](processed/million/hydrographs_900s.csv)

| Rain | Hours | Solver | Rain (m³) | Infiltration (m³) | Outlet volume (m³) | Final storage (m³) | Final wet area (%) | Peak / final discharge (m³/s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| moderate | 24 | fv1 | 8,650,752.0 | 4,448,417.9 | 1,947,778.9 | 2,254,555.2 | 63.28 | 30.206 / 18.797 |
| moderate | 24 | hancock_dense | 8,650,752.0 | 4,638,955.7 | 1,505,457.9 | 2,506,338.4 | 71.42 | 24.691 / 13.080 |
| spatial | 48 | fv1 | 873,209.6 | 873,209.6 | 0.0 | 0.0 | 0.00 | 0.000 / 0.000 |
| spatial | 48 | hancock_dense | 873,209.6 | 873,209.6 | 0.0 | 0.0 | 0.00 | 0.000 / 0.000 |

## Mechanisms, measured cost and Pareto evidence

Across the 8 primary events Hancock accepted 294,040 steps and rejected 572 trials (0.19% of trials). FV1 accepted 219,571 steps. The incised case had the largest accepted-step ratio (1.62×), with 36 retries for 45,545 accepted steps. Rejections explain little of the event-wide step count. Earlier instrumented wet fronts located the restrictive signal at a reconstructed wet/dry face. A separate incised outlet case was limited by the cell-centred gravity-wave speed. The million-cell receipts measure the accepted and rejected counts. Identifying the limiting face or cell at each step would require instrumenting those runs. [Per-event timestep records](processed/million/observations.csv) · [Earlier CFL attribution](../../validation/hancock_long_storm_review/REPORT.md#timestep-attribution)

The steps-ratio column isolates restrictions from the remaining per-step cost. The scheduling table records wet and padded work fractions. The complete graph replays measure individual kernel categories independently of uninstrumented wall time. Dominated, tied and unresolved configurations are traceable to individual runs. Allocation footprint and runtime each have a distinct objective. [Execution rows](processed/million/policies.csv) · [Large-domain Pareto decisions](processed/million/pareto.csv) · [Kernel attribution](processed/million/graph_profile.csv) · [Equal-budget comparisons](tables/08_cost_accuracy.md)

## Reference and terrain evidence

Depth, wet-depth and maximum-depth comparisons use area averages on the common physical support specified by each case. Final inundation IoU uses thresholded common-support depths at 1, 5 and 10 cm. Native-grid component counts and wet-area differences are also recorded. A final-depth metric cannot measure a missing connection by itself. The prospective reference supplement retains 60-second interval discharge in fast cases and exact volume aggregation to 900 seconds.

| Observable | Reassessed small cases | Originally qualified | Qualified on both finer checks |
|---|---:|---:|---:|
| depth | 6 | 3 | 5 |
| discharge | 6 | 4 | 5 |
| inundation | 6 | 1 | 4 |
| maximum_depth | 6 | 4 | 6 |
| peak_discharge | 6 | 4 | 5 |
| peak_timing | 6 | 6 | 5 |

The finest spacing is 1 m, matching the available terrain data. The planned 0.5 m FV2 and Hancock forecasts were withdrawn. The interrupted run contributed no measured result. Without a finer reference, the 1 m configuration on the matched 1 km² plain remains unresolved for absolute accuracy. [Scope amendment](qualification/resolution_scope.json)

The original 48 sampled-terrain storm comparisons had unresolved references in 23 final-depth, 34 inundation and 12 hydrograph cases. The fine-cadence subset has a separate measured Pareto frontier. Its final 1 cm inundation and 60-second hydrographs retain their own reference and timing identities. [Original screening](processed/metrics/reference_uncertainty.csv) · [Short-interval frontier and eligibility](processed/million/short_interval_pareto.csv)

These paired Hancock controls hold the physical domain and continuous forcing fixed. A change in hydraulic spacing can change the cell-sampled rain integral. At fixed hydraulic spacing the bed comparisons reuse the same rainfall raster. RMSE measures the difference between two forecasts. The signed ledger columns give the second forecast minus the first. [All methods, run IDs and formulation interactions](processed/million/resolution_components.csv)

| Case | Paired change | Depth difference (mm RMSE) | Initial storage Δ (m³) | Integrated rain Δ (m³) | Infiltration Δ (m³) |
|---|---|---:|---:|---:|---:|
| plain-709-spatial-1000x1000 | grid on fixed 10 m bed | 0.724 | +0.000 | -0.108 | -95.415 |
| plain-709-spatial-1000x1000 | resampled bed at 1 m | 2.034 | +0.000 | -0.000 | +1.266 |
| floodplain-702-moderate-10240x10240 | grid on fixed 20 m bed and initial map | 20.951 | +0.000 | +0.000 | +35370.691 |
| floodplain-702-moderate-10240x10240 | initial water mapping on fixed bed | 1.091 | -0.070 | +0.000 | +117.491 |
| floodplain-702-moderate-10240x10240 | resampled bed with matched water | 21.808 | +0.000 | +0.000 | -33523.170 |
| coastal-703-moderate-10240x10240 | grid on fixed 20 m bed and initial map | 0.865 | +0.000 | -0.000 | +2587.653 |
| coastal-703-moderate-10240x10240 | initial water mapping on fixed bed | 0.019 | -0.070 | +0.000 | +0.006 |
| coastal-703-moderate-10240x10240 | resampled bed with matched water | 1.395 | +0.000 | +0.000 | -1837.590 |

The fast-event outlet records support a direct comparison of interval means from the same accepted face volumes. Times below are the ends of the intervals containing the maximum mean discharge. [Fine-interval peak observations](processed/million/observations.csv) · [15-minute hydrographs](processed/million/hydrographs_900s.csv)

| Landform | Method | Peak at 60 s (m³/s) | Peak at 900 s (m³/s) | Attenuation from averaging | Peak interval end, 60 s / 900 s (s) |
|---|---|---:|---:|---:|---:|
| incised | fv1 | 48.367 | 48.285 | 0.17% | 19440 / 19800 |
| incised | hancock_dense | 47.527 | 47.170 | 0.75% | 17100 / 17100 |
| mixed | fv1 | 41.363 | 41.153 | 0.51% | 27060 / 27900 |
| mixed | hancock_dense | 38.700 | 38.542 | 0.41% | 26160 / 26100 |

The following medians use one 80 × 80 m control window per terrain realisation at each diagnostic resolution. The opening is eight metres wide on the continuous bed. The displaced-outlet fraction counts native cells whose filled-DEM drainage outlet moves more than two model cells.

| Diagnostic grid (m) | Width of opening (cells) | Connections lost / windows | Barrier lowering, p95 (m) | Saddle elevation RMSE (m) | Change in depression storage (m³) | Displaced outlets |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0 / 15 | 0.0000 | 0.0000 | +0.00 | 0.0% |
| 5 | 1.6 | 9 / 15 | 0.0422 | 0.0016 | +2.47 | 17.6% |
| 10 | 0.8 | 11 / 15 | 0.1172 | 0.0059 | +18.17 | 35.9% |

At the prescribed test stage, 11 of 15 independent case windows lost the metre-grid connection on the 10 m represented terrain. The stage is 1 cm above the lowest native through-going spill elevation. Each window covers 80 × 80 m around the known control and retains its physical coordinates. Whole-domain metrics use the full represented raster. [Terrain diagnostics](processed/million/terrain.csv) · [Fixed-bed and initial-water controls](processed/million/resolution_components.csv) · [Reference screens](processed/million/reference_checks.csv)

## Held-out operational recommendations

| Rule, at least one million cells | Decisions | Qualified feasible opportunities | Covered (%) | Abstained (%) | False feasible | Cheapest / opportunities (%) | Conditional regret (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| advisor | 30 | 0 | unresolved | 100 | 0 | unresolved | unresolved |
| fv1 | 30 | 0 | unresolved | 0 | 12 | unresolved | unresolved |
| hancock_dense | 30 | 0 | unresolved | 0 | 12 | unresolved | unresolved |
| slope | 30 | 0 | unresolved | 100 | 0 | unresolved | unresolved |
| oracle | 30 | 0 | unresolved | 100 | 0 | unresolved | unresolved |

The previously audited evaluation selected a method on 62 of 120 decisions. It recorded 41 successful selections, 58 abstentions, 156.0% median conditional regret and 81.4% empirical interval coverage. [Original immutable validation](raw_results/advisor-validate/b23f24375a6d611f22d5eca6/7a601da9a91c431f/summary.csv) · [Matched old and revised decisions](processed/advisor/previous_comparison.csv)

The advisor abstained on 30 of 30 million-cell decisions. No measured option satisfied both the hydraulic acceptance criterion and terrain admissibility. This test cannot establish an advantage over the simple rules or estimate operational regret on feasible choices. Of 30 decisions, 12 had a qualified reference, 10 had a measured method within hydraulic tolerance and 0 also had admissible terrain. [Selection eligibility by case](processed/advisor/eligibility_breakdown.csv) The runtime-only comparison and the objective-specific feasibility comparison answer separate questions. All rows from an independent terrain realisation belong to the same split. Pre-run inputs describe terrain and prescribed forcing. Realised wet fraction is used only to stratify results after validation. [Held-out individual decisions](processed/advisor/heldout.csv) · [By landform, grid, size and wet fraction](processed/advisor/validation_strata.csv) · [Runtime-only decisions](processed/advisor/runtime_only.csv)

## Engineering recommendation and remaining limits

Keep adaptive Hancock as the development formulation and explicit dense execution as its default. Tiled execution is competitive on adequately large, spatially restricted storms. The optional single-decision preflight uses the measured work crossover on the reference GPU. Each explicit scheduler remains available. Its selection cost is paid during setup. This study does not qualify automatic switching every timestep.

The primary event timings cover a single reference GPU and normal timestep policy. Local reference windows do not establish full-catchment hydraulic truth. Unqualified objectives, zero or unresolved peaks, the first-order time split of hydraulic structures, shared outlet closures, and missing fine-grid comparisons constrain release claims. A measured difference smaller than the reference change cannot establish a robust accuracy ranking. The Pareto archive describes empirical reference errors, including close unresolved comparisons. Release qualification requires representative independent forcing and boundary verification.

[Reproduction instructions](README.md) · [Correction regression tests](../../tests/test_million_qualification.py) · [Actual run provenance](processed/million/archive.csv)

## Completion by registered stage

| Numerical stage | Complete | Registered |
|---|---:|---:|
| million-terrain | 18 | 18 |
| million-pilots | 6 | 6 |
| million-storms | 62 | 62 |
| million-auto | 18 | 18 |
| large-references | 77 | 77 |
| large-refinements | 27 | 27 |
| heldout-references | 5 | 5 |
| million-profiles | 9 | 9 |
| million-repeats | 12 | 12 |
| recession-controls | 8 | 8 |

The prevalidation model used 182 development rows from 14 terrain groups. Parameter selection used 11 validation terrain groups. The current model reproduces every frozen training value, parameter and uncertainty interval. The comparison excludes only the reporting revision string. [Freeze manifest and model checksum](manifests/prevalidation_advisor_freeze.json)

The catalogue retains 2 interrupted or failed numerical attempts outside all hydraulic and timing comparisons. [Attempt catalogue](processed/metrics/attempt_catalog.csv)
- large-references, plain-701-spatial-10240x10240, hancock_dense — incomplete, attempt `affc4e9a96104629`.
- million-storms, plain-709-spatial-1000x1000, adaptive — incomplete, attempt `dc1995df12c544a8`.

## Complete graph replay timings

| Case | Configuration | Largest kernel category | Device seconds | Share of measured kernel time |
|---|---|---|---:|---:|
| floodplain-702-moderate-10240x10240 | fv1 | flux and update | 80.278 | 79.8% |
| floodplain-702-moderate-10240x10240 | hancock_adaptive | flux and update | 69.086 | 39.4% |
| floodplain-702-moderate-10240x10240 | hancock_dense | flux and update | 69.043 | 40.3% |
| mixed-708-spatial-10240x10240 | fv1 | flux and update | 14.330 | 47.0% |
| mixed-708-spatial-10240x10240 | hancock_adaptive | flux and update | 16.156 | 31.4% |
| mixed-708-spatial-10240x10240 | hancock_dense | flux and update | 67.748 | 54.4% |
| plain-701-spatial-10240x10240 | fv1 | flux and update | 4.010 | 46.1% |
| plain-701-spatial-10240x10240 | hancock_adaptive | flux and update | 3.007 | 30.9% |
| plain-701-spatial-10240x10240 | hancock_dense | flux and update | 14.880 | 54.5% |

Across 6 complete timing controls with at least three runs, observed hydraulic runtime ranges spanned 0.08% to 0.68% of the within-control median. [Individual timing ranges](processed/million/repeat_times.csv)

The prospective preflight used tiled execution in 10 of 16 measured grids of at least one million cells. Its measured excess runtime over the fastest explicit observation had a median of -0.2% and a maximum of +7.7%. Separate executions include timing variability.

Preflight selected the faster explicit schedule on 15 of 16 measured grids of at least one million cells. The work estimate uses the initial state and prescribed rain footprint. The matched tile run measures the work actually scheduled over the entire event. [Per-event estimates and observations](processed/million/policies.csv)

| Event | Δx (m) | Chosen policy | Rain footprint | Estimated scheduled | Observed tiled work | Auto cost above fastest |
|---|---:|---|---:|---:|---:|---:|
| rolling-704-soil-5120x5120 | 5 | dense | 100.0% | 100.0% | 80.2% | +7.7% |

## Measured configurations by observable

| Observable | Eligible | Empirical frontier | Frontier under reference sensitivity | Unresolved |
|---|---:|---:|---:|---:|
| depth | 43 | 21 | 21 | 45 |
| discharge | 22 | 9 | 11 | 66 |
| inundation | 43 | 18 | 20 | 45 |
| maximum_depth | 47 | 23 | 23 | 41 |
| memory | 88 | 18 | 18 | 0 |
| peak_discharge | 16 | 8 | 8 | 72 |
| peak_timing | 18 | 7 | 7 | 70 |

Final 1 cm inundation is the common prediction target in the enlarged advisor. The historical paper also retains time-integrated inundation results. Costs are stratified by the Hancock wetted and scheduled fractions at the end of each 15-minute window and use the FV1 time for exactly the same physical window. [Window-level costs](processed/million/wetness_cost.csv) · [Joint solver and resolution validation](processed/advisor/joint_resolution.csv)

## Cheapest measured method and grid meeting tolerance

Each entry uses the measured error plus the reported refinement sensitivity. Tolerances are prescribed in the advisor configuration. No tested option means a qualified comparison existed but the conservative criterion was missed.

| Physical event | Final depth | Final inundation | Outlet hydrograph |
|---|---|---|---|
| coastal-703-moderate-10240x10240 | unresolved | fv1, 10 m | unresolved |
| fan-705-spatial-5120x5120 | fv1, 5 m | unresolved | no tested option |
| fan-711-spatial-10000x5000 | fv1, 10 m | no tested option | no tested option |
| floodplain-702-moderate-10240x10240 | unresolved | unresolved | unresolved |
| hill-706-back-10240x10240 | unresolved | unresolved | unresolved |
| incised-707-spatial-10240x10240 | unresolved | unresolved | unresolved |
| mixed-708-spatial-10240x10240 | fv1, 10 m | hancock_auto, 10 m | no tested option |
| mixed-714-spatial-20480x10240 | unresolved | unresolved | unresolved |
| plain-701-moderate-10240x10240-24h | unresolved | unresolved | unresolved |
| plain-701-spatial-10240x10240 | fv1, 10 m | hancock_adaptive, 10 m | unresolved |
| plain-701-spatial-10240x10240-48h | unresolved | unresolved | unresolved |
| plain-709-spatial-1000x1000 | fv1, 10 m | hancock_auto, 5 m | fv1, 10 m |
| plain-710-spatial-5000x5000 | fv1, 10 m | hancock_auto, 10 m | unresolved |
| plain-712-spatial-20480x10240 | unresolved | unresolved | unresolved |
| plain-715-spatial-10240x10240 | fv1, 10 m | fv1, 10 m | unresolved |
| rolling-704-soil-5120x5120 | unresolved | unresolved | unresolved |
| rolling-713-moderate-20480x10240 | unresolved | unresolved | unresolved |

## Hydrograph fidelity through completed recessions

Of 48 small events with six hours of rain, 31 required a longer recession. The longest matched simulation reached 76 hours. For the 12 storms with independently qualified normalised outlet shape at 10 m, the Hancock–FV2 difference in absolute shape error exceeded the combined reference sensitivity in 0 cases. That comparison concerns the timing distribution of discharged volume. Peak and arrival-time conclusions have separate reference screens. Among 6 FV1 storms with independently assessable peaks and final depth RMSE within 1 mm, 3 had an absolute peak bias above 10%. [Complete small-storm comparisons](fidelity/small_storms/README.md) · [Million-cell export after the original stop time](fidelity/NATIVE_RECESSIONS.md)

## DEM aggregation and extent thresholds

Earlier resolution experiments used cell-centre sampling or repeated parent elevations. The incised fixed-bed refinement created artificial treads that substantially altered runoff. These historical experiments do not establish accuracy at a practical resolution using an area-averaged DEM. The corrected series averages 25 native 1 m elevations per 5 m cell, then four 5 m elevations per 10 m cell. Initial water, momentum, rainfall footprint and infiltration also come from the native metre source. All practical runs start at time zero or resume their own verified checkpoint. [Native-input storm windows, budgets and reference qualification](fidelity/native_results/README.md) Report practical extent separately at 5 and 10 cm. Historical 1 cm extent is a shallow-water diagnostic. [DEM construction, numerical controls and corrected outcomes](fidelity/DEM_CORRECTION.md)

The native-grid incised results also show sensitivity to the downstream FREE boundary. The steady-sheet experiment checks this effect independently against exact discharge. [Outlet placement and terminal storage](fidelity/native_outlet/README.md)

With the pre-repair FREE boundary, through the matched 16-hour window, the highest recorded FV2 RK2 outlet interval means were 66.40 m³/s at 10 h 51 min on the 10 m grid and 48.33 m³/s at 4 h 48 min on the 5 m grid. Each interval spans 60 seconds. Disagreement in these two measured maxima establishes grid sensitivity. The physical storm peak remains unresolved.

Runoff travel time and catchment storage allow the outlet maximum to follow rainfall. The incised diagnostic compares accepted flow at successive cross-sections with storage and infiltration over the intervening reach. The matched finer-grid checks test how strongly the outlet timing and magnitude depend on representation. [Recession water balance](fidelity/native_tail/README.md)

[Matched native-grid costs and observable-specific reference eligibility](fidelity/native_publication/README.md)

The pre-repair replay measured the source of discharge during the highest post-hour-eight minute at each spacing. Net drawdown in the last 40 m supplied 100.00% on the 5 m grid and 99.76% on the 10 m grid. That water may have entered the reach earlier. [Crest-minute budgets](fidelity/native_tail/CREST.md)

With the pre-repair FREE boundary on the 5 m grid, FV1 exported 100,848 m³ more water than Hancock by hour 16. Lower cumulative infiltration accounted for 100,268 m³ of the difference and lower final storage for 579 m³. The net infiltration difference arose on cells outside the rainfall footprint. These terms close the water balance under the same prescribed infiltration capacity. [Runoff and infiltration by grid](fidelity/native_outlet/runoff_partition.csv)

## Repaired FREE outlet

On the same 10 m incised terrain, the corrected 16-hour Hancock and RK2 peaks were 47.18 and 47.66 m³/s. Their earlier late crests disappeared. These are one-minute means at the original outlet. [Fixed-gauge boundary experiments and water budgets](fidelity/outlet_results/README.md)

The pre-repair and corrected hydraulic sources match commits `0afdd684d2d1` and `46afa3f280a5`. The corrected FP64 storm measurements used NVIDIA GeForce RTX 4070 Ti SUPER, 595.99.02, 16376 MiB. [Source and workstation receipts](fidelity/outlet_results/revision_evidence.json)

Of 342 core exact-solution controls, 342 reproduced the pre-repair sampled fields and gauge transport exactly. 342 retained the same ordered accepted timesteps. The maximum relative mass-balance residual was 1.61e-12 and the lake-at-rest depth error was 8.88e-16 m. Measuring the actual steep FREE face exposed a defect that these sampled controls did not detect. [Paired exact references, results and run identities](fidelity/analytical_revision/comparisons.csv)

At the fixed 120 m-wide incised section on the 10 m grid, adding a 100 m downstream buffer changed mean RK2 discharge from 2.46253 to 3.32420 m³/s before the repair and from 3.32413 to 3.32420 m³/s after it. These accepted volumes between 30 and 60 s measure sensitivity to outlet placement. [Fixed-section measurements on the 1, 5 and 10 m grids](fidelity/outlet_results/incised_reach.csv)

At 10 m, FV1 minus Hancock net export was +41,131 m³. The corresponding cumulative infiltration differences were +17,863 m³ inside the rainfall footprint and -59,098 m³ outside. The final storage difference was +104 m³. At hour eight their areas above 5 cm were 0.7233 and 0.0000 km².

At 5 m, FV1 minus Hancock net export was +81,117 m³. The corresponding cumulative infiltration differences were +6,211 m³ inside the rainfall footprint and -87,383 m³ outside. The final storage difference was +56 m³. At hour eight their areas above 5 cm were 0.5847 and 0.0000 km².

At rain cessation (6 h), the direct 5 m FV1-minus-Hancock differences were -10,822 m³ in cumulative export and +85,815 m³ in remaining storage. During the common recession alone FV1 exported +91,939 m³ relative to Hancock. [Rain and recession water accounts](fidelity/outlet_results/phase_volume_differences.csv)

Across 65 snapshots at 15-minute intervals, the largest simultaneous areas above 5 cm were 3.7250, 0.0542 and 0.0535 km² for 5 m FV1, Hancock and RK2. These measurements cover the full direct-outlet storm and recession. Cumulative inundation also includes cells that flooded at different times. [Sampled maxima and cumulative extent](fidelity/outlet_results/peak_extent.csv)

Of the hour-eight area above 5 cm in FV1 alone on the direct 5 m grid, 100.00% lay downstream of the independent internal gauge and 100.00% lay outside the rainfall footprint. The final 40 m contained 2.36% of this disagreement. [Reach and rainfall-footprint decomposition](fidelity/outlet_results/regional_contrasts.csv) · [Maps of inundation and cumulative infiltration](supplementary/incised/hour_eight_water.pdf)

On the 10 m direct-outlet grid, Hancock required 34,118 accepted steps versus 48,782 for FV1. Across 3 and 3 warmed timings respectively, the median Hancock runtime was 1.47× as much per accepted step and 1.029× as much over the full 16 hours. [Independent timings and execution policies](fidelity/outlet_results/independent_timings.csv)

On the 5 m direct-outlet grid, Hancock required 68,115 accepted steps versus 99,493 for FV1. Across 3 and 3 warmed timings respectively, the median Hancock runtime was 1.52× as much per accepted step and 1.041× as much over the full 16 hours. [Independent timings and execution policies](fidelity/outlet_results/independent_timings.csv)

During the corrected 10 m Hancock crest minute, 2,834.2 m³ entered the final 40 m, 2,831.0 m³ passed the FREE face, 2.8 m³ infiltrated, and surface storage changed by +0.4 m³. Over the post-rain recession, drawdown in that strip after infiltration accounted for 0.63% of outlet export. [Accepted transport and strip water accounts](fidelity/outlet_results/last_40m.csv)

During the corrected 10 m RK2 crest minute, 2,862.9 m³ entered the final 40 m, 2,859.6 m³ passed the FREE face, 2.7 m³ infiltrated, and surface storage changed by +0.5 m³. Over the post-rain recession, drawdown in that strip after infiltration accounted for 0.62% of outlet export. [Accepted transport and strip water accounts](fidelity/outlet_results/last_40m.csv)

Halving the RK2 CFL on the 5 m grid changed the one-minute peak from 48.115 to 48.123 m³/s. Net export changed by -17,792 m³, infiltration by +17,792 m³ and final storage by +0 m³. The normal and half-CFL runs accepted 77,045 and 146,457 timesteps.

The direct-outlet observables passing all three reference checks were peak magnitude, net exported volume. These checks support comparisons on the coarser grid. Practical 5 m absolute accuracy remains unresolved. [Reference qualification by observable](fidelity/outlet_results/reference_checks.csv)

For all outward discharge over the fixed 16-hour window, the fraction of volume redistributed in time relative to the normal 5 m RK2 forecast was 14.53% for 10 m FV1 and 3.69% for 10 m Hancock. Their centroid shifts were +28.8 and -7.4 minutes. Applying the same shape diagnostic to the spatial, fine-formulation and temporal controls gave 2.51%, 0.54% and 1.05%. The initial moving channel contributes to these distributions. Rainfall-only hydrograph shape remains unresolved. [Fixed-window comparisons and reference controls](fidelity/outlet_results/fixed_window_shape.csv)

With a 5% tolerance on peak discharge, FV1 at 10 m had the lowest measured hydraulic cost (38.4 s). Its absolute error against the 5 m RK2 reference was 0.53%, rising to 1.48% when reference sensitivity was included.

With a 10% tolerance on net exported volume, adaptive Hancock at 10 m had the lowest measured hydraulic cost (39.5 s). Its absolute error against the 5 m RK2 reference was 5.62%, rising to 9.78% when reference sensitivity was included.

The two decision tables retain the event endpoint, output cadence and reference for each observable. The hydraulic costs use independent timing medians where available. [Cost at a stated tolerance](tables/19_incised_cost_at_tolerance.md) · [Error within a measured runtime](tables/20_incised_error_at_budget.md)

Under intense rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at 4.331 and 6.806 m³/s. They exported 28,772 and 109,391 m³ through the common 32-hour endpoint. Under moderate rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at 0.894 and 0.915 m³/s. They exported 24,912 and 38,983 m³ through the common 40-hour endpoint. Both gauges passed the recession criterion. These CPU controls measure formulation disagreement. [Six-hour rainfall and complete recession accounts](fidelity/strong_storms/completed.csv)

Across 75 completed native storm configurations, 13 restart records inherited initial-window timings from a different execution of the same hydraulics. The revised costs use each configuration's own measured initial and recession intervals. Both gauges and accepted step counts agreed at all 15,648 initial intervals checked. All operational costs, budget decisions and advisor inputs use this attribution. [Individual timing corrections and run identities](fidelity/native_results/timing_provenance.csv)

## Fixed-area storms and the hydraulic objective

The table gives the cheapest terrain-admissible measured configuration at a 5% tolerance on each observable. Compliance includes the measured reference sensitivity. Each row fixes the storm, reference, reporting interval and recession endpoint. Shape measures redistribution of normalised discharge volume, and width measures the central 80% of that volume.

| Storm | End (h) | Interval (s) | Outlet peak | Outlet shape | Net export | Infiltration | Outlet width | Internal peak | Internal shape |
|---|---:|---:|---|---|---|---|---|---|---|
| hill-8500-back-1000x1000 | 12 | 60 | FV1 10 m (2.15 s) | Unresolved | FV1 10 m (2.15 s) | Outside tolerance | Unresolved | FV1 10 m (2.15 s) | Unresolved |
| plain-709-moderate-1000x1000 | 40 | 60 | Unresolved | Unresolved | Unresolved | Unresolved | Unresolved | Refine terrain | Refine terrain |
| plain-709-spatial-1000x1000 | 20 | 300 | Unresolved | Unresolved | Unresolved | Refine terrain | Refine terrain | Refine terrain | Refine terrain |

Depth decisions use an RMSE tolerance of 1 cm. Maximum extent uses the cellwise maximum-depth map and a 5% tolerance on one minus inundation IoU.

| Storm | End (h) | Interval (s) | Final depth | Maximum-depth map | Depth on 5 cm wet union | Depth on 10 cm wet union | Maximum 5 cm extent | Maximum 10 cm extent |
|---|---:|---:|---|---|---|---|---|---|
| hill-8500-back-1000x1000 | 12 | 60 | FV1 10 m (2.15 s) | Hancock dense 10 m (2.28 s) | Unresolved | Unresolved | Unresolved | Unresolved |
| plain-709-moderate-1000x1000 | 40 | 60 | Refine terrain | Refine terrain | Unresolved | Unresolved | Refine terrain | Refine terrain |
| plain-709-spatial-1000x1000 | 20 | 300 | Refine terrain | Refine terrain | Unresolved | Unresolved | Unresolved | Refine terrain |

Unresolved marks observations without an eligible hydraulic reference. Outside tolerance marks a qualified comparison in which every terrain-admissible candidate exceeds the error bound. Refine terrain marks a hydraulically assessable comparison where each candidate grid failed the pre-run terrain check. The accompanying tables retain the hydraulic-only and terrain-admissible decisions.

| Storm | RK2 spatial peak difference (%) | Fine formulation difference (%) | Half-CFL difference (%) | Peak reference |
|---|---:|---:|---:|---|
| hill-8500-back-1000x1000 | 0.1293 | 0.0006244 | 0.003383 | Qualified |
| plain-709-moderate-1000x1000 | 5.04 | 0.0341 | 0.0005529 | Unresolved |
| plain-709-spatial-1000x1000 | 14.78 | 0.1426 | 0.01738 | Unresolved |

The static connection test uses an 80 m square around the prescribed opening and a water level 1 cm above the native spill elevation. These grids lose the connection even where a hydraulic observable meets tolerance.

| Storm | Grid (m) | Maximum local barrier lowering (cm) |
|---|---:|---:|
| plain-709-moderate-1000x1000 | 5 | 18.04 |
| plain-709-moderate-1000x1000 | 10 | 21.07 |
| plain-709-spatial-1000x1000 | 5 | 18.04 |
| plain-709-spatial-1000x1000 | 10 | 21.07 |

[Control maps and spill-height sensitivity](fidelity/native_publication/control_stage_sensitivity.csv)

On the 10 m assessment grid for hill-8500-back-1000x1000, the fine reference retained 0 cells (0 m²) at or above 5 cm at the endpoint. The area-averaged cell maxima reached 10 cm on 0 cells (0 m²).

On the 10 m assessment grid for plain-709-moderate-1000x1000, the fine reference retained 7 cells (700 m²) at or above 5 cm at the endpoint. The area-averaged cell maxima reached 10 cm on 2100 cells (210000 m²).

On the 10 m assessment grid for plain-709-spatial-1000x1000, the fine reference retained 7 cells (700 m²) at or above 5 cm at the endpoint. The area-averaged cell maxima reached 10 cm on 2 cells (200 m²).

[Cost and error bounds at a stated tolerance](tables/21_native_cost_at_tolerance.md) · [Error within an FV1 runtime budget](tables/22_native_error_at_budget.md) · [Timing repeats and spread](fidelity/native_results/timing_distribution.csv) · [Depth and wet-area support](fidelity/native_results/recession_depth.csv)

## Complete storms on native-metre terrain at 250,000 cells and above

Each row compares solvers on the same event and physical footprint. Domain area varies between rows. Peaks are interval means of accepted outlet transport. The volume columns give FV1 minus Hancock. Costs use the fastest measured explicit Hancock policy through the common recession endpoint and the median hydraulic time where repeats exist.

| Storm | Cells | Area (km²) | Grid (m) | End (h) | Interval (s) | FV1 peak (m³/s) | Hancock peak (m³/s) | Δ net export (m³) | Δ infiltration (m³) | Measured Hancock policy | Hancock / FV1 time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| coastal-703-moderate-10240x10240 | 1,048,576 | 104.858 | 10 | 32 | 300 | 131.72 | 112.55 | +870,045 | -835,214 | tiles | 1.752 |
| fan-705-spatial-5120x5120 | 1,048,576 | 26.214 | 5 | 12 | 300 | 10.59 | 10.125 | -3,662 | +2,883 | tiles | 1.245 |
| fan-711-spatial-10000x5000 | 500,000 | 50.000 | 10 | 16 | 300 | 15.22 | 15.825 | -28,081 | +25,783 | tiles | 1.067 |
| floodplain-702-moderate-10240x10240 | 1,048,576 | 104.858 | 10 | 28 | 300 | 141.05 | 123.69 | +1,123,372 | -1,137,291 | tiles | 1.954 |
| hill-706-back-10240x10240 | 1,048,576 | 104.858 | 10 | 16 | 300 | 1174.7 | 1251.6 | +128,872 | -130,907 | tiles | 1.477 |
| hill-8500-back-1000x1000 | 1,000,000 | 1.000 | 1 | 12 | 60 | 16.126 | 16.159 | +726 | -726 | dense only measured | 1.768 |
| plain-701-spatial-10240x10240 | 1,048,576 | 104.858 | 10 | 52 | 300 | 0 | 0 | +0 | +0 | tiles | 0.943 |
| plain-709-moderate-1000x1000 | 1,000,000 | 1.000 | 1 | 40 | 60 | 0.52503 | 0.91794 | -16,994 | +11,334 | dense only measured | 1.699 |
| plain-709-spatial-1000x1000 | 1,000,000 | 1.000 | 1 | 20 | 300 | 0.015501 | 0.035363 | -256 | +230 | tiles | 1.223 |
| plain-710-spatial-5000x5000 | 250,000 | 25.000 | 10 | 60 | 300 | 0 | 0 | +0 | -0 | tiles | 0.878 |
| plain-712-spatial-20480x10240 | 2,097,152 | 209.715 | 10 | 40 | 300 | 0 | 0 | +0 | +0 | tiles | 1.094 |
| rolling-704-soil-5120x5120 | 1,048,576 | 26.214 | 5 | 12 | 300 | 61.208 | 25.644 | +161,747 | -161,747 | tiles | 2.002 |
| rolling-713-moderate-20480x10240 | 2,097,152 | 209.715 | 10 | 24 | 300 | 464.91 | 330.56 | +1,070,101 | -1,073,309 | tiles | 2.263 |

### Internal section and outlet peaks

Each section spans the domain width at the stated distance from its upstream grid edge. The signed percentage is 100 × (FV1 peak / Hancock peak − 1). Hancock supplies the comparison scale. Negligible outward flow gives an unresolved percentage. The peaks at the two gauges may occur at different times. Rainfall, infiltration and storage between the gauges also affect the outlet response.

| Storm | Grid (m) | End (h) | Section x (m) | Section FV1 (m³/s) | Section Hancock (m³/s) | Difference (%) | Outlet FV1 (m³/s) | Outlet Hancock (m³/s) | Difference (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal-703-moderate-10240x10240 | 10 | 32 | 5120 | 86.612 | 73.261 | +18.22 | 131.72 | 112.55 | +17.03 |
| fan-705-spatial-5120x5120 | 5 | 12 | 2560 | 14.151 | 13.657 | +3.62 | 10.59 | 10.125 | +4.59 |
| fan-711-spatial-10000x5000 | 10 | 16 | 5000 | 25.075 | 24.519 | +2.27 | 15.22 | 15.825 | -3.82 |
| floodplain-702-moderate-10240x10240 | 10 | 28 | 5120 | 139.44 | 111.79 | +24.73 | 141.05 | 123.69 | +14.03 |
| hill-706-back-10240x10240 | 10 | 16 | 5120 | 698.38 | 729.51 | -4.27 | 1174.7 | 1251.6 | -6.14 |
| hill-8500-back-1000x1000 | 1 | 12 | 500 | 8.2382 | 8.2497 | -0.14 | 16.126 | 16.159 | -0.20 |
| plain-701-spatial-10240x10240 | 10 | 52 | 5120 | 11.935 | 10.643 | +12.14 | 0 | 0 | unresolved |
| plain-709-moderate-1000x1000 | 1 | 40 | 500 | 1.6288 | 1.657 | -1.70 | 0.52503 | 0.91794 | -42.80 |
| plain-709-spatial-1000x1000 | 1 | 20 | 500 | 0.3576 | 0.35591 | +0.47 | 0.015501 | 0.035363 | -56.17 |
| plain-710-spatial-5000x5000 | 10 | 60 | 2500 | 4.6395 | 4.1863 | +10.83 | 0 | 0 | unresolved |
| plain-712-spatial-20480x10240 | 10 | 40 | 10240 | 7.4985 | 6.4621 | +16.04 | 0 | 0 | unresolved |
| rolling-704-soil-5120x5120 | 5 | 12 | 2560 | 74.948 | 63.865 | +17.35 | 61.208 | 25.644 | +138.69 |
| rolling-713-moderate-20480x10240 | 10 | 24 | 10240 | 395.76 | 330.54 | +19.73 | 464.91 | 330.56 | +40.64 |

### Storm input and remaining inundation

Catchment-mean rainfall divides the integrated six-hour rainfall volume by the whole modelled area. A spatially concentrated storm can deliver a much greater depth inside its rainfall footprint. The areas count original hydraulic cells at the common end time. Maximum extent during the storm is a separate observable.

| Storm | Grid (m) | End (h) | Rain (m³) | Catchment mean (mm) | FV1 storage (m³) | Hancock storage (m³) | FV1 ≥5 cm (m²) | Hancock ≥5 cm (m²) | FV1 ≥10 cm (m²) | Hancock ≥10 cm (m²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal-703-moderate-10240x10240 | 10 | 32 | 8,650,752 | 82.50 | 64,521 | 99,353 | 292,900 | 496,100 | 7,700 | 9,000 |
| fan-705-spatial-5120x5120 | 5 | 12 | 218,302 | 8.33 | 805 | 27 | 5,950 | 0 | 2,850 | 0 |
| fan-711-spatial-10000x5000 | 10 | 16 | 416,379 | 8.33 | 2,297 | 0 | 16,900 | 0 | 9,100 | 0 |
| floodplain-702-moderate-10240x10240 | 10 | 28 | 8,650,752 | 82.50 | 32,309 | 18,390 | 198,200 | 43,300 | 74,400 | 200 |
| hill-706-back-10240x10240 | 10 | 16 | 14,889,779 | 142.00 | 2,035 | 0 | 5,800 | 0 | 4,400 | 0 |
| hill-8500-back-1000x1000 | 1 | 12 | 142,000 | 142.00 | 0 | 0 | 0 | 0 | 0 | 0 |
| plain-701-spatial-10240x10240 | 10 | 52 | 873,210 | 8.33 | 0 | 0 | 0 | 0 | 0 | 0 |
| plain-709-moderate-1000x1000 | 1 | 40 | 82,500 | 82.50 | 5,736 | 77 | 2,605 | 632 | 720 | 255 |
| plain-709-spatial-1000x1000 | 1 | 20 | 8,328 | 8.33 | 111 | 84 | 771 | 677 | 336 | 286 |
| plain-710-spatial-5000x5000 | 10 | 60 | 208,189 | 8.33 | 0 | 0 | 0 | 0 | 0 | 0 |
| plain-712-spatial-20480x10240 | 10 | 40 | 1,746,419 | 8.33 | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling-704-soil-5120x5120 | 5 | 12 | 3,080,192 | 117.50 | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling-713-moderate-20480x10240 | 10 | 24 | 17,301,504 | 82.50 | 3,208 | 0 | 20,600 | 0 | 0 | 0 |

The differences measure sensitivity to formulation on the same represented terrain. Accuracy decisions use the independently screened reference for each observable. [Water budgets and internal gauges](fidelity/native_results/measured_windows.csv) · [Policy-specific observations and reference eligibility](fidelity/native_publication/README.md)

On plain-709-moderate-1000x1000 at 10 m, reducing FV1 CFL from 0.4 to 0.2 changed the one-minute outlet peak by +0.160% and net export by +14.8 m³ through 40 h. CPU runs used the same initial fields and forcing. [Time refinement, both gauge records and water accounts](fidelity/native_results/fv1_temporal_sensitivity.csv)

On plain-709-intense-1000x1000 at 10 m, reducing FV1 CFL from 0.4 to 0.2 changed the one-minute outlet peak by +0.175% and net export by -14.9 m³ through 32 h. CPU runs used the same initial fields and forcing. [Time refinement, both gauge records and water accounts](fidelity/native_results/fv1_temporal_sensitivity.csv)

On plain-709-moderate-1000x1000 at 10 m, reducing Hancock CFL from 0.4 to 0.2 changed the one-minute outlet peak by +0.109% and net export by +37.3 m³ through 40 h. CPU runs used the same initial fields and forcing. [Time refinement, both gauge records and water accounts](fidelity/native_results/hancock_temporal_sensitivity.csv)

On plain-709-intense-1000x1000 at 10 m, reducing Hancock CFL from 0.4 to 0.2 changed the one-minute outlet peak by +0.285% and net export by +93.1 m³ through 32 h. CPU runs used the same initial fields and forcing. [Time refinement, both gauge records and water accounts](fidelity/native_results/hancock_temporal_sensitivity.csv)

### Hydraulic grid and formulation on the same plain

All grids cover the same 1 km². The comparison across grid spacings includes changes to both the represented bed and the hydraulic discretisation. The FV1 and Hancock columns at 1 m use identical terrain and forcing. Initial-window peaks are the largest recorded through hour eight. Rain stops at hour six, and a later crest can exceed the initial-window peak.

| Storm | Window | End (h) | Interval (s) | Observable | FV1, 1 m | Hancock, 1 m | Hancock, 5 m | Hancock, 10 m |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Intense | Initial window | 8 | 60 | Peak (m³/s) | 2.481 | 3.044 | 6.949 | 6.806 |
| Intense | Initial window | 8 | 60 | Export (m³) | 27,557 | 39,616 | 76,714 | 74,310 |
| Moderate | Initial window | 8 | 60 | Peak (m³/s) | 0.525 | 0.9179 | 0.9629 | 0.9154 |
| Moderate | Initial window | 8 | 60 | Export (m³) | 8,446 | 17,236 | 16,401 | 15,411 |
| Moderate | Common recession | 40 | 60 | Peak (m³/s) | 0.525 | 0.9179 | 0.9629 | 0.9154 |
| Moderate | Common recession | 40 | 60 | Export (m³) | 26,394 | 43,388 | 40,372 | 38,983 |

[Per-configuration observations and physical input identities](fidelity/native_results/measured_windows.csv)

### Flow across the transverse embankment

The embankment crest is centred at x = 875 m. At x = 880 m the table partitions accepted eastward transport into the opening band (northings 380–460 m) and the rest of the face. The bed diagnostic takes the highest elevation between eastings 830 and 910 m on each east–west transect. Its reported shift is the median represented-minus-native crest elevation outside the opening band. Each FV1, Hancock and RK2 profile retains signed volumes, including any reversal.

| Storm | Window | End (h) | Hancock grid (m) | In opening band (m³) | Elsewhere (m³) | Median crest shift elsewhere (cm) |
|---|---|---:|---:|---:|---:|---:|
| Intense | Initial | 8 | 1 | 25,457 | 8,043 | +0.00 |
| Intense | Initial | 8 | 5 | 20,310 | 47,517 | -5.89 |
| Intense | Initial | 8 | 10 | 20,267 | 45,924 | -5.93 |
| Moderate | Initial | 8 | 1 | 13,294 | 345 | +0.00 |
| Moderate | Initial | 8 | 5 | 11,248 | 1,170 | -5.89 |
| Moderate | Initial | 8 | 10 | 10,887 | 888 | -5.93 |
| Moderate | Common recession | 40 | 1 | 39,052 | 345 | +0.00 |
| Moderate | Common recession | 40 | 5 | 35,022 | 1,197 | -5.89 |
| Moderate | Common recession | 40 | 10 | 34,284 | 917 | -5.93 |

Through hour 8 of the moderate storm, the portion of net eastward transport outside the band was 2.5% on the 1 m grid and 9.4% on the 5 m grid.

Through hour 8 of the intense storm, the portion of net eastward transport outside the band was 24.0% on the 1 m grid and 70.1% on the 5 m grid.

This exploratory fixed-section measurement locates changes in net routing during the storm. The opening band includes adjacent land, and grid refinement changes both the represented crest and the hydraulic solution. [All formulations, crest elevations and accepted face-volume identities](fidelity/native_results/embankment_transport.csv)

[Transverse profiles of crest elevation and transport](supplementary/operational_storms/plain_709_embankment_profiles.pdf). Shading marks the opening band.

### Where export differences develop

The table gives FV1 minus dense Hancock on either side of the internal gauge. The upstream east face is the gauge and the downstream east face is the outlet. Infiltration and retained storage come from separate cell fields. Each reach closes against its initial water, rain and signed transport on all four sides.

| Storm | Grid (m) | End (h) | Cadence (s) | Gauge x (m) | Reach | Δ eastward export (m³) | Δ infiltration (m³) | Δ retained storage (m³) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| plain-709-moderate-1000x1000 | 1 | 40 | 60 | 500 | upstream | -1,461 | +1,461 | +0 |
| plain-709-moderate-1000x1000 | 1 | 40 | 60 | 500 | downstream | -16,994 | +9,873 | +5,659 |
| plain-709-moderate-1000x1000 | 5 | 40 | 60 | 500 | upstream | -1,146 | +1,146 | +0 |
| plain-709-moderate-1000x1000 | 5 | 40 | 60 | 500 | downstream | -14,311 | +7,943 | +5,222 |
| plain-709-moderate-1000x1000 | 10 | 40 | 60 | 500 | upstream | -756 | +756 | +0 |
| plain-709-moderate-1000x1000 | 10 | 40 | 60 | 500 | downstream | -14,071 | +7,094 | +6,221 |

[Per-formulation water accounts and field identities](fidelity/native_results/regional_water.csv)

## Native hydraulic cell comparisons

Each row compares fields at the stated physical time on identical hydraulic cells against ordinary-CFL RK2 for the same storm. Depth RMSE covers all cells. Extent uses thresholds on individual cells. The half-CFL RK2 rows measure temporal sensitivity. Absolute practical accuracy at the native terrain floor remains unresolved.

| Storm | End (h) | Grid (m) | Compared formulation | CFL factor | Depth RMSE (mm) | Maximum-depth RMSE (mm) | 5 cm IoU | 10 cm IoU |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| hill-8500-back-1000x1000 | 8.0 | 1 | RK2 | 0.5 | 0.0083048 | 0.0084534 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 1 | RK2 | 0.5 | 0 | 0.0084534 | empty union | empty union |
| hill-8500-back-1000x1000 | 8.0 | 1 | FV1 | 1.0 | 0.10716 | 6.2974 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 1 | FV1 | 1.0 | 0.0092575 | 6.2974 | empty union | empty union |
| hill-8500-back-1000x1000 | 8.0 | 1 | Hancock | 1.0 | 0.010741 | 0.018788 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 1 | Hancock | 1.0 | 0 | 0.018788 | empty union | empty union |
| plain-709-intense-1000x1000 | 8.0 | 1 | RK2 | 0.5 | 0.00029241 | 0.00049697 | 1.00000 | 1.00000 |
| plain-709-intense-1000x1000 | 8.0 | 10 | FV1 | 1.0 | 51.745 | 28.641 | 0.72401 | 0.44878 |
| plain-709-intense-1000x1000 | 8.0 | 5 | FV1 | 1.0 | 53.231 | 29.119 | 0.71496 | 0.43857 |
| plain-709-intense-1000x1000 | 8.0 | 1 | FV1 | 1.0 | 26.762 | 20.658 | 0.97715 | 0.80541 |
| plain-709-intense-1000x1000 | 8.0 | 1 | Hancock | 1.0 | 0.02042 | 0.012255 | 0.99992 | 0.99966 |
| plain-709-intense-1000x1000 | 8.0 | 5 | Hancock | 1.0 | 0.076497 | 0.078499 | 0.99961 | 0.99843 |
| plain-709-intense-1000x1000 | 8.0 | 10 | Hancock | 1.0 | 0.39292 | 0.27157 | 0.99638 | 0.98982 |
| plain-709-moderate-1000x1000 | 8.0 | 1 | RK2 | 0.5 | 3.3885e-05 | 0.001106 | 1.00000 | 1.00000 |
| plain-709-moderate-1000x1000 | 32.0 | 1 | RK2 | 0.5 | 3.7162e-05 | 0.001106 | 1.00000 | 1.00000 |
| plain-709-moderate-1000x1000 | 40.0 | 1 | RK2 | 0.5 | 5.6854e-06 | 0.001106 | 1.00000 | 1.00000 |
| plain-709-moderate-1000x1000 | 8.0 | 5 | FV1 | 1.0 | 19.793 | 16.335 | 0.79376 | 0.88864 |
| plain-709-moderate-1000x1000 | 40.0 | 5 | FV1 | 1.0 | 11.079 | 16.335 | 0.36364 | 0.25926 |
| plain-709-moderate-1000x1000 | 8.0 | 10 | FV1 | 1.0 | 20.303 | 17.261 | 0.79584 | 0.84037 |
| plain-709-moderate-1000x1000 | 40.0 | 10 | FV1 | 1.0 | 12.611 | 17.261 | 0.19355 | 0.00000 |
| plain-709-moderate-1000x1000 | 8.0 | 1 | FV1 | 1.0 | 19.012 | 15.655 | 0.77918 | 0.79998 |
| plain-709-moderate-1000x1000 | 40.0 | 1 | FV1 | 1.0 | 12.184 | 15.73 | 0.24261 | 0.35417 |
| plain-709-moderate-1000x1000 | 8.0 | 5 | Hancock | 1.0 | 0.041501 | 0.027549 | 0.99926 | 0.99880 |
| plain-709-moderate-1000x1000 | 40.0 | 5 | Hancock | 1.0 | 0.0032404 | 0.027549 | 1.00000 | 1.00000 |
| plain-709-moderate-1000x1000 | 8.0 | 10 | Hancock | 1.0 | 0.09841 | 0.059938 | 0.99866 | 0.99595 |
| plain-709-moderate-1000x1000 | 40.0 | 10 | Hancock | 1.0 | 0.025406 | 0.059938 | 1.00000 | empty union |
| plain-709-moderate-1000x1000 | 8.0 | 1 | Hancock | 1.0 | 0.0053159 | 0.0035165 | 0.99991 | 0.99974 |
| plain-709-moderate-1000x1000 | 32.0 | 1 | Hancock | 1.0 | 0.0037484 | 0.0035165 | 1.00000 | 1.00000 |
| plain-709-moderate-1000x1000 | 40.0 | 1 | Hancock | 1.0 | 0.00052723 | 0.0035165 | 1.00000 | 1.00000 |
| plain-709-spatial-1000x1000 | 8.0 | 1 | RK2 | 0.5 | 0.00094817 | 0.0012903 | 1.00000 | 1.00000 |
| plain-709-spatial-1000x1000 | 20.0 | 1 | RK2 | 0.5 | 3.8052e-05 | 0.0012476 | 1.00000 | 1.00000 |
| plain-709-spatial-1000x1000 | 8.0 | 1 | FV1 | 1.0 | 0.5193 | 0.54444 | 0.98380 | 0.99025 |
| plain-709-spatial-1000x1000 | 20.0 | 1 | FV1 | 1.0 | 0.40509 | 1.0658 | 0.87808 | 0.85119 |
| plain-709-spatial-1000x1000 | 8.0 | 1 | Hancock | 1.0 | 0.010704 | 0.010765 | 1.00000 | 0.99803 |
| plain-709-spatial-1000x1000 | 20.0 | 1 | Hancock | 1.0 | 0.00011577 | 0.0071151 | 1.00000 | 1.00000 |

[Wet-union errors, connectivity and map identities](fidelity/native_results/native_grid_field_differences.csv)

The maps show the same plain two hours after rainfall ended for the moderate and intense storms. Shading preserves the fraction of hydraulic cells reaching 10 cm in each 10 m reporting cell. The dashed line marks the internal gauge. [1 m hydraulic grid](supplementary/operational_storms/plain_709_extent_10cm_1m.pdf) · [5 m hydraulic grid](supplementary/operational_storms/plain_709_extent_10cm_5m.pdf) · [10 m hydraulic grid](supplementary/operational_storms/plain_709_extent_10cm_10m.pdf)

## Practical grid selection on native terrain

The validation design includes 2 independent terrains with an admissible 5 and 10 m pair and all scheduled depth reference controls (hill-8200, rolling-8105). Completed runs currently qualify 0 terrain pairs for depth. [Design and measured eligibility by observable](fidelity/advisor_design/paired.json)

The test design includes 2 independent terrains with an admissible 5 and 10 m pair and all scheduled depth reference controls (fan-8400, rolling-8300). Completed runs currently qualify 0 terrain pairs for depth. [Design and measured eligibility by observable](fidelity/advisor_design/paired.json)

The paired depth design tests forecasts of 40,000 cells at 5 m and 10,000 cells at 10 m against a 1,000,000-cell 1 m numerical reference on the same footprint. Independent validation of larger-domain grid choices requires admissible comparisons at those scales. [Candidate and reference grids, CFL controls and measured endpoints](fidelity/advisor_design/scheduled.csv)

On floodplain-8103 at 5 m, the installed DEM reader measured maximum local barrier lowering of 0.183 m. Its solver decision was `refine_terrain`. [Installed CLI and archived features](fidelity/advisor_design/installed_cli.csv)

On floodplain-8103 at 10 m, the installed DEM reader measured maximum local barrier lowering of 0.169 m. Its solver decision was `refine_terrain`. [Installed CLI and archived features](fidelity/advisor_design/installed_cli.csv)

Joint method and grid selection on the new final terrains awaits the held-out evaluation.

With quarter-second accepted transport at 200 m, 5 m Hancock took 0.149 s and 1 m FV1 took 1.348 s (9.03×). Their respective sharp-pulse peak biases were -54.09% and -58.60%. Refining Hancock to 1 m reduced its peak bias to -7.94%. [Quarter-second analytical comparisons](fidelity/results/sharp_pulses.csv)
