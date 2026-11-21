# Corrective numerical and GPU audit

**Qualification in progress.** 124 numerical records remain outstanding. Values below cover completed events.

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

| Hancock execution | Events | Median / FV1 | Events ≤1.5× |
|---|---:|---:|---:|
| Dense | 48 | 1.352 | 30 |
| Tiles | 48 | 2.220 | 0 |
| Fastest measured | 48 | 1.352 | 30 |

For 96 cells, compacting the footprint costs extra launches and leaves little hydraulic work to skip. On large sparse rasters, each omitted tile avoids reconstruction, HLL flux, draining, update and accounting work. The separate common-cap measurements isolate this effect. Normal-policy storm times also include differences in accepted timesteps and rejected trials.

## Operational storm matrix

The 8 matched 1,024 × 1,024 comparisons covered 8 landform families. The faster verified explicit Hancock schedule met the 1.5× FV1 target in 1 of 8 complete events. Its median ratio was 1.69×. This retrospective choice gives an attainable measured cost for each event. The automatic runs used a policy frozen before these test observations.

| Complete event | Grid cells | Δx (m) | Hours | FV1 (s) | Hancock dense (s) | Tiles (s) | Auto (s) | Fastest explicit / FV1 | Hancock / FV1 steps | Fastest / FV1 per step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal-703-moderate-10240x10240 | 1048576 | 10 | 8 | 67.651 | 105.3 | 107.89 | 104.32 | 1.56 | 1.06 | 1.47 |
| fan-705-spatial-5120x5120 | 1048576 | 5 | 8 | 22.542 | 78.177 | 37.264 | 37.128 | 1.65 | 1.42 | 1.16 |
| floodplain-702-moderate-10240x10240 | 1048576 | 10 | 12 | 101 | 171.69 | 175.58 | 170.57 | 1.7 | 1.12 | 1.51 |
| hill-706-back-10240x10240 | 1048576 | 10 | 8 | 104.27 | 226.84 | 232.48 | 225.29 | 2.18 | 1.45 | 1.5 |
| incised-707-spatial-10240x10240 | 1048576 | 10 | 8 | 28.372 | 118.26 | 56.092 | 55.975 | 1.98 | 1.62 | 1.22 |
| mixed-708-spatial-10240x10240 | 1048576 | 10 | 8 | 30.451 | 124.16 | 51.032 | 50.9 | 1.68 | 1.47 | 1.14 |
| mixed-714-spatial-20480x10240 | 2097152 | 10 | 8 | 59.353 | 231.01 | 87.453 | 87.417 | 1.47 | 1.33 | 1.11 |
| plain-701-spatial-10240x10240 | 1048576 | 10 | 8 | 8.633 | 27.231 | 9.6535 | 9.6754 | 1.12 | 1.1 | 1.02 |
| plain-709-spatial-1000x1000 | 1000000 | 1 | 8 | 112.72 | 251.72 | 150.23 | 149.92 | 1.33 | 1.03 | 1.29 |
| plain-712-spatial-20480x10240 | 2097152 | 10 | 24 | 48.748 | 164.42 | 53.375 | 53.353 | 1.09 | 1.11 | 0.983 |
| plain-715-spatial-10240x10240 | 4194304 | 5 | 8 | 65.287 | 219.66 | 71.023 | 71.239 | 1.09 | 1.09 | 1 |
| plain-715-spatial-10240x10240 | 1048576 | 10 | 8 | 8.6163 | 26.982 | 9.5917 | 9.5797 | 1.11 | 1.1 | 1.02 |
| rolling-704-soil-5120x5120 | 1048576 | 5 | 8 | 70.34 | 156.65 | 144.93 | 156.03 | 2.06 | 1.28 | 1.61 |
| rolling-713-moderate-20480x10240 | 2097152 | 10 | 8 | 120.76 | 242.01 | 249.96 | 242.08 | 2 | 1.33 | 1.51 |

The archive contains 80 complete operational forecasts across the scale and fixed-area designs. The largest measured primary grid has 4,194,304 cells. The memory observations are live device-pool allocations. Compilation, graph capture, hydraulic execution, extraction and receipt wall time are separate. For changing coastal stages the frame hydraulic timer surrounds the advance call. Prescribed boundary assignments between calls are included in end-to-end time.

## Mechanisms, measured cost and Pareto evidence

Across the 8 primary events Hancock accepted 294,040 steps and rejected 572 trials (0.19% of trials). FV1 accepted 219,571 steps. The incised case had the largest accepted-step ratio (1.62×), with 36 retries for 45,545 accepted steps. Rejections explain little of the event-wide step count. Earlier instrumented wet fronts located the restrictive signal at a reconstructed wet/dry face. A separate incised outlet case was limited by the cell-centred gravity-wave speed. The million-cell receipts measure the accepted and rejected counts. Identifying the limiting face or cell at each step would require instrumenting those runs. [Per-event timestep records](processed/million/observations.csv) · [Earlier CFL attribution](../../validation/hancock_long_storm_review/REPORT.md#timestep-attribution)

The steps-ratio column isolates restrictions from the remaining per-step cost. The scheduling table records wet and padded work fractions. The complete graph replays measure individual kernel categories independently of uninstrumented wall time. Dominated, tied and unresolved configurations are traceable to individual runs. Allocation footprint and runtime each have a distinct objective. [Execution rows](processed/million/policies.csv) · [Large-domain Pareto decisions](processed/million/pareto.csv) · [Kernel attribution](processed/million/graph_profile.csv) · [Equal-budget comparisons](tables/08_cost_accuracy.md)

## Reference and terrain evidence

Depth, wet-depth and maximum-depth comparisons use area averages on the common physical support specified by each case. Final inundation IoU uses thresholded common-support depths at 1, 5 and 10 cm. Native-grid component counts and wet-area differences are also recorded. A final-depth metric cannot measure a missing connection by itself. The prospective reference supplement retains 60-second interval discharge in fast cases and exact volume aggregation to 900 seconds.

| Observable | Reassessed small cases | Originally qualified | Qualified on both finer checks |
|---|---:|---:|---:|

The fast-event outlet records support a direct comparison of interval means from the same accepted face volumes. Times below are the ends of the intervals containing the maximum mean discharge. [Fine-interval peak observations](processed/million/observations.csv) · [15-minute hydrographs](processed/million/hydrographs_900s.csv)

| Landform | Method | Peak at 60 s (m³/s) | Peak at 900 s (m³/s) | Attenuation from averaging | Peak interval end, 60 s / 900 s (s) |
|---|---|---:|---:|---:|---:|
| incised | fv1 | 48.367 | 48.285 | 0.17% | 19440 / 19800 |
| incised | hancock_dense | 47.527 | 47.170 | 0.75% | 17100 / 17100 |
| mixed | fv1 | 41.363 | 41.153 | 0.51% | 27060 / 27900 |
| mixed | hancock_dense | 38.700 | 38.542 | 0.41% | 26160 / 26100 |

At the prescribed test stage, 11 of 15 independent case windows lost the metre-grid connection on the 10 m represented terrain. The stage is 1 cm above the lowest native through-going spill elevation. Each window covers 80 × 80 m around the known control and retains its physical coordinates. Whole-domain metrics use the full represented raster. [Terrain diagnostics](processed/million/terrain.csv) · [Fixed-bed and initial-water controls](processed/million/resolution_components.csv) · [Reference screens](processed/million/reference_checks.csv)

## Held-out operational recommendations

| Rule, at least one million cells | Decisions | Qualified feasible opportunities | Covered (%) | Abstained (%) | False feasible | Cheapest / opportunities (%) | Conditional regret (%) |
|---|---:|---:|---:|---:|---:|---:|---:|

The previously audited evaluation selected a method on 62 of 120 decisions. It recorded 41 successful selections, 58 abstentions, 156.0% median conditional regret and 81.4% empirical interval coverage. [Original immutable validation](raw_results/advisor-validate/b23f24375a6d611f22d5eca6/7a601da9a91c431f/summary.csv) · [Matched old and revised decisions](processed/advisor/previous_comparison.csv)

Independent operational advisor validation is pending.

The held-out physical cases determine whether a measured improvement in computational regret also improves feasible coverage. The runtime-only comparison and the objective-specific feasibility comparison answer separate questions. All rows from an independent terrain realisation belong to the same split. Pre-run inputs describe terrain and prescribed forcing. Realised wet fraction is used only to stratify results after validation. [Held-out individual decisions](processed/advisor/heldout.csv) · [By landform, grid, size and wet fraction](processed/advisor/validation_strata.csv) · [Runtime-only decisions](processed/advisor/runtime_only.csv)

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
| large-references | 4 | 77 |
| large-refinements | 0 | 29 |
| heldout-references | 0 | 5 |
| million-profiles | 0 | 9 |
| million-repeats | 12 | 12 |
| recession-controls | 0 | 8 |

The full CUDA regression receipt records **1893 passed, 25 skipped, 743 warnings in 124.25s (0:02:04)**. [Test log](logs/check/9627291943e14358.log) · [Committed numerical source proof](manifests/corrective_source_tree.json)

The catalogue retains 2 interrupted or failed numerical attempts outside all hydraulic and timing comparisons. [Attempt catalogue](processed/metrics/attempt_catalog.csv)
- large-references, plain-701-spatial-10240x10240, hancock_dense — incomplete, attempt `affc4e9a96104629`.
- million-storms, plain-709-spatial-1000x1000, adaptive — incomplete, attempt `dc1995df12c544a8`.

Across 6 complete timing controls with at least three runs, observed hydraulic runtime ranges spanned 0.08% to 0.68% of the within-control median. [Individual timing ranges](processed/million/repeat_times.csv)

The prospective preflight used tiled execution in 9 of 14 measured grids of at least one million cells. Its measured excess runtime over the fastest explicit observation had a median of -0.2% and a maximum of +7.7%. Separate executions include timing variability.

Preflight selected the faster explicit schedule on 13 of 14 measured grids of at least one million cells. The work estimate uses the initial state and prescribed rain footprint. The matched tile run measures the work actually scheduled over the entire event. [Per-event estimates and observations](processed/million/policies.csv)

| Event | Δx (m) | Chosen policy | Rain footprint | Estimated scheduled | Observed tiled work | Auto cost above fastest |
|---|---:|---|---:|---:|---:|---:|
| rolling-704-soil-5120x5120 | 5 | dense | 100.0% | 100.0% | 80.2% | +7.7% |

## Measured configurations by observable

| Observable | Eligible | Empirical frontier | Frontier under reference sensitivity | Unresolved |
|---|---:|---:|---:|---:|
| depth | 15 | 8 | 9 | 65 |
| discharge | 0 | 0 | 0 | 80 |
| inundation | 15 | 5 | 9 | 65 |
| maximum_depth | 15 | 8 | 9 | 65 |
| memory | 80 | 15 | 15 | 0 |
| peak_discharge | 0 | 0 | 0 | 80 |
| peak_timing | 6 | 1 | 1 | 74 |

Final 1 cm inundation is the common prediction target in the enlarged advisor. The historical paper also retains time-integrated inundation results. Costs are stratified by the Hancock wetted and scheduled fractions at the end of each 15-minute window and use the FV1 time for exactly the same physical window. [Window-level costs](processed/million/wetness_cost.csv) · [Joint solver and resolution validation](processed/advisor/joint_resolution.csv)

## Cheapest measured method and grid meeting tolerance

Each entry uses the measured error plus the reported refinement sensitivity. Tolerances are prescribed in the advisor configuration. No tested option means a qualified comparison existed but the conservative criterion was missed.

| Physical event | Final depth | Final inundation | Outlet hydrograph |
|---|---|---|---|
| coastal-703-moderate-10240x10240 | unresolved | unresolved | unresolved |
| fan-705-spatial-5120x5120 | unresolved | unresolved | unresolved |
| fan-711-spatial-10000x5000 | unresolved | unresolved | unresolved |
| floodplain-702-moderate-10240x10240 | unresolved | unresolved | unresolved |
| hill-706-back-10240x10240 | unresolved | unresolved | unresolved |
| incised-707-spatial-10240x10240 | unresolved | unresolved | unresolved |
| mixed-708-spatial-10240x10240 | unresolved | unresolved | unresolved |
| mixed-714-spatial-20480x10240 | unresolved | unresolved | unresolved |
| plain-701-spatial-10240x10240 | unresolved | unresolved | unresolved |
| plain-709-spatial-1000x1000 | fv1, 10 m | hancock_auto, 5 m | unresolved |
| plain-710-spatial-5000x5000 | unresolved | unresolved | unresolved |
| plain-712-spatial-20480x10240 | unresolved | unresolved | unresolved |
| plain-715-spatial-10240x10240 | fv1, 10 m | fv1, 10 m | unresolved |
| rolling-704-soil-5120x5120 | unresolved | unresolved | unresolved |
| rolling-713-moderate-20480x10240 | unresolved | unresolved | unresolved |
