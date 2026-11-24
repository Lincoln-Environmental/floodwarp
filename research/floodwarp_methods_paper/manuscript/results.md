# Results

The original analytical, synthetic, calibration and advisor tables use their archived numerical revisions. The repaired FREE-outlet measurements are reported separately under [Repaired FREE outlet](#repaired-free-outlet).

## Exact and manufactured controls

The 342 primary forecasts had a minimum depth of 0 m. The largest relative mass residual was 1.61e-12. The independent lake-at-rest control had a maximum depth error of 8.88e-16 m. The refinement table includes spatial, fixed-grid temporal and source-integration sequences. [Analytical errors](../tables/04_analytical_accuracy.md) · [Convergence](../tables/05_convergence.md)

## Flood-wave fidelity per unit cost

Of 48 small events with six hours of rain, 31 required a longer recession. The longest matched simulation reached 76 hours. For the 12 storms with independently qualified normalised outlet shape at 10 m, the Hancock–FV2 difference in absolute shape error exceeded the combined reference sensitivity in 0 cases. That comparison concerns the timing distribution of discharged volume. Peak and arrival-time conclusions have separate reference screens. Among 6 FV1 storms with independently assessable peaks and final depth RMSE within 1 mm, 3 had an absolute peak bias above 10%. [Complete small-storm comparisons](../fidelity/small_storms/README.md) · [Million-cell export after the original stop time](../fidelity/NATIVE_RECESSIONS.md)

## Original fixed-window depth and discharge diagnostics

The 48 prospective physical events included six hours of rain and a common recession for each matched comparison. The table below gives medians across the events that passed the refinement test for each observable. LI uses its native momentum equation. All methods and the error associated with reference refinement appear in the supporting tables. [Six-hour storm comparisons](../tables/06_storm_accuracy.md) · [Reference uncertainty](../processed/metrics/reference_uncertainty.csv)

| Observable | Qualified events | FV1 | Tiled Hancock | FV2 RK2 |
|---|---:|---:|---:|---:|
| depth (m) | 25 | 0.001977 | 0.001996 | 0.002056 |
| discharge (m³/s) | 36 | 0.002054 | 0.001343 | 0.001333 |

## Original-window efficiency and terrain representation

On the 96-cell controls, dense Hancock was faster than tiles in 48 of 48 original fixed-duration forecasts. Relative to sparse FV1 its median runtime was 1.352×. The tiled median was 2.220×. Selecting the faster qualified execution retrospectively gave 1.352×. Both schedules reproduced the same Hancock fields. With only 96 cells, marking and compacting tiles adds GPU launches while eliminating little work. The large-domain suite measures how this balance changes with the scheduled footprint. [Measured GPU costs](../tables/07_runtime.md) · [Full performance qualification](../../../validation/hancock_long_storm_review/DECISION.md)

Fixed-geometry and resampled-geometry experiments use the same physical extent. Their objective-specific rankings and measured cost ratios are available for the plain, fan and incised development realisations. An unresolved reference is an explicit missing comparison. [Resolution profiles](../figures/06_resolution.pdf) · [Equal-cost results](../tables/08_cost_accuracy.md) · [Least-cost feasible configurations](../processed/pareto/least_cost.csv)

## Roughness transfer

The 9 nested training fits selected multipliers between 0.8 and 0.95. Across 105 reference-qualified transfer rows, 64 scored better than physical roughness and 41 scored worse. Multiple bounds can select the same roughness. These counts include repeated bounds and objectives on the same terrain. The group intervals preserve that dependence. [Bounded calibration](../tables/09_calibration.md) · [Grouped transfer intervals](../processed/comparisons/calibration_intervals.csv)

## Held-out solver decisions

The advisor made 16 selections in 150 held-out objective decisions, including 6 with unresolved reference evidence. Among 10 assessable selections, 100% satisfied the requested accuracy and 0 exceeded it. Agreement with the measured least-cost method was 10%. The median conditional runtime regret was 258.6%. The empirical interval covered the observed error in 80% of assessable predictions. [Held-out rule comparison](../tables/10_advisor.md) · [Individual decisions](../processed/advisor/heldout.csv)

## Formulation at matched resolution

The steep moving front and the smooth gravity pulse illustrate different limiter and timestep regimes. All values below are exact-solution RMSE at 5 m.

| Case | Method | Depth (mm) | Face discharge (m²/s) | Accepted steps |
|---|---|---:|---:|---:|
| front_steep | fv1 | 35.46 | 0.1655 | 62 |
| front_steep | hancock_adaptive | 3.208 | 0.01163 | 126 |
| front_steep | adaptive | 3.197 | 0.01147 | 129 |
| gravity_pulse | fv1 | 0.8559 | 0.001149 | 72 |
| gravity_pulse | hancock_adaptive | 0.2311 | 0.0002727 | 72 |
| gravity_pulse | adaptive | 0.2678 | 0.0003071 | 72 |

On the archived 1 m weak dam, LI's depth RMSE was 1.844 mm against its own momentum equations and 1.863 mm against full SWE. Each equation uses its own independent Riemann reference. [Equation-specific errors](../processed/metrics/archival_reduced_equations.csv)

## Resolved differences on storm terrain

A pairwise difference is labelled resolved here when its magnitude exceeds twice the observed reference-refinement change and the objective passes its individual refinement screen. The margin uses the observed change as an uncertainty estimate. It does not establish a rigorous bound on continuum error.

| Observable, 10 m | Comparison method | Hancock lower error | Alternative lower error | Inside margin | Reference unresolved |
|---|---|---:|---:|---:|---:|
| depth | fv1 | 2 | 2 | 21 | 23 |
| depth | adaptive | 1 | 1 | 23 | 23 |
| depth | li | 5 | 6 | 14 | 23 |
| discharge | fv1 | 14 | 0 | 22 | 12 |
| discharge | adaptive | 0 | 0 | 36 | 12 |
| discharge | li | 13 | 5 | 18 | 12 |
| inundation | fv1 | 4 | 0 | 10 | 34 |
| inundation | adaptive | 1 | 0 | 13 | 34 |
| inundation | li | 8 | 0 | 6 | 34 |

The fine-grid cross-formulation file reports each available Hancock–FV2 difference on identical 1 m geometry. The per-family file gives the operational timing and timestep ratios for all eight landforms. [Fine-grid comparisons](../processed/metrics/independent_fine_formulation.csv) · [Landform results](../processed/comparisons/landform_results.csv)

## Million-cell operational controls

The 8 matched 1,024 × 1,024 comparisons covered 8 landform families. Across the original fixed-duration forecasts, the median runtime of the faster verified explicit Hancock schedule relative to FV1 was 1.69×. This retrospective choice gives an attainable measured cost for each event. The automatic runs used a policy frozen before these test observations.

The partial-domain rain and widespread inundation cases have different scheduling and timestep costs. [Complete operational records](../tables/11_execution.md) · [Costs by scheduled footprint](../figures/13_execution_cost.pdf) · [Finer reference checks](../tables/12_reference_qualification.md) · [Operational hydrographs](../figures/18_operational_storms.pdf).

## Held-out evidence at operational scale

The advisor abstained on 30 of 30 million-cell decisions. No measured option satisfied both the hydraulic acceptance criterion and terrain admissibility. This test cannot establish an advantage over the simple rules or estimate operational regret on feasible choices. Of 30 decisions, 12 had a qualified reference, 10 had a measured method within hydraulic tolerance and 0 also had admissible terrain. [Selection eligibility by case](processed/advisor/eligibility_breakdown.csv) [Scale-specific denominators](../tables/13_advisor_scale.md).

## DEM aggregation and extent thresholds

Earlier resolution experiments used cell-centre sampling or repeated parent elevations. The incised fixed-bed refinement created artificial treads that substantially altered runoff. These historical experiments do not establish accuracy at a practical resolution using an area-averaged DEM. The corrected series averages 25 native 1 m elevations per 5 m cell, then four 5 m elevations per 10 m cell. Initial water, momentum, rainfall footprint and infiltration also come from the native metre source. All practical runs start at time zero or resume their own verified checkpoint. [Native-input storm windows, budgets and reference qualification](../fidelity/native_results/README.md) Report practical extent separately at 5 and 10 cm. Historical 1 cm extent is a shallow-water diagnostic. [DEM construction, numerical controls and corrected outcomes](../fidelity/DEM_CORRECTION.md)

The native-grid incised results also show sensitivity to the downstream FREE boundary. The steady-sheet experiment checks this effect independently against exact discharge. [Outlet placement and terminal storage](../fidelity/native_outlet/README.md)

With the pre-repair FREE boundary, through the matched 16-hour window, the highest recorded FV2 RK2 outlet interval means were 66.40 m³/s at 10 h 51 min on the 10 m grid and 48.33 m³/s at 4 h 48 min on the 5 m grid. Each interval spans 60 seconds. Disagreement in these two measured maxima establishes grid sensitivity. The physical storm peak remains unresolved.

Runoff travel time and catchment storage allow the outlet maximum to follow rainfall. The incised diagnostic compares accepted flow at successive cross-sections with storage and infiltration over the intervening reach. The matched finer-grid checks test how strongly the outlet timing and magnitude depend on representation. [Recession water balance](../fidelity/native_tail/README.md)

[Matched native-grid costs and observable-specific reference eligibility](../fidelity/native_publication/README.md)

The pre-repair replay measured the source of discharge during the highest post-hour-eight minute at each spacing. Net drawdown in the last 40 m supplied 100.00% on the 5 m grid and 99.76% on the 10 m grid. That water may have entered the reach earlier. [Crest-minute budgets](../fidelity/native_tail/CREST.md)

With the pre-repair FREE boundary on the 5 m grid, FV1 exported 100,848 m³ more water than Hancock by hour 16. Lower cumulative infiltration accounted for 100,268 m³ of the difference and lower final storage for 579 m³. The net infiltration difference arose on cells outside the rainfall footprint. These terms close the water balance under the same prescribed infiltration capacity. [Runoff and infiltration by grid](../fidelity/native_outlet/runoff_partition.csv)

## Repaired FREE outlet

On the same 10 m incised terrain, the corrected 16-hour Hancock and RK2 peaks were 47.18 and 47.66 m³/s. Their earlier late crests disappeared. These are one-minute means at the original outlet. [Fixed-gauge boundary experiments and water budgets](../fidelity/outlet_results/README.md)

The pre-repair and corrected hydraulic sources match commits `0afdd684d2d1` and `46afa3f280a5`. The corrected FP64 storm measurements used NVIDIA GeForce RTX 4070 Ti SUPER, 595.99.02, 16376 MiB. [Source and workstation receipts](../fidelity/outlet_results/revision_evidence.json)

Of 342 core exact-solution controls, 342 reproduced the pre-repair sampled fields and gauge transport exactly. 342 retained the same ordered accepted timesteps. The maximum relative mass-balance residual was 1.61e-12 and the lake-at-rest depth error was 8.88e-16 m. Measuring the actual steep FREE face exposed a defect that these sampled controls did not detect. [Paired exact references, results and run identities](../fidelity/analytical_revision/comparisons.csv)

At the fixed 120 m-wide incised section on the 10 m grid, adding a 100 m downstream buffer changed mean RK2 discharge from 2.46253 to 3.32420 m³/s before the repair and from 3.32413 to 3.32420 m³/s after it. These accepted volumes between 30 and 60 s measure sensitivity to outlet placement. [Fixed-section measurements on the 1, 5 and 10 m grids](../fidelity/outlet_results/incised_reach.csv)

At 10 m, FV1 minus Hancock net export was +41,131 m³. The corresponding cumulative infiltration differences were +17,863 m³ inside the rainfall footprint and -59,098 m³ outside. The final storage difference was +104 m³. At hour eight their areas above 5 cm were 0.7233 and 0.0000 km².

At 5 m, FV1 minus Hancock net export was +81,117 m³. The corresponding cumulative infiltration differences were +6,211 m³ inside the rainfall footprint and -87,383 m³ outside. The final storage difference was +56 m³. At hour eight their areas above 5 cm were 0.5847 and 0.0000 km².

At rain cessation (6 h), the direct 5 m FV1-minus-Hancock differences were -10,822 m³ in cumulative export and +85,815 m³ in remaining storage. During the common recession alone FV1 exported +91,939 m³ relative to Hancock. [Rain and recession water accounts](../fidelity/outlet_results/phase_volume_differences.csv)

Across 65 snapshots at 15-minute intervals, the largest simultaneous areas above 5 cm were 3.7250, 0.0542 and 0.0535 km² for 5 m FV1, Hancock and RK2. These measurements cover the full direct-outlet storm and recession. Cumulative inundation also includes cells that flooded at different times. [Sampled maxima and cumulative extent](../fidelity/outlet_results/peak_extent.csv)

Of the hour-eight area above 5 cm in FV1 alone on the direct 5 m grid, 100.00% lay downstream of the independent internal gauge and 100.00% lay outside the rainfall footprint. The final 40 m contained 2.36% of this disagreement. [Reach and rainfall-footprint decomposition](../fidelity/outlet_results/regional_contrasts.csv) · [Maps of inundation and cumulative infiltration](../supplementary/incised/hour_eight_water.pdf)

On the 10 m direct-outlet grid, Hancock required 34,118 accepted steps versus 48,782 for FV1. Across 3 and 3 warmed timings respectively, the median Hancock runtime was 1.47× as much per accepted step and 1.029× as much over the full 16 hours. [Independent timings and execution policies](../fidelity/outlet_results/independent_timings.csv)

On the 5 m direct-outlet grid, Hancock required 68,115 accepted steps versus 99,493 for FV1. Across 3 and 3 warmed timings respectively, the median Hancock runtime was 1.52× as much per accepted step and 1.041× as much over the full 16 hours. [Independent timings and execution policies](../fidelity/outlet_results/independent_timings.csv)

During the corrected 10 m Hancock crest minute, 2,834.2 m³ entered the final 40 m, 2,831.0 m³ passed the FREE face, 2.8 m³ infiltrated, and surface storage changed by +0.4 m³. Over the post-rain recession, drawdown in that strip after infiltration accounted for 0.63% of outlet export. [Accepted transport and strip water accounts](../fidelity/outlet_results/last_40m.csv)

During the corrected 10 m RK2 crest minute, 2,862.9 m³ entered the final 40 m, 2,859.6 m³ passed the FREE face, 2.7 m³ infiltrated, and surface storage changed by +0.5 m³. Over the post-rain recession, drawdown in that strip after infiltration accounted for 0.62% of outlet export. [Accepted transport and strip water accounts](../fidelity/outlet_results/last_40m.csv)

Halving the RK2 CFL on the 5 m grid changed the one-minute peak from 48.115 to 48.123 m³/s. Net export changed by -17,792 m³, infiltration by +17,792 m³ and final storage by +0 m³. The normal and half-CFL runs accepted 77,045 and 146,457 timesteps.

The direct-outlet observables passing all three reference checks were peak magnitude, net exported volume. These checks support comparisons on the coarser grid. Practical 5 m absolute accuracy remains unresolved. [Reference qualification by observable](../fidelity/outlet_results/reference_checks.csv)

For all outward discharge over the fixed 16-hour window, the fraction of volume redistributed in time relative to the normal 5 m RK2 forecast was 14.53% for 10 m FV1 and 3.69% for 10 m Hancock. Their centroid shifts were +28.8 and -7.4 minutes. Applying the same shape diagnostic to the spatial, fine-formulation and temporal controls gave 2.51%, 0.54% and 1.05%. The initial moving channel contributes to these distributions. Rainfall-only hydrograph shape remains unresolved. [Fixed-window comparisons and reference controls](../fidelity/outlet_results/fixed_window_shape.csv)

With a 5% tolerance on peak discharge, FV1 at 10 m had the lowest measured hydraulic cost (38.4 s). Its absolute error against the 5 m RK2 reference was 0.53%, rising to 1.48% when reference sensitivity was included.

With a 10% tolerance on net exported volume, adaptive Hancock at 10 m had the lowest measured hydraulic cost (39.5 s). Its absolute error against the 5 m RK2 reference was 5.62%, rising to 9.78% when reference sensitivity was included.

The two decision tables retain the event endpoint, output cadence and reference for each observable. The hydraulic costs use independent timing medians where available. [Cost at a stated tolerance](../tables/19_incised_cost_at_tolerance.md) · [Error within a measured runtime](../tables/20_incised_error_at_budget.md)

Under intense rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at 4.331 and 6.806 m³/s. They exported 28,772 and 109,391 m³ through the common 32-hour endpoint. Under moderate rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at 0.894 and 0.915 m³/s. They exported 24,912 and 38,983 m³ through the common 40-hour endpoint. Both gauges passed the recession criterion. These CPU controls measure formulation disagreement. [Six-hour rainfall and complete recession accounts](../fidelity/strong_storms/completed.csv)

## Complete 1,048,576-cell storms on native-metre terrain

Peaks are interval means of accepted outlet transport. The volume columns give FV1 minus Hancock. Costs use the faster of the two verified explicit Hancock policies through the common recession endpoint and the median hydraulic time where repeats exist.

| Storm | Grid (m) | End (h) | Interval (s) | FV1 peak (m³/s) | Hancock peak (m³/s) | Δ net export (m³) | Δ infiltration (m³) | Hancock / FV1 time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coastal-703-moderate-10240x10240 | 10 | 32 | 300 | 131.72 | 112.55 | +870,045 | -835,214 | 1.752 |
| fan-705-spatial-5120x5120 | 5 | 12 | 300 | 10.59 | 10.12 | -3,662 | +2,883 | 1.245 |
| floodplain-702-moderate-10240x10240 | 10 | 28 | 300 | 141.05 | 123.69 | +1,123,372 | -1,137,291 | 1.954 |
| hill-706-back-10240x10240 | 10 | 16 | 300 | 1174.70 | 1251.56 | +128,872 | -130,907 | 1.477 |
| plain-701-spatial-10240x10240 | 10 | 52 | 300 | 0.00 | 0.00 | +0 | +0 | 0.943 |
| rolling-704-soil-5120x5120 | 5 | 12 | 300 | 61.21 | 25.64 | +161,747 | -161,747 | 2.002 |

The differences measure sensitivity to formulation on the same represented terrain. Accuracy decisions use the independently screened reference for each observable. [Water budgets and internal gauges](../fidelity/native_results/measured_windows.csv) · [Policy-specific observations and reference eligibility](../fidelity/native_publication/README.md)

## Practical grid selection on native terrain

The validation design includes 2 independent terrains with an admissible 5 and 10 m pair and all scheduled depth reference controls (hill-8200, rolling-8105). Completed runs currently qualify 0 terrain pairs for depth. [Design and measured eligibility by observable](../fidelity/advisor_design/paired.json)

The test design includes 2 independent terrains with an admissible 5 and 10 m pair and all scheduled depth reference controls (fan-8400, rolling-8300). Completed runs currently qualify 0 terrain pairs for depth. [Design and measured eligibility by observable](../fidelity/advisor_design/paired.json)

The paired depth design tests forecasts of 40,000 cells at 5 m and 10,000 cells at 10 m against a 1,000,000-cell 1 m numerical reference on the same footprint. Independent validation of larger-domain grid choices requires admissible comparisons at those scales. [Candidate and reference grids, CFL controls and measured endpoints](../fidelity/advisor_design/scheduled.csv)

On floodplain-8103 at 5 m, the installed DEM reader measured maximum local barrier lowering of 0.183 m. Its solver decision was `refine_terrain`. [Installed CLI and archived features](../fidelity/advisor_design/installed_cli.csv)

On floodplain-8103 at 10 m, the installed DEM reader measured maximum local barrier lowering of 0.169 m. Its solver decision was `refine_terrain`. [Installed CLI and archived features](../fidelity/advisor_design/installed_cli.csv)

Joint method and grid selection on the new final terrains awaits the held-out evaluation.

With quarter-second accepted transport at 200 m, 5 m Hancock took 0.149 s and 1 m FV1 took 1.348 s (9.03×). Their respective sharp-pulse peak biases were -54.09% and -58.60%. Refining Hancock to 1 m reduced its peak bias to -7.94%. [Quarter-second analytical comparisons](../fidelity/results/sharp_pulses.csv)
