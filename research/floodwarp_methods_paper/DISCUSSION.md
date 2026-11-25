# Discussion

Tiled execution preserves Hancock's conservative update. Differences between Hancock and RK2 depend on the physical observable, timestep history and limiter response. The median storm errors in the results provide evidence at the tested scales. Small discharge peaks and controls that shift on refinement can make percentage comparisons unstable. A solver that attenuates a front may still predict a satisfactory final storage.

The practical outcome depends on the accuracy objective. The Pareto archive identifies measured configurations dominated in both error and runtime within an individual event. A coarse DEM can remove drainage connections or lower a barrier. Comparing fixed represented geometry with continuous terrain resampling quantifies how this changes the error attributable to the hydraulic grid. Flow accumulation on a filled DEM describes potential drainage. Stage control, infiltration and depressional ponding can change the actual flow path.

Bounded Manning fits target integrated transport or depth and extent. A change that compensates for attenuation on one hydrograph can alter recession or peak timing on the next. The grouped transfer results quantify the effect of applying the fitted physical value unchanged. The first- and higher-order solvers at the prescribed roughness remain the comparison controls. [Transfer trials](processed/comparisons/transfer.csv)

The domain-wide solver advisor abstained on 134 held-out objectives. Its runtime forecast had median absolute percentage error 44.63% for assessable choices. Compare coverage and failed choices together with cost. A low conditional regret can reflect abstention on difficult cases. The original small-domain training cases covered 9,600 m². The expanded splits include independent larger terrain and event realisations. Large-domain accuracy recommendations require a qualified reference for the requested observable, sufficient training support at the requested scale and admissible terrain representation. Additional independent domains, stage controls, storms, GPUs and surveyed hydraulic structures are needed before using forecasts for release decisions.

Retain adaptive Hancock as the development default. Use FV1 or LI when the error for the intended observable is acceptable at lower measured cost. LI also needs a suitable reduced-momentum regime. Publication or operational use of a method recommendation requires a qualified reference and credible predictive coverage on the intended domain. [Rule baselines](tables/10_advisor.md)

Across held-out objectives with at least one empirically feasible option, the proportions receiving a qualified feasible selection were advisor 17.54%, fv1 89.47%, hancock_dense 91.23%, oracle 100%, slope 3.509%. This denominator includes abstentions. The stratified validation tables identify the resolutions and sizes with independent predictive evidence.

Across the 8 primary million-cell events, the faster measured explicit Hancock policy took a median of 1.69 times the FV1 runtime over the original observation windows. The accepted-step ratios ranged from 1.06 to 1.62, and the corresponding mean cost per accepted step ranged from 1.02 to 1.61 times FV1. These factors multiply within each event. The normal timestep policy therefore matters even where tile scheduling removes most of the dry-domain arithmetic. [Per-event costs](tables/11_execution.md)

## Flood-wave interpretation

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

### Outlet hydrograph sensitivity on the native 1 m grid

Each percentage uses ordinary-CFL RK2 on the same bed and event as the comparison scale. Negative peak differences indicate attenuation. Positive centroid delays indicate later transport. Centroids use interval-integrated transport. Subinterval differences describe these integrals at the displayed cadence. Width is the time taken to pass the central 80% of outward volume. Shape is total variation between volume-normalised hydrographs on the original clock. Initial flow contributes to all reported whole-window metrics on events that begin with an active outlet. Absolute storm accuracy at the native terrain floor remains unresolved.

| Event | End (h) | Cadence (s) | Method | Peak difference (%) | Net export difference (%) | Centroid delay (min) | Width difference (%) | Shape difference (%) | Window |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| hill-8500-back-1000x1000 | 12 | 60 | FV1 | -0.205 | +0.551 | -0.679 | +1.93 | 1.23 | includes initial flow |
| hill-8500-back-1000x1000 | 12 | 60 | Hancock | -0.000624 | +0.000946 | +0.00529 | -0.00262 | 0.00374 | includes initial flow |
| plain-709-moderate-1000x1000 | 40 | 60 | FV1 | -42.8 | -39.2 | +104 | +11 | 7.76 | whole storm |
| plain-709-moderate-1000x1000 | 40 | 60 | Hancock | -0.0341 | -0.0136 | +0.0994 | +0.0242 | 0.0103 | whole storm |
| plain-709-spatial-1000x1000 | 20 | 300 | FV1 | -56.2 | -47.9 | +19.2 | +16.2 | 8.53 | whole storm |
| plain-709-spatial-1000x1000 | 20 | 300 | Hancock | -0.143 | -0.177 | +0.175 | -0.028 | 0.0752 | whole storm |

[Paired run IDs, internal and outlet hydrograph metrics, temporal screens and reference checks](fidelity/native_results/hydraulic_fidelity.csv)

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
| hill-8500-back-1000x1000 | 8.0 | 5 | FV1 | 1.0 | 0.5318 | 15.459 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 5 | FV1 | 1.0 | 0.30703 | 15.459 | empty union | empty union |
| hill-8500-back-1000x1000 | 8.0 | 1 | FV1 | 1.0 | 0.10716 | 6.2974 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 1 | FV1 | 1.0 | 0.0092575 | 6.2974 | empty union | empty union |
| hill-8500-back-1000x1000 | 8.0 | 10 | FV1 | 1.0 | 1.5003 | 19.74 | 0.00000 | empty union |
| hill-8500-back-1000x1000 | 12.0 | 10 | FV1 | 1.0 | 1.0526 | 19.74 | 0.00000 | empty union |
| hill-8500-back-1000x1000 | 8.0 | 5 | Hancock | 1.0 | 0 | 0.14707 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 5 | Hancock | 1.0 | 0 | 0.14707 | empty union | empty union |
| hill-8500-back-1000x1000 | 8.0 | 1 | Hancock | 1.0 | 0.010741 | 0.018788 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 1 | Hancock | 1.0 | 0 | 0.018788 | empty union | empty union |
| hill-8500-back-1000x1000 | 8.0 | 10 | Hancock | 1.0 | 0 | 0.27301 | empty union | empty union |
| hill-8500-back-1000x1000 | 12.0 | 10 | Hancock | 1.0 | 0 | 0.27301 | empty union | empty union |
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
| plain-709-spatial-1000x1000 | 8.0 | 5 | FV1 | 1.0 | 2.1735 | 2.1572 | 0.97959 | 0.85000 |
| plain-709-spatial-1000x1000 | 20.0 | 5 | FV1 | 1.0 | 0.50372 | 2.097 | 0.81250 | 0.53846 |
| plain-709-spatial-1000x1000 | 8.0 | 1 | FV1 | 1.0 | 0.5193 | 0.54444 | 0.98380 | 0.99025 |
| plain-709-spatial-1000x1000 | 20.0 | 1 | FV1 | 1.0 | 0.40509 | 1.0658 | 0.87808 | 0.85119 |
| plain-709-spatial-1000x1000 | 8.0 | 10 | FV1 | 1.0 | 3.4477 | 3.6844 | 0.92857 | 0.50000 |
| plain-709-spatial-1000x1000 | 20.0 | 10 | FV1 | 1.0 | 0.56011 | 3.3474 | 1.00000 | empty union |
| plain-709-spatial-1000x1000 | 8.0 | 5 | Hancock | 1.0 | 0.024094 | 0.027654 | 1.00000 | 1.00000 |
| plain-709-spatial-1000x1000 | 20.0 | 5 | Hancock | 1.0 | 0.00074614 | 0.016399 | 1.00000 | 1.00000 |
| plain-709-spatial-1000x1000 | 8.0 | 10 | Hancock | 1.0 | 0.041748 | 0.056641 | 1.00000 | 1.00000 |
| plain-709-spatial-1000x1000 | 20.0 | 10 | Hancock | 1.0 | 0.0035856 | 0.050241 | 1.00000 | empty union |
| plain-709-spatial-1000x1000 | 8.0 | 1 | Hancock | 1.0 | 0.010704 | 0.010765 | 1.00000 | 0.99803 |
| plain-709-spatial-1000x1000 | 20.0 | 1 | Hancock | 1.0 | 0.00011577 | 0.0071151 | 1.00000 | 1.00000 |

[Wet-union errors, connectivity and map identities](fidelity/native_results/native_grid_field_differences.csv)

The following figures pair outlet peak discharge with the area that reached 5 cm or 10 cm at any time before the common recession endpoint. The inundation measurement counts original hydraulic cells. The outlet and the inundation footprint can respond quite differently to the choice of formulation. [hill-8500-back-1000x1000, 12 h](supplementary/native_grid_extent/hill-8500-back-1000x1000_12h.pdf) · [plain-709-moderate-1000x1000, 40 h](supplementary/native_grid_extent/plain-709-moderate-1000x1000_40h.pdf) · [plain-709-spatial-1000x1000, 20 h](supplementary/native_grid_extent/plain-709-spatial-1000x1000_20h.pdf)

The maps show the same plain two hours after rainfall ended for the moderate and intense storms. Shading preserves the fraction of hydraulic cells reaching 10 cm in each 10 m reporting cell. The dashed line marks the internal gauge. [1 m hydraulic grid](supplementary/operational_storms/plain_709_extent_10cm_1m.pdf) · [5 m hydraulic grid](supplementary/operational_storms/plain_709_extent_10cm_5m.pdf) · [10 m hydraulic grid](supplementary/operational_storms/plain_709_extent_10cm_10m.pdf)

## Practical grid selection on native terrain

The validation design includes 2 independent terrains with an admissible 5 and 10 m pair and all scheduled depth reference controls (hill-8200, rolling-8105). Completed runs currently qualify 0 terrain pairs for depth. [Design and measured eligibility by observable](fidelity/advisor_design/paired.json)

The test design includes 2 independent terrains with an admissible 5 and 10 m pair and all scheduled depth reference controls (fan-8400, rolling-8300). Completed runs currently qualify 0 terrain pairs for depth. [Design and measured eligibility by observable](fidelity/advisor_design/paired.json)

The paired depth design tests forecasts of 40,000 cells at 5 m and 10,000 cells at 10 m against a 1,000,000-cell 1 m numerical reference on the same footprint. Independent validation of larger-domain grid choices requires admissible comparisons at those scales. [Candidate and reference grids, CFL controls and measured endpoints](fidelity/advisor_design/scheduled.csv)

On floodplain-8103 at 5 m, the installed DEM reader measured maximum local barrier lowering of 0.183 m. Its solver decision was `refine_terrain`. [Installed CLI and archived features](fidelity/advisor_design/installed_cli.csv)

On floodplain-8103 at 10 m, the installed DEM reader measured maximum local barrier lowering of 0.169 m. Its solver decision was `refine_terrain`. [Installed CLI and archived features](fidelity/advisor_design/installed_cli.csv)

Joint method and grid selection on the new final terrains awaits the held-out evaluation.

With quarter-second accepted transport at 200 m, 5 m Hancock took 0.149 s and 1 m FV1 took 1.348 s (9.03×). Their respective sharp-pulse peak biases were -54.09% and -58.60%. Refining Hancock to 1 m reduced its peak bias to -7.94%. [Quarter-second analytical comparisons](fidelity/results/sharp_pulses.csv)
