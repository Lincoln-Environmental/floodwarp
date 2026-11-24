# Discussion

Tiled execution preserves Hancock's conservative update. Differences between Hancock and RK2 depend on the physical observable, timestep history and limiter response. The median storm errors in the results provide evidence at the tested scales. Small discharge peaks and controls that shift on refinement can make percentage comparisons unstable. A solver that attenuates a front may still predict a satisfactory final storage.

The practical outcome depends on the accuracy objective. The Pareto archive identifies measured configurations dominated in both error and runtime within an individual event. A coarse DEM can remove drainage connections or lower a barrier. Comparing fixed represented geometry with continuous terrain resampling quantifies how this changes the error attributable to the hydraulic grid. Flow accumulation on a filled DEM describes potential drainage. Stage control, infiltration and depressional ponding can change the actual flow path.

Bounded Manning fits target integrated transport or depth and extent. A change that compensates for attenuation on one hydrograph can alter recession or peak timing on the next. The grouped transfer results quantify the effect of applying the fitted physical value unchanged. The first- and higher-order solvers at the prescribed roughness remain the comparison controls. [Transfer trials](../processed/comparisons/transfer.csv)

The domain-wide solver advisor abstained on 134 held-out objectives. Its runtime forecast had median absolute percentage error 44.63% for assessable choices. Compare coverage and failed choices together with cost. A low conditional regret can reflect abstention on difficult cases. The original small-domain training cases covered 9,600 m². The expanded splits include independent larger terrain and event realisations. Large-domain accuracy recommendations require a qualified reference for the requested observable, sufficient training support at the requested scale and admissible terrain representation. Additional independent domains, stage controls, storms, GPUs and surveyed hydraulic structures are needed before using forecasts for release decisions.

Retain adaptive Hancock as the development default. Use FV1 or LI when the error for the intended observable is acceptable at lower measured cost. LI also needs a suitable reduced-momentum regime. Publication or operational use of a method recommendation requires a qualified reference and credible predictive coverage on the intended domain. [Rule baselines](../tables/10_advisor.md)

Across held-out objectives with at least one empirically feasible option, the proportions receiving a qualified feasible selection were advisor 17.54%, fv1 89.47%, hancock_dense 91.23%, oracle 100%, slope 3.509%. This denominator includes abstentions. The stratified validation tables identify the resolutions and sizes with independent predictive evidence.

Across the 8 primary million-cell events, the faster measured explicit Hancock policy took a median of 1.69 times the FV1 runtime over the original observation windows. The accepted-step ratios ranged from 1.06 to 1.62, and the corresponding mean cost per accepted step ranged from 1.02 to 1.61 times FV1. These factors multiply within each event. The normal timestep policy therefore matters even where tile scheduling removes most of the dry-domain arithmetic. [Per-event costs](../tables/11_execution.md)

## Flood-wave interpretation

Of 48 small events with six hours of rain, 31 required a longer recession. The longest matched simulation reached 76 hours. For the 12 storms with independently qualified normalised outlet shape at 10 m, the Hancock–FV2 difference in absolute shape error exceeded the combined reference sensitivity in 0 cases. That comparison concerns the timing distribution of discharged volume. Peak and arrival-time conclusions have separate reference screens. Among 6 FV1 storms with independently assessable peaks and final depth RMSE within 1 mm, 3 had an absolute peak bias above 10%. [Complete small-storm comparisons](../fidelity/small_storms/README.md) · [Million-cell export after the original stop time](../fidelity/NATIVE_RECESSIONS.md)

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

At 10 m, FV1 minus Hancock net export was +41,131 m³. The corresponding cumulative infiltration differences were +17,863 m³ inside the rainfall footprint and -59,098 m³ outside. The final storage difference was +104 m³. At hour eight their areas above 5 cm were 0.7233 and 0.0000 km².

At 5 m, FV1 minus Hancock net export was +81,117 m³. The corresponding cumulative infiltration differences were +6,211 m³ inside the rainfall footprint and -87,383 m³ outside. The final storage difference was +56 m³. At hour eight their areas above 5 cm were 0.5847 and 0.0000 km².

Of the hour-eight area above 5 cm in FV1 alone on the direct 5 m grid, 100.00% lay downstream of the independent internal gauge and 100.00% lay outside the rainfall footprint. The final 40 m contained 2.36% of this disagreement. [Reach and rainfall-footprint decomposition](../fidelity/outlet_results/regional_contrasts.csv)

Halving the RK2 CFL on the 5 m grid changed the one-minute peak from 48.115 to 48.123 m³/s. Net export changed by -17,792 m³, infiltration by +17,792 m³ and final storage by +0 m³. The normal and half-CFL runs accepted 77,045 and 146,457 timesteps.

The direct-outlet observables passing all three reference checks were peak magnitude, net exported volume. These checks support comparisons on the coarser grid. Practical 5 m absolute accuracy remains unresolved. [Reference qualification by observable](../fidelity/outlet_results/reference_checks.csv)

Under intense rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at 4.331 and 6.806 m³/s. They exported 28,772 and 109,391 m³ through the common 32-hour endpoint. Under moderate rainfall on the 1 km² plain, 10 m FV1 and Hancock peaked at 0.894 and 0.915 m³/s. They exported 24,912 and 38,983 m³ through the common 40-hour endpoint. Both gauges passed the recession criterion. These CPU controls measure formulation disagreement. [Six-hour rainfall and complete recession accounts](../fidelity/strong_storms/completed.csv)

With quarter-second accepted transport at 200 m, 5 m Hancock took 0.149 s and 1 m FV1 took 1.348 s (9.03×). Their respective sharp-pulse peak biases were -54.09% and -58.60%. Refining Hancock to 1 m reduced its peak bias to -7.94%. [Quarter-second analytical comparisons](../fidelity/results/sharp_pulses.csv)
