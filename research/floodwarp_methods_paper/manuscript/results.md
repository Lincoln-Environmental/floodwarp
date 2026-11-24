# Results

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

Through the matched 16-hour window, the highest recorded FV2 RK2 outlet interval means were 66.40 m³/s at 10 h 51 min on the 10 m grid and 48.33 m³/s at 4 h 48 min on the 5 m grid. Each interval spans 60 seconds. Disagreement in these two measured maxima establishes grid sensitivity. The physical storm peak remains unresolved.

Runoff travel time and catchment storage allow the outlet maximum to follow rainfall. The incised diagnostic compares accepted flow at successive cross-sections with storage and infiltration over the intervening reach. The matched finer-grid checks test how strongly the outlet timing and magnitude depend on representation. [Recession water balance](../fidelity/native_tail/README.md)

The replay measured the source of discharge during the highest post-hour-eight minute at each spacing. Net drawdown in the last 40 m supplied 99.76% on the 10 m grid. That water may have entered the reach earlier. [Crest-minute budgets](../fidelity/native_tail/CREST.md)

On the 5 m grid, FV1 exported 100,848 m³ more water than Hancock by hour 16. Lower cumulative infiltration accounted for 100,268 m³ of the difference and lower final storage for 579 m³. The net infiltration difference arose on cells outside the rainfall footprint. These terms close the water balance under the same prescribed infiltration capacity. [Runoff and infiltration by grid](../fidelity/native_outlet/runoff_partition.csv)
