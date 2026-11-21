# Results

## Exact and manufactured controls

The 342 primary forecasts had a minimum depth of 0 m. The largest relative mass residual was 1.61e-12. The independent lake-at-rest control had a maximum depth error of 8.88e-16 m. The refinement table includes spatial, fixed-grid temporal and source-integration sequences. [Analytical errors](tables/04_analytical_accuracy.md) · [Convergence](tables/05_convergence.md)

## Storm hydrographs and depth

The 48 prospective physical events included six hours of rain and a common recession for each matched comparison. The table below gives medians across the events that passed the refinement test for each observable. LI uses its native momentum equation. All methods and the error associated with reference refinement appear in the supporting tables. [Six-hour storm comparisons](tables/06_storm_accuracy.md) · [Reference uncertainty](processed/metrics/reference_uncertainty.csv)

| Observable | Qualified events | FV1 | Tiled Hancock | FV2 RK2 |
|---|---:|---:|---:|---:|
| depth (m) | 25 | 0.001977 | 0.001996 | 0.002056 |
| discharge (m³/s) | 36 | 0.002054 | 0.001343 | 0.001333 |

## Efficiency and terrain representation

On the 48 prospective 120 × 80 m events at 10 m, dense Hancock was faster in 48 cases. Its median complete-event runtime was 1.352× FV1, and 30 cases met the 1.5× target. Tiled Hancock had a median of 2.220× FV1 and met the target in 0 cases. The runs produced identical Hancock fields. Each grid had 96 cells. The ongoing operational qualification reports the measured million-cell timings separately. [Both execution modes](processed/million/policies.csv) · [Corrective audit](CORRECTIVE_AUDIT.md)

Fixed-geometry and resampled-geometry experiments use the same physical extent. Their objective-specific rankings and measured cost ratios are available for the plain, fan and incised development realisations. An unresolved reference is an explicit missing comparison. [Resolution profiles](figures/06_resolution.pdf) · [Equal-cost results](tables/08_cost_accuracy.md) · [Least-cost feasible configurations](processed/pareto/least_cost.csv)

## Roughness transfer

The 9 nested training fits selected multipliers between 0.8 and 0.95. Across 105 reference-qualified transfer rows, 64 scored better than physical roughness and 41 scored worse. Multiple bounds can select the same roughness. These counts include repeated bounds and objectives on the same terrain. The group intervals preserve that dependence. [Bounded calibration](tables/09_calibration.md) · [Grouped transfer intervals](processed/comparisons/calibration_intervals.csv)

## Held-out solver decisions

The advisor made 62 selections in 120 held-out objective decisions, including 19 with unresolved reference evidence. Among 43 assessable selections, 95.35% satisfied the requested accuracy and 2 exceeded it. Agreement with the measured least-cost method was 41.86%. The median conditional runtime regret was 156%. The empirical interval covered the observed error in 81.4% of assessable predictions. [Held-out rule comparison](tables/10_advisor.md) · [Individual decisions](processed/advisor/heldout.csv)

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

On the archived 1 m weak dam, LI's depth RMSE was 1.844 mm against its own momentum equations and 1.863 mm against full SWE. Each equation uses its own independent Riemann reference. [Equation-specific errors](processed/metrics/archival_reduced_equations.csv)

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

The fine-grid cross-formulation file reports each available Hancock–FV2 difference on identical 1 m geometry. The per-family file gives the operational timing and timestep ratios for all eight landforms. [Fine-grid comparisons](processed/metrics/independent_fine_formulation.csv) · [Landform results](processed/comparisons/landform_results.csv)
