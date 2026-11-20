# Can Manning roughness compensate for FV1 diffusion?

**Retain prescribed physical roughness.** Bounded calibration sometimes improved an individual FV1 result (outcome B). On moving fronts and transported pulses it generally left substantially more analytical error than reconstruction (outcome C). The experiments provide no support for a transferable automatic correction (outcome A).

## Design and independent validation

Seventeen positive-friction analytical conditions at 1, 5 and 10 m gave **459 bounded fits** across three objectives and three bounds. Two frictionless pulse controls retained analytical n = 0. The pre-existing inviscid dam breaks, radial basin, shear, jumps and manufactured controls also retain their prescribed physical values. Distinct full-SWE and reduced-equation targets for LI are documented separately. Across the 19 primary domains the comparison matrix contains 1,368 rows. All FV1 calibration trials use the original contact-HLL update with its ordinary hydraulic parameters. The bed, initial state, source rates, true Manning coefficient defining the analytical solution, and analytical discharge and depth are frozen.

For the flow objective the fit uses interval hydrograph RMSE, peak error, peak-time error and net transported volume. Discharge errors share a scale equal to the larger of 0.0001 m²/s or 5% of the exact peak magnitude. Time error uses one sampling interval. For prescribed steady discharge the peak time is undefined and contributes zero. The spatial objective uses depth RMSE and maximum-depth RMSE at a 5 mm scale, 1 mm wet-threshold IoU loss at a 0.05 scale, extent error in grid cells and arrival error in sampling intervals. Each component remains available in physical units. The joint fit minimises the larger of these two objective values. The aggregate objective is used only to select n. Model assessments use separate physical quantities. Piecewise-constant threshold and timing objectives can have multiple near-optima.

Each bounded search samples a bracket, refines sampled minima continuously, retains endpoints and tests the physical value. Among the evaluated candidates it selects the smallest adjustment attaining 99% of the available objective improvement. A second column of evidence gives the least sampled adjustment meeting the objective tolerance, including zero when the original value suffices. Fits on the 0.05% wet front, 20 mm/h event, 1% routing sheet and damped shear pulse were frozen before transfer. The **1,944 transfer records** vary event or grid spacing with the fitted value fixed in runs sharing the same physical n. Transfers to materials with a different true n preserve the multiplier on the new material. Event-specific refits on held-out conditions were performed afterwards and never used to replace the transfer results. [Search and physical objectives](../fv1_diffusion_calibrate.py) · [Minimum change meeting tolerance](calibration/minimum_change.csv) · [Independent validation in physical units](calibration/heldout.csv)

## Bounds and trade-offs

Joint fit on the gentle frictional wet front at 5 m, true n = 0.040. Depth comparison uses the boundary-independent interior and discharges use three directly integrated accepted-face sections.

| Method | Manning n | Change | Depth RMSE mm | 1 cm IoU | Hydrograph RMSE m²/s |
|---|---:|---:|---:|---:|---:|
| FV1 physical | 0.040000 | 0% | 20.06 | 0.815 | 0.02917 |
| FV1 ±5% | 0.041955 | +4.89% | 18.59 | 0.824 | 0.02514 |
| FV1 ±10% | 0.043910 | +9.77% | 17.65 | 0.833 | 0.02266 |
| FV1 ±20% | 0.044944 | +12.36% | 17.37 | 0.843 | 0.02200 |
| Hancock MC physical | 0.040000 | 0% | 3.05 | 0.974 | 0.00510 |
| FV2 MC physical | 0.040000 | 0% | 3.12 | 0.974 | 0.00515 |

The 5 m joint-fit forecast took 13.49 ms versus 13.47 ms at physical n, 19.44 ms for Hancock MC and 26.19 ms for FV2 MC. The faster calibrated run still had **5.6 times the FV2 depth error and 4.3 times its discharge error**. For the separate flow-only and spatial-only fits at this grid the ±20% search selected +9.05% and +12.36%, respectively. The corresponding result for the 5% grade, supercritical front had the opposite sign. Its individual joint fit selected −19.83%, reducing FV1 discharge RMSE from 0.16324 to 0.12902 m²/s. Physical-n Hancock and FV2 gave 0.01165 and 0.01155 m²/s.

**A lower n can raise a damped peak without correcting its shape.** In the 5 m damped training event the exact centre-section peak was 0.11031 m²/s. Lowering n from 0.02500 to 0.02048 raised FV1's peak from 0.03293 to 0.04049 m²/s. The timing error grew from 2.5 s early to 5 s early and the absolute cumulative transport error grew from 1.087 to 1.148 m². Three-section hydrograph RMSE improved only from 0.03696 to 0.03605 m²/s. Physical-n Hancock gave 0.01339 and FV2 gave 0.01371 m²/s. In the independent frictionless Gaussian pulse, roughness calibration is physically inadmissible. That control isolates the reconstruction benefit even when the drag parameter is absent. [Sensitivity curves](calibration/roughness_sensitivity.png) · [Individual, joint and physical-value comparisons](calibration/comparison.csv)

## Transfer across events and grids

Selected joint changes within the ±20% band are percentages of each prescribed physical coefficient.

| Training event | 10 m | 5 m | 1 m |
|---|---:|---:|---:|
| Gentle moving front | +16.91 | +12.36 | +6.70 |
| 20 mm/h rainfall | +19.72 | +10.95 | +1.71 |
| Moderate routing | +3.09 | +0.79 | +0.06 |
| Damped shear | −0.73 | −18.09 | −8.09 |

Two cases support a larger compensation on coarser grids. The damped pulse shows that relationship can fail. Manning n is physically prescribed in these tests, so the fitted deviation measures discretisation compensation. A value inside a plausible roughness range can still be inappropriate at another grid or discharge.

| 5 m held-out event | Training event | Multiplier | Physical FV1 Q RMSE | Transferred FV1 Q RMSE (m²/s) |
|---|---|---:|---:|---:|
| Gentler, rougher front | Gentle front | 1.123607 | 0.02875 | 0.02080 |
| Steep front | Gentle front | 1.123607 | 0.16324 | 0.17496 |
| 50 mm/h rain | 20 mm/h rain | 1.109521 | 0.0000380 | 0.0000923 |
| 100 mm/h rain | 20 mm/h rain | 1.109521 | 0.0000898 | 0.002191 |
| Deeper, faster damped pulse | Damped pulse | 0.819098 | 0.12031 | 0.12115 |

The 20 and 50 mm/h cases both have true n = 0.04. The same absolute roughness, 0.04438, was transferred between them. Uniform rainfall depth is fixed by continuity, so depth agreement on that test cannot independently identify n. On the steep-front transfer, depth RMSE increased from 34.82 to 37.95 mm and 1 cm wet IoU fell from 0.906 to 0.886. Among 192 matched-grid, held-out ±20% joint applications, discharge RMSE improved in 80 and worsened in 112. Many trials share a domain or fitting event. These counts describe this design and carry no population probability interpretation.

Boundary distance affects the depth comparison in the deeper damped event. Expanding the domain from 480 × 320 m to 560 × 400 and then 640 × 480 m while keeping the original 160 m observation window changed the 5 m physical-n depth RMSE from 4.08 to 3.13 and 3.12 mm. The transferred coefficient gave 5.03, 2.85 and 2.83 mm. Its apparent depth ranking therefore reverses with domain size. The discharge conclusion survived the check. The largest domain gave 0.12018 m²/s at physical n and 0.12130 m²/s with the transferred n. Small depth differences on the original damped-event domain are inconclusive. [Expanded-domain transfer](damped_transfer_support.json) · [Wall sensitivity by method and grid](wall_sensitivity.json)

## Two roughness zones and forecast cost

A second experiment uses two physically defined parallel benches separated by an impermeable levee, true n = 0.040 on the floodplain and 0.025 in the channel. The wider domain retains a wall-independent interior at the faster channel speed. Stage and face transport were checked by solving the two benches together on CUDA. The assembled final states matched their independent strip solutions exactly at all tested spacings and adjustments. At 5 m the uniform multiplier fit was 1.12943. Separate floodplain and channel fits gave 1.12361 and 1.12693. On the lower-speed held-out event, the worst-zone joint score fell from 3.532 at physical n to 2.573 with one parameter. Two parameters gave 2.603. Additional degrees of freedom brought no material benefit here. Both zones remain hydraulically isolated in this control. Connected overbank exchange remains untested. [Two-zone results](calibration/zones.json)

Searching and evaluating 3,449 distinct FV1 forecasts took 176.0 s in total, including evaluation overhead. The four designated training events alone used 40.2 s. These costs are separate from the forecast runtimes. The deployment study applies the fitted roughness unchanged under common caps and individual CFL policies. For the gentler held-out front under the normal policy, the trained 5 m FV1 gave 0.01960 m²/s RMSE in 7.55 ms. With the same budget, physical-n Hancock at 10 m gave 0.01188 m²/s in 4.96 ms. On the steep-front transfer the FV1 budget was too short for any tested Hancock grid. A missing affordable configuration cannot establish a superior result at equal runtime. [Deployment hydrographs](calibration/deployment.json) · [Measured equal-budget choices](calibration/deployment_budgets.json)

Using separate illustrative thresholds of depth RMSE ≤5 mm, 1 cm IoU ≥0.95 and section RMSE ≤0.001 m²/s, physical-n FV1 met all three on **20 of 51** positive-friction case/grid pairs. Even allowing independent event-specific joint calibration up to ±20% raised that count to **21**. Physical-n Hancock MC and FV2 MC each met all three on **32**, and LI on **22**. The absolute thresholds already admit routine FV1 predictions in some slow sheet flows. Adjusting n there provides little operational value. In front-sensitive and advective cases the main error includes spatial redistribution and hydrograph broadening that a scalar friction parameter cannot reliably correct. No automatic roughness or infiltration adjustment is warranted.
