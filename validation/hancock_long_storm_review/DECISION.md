# Engineering decision for adaptive Hancock

All wall times below use warmed, synchronised FP64 execution on the RTX 4070 Ti SUPER. The original development head was `0180327`. The tiled scheduler is available at `2914077` and the constant-capacity source operator at `e63c936`. The original timings and physical arrays remain available for comparison.

## Fixed-timestep cost

| Case | Baseline FV1 ms | Baseline dense ms | Integrated FV1 ms | Integrated dense ms | Integrated tiles ms | Tiles/FV1 |
|---|---|---|---|---|---|---|
| stoker | 153.38 | 147.02 | 154.48 | 147.32 | 153.48 | 0.99× |
| dry_patch | 31.78 | 90.70 | 32.05 | 90.68 | 32.73 | 1.02× |

## Production timestep cost

Primary events cover eight landforms and six rainfall patterns on a common 384-cell grid. Each ran through six hours of rainfall and two hours of recession. The moderate plain ran for 24 hours.

The median tiled Hancock/FV1 runtime ratio was 1.649×. 2 of 48 pairs met 1.5×. The largest ratio was 2.583× for mixed terrain under spatial rainfall. It required 12,455 accepted steps versus 9,477 for FV1.

| Landform | Median tiles/FV1 wall | Median tiles/FV1 steps | Largest sampled cell Froude |
|---|---|---|---|
| coastal | 1.521 | 1.065 | 0.15 |
| fan | 1.610 | 1.074 | 0.68 |
| floodplain | 1.620 | 1.138 | 0.31 |
| hill | 1.808 | 1.177 | 1.02 |
| incised | 1.820 | 1.178 | 1.46 |
| mixed | 2.160 | 1.350 | 1.06 |
| plain | 1.635 | 1.049 | 0.10 |
| rolling | 1.613 | 1.095 | 0.56 |

| Landform | Rain | Cells | Wet % | Scheduled % | FV1 s | Dense s | Tiles s | Tiles/FV1 |
|---|---|---|---|---|---|---|---|---|
| coastal | moderate | 250,000 | 99.5 | 100.0 | 10.29 | 16.37 | 17.53 | 1.703 |
| mixed | spatial | 250,000 | 13.6 | 27.7 | 6.27 | 19.80 | 11.52 | 1.836 |
| plain | moderate | 250,000 | 99.9 | 100.0 | 9.17 | 13.82 | 14.83 | 1.616 |
| plain | moderate | 500,000 | 100.0 | 100.0 | 17.17 | 27.67 | 29.40 | 1.712 |
| plain | moderate | 1,000,000 | 100.0 | 100.0 | 34.99 | 55.68 | 58.38 | 1.668 |
| plain | moderate | 2,000,000 | 100.0 | 100.0 | 69.73 | 110.62 | 115.09 | 1.651 |
| plain | moving | 250,000 | 0.8 | 2.8 | 0.65 | 1.97 | 0.78 | 1.207 |
| plain | spatial | 250,000 | 16.2 | 26.4 | 2.38 | 5.61 | 3.15 | 1.326 |
| plain | spatial | 500,000 | 15.0 | 21.0 | 4.20 | 11.79 | 4.94 | 1.177 |
| plain | spatial | 1,000,000 | 13.9 | 19.1 | 8.23 | 25.63 | 9.25 | 1.124 |
| plain | spatial | 2,000,000 | 13.5 | 16.8 | 15.94 | 53.05 | 16.96 | 1.064 |

## Hydraulic accuracy

The source specialisation matched the stored hydraulic outputs in 265 replays. The dense and sparse schedules agreed in all 96 small primary and 20 large pairs. All six independent accepted-timestep histories matched across dense, fixed-mask and tiles. Of the 48 finely resampled terrain/event references, 35 met both the depth and hydrograph refinement screens. The landform and 1, 5 and 10 m tables report errors and identify unresolved comparisons. [Accuracy and resolution tables](COMPARISONS.md)

## Recommendation

Keep adaptive Hancock as the development default. Select tiled execution explicitly for spatially restricted events on larger domains. Widespread rainfall, fine-grid outlet storage and shallow moving fronts can each move the full-event runtime above the target. FV1 remains appropriate where its errors meet the application tolerance. LI requires independent qualification of its reduced momentum and outlet boundary for the intended observable. The CPU and CUDA structure tests qualify quasi-steady links, with a first-order time split and a separate structure-volume ledger.

A single-GPU strip crossover and one expanding footprint support an experimental switching policy. Additional GPU architectures, independent downstream controls and broader accuracy qualification are needed before changing public release or scheduling defaults. [Architecture, hydraulic mechanisms and limitations](REPORT.md) · [Complete operational metrics](event_summary.csv) · [Physical input and test provenance](provenance.json)
