# Recession in the 48 small six-hour rainfall events

All 48 events completed at 10 m on the same 120 × 80 m physical domain, with 5 m and 2.5 m refinements. 31 needed additional recession after hour 8. Each group uses its slowest passing configuration to determine the common end time.

| Common end (h) | Events |
|---:|---:|
| 8 | 17 |
| 12 | 21 |
| 16 | 2 |
| 20 | 2 |
| 24 | 2 |
| 40 | 1 |
| 48 | 1 |
| 68 | 1 |
| 76 | 1 |

Each entry below is an outlet observation with independent spatial, temporal and cross-formulation checks. Positive centroid bias indicates later passage. Negative peak bias indicates attenuation. For the signed quantities, the final two columns count comparisons whose bias exceeds the reference-sensitivity envelope. Width is the central 80% discharge duration.

| 10 m method | Observable | Qualified / 48 | Median signed (%) | Median absolute (%) | Below envelope | Above envelope |
|---|---|---:|---:|---:|---:|---:|
| fv1 | peak_bias_pct | 24 | +0.08 | 2.06 | 10 | 11 |
| fv1 | volume_bias_pct | 10 | -1.02 | 1.02 | 9 | 0 |
| fv1 | centroid_delay_pct | 13 | +1.92 | 2.62 | 5 | 7 |
| fv1 | width80_bias_pct | 13 | +1.26 | 6.79 | 6 | 7 |
| fv1 | shape_tv_pct | 12 | +2.75 | 2.75 | 0 | 12 |
| hancock_dense | peak_bias_pct | 24 | -0.35 | 2.63 | 11 | 7 |
| hancock_dense | volume_bias_pct | 10 | -1.10 | 1.10 | 9 | 0 |
| hancock_dense | centroid_delay_pct | 13 | +1.13 | 2.12 | 4 | 8 |
| hancock_dense | width80_bias_pct | 13 | +1.06 | 5.32 | 6 | 7 |
| hancock_dense | shape_tv_pct | 12 | +2.51 | 2.51 | 0 | 12 |
| adaptive | peak_bias_pct | 24 | +0.00 | 2.00 | 10 | 5 |
| adaptive | volume_bias_pct | 10 | -1.01 | 1.01 | 9 | 0 |
| adaptive | centroid_delay_pct | 13 | +1.09 | 2.04 | 4 | 8 |
| adaptive | width80_bias_pct | 13 | +0.82 | 5.29 | 6 | 7 |
| adaptive | shape_tv_pct | 12 | +2.56 | 2.56 | 0 | 12 |

These grids contain 96 cells at the candidate resolution. The table supports event-scale numerical comparisons on this geometry. The million-cell operational cases have separate measurements.

[Paired outlet results and reference IDs](outlet_metrics.csv) · [Matched horizons](recession_horizons.csv) · [Final two-hour transport and storage](recession_terminal.csv)
