# FREE outlet qualification

The exact Manning reference is steady uniform flow on a 7.329% plane. Discharge below uses the accepted face volume between 30 and 60 s at the same section, 400 m from the upstream boundary.

| Method | Grid (m) | Cap (s) | Original face bias (%) | Repaired face bias (%) | Buffered section bias (%) |
|---|---:|---:|---:|---:|---:|
| fv2 | 1 | 0.1 | +0.1158 | +0.1158 | +0.1158 |
| fv2 | 5 | 0.1 | -6.9367 | +0.1158 | +0.1158 |
| fv2 | 10 | 0.1 | -30.7219 | +0.1158 | +0.1158 |
| fv2 | 10 | 0.05 | -30.5563 | +0.0290 | +0.0290 |
| hancock | 1 | 0.1 | +0.1158 | +0.1158 | +0.1158 |
| hancock | 5 | 0.1 | -6.6538 | +0.1158 | +0.1158 |
| hancock | 10 | 0.1 | -30.4284 | +0.1158 | +0.1158 |
| hancock | 10 | 0.05 | -30.4567 | +0.0290 | +0.0290 |
| fv1 | 1 | 0.1 | -36.8162 | -36.8162 | -36.8162 |
| fv1 | 5 | 0.1 | -54.5994 | -54.5994 | -44.8927 |
| fv1 | 10 | 0.1 | -72.4977 | -72.4977 | -46.8352 |
| fv1 | 10 | 0.05 | -72.4466 | -72.4466 | -46.8188 |

The terminal slope previously vanished when the reconstruction stencil reached the FREE face. This left a half-cell bed discontinuity at the last internal face and suppressed the surface-gradient momentum source. Hancock and RK2 now extrapolate the bed and surface increments from the wet interior at an explicitly open face, with zero normal velocity gradient and the existing dry and positivity checks. The residual uniform-sheet error decreases by a factor of four when the timestep cap halves, consistent with the time discretisation of the source balance. FV1 reconstructs from piecewise-constant cell beds. At this grade, the bed drops 0.07329 m per metre, exceeding the 0.03 m sheet depth even on the finest tested grid. Its hydrostatic cut to the higher cell bed clips the downstream state. At the 0.1 s cap the buffered FV1 deficit drops from 46.84% on 10 m cells to 36.82% on 1 m cells. The extra face deficit falls from 25.66 percentage points to below the displayed precision.

## Full incised storm

The gauge remains at the eastern edge of the original 104.8576 km² catchment. The extension continues the 1 m valley, initial channel state and soil. Its additional rainfall is zero. The source fields on the original footprint retain their historical values.

| Method | Grid (m) | Extension (m) | CFL factor | Pre-fix direct peak (m³/s) | Peak (m³/s) | Minute ending (h) | Fixed 15 min peak (m³/s) | Moving 15 min peak (m³/s) | Export at gauge (m³) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive | 10 | 0 | 1.0 | 66.399 | 47.660 | 4.800 | 47.495 | 47.590 | 857,174.3 |
| fv1 | 10 | 0 | 1.0 | 48.371 | 48.371 | 5.400 | 48.290 | 48.293 | 910,216.6 |
| hancock_adaptive | 10 | 0 | 1.0 | 79.651 | 47.184 | 4.783 | 47.021 | 47.120 | 869,085.7 |
| adaptive | 10 | 400 | 1.0 | — | 47.660 | 4.800 | 47.495 | 47.590 | 857,182.5 |
| fv1 | 10 | 400 | 1.0 | — | 48.319 | 5.400 | 48.236 | 48.241 | 910,412.8 |
| hancock_adaptive | 10 | 400 | 1.0 | — | 47.185 | 4.783 | 47.021 | 47.120 | 869,078.9 |
| hancock_adaptive | 5 | 0 | 1.0 | 48.286 | 48.019 | 4.800 | 47.849 | 47.934 | 829,026.1 |

7 of 14 trajectories have completed the historical 16-hour comparison. Each hydrograph is an interval average of accepted transport. On these irregular terrains a solver difference alone does not establish accuracy.

[Water accounts, infiltration inside and outside the rainfall footprint](events.csv) · [Crest and whole-recession last 40 m budgets](last_40m.csv) · [Spatial extent](extent.csv) · [Same-grid wet-union depth differences](depth_disagreement.csv)
