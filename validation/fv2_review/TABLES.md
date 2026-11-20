# Measured FV2 verification results

## Smooth manufactured solution

The spatial study compares cell averages with the exact field. The temporal study fixes the grid and refines time against the same spatial operator with a further 16× smaller step.

| Study | Cells/axis | dt (s) | h RMSE (m) | hu RMSE (m²/s) | hv RMSE (m²/s) | p(h) | p(hu) | p(hv) |
|---|---|---|---|---|---|---|---|---|
| space | 16 | 0.0078125 | 0.00053068 | 0.002802 | 0.0029215 | — | — | — |
| space | 32 | 0.0039062 | 0.00013741 | 0.00047514 | 0.00053181 | 1.9494 | 2.56 | 2.4577 |
| space | 64 | 0.0019531 | 2.6762e-05 | 9.8183e-05 | 9.9696e-05 | 2.3602 | 2.2748 | 2.4153 |
| space | 128 | 0.00097656 | 5.5552e-06 | 2.0572e-05 | 1.9879e-05 | 2.2683 | 2.2548 | 2.3263 |
| time | 16 | 0.015625 | 7.0123e-06 | 1.4741e-05 | 1.3978e-05 | — | — | — |
| time | 16 | 0.0078125 | 1.7708e-06 | 3.6893e-06 | 3.4668e-06 | 1.9855 | 1.9984 | 2.0115 |
| time | 16 | 0.0039062 | 4.4245e-07 | 9.1339e-07 | 8.6367e-07 | 2.0008 | 2.014 | 2.005 |
| time | 16 | 0.0019531 | 1.1105e-07 | 2.2939e-07 | 2.1517e-07 | 1.9944 | 1.9934 | 2.005 |

## Analytical controls at 5 m

RMSE units are metres for depth and m²/s for momentum. All methods use contact-HLL. The oblique, smooth-shear and steady-channel norms use interior windows before reflected boundary information arrives.

| Case | FV1 h | Minmod h | MC h | FV1 hu | FV1 hv | MC hu | MC hv | Max ledger residual (m³) |
|---|---|---|---|---|---|---|---|---|
| lake | 0 | 6.8571e-19 | 1.7347e-18 | 3.0086e-16 | 3.347e-16 | 6.5284e-16 | 7.7936e-16 | 0 |
| ritter | 0.0322 | 0.012548 | 0.0068095 | 0.12426 | 0 | 0.030784 | 0 | 3.638e-12 |
| stoker | 0.046632 | 0.020756 | 0.013928 | 0.1759 | 0 | 0.058756 | 0 | 1.0914e-11 |
| thacker | 0.023431 | 0.0054856 | 0.0047616 | 0.090093 | 0 | 0.010849 | 0 | 3.638e-12 |
| shear | 9.8131e-18 | 9.8131e-18 | 9.8131e-18 | 0 | 0.15601 | 0 | 0.072374 | 2.5466e-11 |
| jump | 0.060682 | 0.04095 | 0.040966 | 0.094662 | 0 | 0.073509 | 0 | 0 |
| jump_offset | 0.042007 | 0.024613 | 0.023394 | 0.089406 | 0 | 0.07638 | 0 | 7.276e-12 |
| manufactured | 0.00015739 | 1.0376e-06 | 2.9392e-07 | 0.00019545 | 0 | 4.7096e-07 | 0 | 2.1828e-11 |
| uniform | 2.9242e-16 | 0 | 6.7911e-16 | 0.00036597 | 0 | 5.9529e-08 | 0 | 3.638e-12 |
| gradual | 0.00010164 | 1.0897e-06 | 4.7409e-08 | 0.00022364 | 0 | 1.7014e-06 | 0 | 3.638e-12 |
| oblique | 0.031767 | 0.015464 | 0.010995 | 0.058213 | 0.026456 | 0.045791 | 0.029365 | 1.1642e-10 |
| smooth_shear | 0 | 0 | 0 | 0 | 0.026541 | 0 | 0.00071985 | 5.8208e-11 |

## Wave positions and smearing

Shorelines use a matched 1 cm depth threshold. The Ritter head uses a matched 1.9 m threshold. An absent width means the corresponding crossing was unavailable or inapplicable.

| Case | Method | Feature | Position error (m) | Displacement error, second half (m) | 10–90% width (m) | Exact cell-average width (m) |
|---|---|---|---|---|---|---|
| ritter | fv1 | rarefaction_head | -18.199 | -3.4463 | — | — |
| ritter | fv1 | front | -26.852 | -16.895 | — | — |
| ritter | mc | rarefaction_head | -2.7092 | 0.062786 | — | — |
| ritter | mc | front | -15.287 | -2.7111 | — | — |
| stoker | fv1 | rarefaction_head | -17.277 | -3.3862 | — | — |
| stoker | fv1 | shock | 0.78817 | -0.092361 | 26.868 | 5.4187 |
| stoker | mc | rarefaction_head | -2.8789 | 0.066934 | — | — |
| stoker | mc | shock | 0.20142 | 0.0092332 | 10.543 | 5.4187 |
| thacker | fv1 | left_front | 1.035 | 1.5641 | — | — |
| thacker | fv1 | right_front | 36.699 | 25.367 | — | — |
| thacker | mc | left_front | 0.36569 | -0.59732 | — | — |
| thacker | mc | right_front | 4.5758 | 0.028362 | — | — |
| shear | fv1 | shear | -0.81177 | -0.023532 | 39.455 | 7.9167 |
| shear | mc | shear | 0.45689 | 0.17081 | 16.768 | 7.9167 |
| jump | fv1 | shock | -0.9151 | -0.015449 | 11.282 | 4 |
| jump | mc | shock | 0.084512 | 0.00013414 | 9.8306 | 4 |
| jump_offset | fv1 | shock | -1.2892 | -0.01079 | 9.8409 | 7.3333 |
| jump_offset | mc | shock | -0.083128 | -0.0097122 | 8.5382 | 7.3333 |

## Coupled rain, exponential infiltration and Manning drag

Absolute error at 2 s in the uniform interior versus an independently integrated DOP853 ODE (relative tolerance 1e-13).

| dt (s) | h error (m) | hu error | hv error | Soil error (m) | p(h) | p(hu) | p(hv) | p(soil) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.00013595 | 4.4245e-05 | 2.5283e-05 | 0.00013595 | — | — | — | — |
| 0.125 | 3.1805e-05 | 1.0226e-05 | 5.8436e-06 | 3.1805e-05 | 2.0958 | 2.1133 | 2.1133 | 2.0958 |
| 0.0625 | 7.6833e-06 | 2.4554e-06 | 1.4031e-06 | 7.6833e-06 | 2.0495 | 2.0582 | 2.0582 | 2.0495 |
| 0.03125 | 1.8879e-06 | 6.0153e-07 | 3.4373e-07 | 1.8879e-06 | 2.025 | 2.0293 | 2.0293 | 2.025 |

## Truncation error on identical represented 5 m terrain

Every reference subcell inherits its 5 m parent's bed, forcing, roughness and initial conserved state. The gap compares the two reference grids using the listed method. IoU and export differences use the 1.25 m MC reference. The JSON also records resolution-dependent sampling of the continuous terrain and the 0.625 m timestep check.

| Case | Method | h vs 1.25 m MC (m) | h vs 1.25 m FV1 (m) | 2.5→1.25 m gap (m) | Wet IoU | Envelope RMSE (m) | Export difference (m³) |
|---|---|---|---|---|---|---|---|
| plane | fv1 | 0.0050416 | 0.0030835 | 0.0012846 | 1 | 0.0021519 | 40.586 |
| plane | mc | 0.0027835 | 0.0011366 | 0.0012951 | 1 | 0.0011302 | 6.5964 |
| irregular | fv1 | 0.0047484 | 0.0029438 | 0.0016211 | 0.97535 | 0.003583 | 1.3177 |
| irregular | mc | 0.0053706 | 0.0050327 | 0.002175 | 0.9892 | 0.0034294 | 0.55729 |
| steep | fv1 | 0.0086528 | 0.0061971 | 0.0032402 | 0.99022 | 0.0087061 | 37.227 |
| steep | mc | 0.0046439 | 0.0047223 | 0.0058062 | 0.98925 | 0.0044696 | 42.96 |
| expansion | fv1 | 0.0053023 | 0.0033979 | 0.0015583 | 1 | 0.0049874 | 2.2138e-09 |
| expansion | mc | 0.0015083 | 0.0014549 | 0.00088001 | 1 | 0.0017329 | -1.3114e-10 |
| rainfall | fv1 | 0.0027857 | 0.0018166 | 0.00079684 | 0.96787 | 0.0027857 | 0.0036257 |
| rainfall | mc | 0.0029673 | 0.0023265 | 0.00080034 | 0.96612 | 0.0029673 | 0.0015226 |
| barrier | fv1 | 0.0035769 | 0.0022241 | 0.0010685 | 0.99412 | 0.0023647 | 1.3037e-34 |
| barrier | mc | 0.001709 | 0.001312 | 0.00089233 | 1 | 0.001155 | 5.5537e-11 |
| oblique_channel | fv1 | 0.0035903 | 0.0033158 | 0.0015208 | 0.99494 | 0.002656 | 8.4788 |
| oblique_channel | mc | 0.0027406 | 0.0029948 | 0.0019787 | 0.99241 | 0.0027578 | 14.9 |

## Orton native terrain control

Tile forcing lasts 240 s. Fields are compared on identical 1 m support against the finest MC solution on the represented 1 m DEM. The independent finest FV1 alternative and timestep study quantify sensitivity.

| DEM (m) | Hydraulic grid (m) | Method | Bed RMSE (m) | h RMSE (mm) | Envelope RMSE (mm) | Wet IoU | Export difference (m³) | Wet component difference |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | fv1 | 0 | 1.0221 | 1.011 | 0.75132 | 0.052441 | -56 |
| 1 | 1 | mc | 0 | 0.79273 | 0.83395 | 0.61292 | 0.021229 | 328 |
| 2 | 2 | fv1 | 0.059944 | 0.92734 | 0.90511 | 0.15253 | 0.02751 | -743 |
| 2 | 2 | mc | 0.059944 | 0.77857 | 0.78454 | 0.15599 | 0.023936 | -601 |
| 5 | 5 | fv1 | 0.13877 | 0.56597 | 0.52671 | 0.071863 | 0.0027411 | -935 |
| 5 | 5 | mc | 0.13877 | 0.58547 | 0.56242 | 0.075401 | 0.0075375 | -909 |
| 10 | 10 | fv1 | 0.22301 | 0.39296 | 0.34588 | 0.046984 | 2.102e-05 | -975 |
| 10 | 10 | mc | 0.22301 | 0.46993 | 0.43413 | 0.048804 | 0.0028354 | -967 |
| 20 | 20 | fv1 | 0.30386 | 0.32554 | 0.28305 | 0.016972 | -0.0012068 | -996 |
| 20 | 20 | mc | 0.30386 | 0.38775 | 0.34127 | 0.036444 | -0.0018148 | -991 |
| 1 | 0.5 | fv1 | — | 0.83756 | 0.82439 | 0.74331 | 0.047026 | -30 |
| 1 | 0.5 | mc | — | 0.21825 | 0.2172 | 0.57754 | 0.0088649 | 331 |
| 1 | 0.25 | fv1 | — | 0.5788 | 0.56651 | 0.73186 | 0.027099 | 31 |
| 1 | 0.25 | mc | — | 0.12014 | 0.12304 | 0.80118 | 0.0029701 | 99 |
| 1 | 0.125 | fv1 | — | 0.33718 | 0.32753 | 0.75348 | 0.014152 | 72 |
| 1 | 0.125 | mc | — | 0 | 0 | 1 | 0 | 0 |

## Purau native terrain control

Tile forcing lasts 600 s. Fields are compared on identical 1 m support against the finest MC solution on the represented 1 m DEM. The independent finest FV1 alternative and timestep study quantify sensitivity.

| DEM (m) | Hydraulic grid (m) | Method | Bed RMSE (m) | h RMSE (mm) | Envelope RMSE (mm) | Wet IoU | Export difference (m³) | Wet component difference |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | fv1 | 0 | 5.9646 | 5.9649 | 0.44012 | 4.5837 | 507 |
| 1 | 1 | mc | 0 | 3.6659 | 3.7318 | 0.46645 | 0.62757 | 1844 |
| 2 | 2 | fv1 | 0.10991 | 7.896 | 7.8951 | 0.48852 | 4.4462 | -69 |
| 2 | 2 | mc | 0.10991 | 6.3857 | 6.4394 | 0.42679 | 0.36461 | 236 |
| 5 | 5 | fv1 | 0.28194 | 6.6984 | 6.6975 | 0.58915 | 2.5876 | -212 |
| 5 | 5 | mc | 0.28194 | 6.7531 | 6.7825 | 0.40908 | -0.0066135 | -155 |
| 10 | 10 | fv1 | 0.52931 | 5.7973 | 5.7957 | 0.65778 | 0.4504 | -220 |
| 10 | 10 | mc | 0.52931 | 6.3774 | 6.4022 | 0.43354 | -0.068599 | -202 |
| 20 | 20 | fv1 | 0.95837 | 5.1049 | 5.1028 | 0.71173 | -0.2555 | -220 |
| 20 | 20 | mc | 0.95837 | 5.5698 | 5.562 | 0.52564 | -0.73652 | -218 |
| 1 | 0.5 | fv1 | — | 4.4765 | 4.477 | 0.53771 | 3.033 | 498 |
| 1 | 0.5 | mc | — | 1.1734 | 1.185 | 0.87775 | 0.40207 | 294 |
| 1 | 0.25 | fv1 | — | 3.1098 | 3.1105 | 0.66523 | 2.0706 | 453 |
| 1 | 0.25 | mc | — | 0.77396 | 0.80987 | 0.88763 | 0.38379 | 151 |
| 1 | 0.125 | fv1 | — | 1.9898 | 1.9907 | 0.78905 | 1.1538 | 352 |
| 1 | 0.125 | mc | — | 0 | 0 | 1 | 0 | 0 |

## Replicated GPU cost

RTX 4070 Ti SUPER, FP64, five synchronised warm repeats. Timing excludes input/output, figure creation and first graph capture. The JSON also contains all individual replicates and per-kernel CUDA timings.

| Case | dx | Method | Execution | Steps | Min dt (s) | Mean dt (s) | Median (ms) | M cell-steps/s | Device arrays (MB) | First graph capture (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| stoker | 20 | fv1 | dense | 36 | 0.39334 | 0.88889 | 2.57 | 7.1719 | 0.13667 | 72.434 |
| stoker | 20 | fv1 | sparse | 36 | 0.39334 | 0.88889 | 3.4988 | 5.2681 | 0.1368 | 0.83279 |
| stoker | 20 | mc | dense | 48 | 0.38121 | 0.66667 | 9.2366 | 2.6607 | 0.26604 | 1.1391 |
| stoker | 20 | mc | sparse | 48 | 0.38121 | 0.66667 | 9.2641 | 2.6528 | 0.26604 | 0.56346 |
| stoker | 10 | fv1 | dense | 71 | 0.39335 | 0.4507 | 5.1232 | 28.382 | 0.53891 | 0.34973 |
| stoker | 10 | fv1 | sparse | 71 | 0.39335 | 0.4507 | 6.5536 | 22.188 | 0.53942 | 0.43084 |
| stoker | 10 | mc | dense | 96 | 0.22494 | 0.33333 | 18.418 | 10.675 | 1.0515 | 0.55256 |
| stoker | 10 | mc | sparse | 96 | 0.22494 | 0.33333 | 18.495 | 10.63 | 1.0515 | 0.56052 |
| stoker | 5 | fv1 | dense | 142 | 0.16758 | 0.22535 | 10.328 | 112.63 | 2.1421 | 0.36541 |
| stoker | 5 | fv1 | sparse | 142 | 0.16758 | 0.22535 | 13.211 | 88.05 | 2.1442 | 0.37646 |
| stoker | 5 | mc | dense | 192 | 0.1328 | 0.16667 | 36.588 | 42.988 | 4.183 | 0.53122 |
| stoker | 5 | mc | sparse | 192 | 0.1328 | 0.16667 | 36.705 | 42.851 | 4.183 | 0.44514 |
| stoker | 2.5 | fv1 | dense | 284 | 0.054702 | 0.11268 | 35.313 | 263.54 | 8.5434 | 0.37133 |
| stoker | 2.5 | fv1 | sparse | 284 | 0.054702 | 0.11268 | 45.397 | 205 | 8.5516 | 0.48334 |
| stoker | 2.5 | mc | dense | 385 | 0.0065018 | 0.083117 | 132.97 | 94.878 | 16.688 | 0.60257 |
| stoker | 2.5 | mc | sparse | 385 | 0.0065018 | 0.083117 | 133.28 | 94.654 | 16.688 | 0.59774 |
| orton_crop | 20 | fv1 | dense | 120 | 2 | 2 | 5.4586 | 2.1984 | 0.027916 | 0.36795 |
| orton_crop | 20 | fv1 | sparse | 120 | 2 | 2 | 6.8769 | 1.745 | 0.027984 | 0.413 |
| orton_crop | 20 | mc | dense | 120 | 2 | 2 | 13.308 | 0.90171 | 0.053772 | 0.57832 |
| orton_crop | 20 | mc | sparse | 120 | 2 | 2 | 12.692 | 0.94546 | 0.054432 | 0.57842 |
| orton_crop | 10 | fv1 | dense | 120 | 2 | 2 | 8.5483 | 5.6151 | 0.10712 | 0.35268 |
| orton_crop | 10 | fv1 | sparse | 120 | 2 | 2 | 10.567 | 4.5425 | 0.10726 | 0.41208 |
| orton_crop | 10 | mc | dense | 138 | 0.56017 | 1.7391 | 36.777 | 1.5009 | 0.20833 | 0.55314 |
| orton_crop | 10 | mc | sparse | 138 | 0.56017 | 1.7391 | 33.825 | 1.6319 | 0.21077 | 0.55686 |
| orton_crop | 5 | fv1 | dense | 120 | 2 | 2 | 8.3102 | 23.104 | 0.42152 | 0.36075 |
| orton_crop | 5 | fv1 | sparse | 120 | 2 | 2 | 10.078 | 19.051 | 0.42192 | 0.44367 |
| orton_crop | 5 | mc | dense | 307 | 0.13777 | 0.78176 | 91.727 | 5.355 | 0.82225 | 0.55652 |
| orton_crop | 5 | mc | sparse | 307 | 0.13777 | 0.78176 | 95.434 | 5.147 | 0.8316 | 0.55225 |
| orton_crop | 2 | fv1 | dense | 176 | 0.18414 | 1.3636 | 11.736 | 149.97 | 2.6127 | 0.41448 |
| orton_crop | 2 | fv1 | sparse | 176 | 0.18414 | 1.3636 | 13.593 | 129.48 | 2.6154 | 0.44167 |
| orton_crop | 2 | mc | dense | 756 | 0.040274 | 0.31746 | 213.48 | 35.413 | 5.1024 | 0.57826 |
| orton_crop | 2 | mc | sparse | 756 | 0.040274 | 0.31746 | 221.3 | 34.162 | 5.1603 | 0.58556 |
| orton_crop | 1 | fv1 | dense | 413 | 0.0070589 | 0.58111 | 41.841 | 394.83 | 10.425 | 0.3745 |
| orton_crop | 1 | fv1 | sparse | 413 | 0.0070589 | 0.58111 | 47.747 | 345.99 | 10.435 | 0.45152 |
| orton_crop | 1 | mc | dense | 1319 | 0.012648 | 0.18196 | 564.46 | 93.47 | 20.364 | 0.58084 |
| orton_crop | 1 | mc | sparse | 1319 | 0.012648 | 0.18196 | 627.97 | 84.016 | 20.593 | 0.62537 |
