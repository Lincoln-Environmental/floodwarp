# Numerical results

Times are seconds unless labelled otherwise. All hydraulic state arrays use FP64.

## Steady hydraulic timing, seven warmed runs

| Domain | Method | Original ms | Final ms | Final / FV1 | Steps | Retries | Mean dt | Minimum dt |
|---|---|---|---|---|---|---|---|---|
| stoker | contact_hll | 40.217 | 38.166 | 1 | 308 | 0 | 0.077922 | 0.015625 |
| stoker | mc | 116.68 | 58.318 | 1.528 | 308 | 0 | 0.077922 | 0.015625 |
| stoker | minmod | 116.68 | 59.362 | 1.5554 | 308 | 0 | 0.077922 | 0.015625 |
| stoker | adaptive | 116.68 | 60.208 | 1.5775 | 308 | 0 | 0.077922 | 0.015625 |
| wet_large | contact_hll | 34.917 | 35.003 | 1 | 103 | 0 | 0.07767 | 0.03125 |
| wet_large | mc | 110.26 | 45.374 | 1.2963 | 103 | 0 | 0.07767 | 0.03125 |
| wet_large | minmod | 110.26 | 45.352 | 1.2957 | 103 | 0 | 0.07767 | 0.03125 |
| wet_large | adaptive | 110.26 | 46.492 | 1.3282 | 103 | 0 | 0.07767 | 0.03125 |
| dry_patch | contact_hll | 14.803 | 14.861 | 1 | 103 | 0 | 0.07767 | 0.03125 |
| dry_patch | mc | 44.951 | 25.532 | 1.7181 | 103 | 0 | 0.07767 | 0.03125 |
| dry_patch | minmod | 44.951 | 25.414 | 1.7102 | 103 | 0 | 0.07767 | 0.03125 |
| dry_patch | adaptive | 44.951 | 26.406 | 1.7769 | 103 | 0 | 0.07767 | 0.03125 |

The original FV2 column uses MC as the common historical baseline for all FV2 rows.

## Incremental implementation timings for FV2 MC

| Revision | Stoker ms | Large wet ms | Dry patch ms |
|---|---|---|---|
| baseline | 116.68 | 110.26 | 44.951 |
| reuse | 109.18 | 102.71 | 43 |
| source | 96.052 | 87.718 | 35.309 |
| warp | 97.436 | 89.117 | 35.535 |
| traces | 72.926 | 60.718 | 33.044 |
| primitive | 69.087 | 55.601 | 29.127 |
| inv_cache | 58.89 | 45.913 | 25.971 |
| fused | 58.324 | 45.692 | 25.793 |
| marked | 58.318 | 45.374 | 25.532 |

## Orton 5 m kernel microseconds per accepted step, uncaptured

| Kernel | baseline fv2_mc | marked fv2_mc | marked fv2_adaptive | marked contact_hll |
|---|---|---|---|---|
| accept_faces | 277.86 | 277.93 | 277.78 | 0 |
| apply_sources | 0 | 0 | 0 | 98.316 |
| begin_step | 8.0864 | 8.16 | 8.1568 | 8.516 |
| cfl_reduce | 0 | 0 | 0 | 117.34 |
| donor_budget | 303.66 | 303.58 | 303.45 | 0 |
| draining | 0 | 0 | 0 | 151.33 |
| euler | 461.47 | 469.59 | 472.58 | 0 |
| faces | 2663.4 | 1401.6 | 1401.6 | 792.15 |
| finish | 5.8432 | 5.9616 | 5.9424 | 0 |
| finish_step | 0 | 0 | 0 | 5.784 |
| limit_faces | 195.78 | 195.64 | 196.46 | 140.63 |
| mark_compressive_stencils | 0 | 0 | 214.67 | 0 |
| predictor | 3.5616 | 3.5488 | 3.4688 | 0 |
| slopes | 998.5 | 636.84 | 635.27 | 0 |
| split_sources | 1150 | 983.03 | 983.28 | 0 |
| update | 0 | 0 | 0 | 366.85 |
| validate_trial | 3.9072 | 3.6864 | 3.8912 | 0 |

## Dynamic scheduling, identical final fields

| Case | dx | Mode | Median ms |
|---|---|---|---|
| dry_patch | 5 | dense | 33.797 |
| dry_patch | 5 | sparse | 26.464 |
| dry_patch | 5 | dynamic | 48.611 |
| wet_large | 5 | dense | 46.676 |
| wet_large | 5 | sparse | 47.105 |
| wet_large | 5 | dynamic | 49.08 |
| Orton | 20 | dense | 11.88 |
| Orton | 20 | sparse | 11.135 |
| Orton | 20 | dynamic | 13.912 |
| Orton | 5 | dense | 526.34 |
| Orton | 5 | sparse | 461.15 |
| Orton | 5 | dynamic | 514.08 |

## Analytical errors at 2.5 m

| Case | Method | Depth RMSE m | Wet depth RMSE m | hu RMSE m²/s | hv RMSE m²/s | Volume residual m³ |
|---|---|---|---|---|---|---|
| lake | fv1 | 0 | 0 | 4.9829e-16 | 4.8481e-16 | -7.276e-12 |
| lake | minmod | 5.682e-18 | 7.2916e-18 | 1.3508e-15 | 1.4301e-15 | -7.276e-12 |
| lake | adaptive | 1.177e-17 | 1.5104e-17 | 1.5272e-15 | 1.6328e-15 | -7.276e-12 |
| ritter | fv1 | 0.021884 | 0.023112 | 0.089349 | 0 | -3.638e-12 |
| ritter | minmod | 0.0070106 | 0.0073299 | 0.03662 | 0 | -3.638e-12 |
| ritter | adaptive | 0.0036577 | 0.0037067 | 0.017733 | 0 | 0 |
| stoker | fv1 | 0.030397 | 0.030397 | 0.11822 | 0 | 0 |
| stoker | minmod | 0.011402 | 0.011402 | 0.049809 | 0 | 0 |
| stoker | adaptive | 0.0068548 | 0.0068548 | 0.031772 | 0 | 7.276e-12 |
| thacker | fv1 | 0.0132 | 0.018204 | 0.050619 | 0 | 0 |
| thacker | minmod | 0.002138 | 0.0033474 | 0.0051519 | 0 | -1.819e-12 |
| thacker | adaptive | 0.0020617 | 0.003309 | 0.0053493 | 0 | 1.819e-12 |
| shear | fv1 | 0 | 0 | 0 | 0.14505 | 1.0914e-11 |
| shear | minmod | 0 | 0 | 0 | 0.092242 | 1.819e-11 |
| shear | adaptive | 0 | 0 | 0 | 0.073984 | 0 |
| jump | fv1 | 0.04286 | 0.04286 | 0.065869 | 0 | 3.638e-12 |
| jump | minmod | 0.028892 | 0.028892 | 0.052545 | 0 | -3.638e-12 |
| jump | adaptive | 0.028975 | 0.028975 | 0.054067 | 0 | 3.638e-12 |
| jump_offset | fv1 | 0.018155 | 0.018155 | 0.060481 | 0 | 7.276e-12 |
| jump_offset | minmod | 0.0075751 | 0.0075751 | 0.028726 | 0 | 0 |
| jump_offset | adaptive | 0.0076487 | 0.0076487 | 0.029398 | 0 | 0 |

## Synthetic 5 m depth RMSE and reference sensitivity, mm

| Case | Fixed 5 m input | FV1 | Minmod | Adaptive | MC grid gap | Roe grid gap | Fine formulation gap | Half dt change |
|---|---|---|---|---|---|---|---|---|
| plane | False | 5.3666 | 3.6107 | 3.2908 | 0.75887 | 0.99948 | 2.0697 | 0.0021551 |
| plane | True | 5.453 | 3.6469 | 3.3168 | 0.84095 | 1.0613 | 2.0477 | 0.0034375 |
| steep | False | 10.868 | 5.7333 | 5.3188 | 1.055 | 1.8225 | 2.7448 | 0.0081436 |
| steep | True | 9.762 | 6.1083 | 5.7215 | 6.9971 | 4.473 | 5.9296 | 0.014001 |
| irregular | False | 6.6052 | 4.1608 | 4.0225 | 0.49368 | 1.2365 | 1.6715 | 0.0013292 |
| irregular | True | 5.0622 | 5.5063 | 5.6334 | 1.275 | 1.3782 | 1.7239 | 0.0020264 |
| expansion | False | 7.9162 | 6.4959 | 6.3156 | 0.83231 | 1.3601 | 1.8086 | 0.0019249 |
| expansion | True | 5.3439 | 2.35 | 1.7052 | 0.73101 | 1.1575 | 1.8809 | 0.0022699 |
| rainfall | False | 2.1922 | 1.1356 | 1.065 | 0.098617 | 0.49657 | 0.7034 | 5.6261e-05 |
| rainfall | True | 2.8268 | 3.1871 | 3.1376 | 0.53277 | 0.60756 | 0.78549 | 0.00013653 |
| barrier | False | 17.196 | 16.943 | 16.759 | 0.71158 | 0.75579 | 1.3443 | 0.00072148 |
| barrier | True | 3.7479 | 2.3967 | 2.211 | 0.8231 | 0.83206 | 1.6092 | 0.00081038 |
| oblique_channel | False | 6.7258 | 6.3441 | 6.3199 | 0.52167 | 0.5512 | 1.5582 | 0.00022339 |
| oblique_channel | True | 3.752 | 2.75 | 2.7528 | 1.3685 | 0.96886 | 1.4311 | 0.0010662 |

## Uniform Manning sheets at 5 m, accepted transport RMSE in m²/s

| Slope % | h m | n | Fr | Exact q | FV1 | Minmod | Adaptive | LI |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.005 | 0.02 | 0.020879 | 2.3116e-05 | 5.3851e-06 | 5.1439e-11 | 5.1439e-11 | 2.0245e-12 |
| 0.01 | 0.02 | 0.02 | 0.083186 | 0.00073681 | 9.8903e-05 | 2.4879e-09 | 2.4879e-09 | 5.526e-11 |
| 0.05 | 0.1 | 0.04 | 0.12162 | 0.012044 | 0.0010255 | 8.7944e-08 | 8.7944e-08 | 6.7341e-10 |
| 0.1 | 0.3 | 0.04 | 0.20655 | 0.10629 | 0.0031107 | 3.8543e-07 | 3.8543e-07 | 2.513e-09 |
| 0.5 | 1 | 0.08 | 0.28225 | 0.88388 | 0.022285 | 1.1274e-05 | 1.1274e-05 | 3.3996e-15 |
| 1 | 0.005 | 0.08 | 0.16506 | 0.00018275 | 0.0005695 | 3.9591e-06 | 3.9591e-06 | 3.8305e-11 |
| 5 | 0.02 | 0.15 | 0.24801 | 0.0021967 | 0.0038547 | 0.00012897 | 0.00012897 | 3.41e-10 |
| 10 | 0.1 | 0.15 | 0.45865 | 0.045419 | 0.025404 | 0.00064085 | 0.00064085 | 4.1298e-09 |

## Lowland hydraulics at 5 m, 0.625 m reference

| Case | Method | Depth RMSE mm | Wet RMSE mm | 1 cm IoU | Mean section RMSE m³/s | Mass residual m³ | Fine method spread mm |
|---|---|---|---|---|---|---|---|
| wide_wave | contact_hll | 1.1609 | 0.7804 | 1 | 0.016091 | -5.6843e-14 | 0.32393 |
| wide_wave | minmod | 0.62817 | 0.34783 | 1 | 0.0058053 | -5.6843e-14 | 0.32393 |
| wide_wave | adaptive | 0.36944 | 0.1733 | 1 | 0.0031033 | -5.6843e-14 | 0.32393 |
| wide_wave | li | 0.18072 | 0.072575 | 1 | 0.0030101 | 5.6843e-13 | 0.32393 |
| backwater | contact_hll | 0.33299 | 0.35578 | 1 | 0.023593 | 0 | 0.12491 |
| backwater | minmod | 0.16916 | 0.18046 | 1 | 0.011466 | 0 | 0.12491 |
| backwater | adaptive | 0.1583 | 0.16886 | 1 | 0.0072092 | 1.1369e-13 | 0.12491 |
| backwater | li | 0.34693 | 0.37088 | 1 | 0.015717 | 2.2737e-13 | 0.12491 |
| depression | contact_hll | 1.5181 | 2.1684 | 0.85915 | 0.009062 | 1.4211e-14 | 0.42934 |
| depression | minmod | 0.84238 | 1.167 | 0.87143 | 0.0032413 | 1.4211e-14 | 0.42934 |
| depression | adaptive | 0.77799 | 1.0737 | 0.87143 | 0.0031603 | 1.4211e-14 | 0.42934 |
| depression | li | 0.39123 | 0.37983 | 0.94203 | 0.0008141 | 1.4211e-14 | 0.42934 |
| overbank | contact_hll | 3.2425 | 3.6146 | 0.86861 | 0.029123 | 0 | 0.88145 |
| overbank | minmod | 1.7591 | 1.821 | 0.96212 | 0.0065186 | 5.6843e-14 | 0.88145 |
| overbank | adaptive | 1.6216 | 1.9875 | 0.9771 | 0.0051152 | 0 | 0.88145 |
| overbank | li | 1.4977 | 1.0916 | 0.97037 | 0.0078765 | 1.7053e-13 | 0.88145 |
| embankment | contact_hll | 14.015 | 12.878 | 0.79076 | 0.26387 | -5.6843e-14 | 0.95014 |
| embankment | minmod | 12.359 | 11.541 | 0.86607 | 0.20294 | 1.1369e-13 | 0.95014 |
| embankment | adaptive | 12.387 | 11.553 | 0.86607 | 0.19583 | 5.6843e-14 | 0.95014 |
| embankment | li | 12.383 | 11.62 | 0.85088 | 0.20658 | 0 | 0.95014 |
| lateral_exchange | contact_hll | 0.9689 | 0.95479 | 0.97778 | 0.007996 | 0 | 0.078746 |
| lateral_exchange | minmod | 0.90367 | 0.90696 | 0.98883 | 0.0043661 | -2.8422e-14 | 0.078746 |
| lateral_exchange | adaptive | 0.8978 | 0.91331 | 0.98883 | 0.0036567 | 2.8422e-14 | 0.078746 |
| lateral_exchange | li | 0.60615 | 0.54167 | 0.97238 | 0.0024606 | 0 | 0.078746 |
| spatial_rain | contact_hll | 0.39535 | 0 | 1 | 0.0054289 | 0 | 0.10539 |
| spatial_rain | minmod | 0.24655 | 0 | 1 | 0.00027282 | 3.5527e-15 | 0.10539 |
| spatial_rain | adaptive | 0.21564 | 0 | 1 | 0.00027916 | 7.1054e-15 | 0.10539 |
| spatial_rain | li | 0.0090879 | 0 | 1 | 1.3355e-05 | -2.4052e-12 | 0.10539 |
| rain_loss | contact_hll | 0.27342 | 0 | 1 | 0.0031998 | 7.1054e-15 | 0.09308 |
| rain_loss | minmod | 0.18603 | 0 | 1 | 0.00019635 | 9.0594e-14 | 0.09308 |
| rain_loss | adaptive | 0.17395 | 0 | 1 | 0.00020092 | 8.3489e-14 | 0.09308 |
| rain_loss | li | 0.010559 | 0 | 1 | 9.4137e-06 | 1.0598e-11 | 0.09308 |
| stage | contact_hll | 0.65535 | 0.7993 | 1 | 0.020168 | 0 | 0.20345 |
| stage | minmod | 0.56429 | 0.68828 | 1 | 0.010731 | 0 | 0.20345 |
| stage | adaptive | 0.5365 | 0.65437 | 1 | 0.0065808 | 0 | 0.20345 |
| stage | li | 0.38392 | 0.46837 | 1 | 0.012314 | 1.7053e-13 | 0.20345 |

## Lowland hydraulics at 5 m, hydraulic refinement below native terrain

| Case | Method | Depth RMSE mm | Wet RMSE mm | 1 cm IoU | Mean section RMSE m³/s | Mass residual m³ | Fine method spread mm |
|---|---|---|---|---|---|---|---|
| wide_wave | contact_hll | 1.1704 | 0.78887 | 1 | 0.01616 | -5.6843e-14 | 0.20206 |
| wide_wave | minmod | 0.63888 | 0.35659 | 1 | 0.0058817 | -5.6843e-14 | 0.20206 |
| wide_wave | adaptive | 0.38024 | 0.18208 | 1 | 0.003193 | -5.6843e-14 | 0.20206 |
| wide_wave | li | 0.191 | 0.077625 | 1 | 0.0028616 | 5.1159e-13 | 0.20206 |
| backwater | contact_hll | 0.34209 | 0.36551 | 1 | 0.023629 | 0 | 0.10278 |
| backwater | minmod | 0.17997 | 0.19204 | 1 | 0.011493 | 0 | 0.10278 |
| backwater | adaptive | 0.17021 | 0.18162 | 1 | 0.0072617 | 1.1369e-13 | 0.10278 |
| backwater | li | 0.3484 | 0.37245 | 1 | 0.015653 | 2.2737e-13 | 0.10278 |
| depression | contact_hll | 1.5525 | 2.219 | 0.85915 | 0.0092387 | 1.4211e-14 | 0.30965 |
| depression | minmod | 0.88187 | 1.2263 | 0.87143 | 0.0034243 | 1.4211e-14 | 0.30965 |
| depression | adaptive | 0.81574 | 1.129 | 0.87143 | 0.003342 | 1.4211e-14 | 0.30965 |
| depression | li | 0.40694 | 0.40096 | 0.94203 | 0.00085708 | 2.8422e-14 | 0.30965 |
| overbank | contact_hll | 3.3567 | 3.7944 | 0.86232 | 0.02927 | 0 | 0.63261 |
| overbank | minmod | 1.8948 | 2.0524 | 0.95489 | 0.0064714 | 5.6843e-14 | 0.63261 |
| overbank | adaptive | 1.7867 | 2.2378 | 0.9697 | 0.0053459 | 0 | 0.63261 |
| overbank | li | 1.4907 | 1.058 | 0.97778 | 0.0075233 | 0 | 0.63261 |
| embankment | contact_hll | 14.081 | 12.839 | 0.78804 | 0.26473 | -5.6843e-14 | 0.6746 |
| embankment | minmod | 12.432 | 11.465 | 0.8631 | 0.2038 | 1.1369e-13 | 0.6746 |
| embankment | adaptive | 12.462 | 11.476 | 0.8631 | 0.19666 | 5.6843e-14 | 0.6746 |
| embankment | li | 12.452 | 11.555 | 0.84795 | 0.20741 | 0 | 0.6746 |
| lateral_exchange | contact_hll | 0.98148 | 0.97065 | 0.97778 | 0.007993 | 0 | 0.12525 |
| lateral_exchange | minmod | 0.91686 | 0.93244 | 0.98883 | 0.0043949 | -2.8422e-14 | 0.12525 |
| lateral_exchange | adaptive | 0.91013 | 0.93916 | 0.98883 | 0.003715 | 2.8422e-14 | 0.12525 |
| lateral_exchange | li | 0.59924 | 0.56039 | 0.97238 | 0.002393 | -8.5265e-14 | 0.12525 |
| spatial_rain | contact_hll | 0.39841 | 0 | 1 | 0.0054645 | 0 | 0.06018 |
| spatial_rain | minmod | 0.2498 | 0 | 1 | 0.00030785 | 3.5527e-15 | 0.06018 |
| spatial_rain | adaptive | 0.21888 | 0 | 1 | 0.00031433 | 7.1054e-15 | 0.06018 |
| spatial_rain | li | 0.0082475 | 0 | 1 | 4.2684e-05 | -2.4016e-12 | 0.06018 |
| rain_loss | contact_hll | 0.27567 | 0 | 1 | 0.0032249 | 7.1054e-15 | 0.053998 |
| rain_loss | minmod | 0.18846 | 0 | 1 | 0.00022076 | 9.0594e-14 | 0.053998 |
| rain_loss | adaptive | 0.17633 | 0 | 1 | 0.00022541 | 8.3489e-14 | 0.053998 |
| rain_loss | li | 0.008856 | 0 | 1 | 2.9404e-05 | 1.0601e-11 | 0.053998 |
| stage | contact_hll | 0.67001 | 0.81719 | 1 | 0.020313 | 0 | 0.21634 |
| stage | minmod | 0.59196 | 0.72205 | 1 | 0.011606 | 0 | 0.21634 |
| stage | adaptive | 0.56218 | 0.68571 | 1 | 0.0077362 | 0 | 0.21634 |
| stage | li | 0.42878 | 0.5231 | 1 | 0.013424 | 0 | 0.21634 |

## Loss of terrain detail at constant hydraulic resolution

| Case | Bed RMS mm | Max crest lowering mm | Storage change at 0.04 m stage m³ | Fine depth disagreement mm | Mean section disagreement m³/s |
|---|---|---|---|---|---|
| wide_wave | 0.74613 | 1.6337 | -0.021281 | 0.726 | 0.00096343 |
| backwater | 0.95107 | 3.8384 | -0.021281 | 0.95702 | 0.00022661 |
| depression | 1.3036 | 10.948 | -0.021281 | 1.1408 | 0.00087154 |
| overbank | 8.6965 | 36.434 | -5.7005 | 8.5917 | 0.0053852 |
| embankment | 8.3749 | 61.688 | -8.542 | 13.137 | 0.17347 |
| lateral_exchange | 7.5396 | 26.614 | -4.4278 | 5.997 | 0.0027306 |
| spatial_rain | 0.74613 | 1.6337 | -0.021281 | 0.35937 | 9.0623e-05 |
| rain_loss | 0.74613 | 1.6337 | -0.021281 | 0.25306 | 5.4706e-05 |
| stage | 0.74613 | 1.6337 | -0.021281 | 0.67526 | 0.00015095 |

## Equal measured budget at the FV1 2.5 m runtime

| Case | Method | Budget ms | Best tested grid m | RMSE | Tolerance | Cheapest passing run ms |
|---|---|---|---|---|---|---|
| barrier | adaptive | 13.884 | 10 | 0.039113 | 0.01 | — |
| barrier | fv1 | 13.884 | 5 | 0.017196 | 0.01 | — |
| barrier | minmod | 13.884 | 10 | 0.039113 | 0.01 | — |
| embankment | adaptive | 24.298 | 10 | 0.014813 | 0.001 | — |
| embankment | fv1 | 24.298 | 2.5 | 0.0070746 | 0.001 | — |
| embankment | li | 24.298 | 5 | 0.012383 | 0.001 | — |
| embankment | minmod | 24.298 | 10 | 0.015065 | 0.001 | — |
| irregular | adaptive | 13.238 | 10 | 0.0086488 | 0.01 | 7.7947 |
| irregular | fv1 | 13.238 | 2.5 | 0.004328 | 0.01 | 3.5693 |
| irregular | minmod | 13.238 | 10 | 0.0087572 | 0.01 | 7.3747 |
| oblique_channel | adaptive | 11.503 | 10 | 0.01773 | 0.01 | 12.702 |
| oblique_channel | fv1 | 11.503 | 2.5 | 0.0027227 | 0.01 | 6.4055 |
| oblique_channel | minmod | 11.503 | 10 | 0.017754 | 0.01 | 12.219 |
| overbank | adaptive | 24.499 | 10 | 0.0025984 | 0.001 | 50.88 |
| overbank | fv1 | 24.499 | 2.5 | 0.0022293 | 0.001 | — |
| overbank | li | 24.499 | 5 | 0.0014977 | 0.001 | — |
| overbank | minmod | 24.499 | 10 | 0.0028787 | 0.001 | 48.714 |
| plane | adaptive | 12.386 | 10 | 0.0060188 | 0.01 | 7.3692 |
| plane | fv1 | 12.386 | 2.5 | 0.0040036 | 0.01 | 3.1922 |
| plane | minmod | 12.386 | 10 | 0.006141 | 0.01 | 7.1157 |
| ritter | adaptive | 30.409 | 5 | 0.006901 | 0.04 | 8.7703 |
| ritter | fv1 | 30.409 | 2.5 | 0.020826 | 0.04 | 9.603 |
| ritter | minmod | 30.409 | 5 | 0.012545 | 0.04 | 7.3716 |
| shear | adaptive | 32.394 | 5 | 0.072423 | 0.15 | 5.0779 |
| shear | fv1 | 32.394 | 2.5 | 0.14312 | 0.15 | 32.394 |
| shear | minmod | 32.394 | 5 | 0.094773 | 0.15 | 4.9746 |
| spatial_rain | adaptive | 23.539 | 10 | 0.00034379 | 0.001 | 10.321 |
| spatial_rain | fv1 | 23.539 | 2.5 | 0.00030459 | 0.001 | 4.8243 |
| spatial_rain | li | 23.539 | 5 | 9.0879e-06 | 0.001 | 6.3169 |
| spatial_rain | minmod | 23.539 | 10 | 0.00035838 | 0.001 | 9.9492 |
| steep | adaptive | 12.43 | 10 | 0.0099007 | 0.01 | 7.24 |
| steep | fv1 | 12.43 | 2.5 | 0.0070252 | 0.01 | 12.43 |
| steep | minmod | 12.43 | 10 | 0.010397 | 0.01 | 12.655 |
| stoker | adaptive | 35.288 | 5 | 0.014009 | 0.04 | 6.5383 |
| stoker | fv1 | 35.288 | 2.5 | 0.02799 | 0.04 | 35.288 |
| stoker | minmod | 35.288 | 5 | 0.020837 | 0.04 | 11.132 |
| wide_wave | adaptive | 24.066 | 10 | 0.00083097 | 0.001 | 10.465 |
| wide_wave | fv1 | 24.066 | 2.5 | 0.00082388 | 0.001 | 24.066 |
| wide_wave | li | 24.066 | 10 | 0.00013666 | 0.001 | 6.4065 |
| wide_wave | minmod | 24.066 | 10 | 0.0010218 | 0.001 | 25.896 |

This budget selects measured runs that fit. Tolerances are illustrative depth RMSE in metres except shear, which uses transverse momentum in m²/s. Missing entries indicate no qualifying measured configuration.

## Nsight Compute, baseline

| Launch | Kernel | Registers/thread | Active warps % | DRAM % sustained peak | DRAM read GB/s |
|---|---|---|---|---|---|
| 0 | split_sources_3e866b96 | 58 | 55.71 | 11.81 | 77.374 |
| 1 | slopes_c1540043 | 70 | 45.52 | 4.99 | 32.666 |
| 2 | faces_f7e69d58 | 72 | 44.94 | 5.8 | 38.025 |
| 3 | faces_f7e69d58 | 72 | 44.98 | 5.97 | 39.105 |

## Nsight Compute, final

| Launch | Kernel | Registers/thread | Active warps % | DRAM % sustained peak | DRAM read GB/s |
|---|---|---|---|---|---|
| 0 | split_sources_ef4fb6e4 | 64 | 56.9 | 66.07 | 431.5 |
| 1 | slopes_88de1a80 | 82 | 29.8 | 30.23 | 197.63 |
| 2 | faces_5972e74c | 72 | 39.46 | 13.65 | 89.431 |
| 3 | faces_5972e74c | 72 | 44.57 | 14.96 | 98.012 |
| 4 | limit_faces_fee2bbb4 | 20 | 84.58 | 57.07 | 371.33 |

## Native 5 m HIRDS continuation, hour 25 to 25 h 10 min

| Method | Export m³ | Rain m³ | Loss m³ | Final wet area m² | Speed p95 m/s | Final depth RMSE vs FV1 m | Mass residual m³ |
|---|---|---|---|---|---|---|---|
| contact_hll | 41707 | 27338 | 7969.6 | 1229025 | 1.5733 | 0 | 2.3283e-10 |
| fv2_adaptive | 47821 | 27338 | 7969.6 | 959425 | 2.7877 | 0.01895 | 2.0373e-10 |
| li | 49049 | 27338 | 7969.6 | 1087700 | 2.2427 | 0.036884 | -5.5297e-10 |

## Complete HIRDS 48-hour event, matched 20 m terrain

| Version | Method | Steps | Hydraulic loop s | Peak m³/s | Peak interval end h | Residual m³ |
|---|---|---|---|---|---|---|
| baseline | fv2_mc | 321184 | 259.08 | 171.72 | 24.433 | -6.7666e-10 |
| final | contact_hll | 263421 | 38.303 | 139.73 | 24.583 | 5.5297e-10 |
| final | fv2_adaptive | 319664 | 192.56 | 172 | 24.433 | 6.5993e-09 |
