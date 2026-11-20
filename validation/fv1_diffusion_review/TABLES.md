# Measured comparisons

All timings are hydraulic seconds unless the column gives milliseconds. Section discharge is per unit width except in the radial-basin table.

## Same-grid GPU cost

| Policy | Method | Pairs | Minimum ×FV1 | Median ×FV1 | Maximum ×FV1 | At most 1.5× |
|---|---|---|---|---|---|---|
| common | fv1 | 57 | 1 | 1 | 1 | 57 |
| common | low_froude_hll | 57 | 0.74044 | 1.0723 | 1.0909 | 57 |
| common | euler_mc | 57 | 0.78126 | 1.2491 | 1.4635 | 57 |
| common | hancock_mc | 57 | 0.98255 | 1.4623 | 1.6547 | 38 |
| common | hancock_adaptive | 57 | 1.0085 | 1.4819 | 1.6793 | 33 |
| common | hancock_low_froude | 57 | 1.02 | 1.551 | 1.7679 | 19 |
| common | mc | 57 | 1.3227 | 1.9627 | 2.2095 | 2 |
| common | adaptive | 57 | 1.3727 | 2.0121 | 2.2487 | 1 |
| common | li | 57 | 1.0077 | 1.6552 | 2.5491 | 18 |
| production | fv1 | 57 | 1 | 1 | 1 | 57 |
| production | low_froude_hll | 57 | 1.0029 | 1.0627 | 1.1548 | 57 |
| production | euler_mc | 57 | 1.0032 | 1.4154 | 3.4076 | 35 |
| production | hancock_mc | 57 | 1.199 | 1.6547 | 3.3681 | 17 |
| production | hancock_adaptive | 57 | 1.2148 | 1.6888 | 3.4693 | 17 |
| production | hancock_low_froude | 57 | 1.3127 | 1.7645 | 3.5019 | 12 |
| production | mc | 57 | 1.6092 | 2.2035 | 4.5886 | 0 |
| production | adaptive | 57 | 1.6331 | 2.2519 | 4.7708 | 0 |
| production | li | 57 | 0.54174 | 2.1964 | 8.393 | 17 |

## Matched 5 m analytical fields and internal sections

| Case | Method | Median Fr | Depth RMSE mm | 1 cm IoU | Q RMSE m²/s | Peak error RMS m²/s | GPU ms |
|---|---|---|---|---|---|---|---|
| front_plain | fv1 | 0.54745 | 20.061 | 0.81522 | 0.029171 | 0.014641 | 13.473 |
| front_plain | low_froude_hll | 0.54745 | 20.018 | 0.81522 | 0.02935 | 0.015522 | 14.447 |
| front_plain | euler_mc | 0.54745 | 3.0529 | 1 | 0.0051301 | 0.0064004 | 16.699 |
| front_plain | hancock_mc | 0.54745 | 3.0515 | 0.97403 | 0.0050979 | 0.0041152 | 19.44 |
| front_plain | hancock_adaptive | 0.54745 | 3.0515 | 0.97403 | 0.0050979 | 0.0041152 | 19.735 |
| front_plain | hancock_low_froude | 0.54745 | 3.069 | 0.97403 | 0.005074 | 0.0037538 | 20.714 |
| front_plain | mc | 0.54745 | 3.1238 | 0.97403 | 0.0051505 | 0.0042837 | 26.187 |
| front_plain | adaptive | 0.54745 | 3.1238 | 0.97403 | 0.0051505 | 0.0042837 | 26.785 |
| front_plain | li | 0.54745 | 6.0719 | 0.97403 | 0.0095898 | 0.007417 | 17.232 |
| front_steep | fv1 | 2.0557 | 34.823 | 0.90588 | 0.16324 | 0.25491 | 16.818 |
| front_steep | low_froude_hll | 2.0557 | 34.823 | 0.90588 | 0.16324 | 0.25491 | 12.453 |
| front_steep | euler_mc | 2.0557 | 6.249 | 0.99216 | 0.023861 | 0.025055 | 14.523 |
| front_steep | hancock_mc | 2.0557 | 3.2437 | 1 | 0.011651 | 0.019448 | 16.524 |
| front_steep | hancock_adaptive | 2.0557 | 3.2437 | 1 | 0.011651 | 0.019448 | 16.96 |
| front_steep | hancock_low_froude | 2.0557 | 3.2437 | 1 | 0.011651 | 0.019448 | 17.155 |
| front_steep | mc | 2.0557 | 3.253 | 0.99609 | 0.011553 | 0.017951 | 22.245 |
| front_steep | adaptive | 2.0557 | 3.253 | 0.99609 | 0.011553 | 0.017951 | 23.086 |
| front_steep | li | 2.0557 | 46.4 | 0.91373 | 0.16136 | 0.405 | 16.947 |
| gravity_pulse | fv1 | 0.0021563 | 0.94462 | 1 | 0.0012805 | 0.0024749 | 26.9 |
| gravity_pulse | low_froude_hll | 0.0021563 | 0.66309 | 1 | 0.00088774 | 0.0017547 | 29.278 |
| gravity_pulse | euler_mc | 0.0021563 | 0.24179 | 1 | 0.00027332 | 0.00070277 | 27.175 |
| gravity_pulse | hancock_mc | 0.0021563 | 0.25778 | 1 | 0.00029724 | 0.00077682 | 32.685 |
| gravity_pulse | hancock_adaptive | 0.0021563 | 0.25778 | 1 | 0.00029724 | 0.00077682 | 33.261 |
| gravity_pulse | hancock_low_froude | 0.0021563 | 0.24479 | 1 | 0.00024836 | 0.00059341 | 36.126 |
| gravity_pulse | mc | 0.0021563 | 0.26398 | 1 | 0.00030323 | 0.00078585 | 45.965 |
| gravity_pulse | adaptive | 0.0021563 | 0.26398 | 1 | 0.00030323 | 0.00078585 | 47.082 |
| gravity_pulse | li | 0.0021563 | 0.39759 | 1 | 0.000495 | 0.00028572 | 28.227 |
| shear_pulse | fv1 | 1.166 | 0.076603 | 1 | 0.040663 | 0.0746 | 12.686 |
| shear_pulse | low_froude_hll | 1.166 | 0.076286 | 1 | 0.040663 | 0.0746 | 13.49 |
| shear_pulse | euler_mc | 1.166 | 0 | 1 | 0.0088964 | 0.013192 | 13.367 |
| shear_pulse | hancock_mc | 1.166 | 7.3485e-14 | 1 | 0.011271 | 0.017813 | 16.161 |
| shear_pulse | hancock_adaptive | 1.166 | 7.3485e-14 | 1 | 0.011271 | 0.017813 | 16.712 |
| shear_pulse | hancock_low_froude | 1.166 | 7.3485e-14 | 1 | 0.011271 | 0.017813 | 17.187 |
| shear_pulse | mc | 1.166 | 0 | 1 | 0.011914 | 0.01886 | 22.157 |
| shear_pulse | adaptive | 1.166 | 0 | 1 | 0.011914 | 0.01886 | 23.286 |
| shear_pulse | li | 1.166 | 6.1898e-13 | 1 | 0.11367 | 0.20723 | 23.666 |
| gentle_rise | fv1 | 0.0024318 | 1.6719e-06 | 1 | 2.3907e-05 | 2.3847e-05 | 13.063 |
| gentle_rise | low_froude_hll | 0.0024318 | 2.0794e-07 | 1 | 2.3907e-05 | 2.3847e-05 | 14.058 |
| gentle_rise | euler_mc | 0.0024318 | 5.1942e-07 | 1 | 7.6597e-07 | 7.6589e-07 | 17.092 |
| gentle_rise | hancock_mc | 0.0024318 | 4.4417e-07 | 1 | 1.8082e-10 | 2.4127e-10 | 19.831 |
| gentle_rise | hancock_adaptive | 0.0024318 | 4.4417e-07 | 1 | 1.8082e-10 | 2.4127e-10 | 20.127 |
| gentle_rise | hancock_low_froude | 0.0024318 | 1.6421e-07 | 1 | 1.8264e-10 | 2.6096e-10 | 21.306 |
| gentle_rise | mc | 0.0024318 | 6.4192e-07 | 1 | 1.802e-10 | 2.2821e-10 | 26.917 |
| gentle_rise | adaptive | 0.0024318 | 6.4192e-07 | 1 | 1.802e-10 | 2.2821e-10 | 27.508 |
| gentle_rise | li | 0.0024318 | 1.0887e-14 | 1 | 7.0071e-07 | 6.3036e-07 | 17.746 |
| routing | fv1 | 0.57864 | 0.013413 | 1 | 0.012431 | 0.0004529 | 13.173 |
| routing | low_froude_hll | 0.57864 | 0.0095576 | 1 | 0.012431 | 0.00045196 | 14.134 |
| routing | euler_mc | 0.57864 | 0.0012014 | 1 | 0.0022917 | 0.0022907 | 16.547 |
| routing | hancock_mc | 0.57864 | 0.00078403 | 1 | 6.8347e-06 | 7.6802e-06 | 19.282 |
| routing | hancock_adaptive | 0.57864 | 0.00078403 | 1 | 6.8347e-06 | 7.6802e-06 | 19.543 |
| routing | hancock_low_froude | 0.57864 | 0.00059119 | 1 | 6.8373e-06 | 7.7041e-06 | 20.663 |
| routing | mc | 0.57864 | 0.0017858 | 1 | 6.8189e-06 | 7.5092e-06 | 25.887 |
| routing | adaptive | 0.57864 | 0.0017858 | 1 | 6.8189e-06 | 7.5092e-06 | 26.469 |
| routing | li | 0.57864 | 5.3981e-12 | 1 | 0.001078 | 5.0871e-05 | 18.392 |
| rain20 | fv1 | 0.016018 | 3.0147e-14 | 1 | 8.9819e-06 | 3.1072e-05 | 13.174 |
| rain20 | low_froude_hll | 0.016018 | 3.0083e-14 | 1 | 8.9819e-06 | 3.1072e-05 | 14.131 |
| rain20 | euler_mc | 0.016018 | 3.7359e-14 | 1 | 4.0465e-07 | 1.4017e-06 | 17.024 |
| rain20 | hancock_mc | 0.016018 | 3.7359e-14 | 1 | 4.0465e-07 | 1.4017e-06 | 19.804 |
| rain20 | hancock_adaptive | 0.016018 | 3.7359e-14 | 1 | 4.0465e-07 | 1.4017e-06 | 20.082 |
| rain20 | hancock_low_froude | 0.016018 | 3.7359e-14 | 1 | 4.0465e-07 | 1.4017e-06 | 21.244 |
| rain20 | mc | 0.016018 | 3.7359e-14 | 1 | 4.0465e-07 | 1.4017e-06 | 26.756 |
| rain20 | adaptive | 0.016018 | 3.7359e-14 | 1 | 4.0465e-07 | 1.4017e-06 | 27.345 |
| rain20 | li | 0.016018 | 3.0083e-14 | 1 | 7.3076e-06 | 2.5267e-05 | 17.572 |
| damped_train | fv1 | 0.57338 | 1.369 | 1 | 0.03696 | 0.068851 | 11.769 |
| damped_train | low_froude_hll | 0.57338 | 1.3922 | 1 | 0.036963 | 0.068857 | 12.49 |
| damped_train | euler_mc | 0.57338 | 1.0124 | 1 | 0.012553 | 0.025441 | 15.764 |
| damped_train | hancock_mc | 0.57338 | 1.031 | 1 | 0.013394 | 0.026906 | 18.515 |
| damped_train | hancock_adaptive | 0.57338 | 1.031 | 1 | 0.013394 | 0.026906 | 18.884 |
| damped_train | hancock_low_froude | 0.57338 | 1.0317 | 1 | 0.013394 | 0.026906 | 19.511 |
| damped_train | mc | 0.57338 | 1.0369 | 1 | 0.013708 | 0.027377 | 24.484 |
| damped_train | adaptive | 0.57338 | 1.0369 | 1 | 0.013708 | 0.027377 | 25.216 |
| damped_train | li | 0.57338 | 0.89731 | 1 | 0.12718 | 0.17535 | 20.604 |

## 400 × 240 cells, 128 accepted steps, seven warmed runs

Uncaptured kernel groups are µs per accepted step. Fused prediction is included in slopes. Face kernels include their wave-speed calculation. FV1 uses different kernel names. Raw names and individual repeat values are in profile.json.

| Case | Mode | Method | Fused | Steps | Wall ms | Owned MiB | Capture ms | slopes | predict | faces | split_sources | euler | limit_faces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stoker | dense | contact_hll | None | 128 | 43.401 | 23.841 | 83.044 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | low_froude_hll | None | 128 | 48.224 | 23.841 | 0.35622 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | euler_mc | None | 128 | 31.848 | 40.701 | 1.0176 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | hancock_mc | False | 128 | 44.173 | 41.068 | 0.61362 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | hancock_mc | True | 128 | 43.26 | 41.068 | 0.5713 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | hancock_low_froude | None | 128 | 48.36 | 41.068 | 0.45252 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | hancock_adaptive | None | 128 | 43.927 | 41.434 | 0.57789 | 0 | 0 | 0 | 0 | 0 | 0 |
| stoker | dense | mc | None | 128 | 56.186 | 45.111 | 0.55579 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | contact_hll | None | 128 | 22.366 | 23.841 | 0.39745 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | low_froude_hll | None | 128 | 23.292 | 23.841 | 0.34448 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | euler_mc | None | 128 | 24.302 | 40.701 | 0.43868 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | hancock_mc | False | 128 | 28.277 | 41.068 | 0.49585 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | hancock_mc | True | 128 | 27.108 | 41.068 | 0.44879 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | hancock_low_froude | None | 128 | 28.22 | 41.068 | 0.45372 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | hancock_adaptive | None | 128 | 27.575 | 41.434 | 0.45788 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | dense | mc | None | 128 | 41.209 | 45.111 | 0.53639 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | contact_hll | None | 128 | 20.274 | 23.864 | 1.2891 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | low_froude_hll | None | 128 | 20.029 | 23.864 | 0.41362 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | euler_mc | None | 128 | 19.674 | 40.701 | 0.5073 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | hancock_mc | False | 128 | 23.687 | 41.068 | 0.46188 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | hancock_mc | True | 128 | 22.489 | 41.068 | 0.4635 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | hancock_low_froude | None | 128 | 23.48 | 41.068 | 0.4523 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | hancock_adaptive | None | 128 | 22.972 | 41.434 | 0.50221 | 0 | 0 | 0 | 0 | 0 | 0 |
| dry_patch | sparse | mc | None | 128 | 32.047 | 45.111 | 0.55854 | 0 | 0 | 0 | 0 | 0 | 0 |

## Transient peak at the central section, 5 m

| Case | Method | Exact peak | Model peak | Attenuation % | Peak time error s | Width s | Exact width s | Speed m/s | Exact speed m/s |
|---|---|---|---|---|---|---|---|---|---|
| gravity_pulse | fv1 | 0.0086471 | 0.0061488 | 28.891 | -2.3693 | 37.473 | 26.142 | 1.0213 | 1.1099 |
| gravity_pulse | low_froude_hll | 0.0086471 | 0.0068382 | 20.918 | -0.5891 | 34.501 | 26.142 | 1.0068 | 1.1099 |
| gravity_pulse | euler_mc | 0.0086471 | 0.0078556 | 9.1531 | -0.69836 | 28.977 | 26.142 | 1.0577 | 1.1099 |
| gravity_pulse | hancock_mc | 0.0086471 | 0.0077913 | 9.8963 | -1.1227 | 29.253 | 26.142 | 1.0534 | 1.1099 |
| gravity_pulse | hancock_low_froude | 0.0086471 | 0.0080246 | 7.1984 | -1.1061 | 28.044 | 26.142 | 1.1026 | 1.1099 |
| gravity_pulse | mc | 0.0086471 | 0.0077813 | 10.012 | -0.94671 | 29.291 | 26.142 | 1.0635 | 1.1099 |
| gravity_pulse | li | 0.0086471 | 0.0084761 | 1.977 | 2.6944 | 27.628 | 26.142 | 1.0087 | 1.1099 |
| shear_pulse | fv1 | 0.22253 | 0.14006 | 37.058 | -0.43214 | 17.644 | 11.346 | 1.9584 | 2 |
| shear_pulse | low_froude_hll | 0.22253 | 0.14006 | 37.058 | -0.43214 | 17.644 | 11.346 | 1.9584 | 2 |
| shear_pulse | euler_mc | 0.22253 | 0.20714 | 6.9124 | -0.22644 | 12.356 | 11.346 | 2.0028 | 2 |
| shear_pulse | hancock_mc | 0.22253 | 0.20171 | 9.3523 | -0.28285 | 12.668 | 11.346 | 2.0016 | 2 |
| shear_pulse | hancock_low_froude | 0.22253 | 0.20171 | 9.3523 | -0.28285 | 12.668 | 11.346 | 2.0016 | 2 |
| shear_pulse | mc | 0.22253 | 0.20034 | 9.9688 | -0.32728 | 12.737 | 11.346 | 2.0015 | 2 |
| shear_pulse | li | 0.22253 | 0.00056254 | 99.747 | -14.75 | None | 11.346 | None | 2 |

## Affordable grids at the measured 5 m FV1 time budget

| Case | Budget ms | Method | Grid m | Time ms | Q RMSE | FV1 5 m RMSE |
|---|---|---|---|---|---|---|
| front_plain | 5.7867 | adaptive | 10 | 5.2297 | 0.012018 | 0.027551 |
| front_plain | 5.7867 | euler_mc | 10 | 3.4798 | 0.012671 | 0.027551 |
| front_plain | 5.7867 | euler_minmod | 10 | 3.4199 | 0.014036 | 0.027551 |
| front_plain | 5.7867 | fv1 | 5 | 5.7867 | 0.027551 | 0.027551 |
| front_plain | 5.7867 | hancock_adaptive | 10 | 3.9497 | 0.012338 | 0.027551 |
| front_plain | 5.7867 | hancock_low_froude | 10 | 4.1147 | 0.012306 | 0.027551 |
| front_plain | 5.7867 | hancock_mc | 10 | 3.9035 | 0.012338 | 0.027551 |
| front_plain | 5.7867 | hancock_minmod | 10 | 3.8404 | 0.016193 | 0.027551 |
| front_plain | 5.7867 | low_froude_hll | 10 | 2.394 | 0.048045 | 0.027551 |
| front_plain | 5.7867 | mc | 10 | 5.1009 | 0.012018 | 0.027551 |
| front_plain | 5.7867 | minmod | 10 | 5.0631 | 0.016877 | 0.027551 |
| front_steep | 4.3893 | fv1 | 5 | 4.3893 | 0.16546 | 0.16546 |
| front_steep | 4.3893 | low_froude_hll | 10 | 2.746 | 0.1744 | 0.16546 |
| gravity_pulse | 5.5768 | adaptive | 10 | 4.6616 | 0.00094199 | 0.0011493 |
| gravity_pulse | 5.5768 | euler_mc | 10 | 2.9362 | 0.00069949 | 0.0011493 |
| gravity_pulse | 5.5768 | euler_minmod | 10 | 2.9114 | 0.0010396 | 0.0011493 |
| gravity_pulse | 5.5768 | fv1 | 5 | 5.5768 | 0.0011493 | 0.0011493 |
| gravity_pulse | 5.5768 | hancock_adaptive | 10 | 3.4107 | 0.00086222 | 0.0011493 |
| gravity_pulse | 5.5768 | hancock_low_froude | 10 | 3.6778 | 0.00065511 | 0.0011493 |
| gravity_pulse | 5.5768 | hancock_mc | 10 | 3.3597 | 0.00086222 | 0.0011493 |
| gravity_pulse | 5.5768 | hancock_minmod | 10 | 3.3298 | 0.0011566 | 0.0011493 |
| gravity_pulse | 5.5768 | low_froude_hll | 10 | 2.7746 | 0.0013319 | 0.0011493 |
| gravity_pulse | 5.5768 | mc | 10 | 4.5615 | 0.00094199 | 0.0011493 |
| gravity_pulse | 5.5768 | minmod | 10 | 4.5255 | 0.0012798 | 0.0011493 |
| shear_pulse | 7.1033 | euler_mc | 10 | 5.7401 | 0.03148 | 0.039242 |
| shear_pulse | 7.1033 | euler_minmod | 10 | 5.7185 | 0.040355 | 0.039242 |
| shear_pulse | 7.1033 | fv1 | 5 | 7.1033 | 0.039242 | 0.039242 |
| shear_pulse | 7.1033 | hancock_adaptive | 10 | 7.024 | 0.033702 | 0.039242 |
| shear_pulse | 7.1033 | hancock_mc | 10 | 6.8253 | 0.033705 | 0.039242 |
| shear_pulse | 7.1033 | hancock_minmod | 10 | 6.782 | 0.041538 | 0.039242 |
| shear_pulse | 7.1033 | low_froude_hll | 10 | 4.0215 | 0.054307 | 0.039242 |

## Circular Thacker basin after one cycle

The selected gauge is 80 m from the centre. The inherited JSON fields named cycle2 refer here to the last complete cycle, which is cycle one.

| dx m | Method | Depth RMSE m | Exact peak m³/s | Model peak m³/s | Q RMSE m³/s | Wet symmetric difference m² | Budget residual m³ |
|---|---|---|---|---|---|---|---|
| 10 | fv1 | 0.28871 | 175.35 | 115.42 | 56.521 | 33600 | -7.276e-12 |
| 10 | euler_mc | 0.11355 | 175.35 | 170.17 | 20.742 | 22000 | 0 |
| 10 | hancock_mc | 0.13039 | 175.35 | 165.09 | 23.477 | 28000 | 0 |
| 10 | hancock_low_froude | 0.13243 | 175.35 | 165.22 | 24.004 | 30000 | 0 |
| 10 | hancock_adaptive | 0.12736 | 175.35 | 165.09 | 23.292 | 28000 | -7.276e-12 |
| 10 | low_froude_hll | 0.27096 | 175.35 | 122.23 | 53.549 | 35200 | 0 |
| 10 | mc | 0.13183 | 175.35 | 166.2 | 23.787 | 27600 | -7.276e-12 |
| 10 | adaptive | 0.12754 | 175.35 | 166.2 | 23.603 | 27600 | 0 |
| 5 | fv1 | 0.21524 | 175.35 | 142.91 | 39.415 | 42300 | 0 |
| 5 | euler_mc | 0.035727 | 175.35 | 176.77 | 7.4213 | 4800 | -7.276e-12 |
| 5 | hancock_mc | 0.051415 | 175.35 | 173.49 | 9.7876 | 3500 | 0 |
| 5 | hancock_low_froude | 0.051218 | 175.35 | 173.52 | 9.8309 | 4100 | 0 |
| 5 | hancock_adaptive | 0.050649 | 175.35 | 173.49 | 9.8579 | 3700 | 0 |
| 5 | low_froude_hll | 0.19837 | 175.35 | 147.36 | 37.692 | 43500 | 0 |
| 5 | mc | 0.052134 | 175.35 | 173.3 | 9.9906 | 3300 | 0 |
| 5 | adaptive | 0.051041 | 175.35 | 173.3 | 10.068 | 3300 | -7.276e-12 |
| 1 | fv1 | 0.078866 | 175.35 | 168.47 | 15.28 | 5324 | 0 |
| 1 | low_froude_hll | 0.074667 | 175.35 | 169.5 | 14.934 | 5456 | 0 |
| 1 | euler_mc | 0.0058976 | 175.35 | 175.99 | 0.57202 | 106 | 0 |
| 1 | hancock_mc | 0.0040478 | 175.35 | 175.29 | 0.82789 | 100 | 0 |
| 1 | mc | 0.0041775 | 175.35 | 175.29 | 0.87845 | 100 | 0 |
| 1 | adaptive | 0.0041547 | 175.35 | 175.29 | 0.8784 | 100 | 0 |

## Jump discharge 40–80 s after initial projection

| Method | dx m | Section RMSE m²/s | Max section std m²/s | Max absolute error m²/s |
|---|---|---|---|---|
| fv1 | 5 | 0.0036238 | 0.0012984 | 0.0070465 |
| fv1 | 2.5 | 0.0028417 | 0.00084424 | 0.0045013 |
| fv1 | 1.25 | 0.0016517 | 0.00047941 | 0.0024262 |
| euler_mc | 5 | 0.0022232 | 0.0013932 | 0.0045512 |
| euler_mc | 2.5 | 0.069968 | 0.106 | 0.15655 |
| euler_mc | 1.25 | 0.060142 | 0.089523 | 0.13406 |
| hancock_mc | 5 | 0.0021128 | 0.0008401 | 0.0030307 |
| hancock_mc | 2.5 | 0.065034 | 0.099087 | 0.1461 |
| hancock_mc | 1.25 | 0.054879 | 0.083779 | 0.12595 |
| hancock_minmod | 5 | 0.0020219 | 0.0011409 | 0.0032145 |
| hancock_minmod | 2.5 | 0.00053406 | 0.00026587 | 0.00086956 |
| hancock_minmod | 1.25 | 0.00015217 | 0.00012694 | 0.00027608 |
| hancock_adaptive | 5 | 0.0018295 | 0.00090464 | 0.003032 |
| hancock_adaptive | 2.5 | 0.00047269 | 0.00027911 | 0.00079081 |
| hancock_adaptive | 1.25 | 0.00014792 | 0.00010228 | 0.00021536 |
| hancock_low_froude | 5 | 0.0021292 | 0.00073275 | 0.0030764 |
| hancock_low_froude | 2.5 | 0.065043 | 0.098198 | 0.14767 |
| hancock_low_froude | 1.25 | 0.055318 | 0.084708 | 0.12467 |
| mc | 5 | 0.0021706 | 0.0011055 | 0.0036403 |
| mc | 2.5 | 0.067892 | 0.1041 | 0.15343 |
| mc | 1.25 | 0.057391 | 0.087758 | 0.13149 |
| adaptive | 5 | 0.0018252 | 0.00091843 | 0.0030321 |
| adaptive | 2.5 | 0.00047131 | 0.00028067 | 0.00079078 |
| adaptive | 1.25 | 0.00014718 | 0.00010229 | 0.00021199 |
