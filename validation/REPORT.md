# Independent numerical validation

The criteria in [ACCEPTANCE.md](ACCEPTANCE.md) were fixed before evaluating the extracted solver outputs. No hydraulic operator, roughness, boundary or tolerance was tuned to obtain passes. These are new CPU runs against compact pre-existing independent LISFLOOD-FP 8.3 reference outputs.

**25 of 32 comparisons pass the applicable gates.** Failures and incomplete coverage remain release limitations. All runs completed with finite, nonnegative depths. The maximum absolute floodwarp mass residual was 4.06e-09 m³.

| Case | Model | Final depth RMSE (m) | Final wet IoU | Outlet NRMSE | Verdict |
|---|---|---:|---:|---:|---|
| acc_flat_10m_0mmh_dt1 | physical_candidate | 3.46945e-18 | 1 | 0.0 | PASS |
| acc_flat_10m_0mmh_dt1 | lisflood_fp_acc | 3.46945e-18 | 1 | 0.0 | PASS |
| acc_steep_10m_0mmh_dt1 | physical_candidate | 0.000654297 | 1 | 0.016622871374171646 | DISCREPANCY |
| acc_steep_10m_0mmh_dt1 | lisflood_fp_acc | 1.31731e-06 | 1 | 0.0007211206968732899 | PASS |
| acc_complex_10m_0mmh_dt1 | physical_candidate | 0.000535614 | 0.921569 | 0.01602257542670255 | DISCREPANCY |
| acc_complex_10m_0mmh_dt1 | lisflood_fp_acc | 1.16019e-09 | 1 | 2.990912994698588e-08 | PASS |
| acc_heterogeneous_10m_0mmh_dt1 | physical_candidate | 0.000598192 | 0.931624 | 0.02609890830889269 | DISCREPANCY |
| acc_heterogeneous_10m_0mmh_dt1 | lisflood_fp_acc | 1.42056e-09 | 1 | 3.940695144840197e-08 | PASS |
| acc_complex_5m_0mmh_dt1 | physical_candidate | 0.000662927 | 0.886957 | 0.023985278039876913 | DISCREPANCY |
| acc_complex_5m_0mmh_dt1 | lisflood_fp_acc | 2.50177e-06 | 0.94375 | 0.00018432782622103988 | DISCREPANCY |
| acc_complex_20m_0mmh_dt1 | physical_candidate | 0.000460841 | 0.964286 | 0.01443855283230644 | PASS |
| acc_complex_20m_0mmh_dt1 | lisflood_fp_acc | 8.06522e-10 | 1 | 2.697312813120953e-08 | PASS |
| acc_heterogeneous_10m_1mmh_dt0.5 | physical_candidate | 0.000660494 | 0.927928 | 0.02716775380817575 | DISCREPANCY |
| acc_heterogeneous_10m_1mmh_dt0.5 | lisflood_fp_acc | 1.18134e-09 | 1 | 3.9521560883072466e-08 | PASS |
| acc_heterogeneous_10m_1mmh_dt0.25 | physical_candidate | 0.000661362 | 0.927928 | 0.02740524118617434 | DISCREPANCY |
| acc_heterogeneous_10m_1mmh_dt0.25 | lisflood_fp_acc | 1.44909e-07 | 1 | 9.425504095385103e-05 | PASS |
| fv1_wet_dam_0.4 | FV1 | 0.0017671 | 1 | not sampled | PASS |
| fv1_wet_dam_0.2 | FV1 | 0.000948581 | 1 | not sampled | PASS |
| fv1_wet_dam_0.1 | FV1 | 0.000501505 | 1 | not sampled | PASS |
| fv1_wet_dam_0.05 | FV1 | 0.000310305 | 1 | not sampled | PASS |
| fv1_strong_shock_0.4 | FV1 | 0.0146176 | 1 | not sampled | PASS |
| fv1_strong_shock_0.2 | FV1 | 0.0081418 | 1 | not sampled | PASS |
| fv1_strong_shock_0.1 | FV1 | 0.00540579 | 1 | not sampled | PASS |
| fv1_strong_shock_0.05 | FV1 | 0.00425004 | 1 | not sampled | PASS |
| fv1_dry_dam_0.4 | FV1 | 0.00141407 | 1 | not sampled | PASS |
| fv1_dry_dam_0.2 | FV1 | 0.000908826 | 0.984127 | not sampled | PASS |
| fv1_dry_dam_0.1 | FV1 | 0.000588596 | 1 | not sampled | PASS |
| fv1_dry_dam_0.05 | FV1 | 0.000458908 | 1 | not sampled | PASS |
| fv1_retained_outlet_crop_0.4 | FV1 | 0.00217378 | 1 | not sampled | PASS |
| fv1_retained_outlet_crop_0.2 | FV1 | 0.00210829 | 1 | not sampled | PASS |
| fv1_retained_outlet_crop_0.1 | FV1 | 0.00209364 | 1 | not sampled | PASS |
| fv1_retained_outlet_crop_0.05 | FV1 | 0.00209012 | 1 | not sampled | PASS |

## Discrepancies

- acc_steep_10m_0mmh_dt1 / physical_candidate: temporal wet, final volume. No acceptance relaxation applied.
- acc_complex_10m_0mmh_dt1 / physical_candidate: final wet, temporal wet. No acceptance relaxation applied.
- acc_heterogeneous_10m_0mmh_dt1 / physical_candidate: final wet. No acceptance relaxation applied.
- acc_complex_5m_0mmh_dt1 / physical_candidate: final wet, temporal depth, temporal wet. No acceptance relaxation applied.
- acc_complex_5m_0mmh_dt1 / lisflood_fp_acc: final wet. No acceptance relaxation applied.
- acc_heterogeneous_10m_1mmh_dt0.5 / physical_candidate: final wet. No acceptance relaxation applied.
- acc_heterogeneous_10m_1mmh_dt0.25 / physical_candidate: final wet. No acceptance relaxation applied.

Physical LI deliberately differs from ACC in source ordering and available-water infiltration, dry-depth threshold, centred resistance and exterior vector friction. Those differences matter in shallow films and wet/dry extent; small depth RMSE does not erase a failed wet gate. The 5 m ACC-compatible complex case also fails its wet gate and is not claimed exact parity.

The independent FREE comparisons explicitly choose local_or_prescribed_slope for LI and native local-slope FREE for ACC. Gravity and CFL are explicitly matched to the reference (9.80665 m/s², 0.4); this benchmark configuration does not change library defaults. Native 1 mm censoring and step-start rainfall sampling remain documented differences. ACC time-dependent depth fields are sampled at 60 s; native initial index 0 is omitted because it follows the first timestep.

## FV1 boundary versus interior

At CFL 0.05, the retained 100×100 two-second outlet case exports 75.778282 m³ versus native 77.185587 m³. Depth RMSE is 0.00209012 m. These values pass the declared field/export gates; they are not state equality.

The three-cell x-boundary band RMSE is 0.00797717 m and the remaining x-interior RMSE is 0.000765263 m. For thin Riemann strips this band is defined along propagation, not as a full 2D exclusion of transverse walls. These are spatial diagnostics, not a unique causal decomposition.

FloodWarp preserves reflecting walls, finite-velocity stage extrapolation, contact-upwind tangential transport and endpoint wave bounds. Native FV1 retains its independent embedded-stage/inactive-halo treatment and flux operators. The external halo is outside the physical 100×100 limit. No external hydraulic code was copied. Historical discharge/legacy compatibility and rejection of incompatible checkpoint identities are covered separately by the regression suite.

## Sensitivity and coverage limits

[Sensitivity results](sensitivity.json) compare ACC complex-terrain grids at 20/10/5 m using conservative averaging of simultaneous final depths. Both LI profiles and independent native outputs are included. Two matched heterogeneous-terrain timestep caps (0.5/0.25 s) show sensitivity, not a convergence order. FV1 compares CFL 0.4/0.2/0.1 against 0.05 for each of the four controls, separately for FloodWarp and native.

Unresolved: independent FV1 grid-refinement fixtures, resolved temporal depth fields and instantaneous outlet hydrographs are not included. Its outlet comparison measures cumulative export and event-mean discharge, not a resolved hydrograph. Riemann controls have no meaningful outlet flow. The retained crop is a two-second operator check, not an event-scale validation. A complete independent FV1 grid/time/outlet qualification remains a release limitation.

## Reproduce

```bash
python -m validation.compare
python -m validation.sensitivity
```

The scripts default to CPU and use only packaged fixtures. Their execution guards hold adjustable paths and devices. Results are written under ignored outputs/. Reference checksums are verified before evaluation. [Fixture manifest](fixtures/manifest.json) records source commit, original files, binary identities, controls and hashes. Reference arrays were never regenerated from FloodWarp.

Extraction QA caught a CPU snapshot-view alias in the new output collector; collectors now copy snapshots, and a regression verifies distinct sampled depths. Metrics above use the corrected snapshots. No numerical operator or acceptance criterion changed.
