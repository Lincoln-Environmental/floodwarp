# Changelog

## 0.2.0.dev0 — development only

- Complete matched 72-hour verification of the modified sparse path: 1512.372 s
  hydraulic time versus 1641.957 s released sparse and 1906.106 s dense;
  all hourly fields, accepted steps and final released-sparse checkpoint arrays agree.
  Timing is one new event against retained controls, not interleaved repetitions.
- Larger static controls and curved-terrain reconstruction diagnostics retain
  the original failed small-case evidence. Broad 512×512 controls improve
  accuracy/runtime cost; the protected channel and real-terrain representation
  remain limitations. Affine free-surface screening is diagnostic only.
  Static methods remain experimental and dynamic AMR is not enabled.
- Retain independently measured released-tile scratch clearing and fused
  source/activity detection. Six interleaved saved-state phases preserve all
  fields and accepted steps; wet-phase hydraulic time falls about 7–8% versus
  v0.1.1 sparse execution. Short dry replay is slightly slower. No new full-event
  speedup was claimed at that stage; subsequent full-event verification is recorded above.
  Forced early dense indexing remains rejected.
- Merge v0.1.1 release fixes and formulation-interpretation guidance forward,
  preserving experimental static-mesh evidence and the production sparse FV1
  default. New performance and adaptive-resolution research remains on dev.
- Agent instruction files remain ignored and local; production release tags
  and immutable benchmark datasets are preserved.

## v0.1.1

- Exact-zero 8×8 sparse execution is now the FV1 default, with dense FP64
  storage, the existing uniform-grid numerical formulation and explicit
  `execution_mode='dense'` reference/fallback. Full-domain sources and CFL,
  conservative transport, maxima and restart identity are preserved.
- The complete matched 72-hour event retained 1,570,053 steps and all recorded
  hourly fields. Hydraulic time was 1641.957 s versus 1906.106 s dense on the
  same RTX 4070 Ti SUPER, a 13.9% reduction. This is one candidate event against
  a retained dense run; high-occupancy windows can be slower.
- Research-software disclaimer and responsible-use guidance, accompanying
  geospatial/figure notice and inclusion in built distributions. AGPL-3.0-only
  and separate elevation-data terms are unchanged.
- Standalone replay includes the exact final time for short and non-divisible
  durations, saves the final checkpoint and refuses to overwrite a prior run.
  Checkpoint restoration remains explicit; no automatic-resume claim is made.
- Original v0.1.0 tag, data assets, numerical arrays and source snapshots are
  preserved. Static/dynamic AMR, wavelets and mixed-precision FV1 are excluded.
- Agent instruction files remain local and are excluded from tracked release
  contents and distributions.

## v0.1.0

Research-grade standalone CPU/CUDA LI and FP64 full shallow-water FV1, copyright
Patrick Durney and Lincoln Agritech Ltd, authorised under AGPL-3.0-only.
Traditional and enhanced SGS remain experimental.

- Complete 5 m Orton Bradley Park conditioned DEM, EPSG:2193 / NZVD2016,
  separate from the 200 m installation crop.
- Verified existing 72-hour results; FP64 maximum-depth/difference GeoTIFFs,
  maximum-event extents, boundaries, minute hydrographs and reproduction archive.
- Licence/authorship metadata, CC BY 4.0 credits, CPU CI and clean release tests.
- Numerical operators and frozen references unchanged. Sparse FV1, AMR and
  further optimisation deferred to research.

The owner controls the separate change of repository visibility to public.
