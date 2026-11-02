# Changelog

## v0.1.3 — spatial local UI on main

- Add aligned spatial Manning roughness, constant-capacity and exponential
  infiltration maps using existing FV1 fields and setters.
- Add explicit Closed/FREE/constant-stage face CSVs with absolute datum checks,
  templates and visible coastal contacts. Reject ambiguous surplus CSV values.
- Load and preview the complete original Orton Bradley setup when available;
  preserve exact FP64 inputs, all 212 coastal contacts and explicit matched
  numerical settings. Never launch its 72-hour simulation automatically.
- Add bundled Leaflet and streamed OpenStreetMap with local terrain, maximum
  and final-depth overlays. Map reprojection is display-only.
- Expose native finite-support rainfall continuation explicitly while preserving
  the default coverage guard, restart identity and all hydraulic operators.
- Separate checkpoint cadence from diagnostics for practical full-domain IO.
  Keep original v0.1.2 artifacts and all historical evidence unchanged.

## v0.1.2 — local publication candidate

- Add an optional localhost rainfall UI (`python -m floodwarp.ui`) with real
  worker-process FV1 execution, domain/rainfall previews, exact final sampling,
  atomic checkpoints, georeferenced outputs and checksummed run provenance.
- Default high-level factories to FV1 deliberately; retain explicit LI/native
  construction, sparse FV1 default and explicit dense execution. SGS selects LI.
- Share exact-final-time sampling in small public examples and the UI runner;
  the downloadable full-domain replay keeps its standalone equivalent.
- Intersect binary domains with valid DEM cells. Add installed synthetic UI
  inputs, optional geospatial dependencies and built-wheel installation CI.
- Transfer historical sparse-performance drivers, snapshots, profiles, memory
  records and publication verification from dev without static-mesh research.
- Promote the full-event-verified modified sparse FV1 default: released-tile
  scratch clearing and fused post-source activity detection. All recorded
  hydraulic fields and accepted steps match dense and prior sparse execution.
  Hydraulic runtime is 1512.372 s, 7.89% below prior sparse and 20.66% below dense.
  Dense override remains available. Existing release tags/assets are unchanged.

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
