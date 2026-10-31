# Changelog

## Unreleased — dev

- Default FV1 execution now uses validated exact-zero 8×8 sparse scheduling
  with dense FP64 storage and the existing uniform-grid numerical methods.
  The matched 72-hour Orton Bradley event retained all hourly fields exactly
  and reduced hydraulic runtime by 13.9% on RTX 4070 Ti SUPER. High-occupancy
  windows can be slower; `execution_mode='dense'` selects the retained reference.
- CPU/CUDA native and factory defaults, dense opt-in and cross-mode restart
  are covered by regressions. Static multiresolution remains experimental.

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
