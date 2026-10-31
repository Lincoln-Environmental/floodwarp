# FloodWarp v0.1.1 — exact-sparse FV1 default

FV1 now defaults to validated exact-zero sparse execution with 8×8 tiles and
`dense_fraction=0.65`. It uses the same uniform-grid FP64 hydraulic formulation,
source/soil updates, full-domain source/boundary-aware CFL and conservative
accounting. Select `execution_mode='dense'` for the retained dense reference.
Checkpoints are portable in both directions.

The complete 72-hour Orton Bradley Park replay used **1641.957 s hydraulic
time versus 1906.106 s dense**, a **13.9% reduction**, with **1,570,053 accepted
steps** and zero differences at all 72 recorded hourly field comparisons.
This was one candidate run against a retained same-device dense benchmark,
not multiple interleaved event repetitions. Widespread inundation can make
sparse execution slower. The dense fallback remains available.

The standalone example now reaches its exact requested final time for short
or non-divisible sampling periods and always saves final state/diagnostics.
It starts fresh; checkpoint restoration is explicit, not automatic.

Read the **[developer disclaimer](https://github.com/Lincoln-Environmental/floodwarp/blob/v0.1.1/DISCLAIMER.md)**
and [release verification report](https://github.com/Lincoln-Environmental/floodwarp/blob/v0.1.1/validation/release_v0.1.1/REPORT.md).
**Research demonstration only — not an official flood-hazard assessment.
Independent site-specific validation is required before decision-making.**
This guidance adds no AGPL restriction. Software copyright Patrick Durney and
Lincoln Agritech Ltd; AGPL-3.0-only. Elevation data retain CC BY 4.0 attribution.

The complete **1123×843, 5 m Orton Bradley Park dataset** is unchanged:
EPSG:2193, NZVD2016 and 559,985 active cells. Use the versioned v0.1.0 assets:

- [Complete DEM/results/reproduction archive](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_v0.1.0.tar.gz)
- [Full DEM GeoTIFF](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_dem_5m.tif)
- [FloodWarp FV1 maximum-depth GeoTIFF](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_floodwarp_fv1_max_depth_5m.tif)
- [All aligned maximum-depth, difference and extent products](https://github.com/Lincoln-Environmental/floodwarp/releases/tag/v0.1.0)
- [Original data checksums](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/SHA256SUMS)

All original assets and archived source/result files were checksum-verified;
independent reproduction passed without a hydraulic rerun. The original tag
and assets are preserved. The accompanying [geospatial/figure notice](https://github.com/Lincoln-Environmental/floodwarp/blob/v0.1.1/examples/orton_bradley/OUTPUT_NOTICE.md)
applies to interpretation of these historical products.

Release checks: full CPU suite **965 passed**; full CPU/CUDA suite **1609 passed**;
focused installed-wheel controls **75 passed**. The actual wheel was installed
outside the checkout in a fresh Python 3.14 environment. Static-mesh modules
are absent. SGS remains experimental; static/dynamic AMR, wavelets and
mixed-precision FV1 are excluded from this release and remain separate on dev.

Visibility remains private until the owner changes it. Authenticated access
passed; anonymous repository/data requests currently return 404. Independent
legal or institutional approval of the developer disclaimer is not claimed.
