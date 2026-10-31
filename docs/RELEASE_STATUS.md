# FloodWarp v0.1.0 release status

The rights holders authorised this research-grade release under **GNU AGPL
version 3 only**, credited to **Patrick Durney and Lincoln Agritech Ltd**.
LICENSE contains the complete text; package metadata uses `AGPL-3.0-only`.
Separate LINZ elevation CC BY 4.0 credits and dependency notices remain in
THIRD_PARTY_NOTICES.md. The owner controls repository visibility; release
preparation does not change that setting.

FV1 is an independent FP64 finite-volume full shallow-water solver. LI is a
reduced-complexity alternative. Both SGS modes remain experimental with visible
warnings. Sparse FV1, AMR, mixed-precision FV1 and further optimisation are
future research, excluded from v0.1.0.

The complete Orton Bradley Park 1123×843, 5 m demonstration has 559,985 active
cells, EPSG:2193 and NZVD2016. Authoritative conditioned elevations are exported
unchanged, using the physical mask. Completed, checksummed 72-hour FV1/LI/external
FV1 runs supply aligned FP64 maximum-depth GeoTIFFs, signed differences,
maximum-event extents at 0.01/0.10 m, minute hydrographs and standalone scripts.
See [the guide](../examples/orton_bradley/README.md) and
[recomputed comparison](../examples/orton_bradley/COMPARISON.md).

The matched FV1 benchmark uses discharge-extrapolated stage ghosts; public
FV1 defaults use velocity extrapolation. Evidence applies to matched settings,
not all hydraulic cases or boundary laws. LI retains material maximum-event
inundation discrepancies. Soil/roughness mappings need site calibration.
Raster footprint, active area and geographical catchment are distinct.

The [historical repair assessment](../validation/repair_review/FINAL.md)
retains analytical, CPU/CUDA, graph, restart and independent controls, including
seven failed LI strict gates out of 32 matched cases. Release does not turn
failures into passes. Original extraction/numerical hashes remain in
`extraction.json` and `numerical_integrity.json`. Extraction baseline:
`Overlander@ba503b8bde51d3f212199e81efc7038d9fffecc8`. Completed evidence anchor:
`183c884`, with exact runtime module/binary hashes in each immutable result record.

Release audit records in `validation/release_v0.1.0` cover clean installation,
synthetic LI/FV1 and crop execution, CPU regressions, available CUDA controls,
independent archive reproduction, raster grid/value/mask checks, package
metadata and checksums. Installation tests never run the full event. A minimal
CPU GitHub workflow is included. Versioned release assets require authentication
while the owner retains private visibility.

No prepared-model package, external solver executable, other HydroOne runtime,
coupled interface, routing, surrogate, campaign or temporal-subcycling dependency
is bundled. Numerical defaults, operator arithmetic and reference fixtures are
unchanged by the release work.
