# FloodWarp v0.1.1 release status

Current main includes the subsequently verified modified sparse FV1 default
(1512.372 s matched-event hydraulic runtime; identical recorded state/history).
This branch promotion does not alter the published v0.1.1 tag or assets.
The following v0.1.1 release record remains historical.

The rights holders authorised this research-grade release under **GNU AGPL
version 3 only**, credited to **Patrick Durney and Lincoln Agritech Ltd**.
LICENSE contains the complete text; package metadata uses `AGPL-3.0-only`.
Separate LINZ elevation CC BY 4.0 credits and dependency notices remain in
THIRD_PARTY_NOTICES.md. The owner controls repository visibility; release
preparation does not change that setting.

FV1 is an independent FP64 finite-volume full shallow-water solver. v0.1.1
defaults to exact-zero sparse uniform execution with 8×8 tiles and a 0.65
dense-indexing fallback fraction. `execution_mode='dense'` retains the dense
reference. This changes scheduling only: dense FP64 state, operators, sources,
full-domain CFL, conservative transport, maxima and restart identity are retained.
LI is a reduced-complexity alternative. Both SGS modes remain experimental.
Static/dynamic AMR, wavelets, mixed-precision FV1, sparse storage and additional
performance research are excluded from v0.1.1 and belong separately on dev.

The [complete exact-sparse event](../validation/sparse_review/FULL_EVENT.md)
retained 1,570,053 steps and zero differences at all recorded hourly field
comparisons. Hydraulic time was 1641.957 s versus 1906.106 s dense (13.9% less),
one candidate event against the retained same-device dense run. It was not a
series of interleaved event repetitions. Widespread-inundation windows can be
slower. Explicit dense mode and the high-occupancy fallback remain available.

The [developer disclaimer](../DISCLAIMER.md) accompanies the README, built
distributions, release notes and [geospatial/figure notice](../examples/orton_bradley/OUTPUT_NOTICE.md).
It adds no AGPL restriction and preserves rights or liabilities that cannot
lawfully be excluded. The developer confirmed proceeding within the private
repository with their own disclaimer; independent legal or institutional
approval is not claimed. The owner retains control over public visibility.
Anonymous installation and asset access cannot be claimed while it is private.

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

Historical audit records in `validation/release_v0.1.0` cover clean installation,
synthetic LI/FV1 and crop execution, CPU regressions, available CUDA controls,
independent archive reproduction, raster grid/value/mask checks, package
metadata and checksums. Installation tests never run the full event. A minimal
CPU GitHub workflow is included. Versioned release assets require authentication
while the owner retains private visibility.

No prepared-model package, external solver executable, other HydroOne runtime,
coupled interface, routing, surrogate, campaign or temporal-subcycling dependency
is bundled. Numerical defaults, operator arithmetic and reference fixtures are
unchanged by the release work. The fixed v0.1.0 tag and its 26 versioned release
assets are preserved. v0.1.1 links the original datasets rather than replacing
them. Current release checks are recorded separately in `validation/release_v0.1.1`.

The standalone example replay includes the exact requested final time and final
checkpoint for durations shorter than or not divisible by its output interval.
It starts fresh, refuses a nonempty output directory and requires explicit
checkpoint restoration; it does not automatically resume.
