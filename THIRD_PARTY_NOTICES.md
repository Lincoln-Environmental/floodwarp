# Attribution and provenance

FloodWarp's hydraulic code is extracted from Lincoln-Environmental/Overlander
at `ba503b8bde51d3f212199e81efc7038d9fffecc8`. Author and contributor rights
must be confirmed before selecting the software licence. No prior Git history
or archived research branches are copied. Per-module source hashes and
extraction changes are in `docs/extraction.json` and `docs/numerical_integrity.json`.

## Purau terrain

Contains data sourced from the LINZ Data Service licensed for reuse under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Follow the [LINZ elevation attribution guidance](https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-elevation-or-aerial-imagery-data).

Credit Environment Canterbury, Christchurch City Council where applicable,
Aerial Surveys, Landpro, AAM NZ, and Toitū Te Whenua Land Information New Zealand.
The local derivative lists multiple source surveys; all their official STAC
collection licences were checked as CC-BY-4.0. Exact catalogue URLs, provider
roles, source hashes and modifications are in `examples/purau/provenance.json`.
The derivative cannot be attributed to a single survey at pixel level.
It had already been hydraulically conditioned. This extraction only crops and
reorients its arrays. Attribution does not imply endorsement.

Official Banks Peninsula surveys use NZVD2016; the local derivative omitted its
vertical datum, so that identity remains unverified for this crop. Elevations
are retained without conversion, and the demonstration uses FREE boundaries.

## Independent numerical references

Validation arrays are outputs of independent LISFLOOD-FP ACC/FV1 runs, not
FloodWarp-generated targets. `validation/fixtures/manifest.json` records their
source paths, binary hashes, version, controls and file hashes. No LISFLOOD
source or executable is included, linked into the package, or required to run
its tests. The source project's inspected upstream distribution did not
establish the main LISFLOOD software licence; do not infer executable/source
redistribution rights from these comparisons.

## Dependencies

NumPy, SciPy, Numba, Warp, optional Rasterio and Matplotlib are installed as
separate distributions. Their licences and notices remain in those installed
distributions; their source trees and binaries are not vendored here.
