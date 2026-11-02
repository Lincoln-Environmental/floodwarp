# Attribution and provenance

FloodWarp's hydraulic code is extracted from Lincoln-Environmental/Overlander
at `ba503b8bde51d3f212199e81efc7038d9fffecc8`. Copyright Lincoln Agritech Ltd. Developed by Patrick Durney. The rights holders authorised release under
**AGPL-3.0-only**; see LICENSE and COPYRIGHT. No prior Git history
or archived research branches are copied. Per-module source hashes and
extraction changes are in `docs/extraction.json` and `docs/numerical_integrity.json`.

## Orton Bradley Park terrain (historical dataset identifier: Purau)

Contains data sourced from the LINZ Data Service licensed for reuse under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Follow the [LINZ elevation attribution guidance](https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-elevation-or-aerial-imagery-data).

Credit Environment Canterbury, Christchurch City Council where applicable,
Aerial Surveys, Landpro, AAM NZ, and Toitū Te Whenua Land Information New Zealand.
The local derivative lists multiple source surveys; all their official STAC
collection licences were checked as CC-BY-4.0. Exact catalogue URLs, provider
roles, source hashes and modifications are in `examples/purau/provenance.json`.
The derivative cannot be attributed to a single survey at pixel level.
It had already been hydraulically conditioned. The full-domain release exports
the authoritative existing 5 m elevations unchanged, with the physical active
mask. The small installation example crops and reorients the same model.
Attribution does not imply endorsement.

Horizontal coordinates are NZTM2000 / EPSG:2193; elevations are metres NZVD2016,
as confirmed by the model owner for this release. No datum conversion or
vertical adjustment was applied. Original metadata and internal dataset names
remain preserved as historical provenance. The small crop uses FREE boundaries;
the full matched benchmark uses its established stage-0 coastal contacts.

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

## Leaflet and OpenStreetMap (development local interface)

Leaflet 1.9.4 is vendored in `src/floodwarp/ui_assets/leaflet.js` and
`leaflet.css`. Copyright (c) 2010–2023 Volodymyr Agafonkin and
Copyright (c) 2010–2011 CloudMade. BSD-2-Clause terms are retained verbatim in
`src/floodwarp/ui_assets/leaflet-LICENSE.txt`. Upstream:
https://github.com/Leaflet/Leaflet/tree/v1.9.4 .

The optional online basemap uses OpenStreetMap tiles. © OpenStreetMap
contributors; data are available under the Open Database License. Visible
attribution links to https://www.openstreetmap.org/copyright . Tile service
use follows https://operations.osmfoundation.org/policies/tiles/ . Basemap
images are display context; they do not supply hydraulic terrain or forcing.
The documented map screenshot retains visible OpenStreetMap attribution.
