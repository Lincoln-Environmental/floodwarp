# Data attribution

The conditioned elevation derivative contains data sourced from the LINZ Data
Service, licensed for reuse under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit **Environment Canterbury; Christchurch City Council where applicable;
Aerial Surveys; Landpro; AAM NZ; and Toitū Te Whenua Land Information New Zealand**.
Follow [LINZ elevation attribution guidance](https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-elevation-or-aerial-imagery-data).
Attribution does not imply endorsement.

The historical model metadata lists multiple surveys. Exact collection URLs,
provider roles, source/content hashes and original hydraulic-conditioning history
are preserved in provenance.json. Conditioning, aggregation to 5 m and model
mask selection predate this release. The release applies no vertical adjustment,
resampling or new terrain conditioning. Pixel-level attribution to a single
survey is not claimed. The model owner confirmed Orton Bradley Park / NZVD2016
identity for the release; historical dataset identifiers remain unchanged.

Rainfall, roughness and infiltration inputs are retained numerical model fields,
with their original source descriptions and modelling assumptions documented in
provenance.json. Original source rainfall grids, land-cover and soil vector
datasets are not bundled. The numerical outputs are products of the credited
FloodWarp and independent LISFLOOD-FP runs, not observed flood measurements.

FloodWarp software and reproduction scripts: copyright Patrick Durney and
Lincoln Agritech Ltd, **AGPL-3.0-only**. See LICENSE, COPYRIGHT and
THIRD_PARTY_NOTICES.md for software, dependencies and independent reference
provenance. AGPL software terms do not replace the separate CC BY 4.0 elevation
attribution requirements.
