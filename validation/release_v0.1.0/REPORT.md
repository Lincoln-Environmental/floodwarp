# v0.1.0 release audit

The rights holders authorised AGPL-3.0-only licensing and Patrick Durney /
Lincoln Agritech Ltd attribution. Repository visibility remains controlled by
the owner and is not changed by this work. The release branch excludes the
separate sparse/AMR research; its numerical operator files match `183c884`.

## Installation and numerical checks

- Fresh isolated Python 3.14.7 environment; pinned NumPy 2.3.5, SciPy 1.16.3,
  Numba 0.67.0 and Warp 1.17.0. Optional versions are recorded separately.
- Wheel and sdist built; package metadata identifies v0.1.0, both credited
  authors, AGPL-3.0-only and complete licence/notice files. `pip check` passes.
- Required CPU suite: **923 passed, 603 skipped**, 350 existing warnings,
  91.54 s. CUDA-dependent skips are not claimed as CPU coverage.
- Available opt-in FV1 CUDA and product controls: **36 passed**. Final focused
  raster/metadata/hydrograph/extent controls: **4 passed**.
- Synthetic FV1/LI and the separately labelled 200 m crop complete in the clean
  environment. Crop residuals are approximately 6.37e-12 m³ (LI) and
  −1.78e-15 m³ (FV1).
- Both full-domain input configurations load and advance for a one-second
  CPU smoke check. No 72-hour simulation was rerun for this release.

## Original-data and GIS verification

The conditioned original DEM, benchmark packet, original active mask, ASCII
grid header and original model metadata establish the exact 1123×843 grid,
5 m spacing, EPSG:2193 and bounds (1574940, 5161660, 1579155, 5167275) m.
All 559,985 active elevations agree exactly. NZVD2016 identity was confirmed
by the model owner; no elevation adjustment or resampling occurred.

Raster footprint is 23.667225 km²; physical active area is 13.999625 km².
Geographical catchment coverage is not inferred. Original DEM valid-data
coverage is broader than the physical hydraulic mask: the exported DEM uses
the independently verified active mask, not an elevation threshold.

All release elevation/depth/difference GeoTIFFs are losslessly compressed FP64,
with numeric dry active zeros and masked outside-domain nodata −9999. Byte
extent/class rasters use nodata 255. Band units, NZVD2016 metadata, exact
transform, mask, orientation, coordinates and every active pixel are checked.
Conversion error is zero. `product_verification.json` independently compares
rasters to original retained arrays and every CSV value to cumulative records.

The standalone archive is extracted outside the repository, its complete
checksum manifest verified, and all figures/rasters/tables regenerated without
reading Overlander or another private working directory. Raw historical
records retain informational source paths and hashes; reproduction never
dereferences those paths. The archive includes scripts, input/result data,
provenance, attribution, raster products, CSV, figures and licences.

## Hydraulic interpretation and presentation

The original completed record payloads and field hashes verify all three
72-hour simulations. The minute hydrograph has 4,320 common intervals. The
external extra 1-second diagnostic is excluded from minute differencing;
cumulative minute-end samples define actual 60 s means.

Recomputed FV1 maximum-envelope RMSE is 0.001434806 m, maximum absolute
difference 0.119908348 m, maximum-event IoU 0.998711452 at 0.01 m and
0.995776095 at 0.10 m. Final depth and event maxima are distinct products.
Runtime, peak flow and accepted-step totals reproduce the retained final table.

The LI maximum/final arrays additionally match its complete final checkpoint
exactly; bed and mask align. Median maximum-envelope difference is 0.00281185 m,
95th percentile 0.041966 m; 97.34% of active cells differ by less than 0.10 m.
Localized channel discrepancies reach 6.243813 m and warrant investigation;
their cause is not established here. `li_envelope_audit.json` records the
independent checkpoint check. A log-scaled error plot overstated the visual
prevalence of small differences and was replaced. Both FV1 and LI signed-error
maps now use full-range, zero-centred **linear** colour scales, without clipping.
Depth panels retain explicitly labelled common logarithmic scales.

The release is research-grade. The matched FV1 benchmark used explicitly
discharge-extrapolated prescribed-stage momentum; default velocity extrapolation
does not acquire universal qualification from it. LI approximation errors,
SGS experimental status and historical failed strict gates remain documented.

## Publication

The v0.1.0 GitHub release distributes the full data archive, individual GIS
products, hydrograph CSV, comparison/provenance files, wheel, sdist and external
asset checksums. The owner will change visibility to public separately.
Versioned release links require authenticated access until that change.
