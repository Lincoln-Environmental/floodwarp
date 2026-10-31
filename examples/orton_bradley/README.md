# Orton Bradley Park — complete 5 m model demonstration

**Research demonstration only — not an official flood-hazard assessment.
Independent site-specific validation is required before decision-making.**
See the [software disclaimer](../../DISCLAIMER.md) and
[notice accompanying all geospatial products and figures](OUTPUT_NOTICE.md).

Orton Bradley Park, Banks Peninsula, Canterbury, New Zealand. This is the
**entire established model raster**, 1123 north–south rows × 843 east–west
columns at 5 m: 23.667225 km² raster footprint, 559,985 active hydraulic cells
and 13.999625 km² active area. The geographical catchment may extend beyond
the model; whole-catchment coverage is not asserted.

Coordinates are **NZTM2000 / EPSG:2193**; elevations are **metres NZVD2016**.
Bounds are west 1574940, south 5161660, east 1579155, north 5167275 m. The
GeoTIFF upper-left origin is (1574940, 5167275), with 5 m pixels. The owner
confirmed the datum/site identity for this release; original internal names
`purau_event_72h` and `purau_point_1575902_5166615_catchment_model` remain in
immutable historical records. No numerical history or elevation was renamed
by changing its values. The separate [200 m installation crop](../purau/README.md)
is a quick test, not this full model.

## Download and regenerate

Download the [complete v0.1.0 data archive](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_v0.1.0.tar.gz)
and [asset checksums](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/SHA256SUMS).
Repository visibility is controlled by the owner; authenticated access is needed
until the owner makes the repository public.

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf orton_bradley_v0.1.0.tar.gz
cd orton_bradley_v0.1.0
sha256sum -c SHA256SUMS
python -m pip install numpy==2.3.5 rasterio matplotlib
python reproduce.py
```

No solver run, external executable, original working directory or Overlander
installation is needed. The script verifies original source and result hashes,
the complete domain, original DEM/mask alignment and the completed record
payloads, then recreates `outputs/` and `figures/`. All editable controls are
under its execution guard. It writes a numerical comparison JSON/Markdown,
raster manifest, orientation/coordinate checks and the 72-hour CSV.

The archive contains original ordinary benchmark input files, the original
conditioned DEM and physical mask, complete checksummed FP64 final/maximum
arrays and diagnostic records for all three solvers, provenance, licences,
scripts, aligned GeoTIFFs, CSV and figures. It is independently usable outside
the repository. The numerical result arrays are south-first; the original
terrain and GeoTIFFs are north-first. The explicit export orientation conversion
is verified against original georeferencing; no interpolation or resampling occurs.

## GIS products

The local example directory is `examples/orton_bradley/outputs/`. Its 14 aligned
GeoTIFFs, hydrographs, boundaries, provenance and checksums are supplied in the
unchanged v0.1.0 release archive. Generated data are Git-ignored; copy the
archive's `outputs/` contents here for a fresh checkout. Keep its complete
`inputs/` and `results/` alongside them when regenerating figures and tables.
v0.1.1 links these immutable products without rewriting their values or metadata;
OUTPUT_NOTICE.md accompanies their interpretation.

Individual files are also available on the [release page](https://github.com/Lincoln-Environmental/floodwarp/releases/tag/v0.1.0):

- [Full conditioned DEM](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_dem_5m.tif).
- [FloodWarp FV1 maximum depth](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_floodwarp_fv1_max_depth_5m.tif).
- [LISFLOOD-FP FV1 maximum depth](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_lisflood_fv1_max_depth_5m.tif).
- [Signed FV1 minus LISFLOOD maximum-depth difference](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_fv1_minus_lisflood_max_depth_5m.tif).
- [FloodWarp LI maximum depth](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_floodwarp_li_max_depth_5m.tif).
- [Signed LI minus LISFLOOD maximum-depth difference](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_li_minus_lisflood_max_depth_5m.tif).
- [FV1 maximum extent ≥0.01 m](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_fv1_max_extent_001m.tif) and [≥0.10 m](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_fv1_max_extent_010m.tif).
- Active mask, coastal outlet mask, corresponding reference extents, extent
  classifications, projected boundary GeoJSON, boundary-face arrays and
  [60-second hydrograph CSV](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_hydrographs_72h.csv).

Every raster has the exact DEM grid and physical mask. Depth/elevation/difference
values use Float64 with lossless DEFLATE compression, nodata −9999 and explicit
dataset masks. Dry active depth is numeric zero. Class rasters use Byte, nodata
255; extent 0=dry and 1=inundated. Combined extent classes are 0=both dry,
1=both inundated, 2=FloodWarp only, 3=LISFLOOD only. Thresholds apply to each
cell's **maximum depth over the event**, not its final depth. Coast GeoJSON
coordinates are projected EPSG:2193, explicitly labelled in the file.

Δh_max = max_t h_FloodWarpFV1 − max_t h_LISFLOODFV1. It compares two
maximum-depth envelopes; it is not max_t of their instantaneous difference.
Plots use a shared logarithmic depth scale and a zero-centred linear signed
error scale spanning the actual extrema. No error is clipped. Pointwise event
maxima are not simultaneous water storage.

## Interpretation and controls

See [COMPARISON.md](COMPARISON.md) for active-domain RMSE, wet-union errors,
maximum-event IoU, false-positive/negative areas, export/infiltration differences,
independent peak magnitude/timing errors and water balances. Full machine-readable
metrics and benchmark signatures are in the archive.

The 72-hour event uses the established 48-hour HIRDS v4 100-year historical
design rainfall and 24-hour recession, spatial rainfall multipliers, Manning
roughness and exponential f0/fd/k infiltration with finite water availability.
There are 212 prescribed stage-0 north-facing coastal contacts; other domain
boundaries are closed. These are model assumptions, not a calibrated forecast.

The matched FloodWarp FV1 run explicitly uses **discharge-extrapolated stage
momentum**, contact HLL, FP64, CFL 0.4, gravity 9.80665 m/s², dry momentum depth
0.0001 m and maximum dt 1 s. The ordinary public FV1 default uses velocity
extrapolation; this benchmark does not qualify every boundary closure. LI is a
reduced-complexity alternative with material maximum-inundation errors here.
External LISFLOOD-FP is an independent validation reference, not a runtime
dependency or bundled implementation.

`run_model.py` can optionally rerun FV1 or LI from the published ordinary arrays
after installing FloodWarp. Its default is the substantial full 72-hour FV1
simulation on CUDA, with minute diagnostics and hourly checkpoints. It is
deliberately separate from `reproduce.py` and normal installation tests.

v0.1.1 uses exact-sparse scheduling for FV1; the original v0.1.0 demonstration
remains the dense numerical reference. The replay reaches the exact requested
end time even if its duration is shorter than or not divisible by the output
interval, and always saves the final checkpoint and diagnostic ledger.

The standalone replay **starts fresh and does not resume automatically**.
It refuses a nonempty output directory. To recover an interrupted simulation,
explicitly restore its checkpoint with `floodwarp.model.load_checkpoint` into
an identically configured model, including sources and soil parameters, before
calling `advance` with the desired absolute end time. The native FV1 checkpoint
is portable between sparse and dense execution; changing scheduling does not
change restart identity. The standalone script has no resume option.

Elevation data credit and licence: [DATA_LICENSE.md](DATA_LICENSE.md).
Software copyright Patrick Durney and Lincoln Agritech Ltd, AGPL-3.0-only.
