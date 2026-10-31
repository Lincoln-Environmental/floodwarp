# FloodWarp

**FloodWarp v0.1.0** is research-grade surface-water simulation software for
CPU and NVIDIA CUDA. Copyright **Patrick Durney and Lincoln Agritech Ltd**.
Software is **[GNU AGPL-3.0-only](LICENSE)**; attributed elevation data retain
CC BY 4.0. FV1 is an independent FP64 first-order finite-volume full
shallow-water solver. LI is a reduced-complexity local-inertial alternative.
Traditional and enhanced SGS modes remain explicitly experimental.
Independent benchmark and analytical evidence do not establish universal
hydraulic qualification.

## Install and quick start

Use Python 3.14 and a fresh environment, from a checkout of this release:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ".[test,raster,plot]"
python -m pip check
python -m examples.run_synthetic
python -m examples.run_purau
python -m pytest -q
```

`pip install .` installs the CPU core. Numerical dependencies are pinned:
NumPy 2.3.5, SciPy 1.16.3, Numba 0.67.0 and Warp 1.17.0. Rasterio adds GeoTIFF
IO; Matplotlib adds figures. Set `device='cuda:0'` for compatible NVIDIA hardware
and drivers; ordinary wheel use does not need a system CUDA toolkit. Release
validation covers Linux x86-64 CPU and RTX 4070 Ti SUPER. Other platforms require
compatible Python 3.14 wheels and local validation. No Overlander/prepared-model
installation or external solver executable is required.

Use `solver_method='fv1'` or `'li'` with `create_model`. FV1 defaults to contact
HLL, velocity-extrapolated stages and FP64; LI defaults to the physical profile.
The [input guide](docs/INPUTS.md) covers arrays, rainfall, infiltration, boundaries,
raster datum/mask handling and restart. [run_direct.py](examples/run_direct.py)
is an editable GeoTIFF example. The historical `run_purau` command is a separate
200 m installation crop, not the full-domain demonstration below.

On `dev`, FV1 now defaults to exact-zero sparse execution with 8×8 tiles.
The complete 72-hour Orton Bradley replay matched every retained hourly dense
field and used **1641.96 s hydraulic time versus 1906.11 s** on the same GPU
(13.9% less runtime). This preserves the uniform grid, dense FP64 state,
sources and CFL policy. Widespread-runoff windows can still be slower.
Pass `execution_mode='dense'` when constructing an FV1 solver/model for the
retained dense path. See [validation and limits](validation/sparse_review/DEFAULT_PROMOTION.md).
The fixed v0.1.0 release and its benchmark results below are unchanged;
static AMR remains experimental.

## Orton Bradley Park — complete 5 m model domain

**Orton Bradley Park, Banks Peninsula, Canterbury, New Zealand**: the entire
established **1123×843 raster at 5 m**, with **559,985 active cells**. Raster
footprint is 23.667225 km²; active hydraulic area is 13.999625 km². These are
model-domain measurements, not a claim of complete geographical catchment
coverage. Coordinates are NZTM2000 / EPSG:2193; elevations are metres NZVD2016.

![Orton Bradley Park — 5 m model terrain](examples/orton_bradley/figures/orton_bradley_dem.png)

Download the [complete data and reproduction archive](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_v0.1.0.tar.gz),
[full DEM GeoTIFF](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_dem_5m.tif),
[FV1 maximum-depth GeoTIFF](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/orton_bradley_floodwarp_fv1_max_depth_5m.tif)
and [checksums](https://github.com/Lincoln-Environmental/floodwarp/releases/download/v0.1.0/SHA256SUMS).
The [demonstration guide](examples/orton_bradley/README.md) links reference,
LI, signed-difference, extent and hydrograph products. Assets need authenticated
access until the owner changes repository visibility to public.

## Maximum inundation and independent comparison

The retained completed **72-hour** runs use matched terrain, rainfall, soil and
coastal contacts. Each maximum map is the greatest depth at each cell over the
event, not the final depth. The signed map is FloodWarp FV1 maximum minus
LISFLOOD-FP FV1 maximum, not the maximum instantaneous difference. All GIS
products use the exact DEM grid and physical mask, with zero FP64 active-cell
conversion error.

![Full-domain maximum-depth and extent comparison](examples/orton_bradley/figures/orton_bradley_fv1_comparison_001m.png)

FV1 maximum-envelope RMSE is **0.001435 m**, maximum absolute difference is
**0.119908 m**, and maximum-event extent IoU is **0.998711 at 0.01 m** and
**0.995776 at 0.10 m**. Final-depth RMSE is 0.0002405 m; final wet-union RMSE
is 0.003930 m using a 0.001 m wet-union threshold. These distinct metrics are
recomputed from checksummed arrays. The [0.10 m map](examples/orton_bradley/figures/orton_bradley_fv1_comparison_010m.png)
and [quantitative report](examples/orton_bradley/COMPARISON.md) include signed
extrema, false-positive/negative areas, source/export differences and balances.

## Full-event hydrographs

![Full 72-hour hydrographs and peak detail](examples/orton_bradley/figures/orton_bradley_hydrographs_72h.png)

| Completed matched run | LISFLOOD-FP FV1 | FloodWarp FV1 | FloodWarp LI |
|---|---:|---:|---:|
| Hydraulic runtime [s] | 1403.56 | 1906.11 | 1019.46 |
| End-to-end component runtime [s] | 1486.87 | 2007.82 | 1126.95 |
| Accepted timesteps | 816,514 | 1,570,053 | 694,389 |
| Peak 60 s mean discharge [m³/s] | 161.483 | 161.173 | 182.167 |
| Peak interval ending [h] | 24.4833 | 24.5000 | 24.4167 |

Actual 60-second interval means come from cumulative diagnostics: 4,320 common
intervals, measured on RTX 4070 Ti SUPER. FV1 hydrograph NRMSE is 0.0002804
normalised by reference peak; peak magnitude error is −0.192%, timing error +60 s.
End-to-end component timing includes diagnostics/checkpoints but excludes some
publication overhead. These are single completed runs, not a repeated hardware
performance survey.

LI is faster here but has **0.25638 m maximum-envelope RMSE**, maximum-event
IoU **0.497596 at 0.01 m**, and a peak 12.81% higher and 240 s earlier.
The median envelope difference is only 2.8 mm, and 97.34% of active cells
differ by less than 0.10 m. Larger discrepancies are spatially concentrated;
the retained LI channel maxima warrant investigation.
The [LI comparison](examples/orton_bradley/figures/orton_bradley_li_comparison.png)
uses identical depth limits and full-range **linear** signed differences to
show their prevalence without stretching small errors.

## Reproduce and interpret

The archive contains original conditioned inputs, complete FP64 fields,
checksummed records, aligned GeoTIFFs, boundaries, provenance, figures and
standalone scripts. Extract it and run `python reproduce.py` with NumPy,
Rasterio and Matplotlib to regenerate products **without rerunning hydraulics**.
An optional full-domain FloodWarp replay script is separate. Installation tests
never run the 72-hour event.

The matched FV1 benchmark explicitly used **discharge-extrapolated** prescribed
stage momentum, contact HLL, FP64, CFL 0.4 and maximum dt 1 s. Public defaults
use velocity extrapolation. Established design rainfall lasts 48 hours, with
recession to 72 hours. Manning and exponential-soil mappings are assumptions
requiring site calibration. Reference agreement is numerical context, not a
calibrated flood forecast or qualification of every closure/scenario. SGS is
experimental. Sparse FV1, AMR and mixed-precision FV1 are outside v0.1.

See [release status](docs/RELEASE_STATUS.md), [numerical assessment](validation/repair_review/FINAL.md),
[changelog](CHANGELOG.md), [attribution](THIRD_PARTY_NOTICES.md) and
[copyright](COPYRIGHT). FloodWarp retains its documented Overlander extraction
provenance without requiring that repository. Elevation source/conditioning
credits are preserved; no endorsement is implied.
