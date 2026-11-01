# Direct hydraulic inputs

All numerical fields use SI: metres, seconds, rainfall/infiltration in m/s,
and Manning n in s/m^(1/3). Solver arrays have row 0 at the south and column 0
at the west. Array loading requires explicit row order, dx, dy and vertical
datum. `terrain_from_array` preserves elevation values, including inactive
nodata; only the solver's inactive working values are filled with zero.
An explicit mask cannot activate nodata. Elevation and domain masks must be 2D.
Fields accept a scalar or the exact terrain shape, with no implicit row/column
broadcasting. Active Manning values must be positive.

`read_terrain(path=..., vertical_datum=...)` accepts a north-up, unrotated,
single-band GeoTIFF with a projected metre CRS. It reads spacing, nodata,
transform, CRS and tags and flips raster rows once. Geographic degrees, rotated
rasters, conflicting resolution, non-metre elevations and mismatched datum
metadata are rejected. Unrecorded elevation units/datum must be supplied by
the caller; FloodWarp does not infer a datum transformation.
`read_aligned_field` validates roughness and other rasters against the terrain.
`read_active_mask` accepts an aligned 0/1 raster with nodata treated as inactive;
`with_active_mask` restricts a domain using an array. `write_raster` restores
north-first orientation and retains georeferencing and datum metadata.
Rasterio is imported only when these IO functions are called.

## Solvers and initial state

`create_model(terrain=..., solver_method='li'|'fv1', ...)` uses the supported
factory, with FV1 as the omitted-method default since v0.1.2. Explicit LI
selection and native LI construction remain available. `initial_depth` or `initial_stage` accepts a scalar or spatial field;
they are mutually exclusive. Stages require `stage_datum` matching terrain.
Native `LocalInertialSolver` and `FullSWESolver` remain public for advanced
array control. LI's ordinary omitted profile resolves to `physical_candidate`.
The historical `default`, `lisflood_fp_acc` and `lisflood_fp_acc_parity` profiles
remain explicit choices. FV1 retains FP64, endpoint wave bounds, reflecting
walls, finite-velocity stages and contact-upwind HLL. Historical
`legacy_hll`/`discharge_extrapolation` remain explicit compatibility options.
Factory timestep defaults remain 0.001–60 s; the native FV1 class retains its
own 0–1 s defaults. Minimum timestep guards never authorise exceeding CFL.

v0.1.1 FV1 defaults to exact-zero sparse execution on the same uniform grid,
with dense FP64 state, 8×8 tiles and a 0.65 dense-indexing fallback fraction.
Native and factory constructors accept `execution_mode='dense'` to use the
retained reference without scheduling overhead. `tile_size` (8, 16 or 32)
and `dense_fraction` tune scheduling only. Existing uniform solvers can switch
with `solver.configure_execution(mode='dense'|'sparse', tile_size=8)`.
Full-domain source/soil updates and CFL remain unchanged, including dry cells
where rain infiltrates. Positive shallow water is retained exactly. Checkpoints
are portable in both directions and reconstruct scheduling caches after restore.

## Forcing and infiltration

`rainfall` accepts a constant or spatial rate field. `rainfall_series=(times,
rates)` accepts finite increasing piecewise-linear nodes starting at zero;
provide explicit zero nodes for recession and cover the simulation duration.
Do not also specify nonzero `rainfall`. To distribute a hyetograph spatially,
use `model.solver.set_rainfall_spatial_multiplier(multiplier=array)`.
Physical LI uses the preserved midpoint convention; ACC-compatible LI uses
step-start sampling. FV1 uses its existing forcing integration. No new source
law is introduced by this interface.

`infiltration` supplies the existing constant-capacity field. LI also accepts
its native `infiltration_method='exponential'` with
`infiltration_f0_m_per_s`, `infiltration_fd_m`, `infiltration_k_m_per_s`, or
`'green_ampt'` with `infiltration_ks_m_per_s`, `infiltration_psi_f_m`,
`infiltration_delta_theta`, passed through `create_model`.
FV1 retains constant and exponential infiltration: configure the latter with
`model.solver.set_exponential_infiltration(f0=..., fd=..., conductivity=...)`.
There is no new FV1 Green-Ampt implementation. Configure identical inputs and
soil parameters before loading a restart. Finite soil-storage limits and
ordinary source fields remain; groundwater adapters and exchange actuators do not.

## Boundaries

`boundary_faces(terrain=...)` returns closed exterior/mask interfaces and
internal code 0. Set selected interface entries to 2 (FREE) or 3 (stage),
leaving closed sections at 1. qx shape is `(ny,nx+1)`; qy is `(ny+1,nx)`.
For example, `qx[3:7,-1]=3` opens only four east faces. Openings on internal
active/active faces and unspecified domain interfaces are rejected. The input
adapter translates masked openings and wholly inactive faces to each solver's
existing representation; it does not alter hydraulic closures.

`stage_qx`/`stage_qy` accept absolute constant or face-shaped stages.
`stage_series_qx`/`stage_series_qy` accept `(times, stages)` for a uniform
piecewise-linear stage on that axis's prescribed faces. Use native setters
for independently varying face fields. LI's native loop supports stage series
and CUDA graphs. FV1 time-varying stages use host stepping, sampled at accepted
step starts; requesting graphs with these series raises explicitly. Constant
FV1 stages support CUDA graphs. Existing FREE closures and SGS restrictions
are preserved; no silent fallback occurs.

## Experimental SGS

Use LI with `hydraulic_geometry_mode='traditional'` (common-stage) or
`'enhanced'` (component-aware), supplying `subgrid_bed_elevation` and
`subgrid_resolution_m`. Native arrays must align exactly with integer coarse
cell subdivisions; the native grid is south-first. `subgrid` is the historical
alias for enhanced. Both constructors emit `ExperimentalSGSWarning`, a visible
standard `UserWarning`, with no prompt. Python's ordinary warning filters
apply. Diagnostics and saved numerical metadata record the resolved mode.
Both retain their legacy `default` physics, not the ordinary physical-LI
friction/boundary formulations.

These modes are not production-qualified. Common-stage storage can connect
hydraulically disconnected subcell areas. Component reduction improves that
representation but depends on finite stage tables, port topology, interpolation,
and transport assumptions; input/table failures must not be interpreted as a
request to fall back to coarse flow. SGS depths and inundation are not certified
fine-scale predictions. Enhanced SGS rejects slope-based FREE override.
Inclined/terrain-pressure SGS and connectivity research prototypes are absent.

## Outputs and restart

`advance` uses absolute end times. `diagnostics` reports the source and boundary
ledger, simultaneous storage, residual, and resolved numerical identity.
CPU getter arrays may share solver storage: use `.copy()` when retaining a
snapshot. Examples do this and save depth series, interval-mean hydrographs,
water balance, optional figures and complete checkpoints.

`save_checkpoint`/`load_checkpoint` bind the direct model's inputs, numerical
controls and soil parameters to a checksum-checked disk state. New direct-model
archives embed their identity and checksums in the NPZ and publish it with a
single atomic replacement after validation and fsync. Interrupted writes retain
the previous complete restart. Legacy NPZ/JSON pairs remain readable.
FV1 retains its baseline disk format; LI/SGS use the extracted complete-state
serializer and an immutable SGS geometry sidecar. Keep the geometry directory
beside the checkpoint (and the JSON sidecar for legacy archives). Restore into
an identically configured fresh model.
External source-slot and environment names were renamed for standalone use;
old LI/SGS disk archives are not silently accepted as new FloodWarp packages.
Historical numerical profiles remain available, and FV1 legacy checkpoint
identity handling is unchanged. Use `FLOODWARP_PRECISION` before import;
FP64 is the default and required for FV1. FP32/mixed LI research settings do
not acquire new qualification from this extraction.

## Local rainfall interface

See [UI inputs and execution](UI.md) for polygon CRS/rasterisation, binary mask
intersection, the hours/mm-per-hour rainfall CSV, dimensionless scaling maps,
FREE exterior interfaces versus enclosed holes, and complete output provenance.
The UI deliberately requires a final zero rainfall node and extends it with
zero recession; the ordinary API retains its explicit forcing-coverage contract.
`floodwarp.sampling.sample_targets` includes the exact requested end time once.
