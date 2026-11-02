# Local rainfall interface — v0.1.2

For the current development UI with spatial coefficients, explicit stage faces,
Orton Bradley and OpenStreetMap, see [the spatial UI guide](UI_SPATIAL.md).
The following records the prepared v0.1.2 interface.

Install on Python 3.14 with `python -m pip install 'floodwarp[ui]'`, or use
`python -m pip install '.[ui]'` from this source checkout. Launch with
`python -m floodwarp.ui` and open **http://127.0.0.1:8765**. The server binds
only to loopback. Use the printed IP address; other Host headers are rejected.
The optional extra adds raster/vector IO and plotting; the browser server uses
Python's standard library. Core solver imports do not load these dependencies.
Stop the interface with Ctrl+C. A launched worker continues independently and
writes its own status, logs and checkpoint; keep its output directory.

This is a single-user local tool. It reads paths on the computer running
Python. Browse selects a path without uploading/copying rasters. It has one
active run per server and no accounts, cloud execution, GIS editing, automatic
restart or scheduling. Polling and refreshing the page do not restart a run.
Restarting the server does not rediscover an old worker; let that worker finish
before launching another job. Port 8765 must be free.

## Small included example

Click **Load small synthetic example**, select an existing output parent,
then **Validate & preview** and **Run**. The 12×16, 5 m synthetic DEM, mask,
dimensionless scaling raster, declared-CRS GeoJSON and rainfall CSV ship in
both distributions. The example lasts 120 s, including 48 s of zero rainfall
after its 72 s event. It is an installation control, not real terrain.
`python -m floodwarp.ui_example` copies the installed input set to a new
`floodwarp-example` directory for editing. It refuses to overwrite that directory.

## Terrain, domain and rainfall

Supply a single-band, north-up, unrotated GeoTIFF with a projected metre CRS.
Elevation values and resolution are preserved. State its vertical datum;
`local` explicitly means an unrecorded local datum, not a datum transformation.
Metadata conflicts are rejected. Row orientation uses `floodwarp.inputs`:
rasters are north-first, solver arrays south-first, and output rows are restored.

Choose all valid DEM cells, an exactly aligned binary GeoTIFF, or polygons in
one GeoPackage layer / a GeoJSON FeatureCollection. Polygon features must be
valid Polygon or MultiPolygon geometries with a declared CRS. GeoJSON must
include a named `crs` member, such as
`"crs":{"type":"name","properties":{"name":"EPSG:2193"}}`;
use GeoPackage for GIS interchange that preserves CRS without this older
GeoJSON convention. Missing/invalid CRS is rejected, never guessed.
Pyogrio reads geometry, Pyproj validates CRS, and Rasterio transforms geometry
into the DEM CRS and rasterises directly on its grid. The inclusion rule is
Rasterio `all_touched=False`: cell centres inside the polygon, with its standard
Bresenham edge treatment ([Rasterio features](https://rasterio.readthedocs.io/en/stable/topics/features.html)).
The resulting domain intersects valid DEM cells. Empty intersections fail.
Mask raster values must be 0/1 or NoData; NoData is inactive. Enclosed holes
remain inactive. No terrain resampling, filling, breaching or conditioning occurs.

All raster fields require the same shape, CRS, transform and cell spacing as
the DEM. A mismatch means exporting the field onto that exact grid in GIS
before running. Scaling maps must cover every active cell with finite,
non-negative values. Zero is valid. NoData within the active domain fails.
There is no normalisation: **local rainfall = hyetograph × multiplier**.
The preview reports the active multiplier range. A multiplier is dimensionless,
not an absolute rainfall-rate raster.

The downloadable example uses exactly `time_h,rainfall_mm_h`. Times start at
zero, are finite and strictly increasing, and have at least two nodes.
Intensities are finite, non-negative mm/h. Linear interpolation follows the
existing FV1 forcing integration. Convert hours ×3600 to seconds and mm/h
÷3,600,000 to m/s. Supply an explicit final zero-intensity node to define the
end of the event; a zero node is appended at the requested simulation end
for recession. Nonzero event endings are rejected, so the final intensity is
never extended through recession. Simulation duration must cover the full
rainfall CSV. Discharge in m³/s is not supported as rainfall.

## Settings and boundaries

The interface selects FP64 uniform-grid FV1, initially dry. It retains sparse
execution, contact HLL, velocity-extrapolated stage formulation, CFL 0.4,
dry depth 1e-4 m and factory timestep bounds 0.001–60 s. CUDA-graph policy is
unchanged (`advance` receives its ordinary default). Resolved numerical
configuration is saved; nothing is tuned to improve benchmark agreement.

Manning n defaults to 0.03 s/m^(1/3). Uniform constant infiltration capacity
defaults to 0 mm/h, explicitly converted to m/s. Automatic device selection
chooses the first available CUDA device, otherwise CPU; CPU and individual
available CUDA devices can be selected. The visible run summary reports the
resolved device, solver, boundary, roughness and infiltration even when settings
are collapsed.

**Closed** reflects at every domain interface. **FREE/outflow** opens active
raster edges and interfaces adjoining inactive cells connected to the outside
through four-neighbour inactive cells. Enclosed inactive/NoData holes remain
closed. Thus polygon/mask exterior boundaries can drain; the preset does not
open every masked interface. The summary reports the number of FREE faces.
No coastal stage is inferred from elevation. More selective outlets and stages
require the ordinary Python API. These are existing closures, not new boundary
laws. Preview the selected domain before choosing the boundary preset.

## Execution, outputs and replay

Run validates configuration before spawning a separate Python worker. Inputs
are checksum-checked again in that worker; changing inputs after validation
fails explicitly. Large arrays are loaded for preparation, then released by
the server; progress requests read only small status files. The worker loads
its run once. Progress advances at the chosen diagnostic interval (default
60 simulated seconds) and can stay at zero while kernels compile. Shorter
intervals give more frequent progress and more checkpoint IO. Every duration
includes the exact final time once, including durations below the interval.

Each run creates a distinct `floodwarp-<UTC timestamp>-<unique suffix>` directory:

| File | Meaning |
|---|---|
| `maximum_depth.tif` | Cellwise event maximum water depth, metres |
| `final_depth.tif` | Simultaneous depth at the requested final time, metres |
| `water_balance.csv`, `diagnostics.json` | Sampled cumulative source/boundary volumes in m³, storage and residual |
| `boundary_flow.csv` | Interval-mean aggregate boundary inflow/outflow in m³/s; not a selected river outlet |
| `checkpoint.npz` | Atomic complete restart at the last completed diagnostic sample |
| `run_config.json` | Inputs/checksums, interpretation, domain/grid, resolved numerical settings, device, version/source and actual final time |
| `status.json`, `run.log` | Progress/completion/failure and detailed worker output |

Output GeoTIFFs preserve the complete DEM grid, CRS, transform, active mask
and datum. Inactive cells are NoData. Maximum depth is not final depth, and
summing cellwise maxima is not simultaneous storage. Only scalar diagnostics
are retained over time; full depth snapshots are not accumulated. An initial
checkpoint and one per completed sample are saved. A later failure retains
the last completed checkpoint and flushed diagnostics. The UI shows the error
and log link, including abnormal worker exits. Automatic restart is excluded.

The reusable `floodwarp.ui_runner` module can prepare and execute the same run
without a browser. For restart, construct `RunConfig` from the saved `config`,
then call `prepare_run`, `create_run_model` and
`floodwarp.model.load_checkpoint(model=model, path=checkpoint_path)`.
Keep input files unchanged and verify their recorded SHA-256 identities.
`create_run_model` binds the spatial multiplier into direct-model checkpoint
identity. Changing formulations or inputs fails rather than reinterpreting an
archive. To continue beyond the original duration, provide an explicitly
extended rainfall series and follow the ordinary API's forcing identity rules;
there is no UI resume operation.

The interface supports rainfall-driven FV1 with uniform constant infiltration
and two boundary presets. Spatial roughness, native exponential infiltration,
LI, experimental SGS and arbitrary stage/outlet control remain Python tasks.
Completion does not establish calibration, observed-event validation or grid
convergence. The existing scientific evidence applies within its recorded scope.
