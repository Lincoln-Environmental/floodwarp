# Orton Bradley FV1 3D animation

This additive workflow uses the published full 1123 × 843, 5 m model with
559,985 active hydraulic cells. It preserves the 48-hour spatial HIRDS v4
100-year design rainfall, 24-hour recession, Manning maps, exponential
infiltration and 212 stage-zero coastal contacts. The simulation is an
**uncalibrated design rainfall demonstration, not an observed flood
reconstruction or a qualified hazard assessment**.

The existing `run_model.py`, published benchmark files, solver methods and
reference arrays are unchanged. The exporter reuses the public direct-input
FV1, diagnostics and checkpoint APIs and the existing binary forcing reader.
It explicitly selects contact HLL, discharge-extrapolated stage momentum,
FP64, CFL 0.4, gravity 9.80665 m/s², dry momentum depth 0.0001 m and maximum
dt 1 s. The existing exact-sparse uniform FV1 scheduler remains in use.
The public adapter's `finite_support` rainfall option preserves the native
zero-rain recession after the original forcing ends at 172800 s.

## Current execution status

Preparation, immutable source checks, native 3D geometry and video encoding
have been exercised using local published files. The retained files contain
final depth, all-step maximum depth, minute diagnostics and a final checkpoint;
they do **not** contain the required five-minute instantaneous series. The
historical installation crop and synthetic outputs cannot substitute for it.

The GPU was occupied by another compute process during setup. No full-domain
hydraulic smoke test or new event has been launched. The preflight video is
explicitly labelled as a camera animation of the **retained final field only**.
It is not the conference deliverable. Current evidence and test logs are in
`outputs/orton_bradley_visualisation/` and remain outside Git. Setup checks
passed 1054 CPU tests, with 642 skipped while CUDA was hidden. The software
preflight renders 60 frames, encodes both MP4s and checks every decoded frame.
These checks do not establish new full-event numerical parity.

## Environment and local inputs

Run from the repository root on `dev`, using the pinned Python 3.14 environment:

```bash
.venv/bin/python -m pip install -r examples/orton_bradley/requirements-visualisation.txt
.venv/bin/python -m pip check
.venv/bin/python -m examples.orton_bradley.export_visualisation
MPLBACKEND=Agg .venv/bin/python -m examples.orton_bradley.render_visualisation
```

Default export action is `prepare`; it never starts the model. Default render
action is the two-second, 960 × 540, 30 fps preflight. All editable variables
are under each script's execution guard. No argparse or other orchestration
framework is used. The optional rendering requirements retain the hydraulic
pins. FFmpeg is supplied by `imageio-ffmpeg`; system FFmpeg and Blender are
not required. The default renderer uses Mesa software EGL, without CUDA,
and removes the inherited X display to avoid this host's display-probe failure.

Local inputs are read from `outputs/orton_bradley_release_staging/`, which
contains the immutable published `inputs/` and `results/`. Preparation verifies
all published SHA256 values, DEM/mask alignment, active bed elevations,
EPSG:2193, NZVD2016 and the 212 contacts. Source locations can be changed
under the execution guard when using an independently unpacked release archive.
The unchanged public replay's DEM is available in
`examples/orton_bradley/outputs/orton_bradley_dem_5m.tif` for the parity check.

## Execute when the device is free

The scripts do not stop, pre-empt or modify other jobs. Each hydraulic entry
checks `nvidia-smi` before model allocation and refuses external compute jobs
or excessive idle GPU utilisation. The same process's own CUDA context is
allowed during the serial parity check. A CUDA query failure stops execution;
there is no silent CPU fallback. The idle utilisation threshold is adjustable.
A new job can still start after this check: it is a preflight guard, not an
exclusive device reservation. No background waiter or scheduled task is installed.

Once the device is free, run these commands in order:

```bash
FLOODWARP_VIS_ACTION=smoke .venv/bin/python -u -m examples.orton_bradley.export_visualisation
FLOODWARP_VIS_ACTION=run .venv/bin/python -u -m examples.orton_bradley.export_visualisation
FLOODWARP_VIS_RENDER=conference MPLBACKEND=Agg .venv/bin/python -u -m examples.orton_bradley.render_visualisation
```

The 600-second smoke uses the actual full domain and writes frames at
0, 300 and 600 s. It compares depth, tracked maxima, timestamps and source /
boundary ledgers with a separate run of the unchanged public example. Its
initial period is infiltration-limited and cannot qualify wet flood dynamics;
wet/dry geometry is checked independently using the genuine retained final
field and geometric tests. The full action requires both successful smoke
parity and a successful decoded render preflight, then resumes the smoke run.
The full event provides the hydrograph and inundation validation coverage.

The previously recorded matched sparse event required about 25 minutes of
hydraulic time on this GPU. That is historical evidence, not a prediction or
runtime measurement for the new export. Allow additional compression,
checkpoint and rendering time. The hourly checkpoints alone need roughly
17 GB; allow at least 30 GB free for checkpoints, frames and rendered PNGs.

## Dataset and recovery

The output root is `outputs/orton_bradley_visualisation/`:

- `metadata/dataset.json`: exact source and numerical hashes, FP64/version
  identity, geotransform, CRS, datum, units, row order, intervals and credits.
- `metadata/terrain.npz`: full DEM, valid-DEM mask, physical hydraulic mask,
  original rainfall nodes and spatial multiplier. Arrays are south-first.
- `frames/depth-NNNNNN.npz`: losslessly compressed FP64 instantaneous depth
  and absolute elapsed seconds. Dry and inactive depth is zero. Compute water
  elevation as `dem + depth`; inactive/dry cells do not define a water surface.
- `diagnostics/ledger.json`: 60-second cumulative rainfall, infiltration,
  boundary import/export, simultaneous storage and residual, plus interval-mean
  net coastal discharge and rainfall volume rate. Divide rainfall volume rate
  by active area and multiply by 3600000 for domain-mean mm/h.
- `checkpoints/restart-NNNNNN.npz` and matching `.json`: independent hourly
  complete model restarts and transaction commit markers. A final checkpoint
  is also retained; the smoke has an additional restart at 600 s.
- `metadata/native_final.npz`: final instantaneous depth and native tracked
  all-step maximum envelope. The latter is never animated as an instantaneous
  flood surface or summed as storage.
- `diagnostics/validation.json` and `frame_manifest.json`: numerical gates,
  coverage and SHA256/size inventory for all exported frames.
- `renders/preflight/`: pipeline-check video, identity, image sequence and report.
- `renders/conference/`: main/preview MP4s, stills, image sequence and report.

Expected event coverage is **865 depth frames**, including t=0 and t=259200 s,
and **4320 minute-end diagnostics**. Each frame is compressed separately;
only a few arrays need to be held in host memory. Water elevation reconstructs
losslessly without storing a redundant full time series. Velocity is omitted;
no solver change is needed or introduced for it.

Re-run the same `smoke`, `run` or `conference` command after interruption.
Checkpoint markers are published only after their NPZ is complete and hashed.
Recovery selects the latest checksum-valid committed restart and its matching
ledger; diagnostics after that commit are recomputed. Already completed frames
are checked against replayed values and retained, with no duplicate frame writes.
Partial temporary files are ignored. Restart checkpoints never serve as animation
frames. Full-run configuration/source changes are rejected once hydraulic data
exist; choose a separate output root for a different experiment.

Rendering publishes individual PNGs atomically and resumes the same settings
without redrawing completed images. It checks field hashes against the validated
frame inventory. Settings/source changes require a separate render destination;
set `render_output` in the execution guard, preserving completed renders. MP4s
are written to temporary paths, decoded fully, checked for resolution, fps and
frame count, then published atomically. Incomplete videos do not replace a
valid previous one.

## Representation and validation

The renderer builds actual terrain and water triangle meshes from native
5 m DEM cell centres. Water vertex heights are exactly the corresponding
`DEM + h`, with vertical scale 1:1. Earth-toned terrain colours use DEM-derived
hillshading; no aerial imagery is implied. All surrounding valid model DEM
terrain is shown, including terrain outside the hydraulic mask.

Only triangles with three active vertices at depth ≥0.01 m are displayed as
water. This conservative centre-mesh representation omits some narrow wet
strips and the outer half-cell fringe; it cannot represent subcell hydraulics.
It never bridges a dry vertex. Depth controls a shared logarithmic blue colour
scale in metres; water opacity is 0.88. Spatial triangle interpolation is a
rendering surface between native samples. Temporal frames are held until the
next five-minute sample; no intermediate hydraulics, depth exaggeration,
flow particles or invented currents are generated. The simulation clock and
both chart indicators display the actual held sample time. A coherent oblique
camera movement rotates by 30 degrees through the event.

The conference default is **1920 × 1080, 72 seconds, 30 fps**. The website
preview is 960 × 540 with the same duration and fps. At least three stills are
selected from development, the maximum sampled wet area at the display threshold,
and final recession; reports record their actual sampled times. These are not
three invented event stages from the retained maximum envelope.

The validation command is independently repeatable after export:

```bash
FLOODWARP_VIS_ACTION=validate .venv/bin/python -m examples.orton_bradley.export_visualisation
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
```

CUDA is hidden for the CPU regression command while another job occupies it.
This does not establish new CUDA parity or replace the deferred full-domain
CUDA smoke. The new controls test lossless output, duplicate protection,
recovery with orphan frames and a corrupt checkpoint, interval ledgers,
orientation, wet/dry triangles, exact DEM+h vertices, busy-device refusal and
real MP4 encoding/decoding.

Full-event gates require exactly 259200 s and the expected complete timestamp
sequences, finite nonnegative depths, inactive zeros, matching final native
fields and geospatial identities. Mass residual must be ≤1e-8 of supplied
volume. Against the published FloodWarp FV1 record, depth/maximum-envelope
RMSE must be ≤0.001 m, discharge NRMSE ≤0.05, and storage/export errors within
5 m³ + 2% of their reference values. These use the frozen acceptance scales;
controls are not tuned to achieve a pass. Five-minute sampled maxima cannot
exceed native all-step maxima beyond 64 FP64 eps scaled by event depth.
Missed interframe peaks are reported as deficits, with no zero-error
requirement on sampled maxima. Validation failures stop conference rendering.
Visual inspection is still required for the final inundation and camera views.

Software: Patrick Durney / Lincoln Agritech Ltd, AGPL-3.0-only. Elevation data:
CC BY 4.0, with the LINZ, Environment Canterbury, Christchurch City Council,
Aerial Surveys, Landpro and AAM NZ attribution preserved in metadata and video.
See [DATA_LICENSE.md](DATA_LICENSE.md), [OUTPUT_NOTICE.md](OUTPUT_NOTICE.md)
and [the source comparison](COMPARISON.md).
