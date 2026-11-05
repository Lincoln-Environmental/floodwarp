# Interactive Orton Bradley 3D viewer

Explore the completed full-domain FV1 event in a local browser. Drag to rotate,
scroll or pinch to zoom, and right-drag to pan. Click or drag the discharge
hydrograph or rainfall hyetograph to select the nearest saved five-minute
result. Both chart indicators, the clock and the water surface update together
only after the selected frame has loaded and passed its checksum check.

This reuses the validated 72-hour dataset. It never starts a hydraulic solver,
changes a numerical timestep or modifies the original benchmark files.

## Open the prepared viewer

From the repository root, on `dev`:

```bash
.venv/bin/python -m examples.orton_bradley.serve_interactive
```

Open **http://127.0.0.1:8766/**. Stop the server with Ctrl+C. The default is
loopback only. The host, port and directory are editable under the script's
execution guard. No account, CDN, external tiles or running model is needed.
Use a current browser with WebGL2 and gzip-stream support; the checked browser
was Chromium with software WebGL. Runtime browser requests stay local.

The viewer initially shows the peak-flow frame at 24:30. Controls include:

- Play/pause, playback speed, previous/next five-minute frame and a time slider.
- Click or drag either chart; focus a chart and use Left/Right or Home/End.
- Direct hour entry, rounded to the nearest saved frame, with actual time shown
  as hours:minutes. The charts show preceding-minute means.
- Start, peak discharge, peak sampled inundation and final recession shortcuts.
- Reset overview, plan view, zoom buttons, water opacity and full screen.
- Bookmarkable times, for example `http://127.0.0.1:8766/#t=88200`.

The 3D view retains the actual 1123 × 843, 5 m DEM and 559,985 active cells.
There is no vertical exaggeration. Water vertices are DEM + instantaneous h.
Only triangles with three wet active native samples are shown, using the same
0.01 m display threshold as the movie. This omits some narrow wet strips and
the outer half-cell fringe. Spatial triangles are a display surface; there is
no temporal interpolation, manufactured flow texture or particle animation.

All clocks and geometry represent an **uncalibrated 100-year design rainfall
demonstration**, not an observed flood reconstruction. The camera can move
freely and the model has no terrain collision system. Reset view restores the
initial overview. The responsive layout is tested, but the full native mesh
still requires a capable WebGL device; mobile rendering speed varies.

## Rebuild without rerunning hydraulics

The already prepared local bundle is
`outputs/orton_bradley_interactive/`. To reproduce it:

```bash
.venv/bin/python -m pip install -r examples/orton_bradley/requirements-visualisation.txt
.venv/bin/python -m examples.orton_bradley.build_interactive
.venv/bin/python -m examples.orton_bradley.serve_interactive
```

Adjust the source dataset, application directory, destination, wet threshold,
colour-scale maximum and compression level in `build_interactive.py`'s guard.
The builder requires the complete successful FV1 validation record and all
865 source frames, including t=0 and t=259200 s. It verifies native frame
hashes against the frozen export manifest. There are no new hydraulic inputs.

The builder writes each display frame and its commit marker atomically. A
repeat build validates and retains completed matching frame payloads. An
interruption resumes at missing frame markers. A changed dataset or conversion
configuration is rejected; use a separate destination and preserve existing
results. Application files can be refreshed by rerunning the same build,
without recompressing completed frame payloads.

The output directory is a static website and can be served by an ordinary HTTP
server. ES modules need HTTP; opening `index.html` with `file://` is insufficient.
No hosting, publication or changes to the existing FloodWarp UI are performed.
Future hosting must retain binary assets, metadata and credits together; no
public deployment is included here.

## Data representation and measured fidelity

The original losslessly compressed **FP64** event remains untouched in
`outputs/orton_bradley_visualisation/`. The browser bundle contains compressed
Float32 display fields, not replacement scientific result arrays. All original
Manning, rainfall, infiltration and boundary assumptions remain those of the
completed benchmark run. Horizontal CRS is EPSG:2193; elevation datum is NZVD2016.

The completed bundle has 865 frames, 703,467,843 compressed frame bytes and
7,258,481 compressed terrain bytes (approximately 711 MB combined). The browser
loads one selected frame at a time and retains at most six decompressed frames.
Terrain remains a native cell-centre mesh, with local metre coordinates to avoid
large NZTM coordinate values in GPU vertex arithmetic. East is +x, up is +y,
and north is −z; geographic origin and geotransform are preserved in metadata.

Wet/dry classification is calculated from the native **FP64** depth before
conversion. Each frame carries an explicit native wet mask, preventing Float32
rounding around 0.01 m from changing inundation classification. The verified
maximum display depth conversion error is **2.3838104 × 10⁻⁷ m**. DEM elevation
conversion error is zero for this original Float32-source terrain. The measured
maximum Float32 `DEM + h` water-elevation rounding error is
**3.0545080 × 10⁻⁵ m**. These are graphics precision effects; they do not change
the hydraulic solution or its native acceptance results.

The browser uses camera-dependent clipping and reversed depth buffering where
available, with a logarithmic-depth fallback. This improves separation of
shallow water and terrain without an artificial elevation offset. Depth-based
cyan-to-blue colouring uses the disclosed logarithmic metre scale. Alpha is a
display control and does not alter depth or wet/dry masks.

`data/manifest.json` binds the source identity, validation, frame inventory and
terrain hashes. It also records timestamps, display errors, native wet areas,
native instantaneous maxima, diagnostic series, geospatial metadata and credits.
Every fetched decompressed binary is length-checked and SHA256-checked before
use. Rapid selections abort earlier requests, and stale responses cannot replace
a later requested frame, including a return to the currently displayed time.

Binary layout, for N = ny × nx, in south-first row order:

| File | Layout after gzip decompression |
| --- | --- |
| `data/terrain.bin.gz` | 3N little-endian Float32 x/y/z positions; 3N UInt8 RGB; N UInt8 valid-DEM mask; N UInt8 active mask. |
| `data/frames/depth-NNNNNN.bin.gz` | N little-endian Float32 depths; N UInt8 native wet flags. |

`data/build_identity.json` and `build_report.json` retain source/configuration
and application hashes. Generated binary data, screenshots and QA reports stay
outside Git. Three.js 0.187.0 and OrbitControls are vendored unchanged, with MIT
licence and source archive/file hashes in `viewer/vendor/`. Their source is the
official npm package; no runtime external script request is made.

## Repeat the checks

The numerical/display unit controls are part of the ordinary suite:

```bash
FLOODWARP_TEST_RENDER=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
```

For independent checks of all 865 converted frames, without a browser/server:

```bash
FLOODWARP_INTERACTIVE_CHECK=data .venv/bin/python -m examples.orton_bradley.check_interactive
```

Browser QA is optional and requires Playwright and a local Chromium executable:

```bash
.venv/bin/python -m pip install -r examples/orton_bradley/requirements-interactive-test.txt
FLOODWARP_INTERACTIVE_CHECK=browser .venv/bin/python -m examples.orton_bradley.check_interactive
```

Start the viewer server first and set `executable` in the QA script's guard to
your Chromium path. This session reused the existing cached Chromium; no browser
was installed or downloaded. The default `all` action runs data then browser QA.
The helper launches a separate headless software-rendered browser and closes it
after checks; it does not take over a user's browser.

Independent data QA passed all timestamps, original FP64 source hashes,
compressed-payload hashes, native masks and Float32 rounding bounds. Browser QA
passed initial/zero/final fields, chart clicks and drags, keyboard stepping,
playback, orbit, wheel/button zoom, pan, reset, rapid selection races, bounded
cache, mobile page layout and the forced logarithmic-depth fallback, with no
page/console errors. The full CPU and
headless-graphics suite passed **1060 tests, 642 skipped**. Evidence is in
`outputs/orton_bradley_interactive/qa/` and test logs under the native dataset's
output root. A later UI change should rerun browser QA.

Software attribution: Patrick Durney / Lincoln Agritech Ltd, AGPL-3.0-only.
DEM data retain CC BY 4.0 credits to LINZ, Environment Canterbury, Christchurch
City Council where applicable, Aerial Surveys, Landpro and AAM NZ. These appear
in the viewer and its metadata. See [DATA_LICENSE.md](DATA_LICENSE.md),
[VISUALISATION.md](VISUALISATION.md) and [OUTPUT_NOTICE.md](OUTPUT_NOTICE.md).
