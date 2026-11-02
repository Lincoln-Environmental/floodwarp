# Spatial local UI development verification

Tested source: `60a570ac24fe453631932e60a7412badfb3e0e39` on dev, Python 3.14.7, pinned core dependencies,
Linux x86-64 and RTX 4070 Ti SUPER. Subsequent commits add evidence only.
See [verification.json](verification.json) and the linked logs.

- Full CPU suite: **1,044 passed, 642 skipped**, with 350 retained experimental
  warnings. CUDA-dependent and opt-in/external controls account for skips.
- Focused FV1 CUDA/graph/sparse/dense, restart and new UI controls: **172 passed**.
- Spatial roughness and constant/exponential infiltration use existing native
  fields and setters. Controls cover direct-API parity, rainfall volume and
  residual, invalid grids/rates, explicit interface/datum rules, exact final
  time, checkpoint cadence and restart identity for finite-support forcing.
- Original-input conversion retains exact FP64 coefficient/forcing arrays and
  original DEM bytes. The real full Orton Bradley preset has 559,985 active
  cells and 212 explicit stage-0 NZVD2016 coastal contacts. A **2 s full-grid
  CUDA prefix** exactly matches independently constructed ordinary-input
  state, maxima and soil arrays. This is not a new 72-hour simulation.
- An actual previous v0.1.2 UI checkpoint restores at 120 s. The default high-level
  forcing-coverage guard and existing serializer formats are unchanged.
- Firefox used actual UI controls: automatic full Orton setup/map; spatial
  roughness/capacity and a custom constant-stage face; Run to exactly 2.3 s;
  terrain/maximum/final overlays; offline basemap toggle and refresh without
  restart. Live OSM tile requests returned 200. A separate screenshot check
  verified 256×256 loaded tiles at opacity 1, rather than treating network
  requests alone as proof of a visible map. See [map screenshot](../../docs/ui-orton-map.png).
- The separate development wheel contains all 57 committed Python modules,
  local Leaflet code/licence and the preset catalogue, with matching hashes.
  Fresh installation outside the checkout passes pip check and default FV1 /
  explicit LI controls; its actual UI launcher and worker complete at 73 s.
  That installed HTTP/worker check is distinct from the Firefox interface check.

Native hydraulic solvers/kernels, sparse execution, static research and pinned
numerical dependencies are unchanged. The explicit new finite-support API
option exposes the native event termination already implemented in FV1;
non-default forcing semantics enter checkpoint identity. Map reprojection is
bounded and display-only. Coefficient inputs and result GeoTIFFs retain the
original grid/mask/CRS and physical units.

This work remains on dev / 0.2.0.dev0. Main, prior release tags and the prepared
v0.1.2 wheel/source archives are unchanged; both stored release SHA-256 checks
pass. No remote push, visibility change, publication or asset upload was done
for this extension. The local `.venv` now uses the editable development checkout.

There is no full-event requalification, calibration, observed-event validation
or new performance claim. Boundaries use explicit face files and constant stages;
time-varying stages remain available through Python. Basemap access is optional
and online; local solver execution and overlays do not need internet.

[User guide](../../docs/UI_SPATIAL.md). Completed spatial browser example:
`/home/patrickdurney/PycharmProjects/floodwarp/outputs/ui-spatial-review/floodwarp-20261102T003223Z-qke8ls9p`.
