# Spatial UI on main — v0.1.3

The [later runtime metrics update](../run_metrics_review/REPORT.md) records
current per-run reporting checks and fresh local distributions under
`outputs/release/v0.1.3-metrics/dist/`. The earlier artifacts and measurements
below are retained with their original identities.

The later [IDE directory correction](DIRECTORY_FIX.md) records the current
fix and corrected local distributions. The measurements below retain their
original source/artifact identity.

Source tested and used to build the distributions: **`72770048f93aeec14bbca3c95d381f625295d245`**.
The following commit records this evidence only. The spatial input/boundary/
map implementation is byte-identical to the previously tested dev implementation;
main's native solvers and kernels remain unchanged. Static-mesh research modules,
tests and development-only history were not promoted.

Main was checked separately in a fresh Python 3.14.7 pinned environment:

- Full CPU source suite: **1,025 passed, 634 skipped**, 350 retained warnings.
- Focused FV1 CUDA/graph/sparse/restart and spatial UI controls: **172 passed**.
- Revision-stamped wheel and source archive built from a clean committed export.
  Their Python module set matches main; neither includes static FV1 runtime/tests.
- Actual v0.1.3 wheel installed outside the checkout: default FV1, explicit LI,
  pip check and the documented UI launcher pass. The installed worker reaches
  exactly 73 s and produces GeoTIFFs, checkpoints and map overlays.

[Main verification record](main_verification.json) contains artifact identities.
[Earlier dev verification](REPORT.md) retains its own source/total and the
actual Firefox/live OSM checks, spatial run and exact 2 s full-domain Orton
Bradley prefix. Those are not relabelled as a full main suite or a new 72-hour
validation. CPU skips include CUDA-dependent and opt-in/external controls;
relevant CUDA controls were run separately. Observed-event validation and grid
convergence are not claimed.

[Current user guide](../../docs/UI_SPATIAL.md). Roughness and constant/exponential
soil maps use native fields/setters. Explicit face CSVs select Closed/FREE/
constant-stage interfaces and absolute datum. Orton Bradley retains original
SI data and 212 coastal contacts; matched settings remain explicit. OSM images
are display context; hydraulic grids are unchanged.

Artifacts are under `/home/patrickdurney/PycharmProjects/floodwarp/outputs/release/v0.1.3/dist`; the earlier v0.1.2 files/checksums
are unchanged. Version/CITATION/README/input/UI guides and built-wheel CI now
identify main v0.1.3. No release, assets or visibility change is published by
this branch update.
