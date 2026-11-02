# Development

The verified modified sparse FV1 path is now the default on both main and dev.
Dense execution remains selectable; static AMR stays separate on dev.

Work on dev. main receives ordinary validated merges/fast-forwards. This
repository has independent history; do not change the original Overlander
research checkout as part of FloodWarp maintenance.

Use a fresh Python 3.14 environment with `pip install '.[test,raster,plot]'`.
The numerical core pins the source baseline's NumPy 2.3.5, SciPy 1.16.3,
Numba 0.67.0 and Warp 1.17.0. Do not upgrade them as housekeeping.

Maintained entry points, from the repository root:

- `python -m examples.run_synthetic`: small LI/FV1 installation smoke test.
- `python -m examples.run_purau`: 40×40 real-terrain CPU rainfall demonstration.
- `python -m examples.run_direct`: user-editable GeoTIFF paths and controls.
- `python -m validation.compare`: bounded independent reference comparisons.
- `python -m validation.sensitivity`: assess completed comparison outputs.
- `python -m pytest -q`: retained numerical and direct-input regressions.
- `OVERLANDER_TEST_FULL_SWE_CUDA=1 python -m pytest -q`: opt-in extra FV1 CUDA controls; historical test flag retained.

Python scripts use named reusable functions, pathlib paths, explicit keyword
arguments and adjustable controls inside `if __name__ == "__main__":`.
No argparse, path injection or other-repository imports are needed.
Numerical library modules are preserved baseline code; avoid incidental
formatting/refactoring that obscures operator provenance.

The standalone precision environment names are FLOODWARP_PRECISION and
FLOODWARP_FLOAT. Existing OVERLANDER_* tuning flags remain for reproducible
kernel/test controls; they do not imply a dependency on another installation.
CPU getters may return views: copy arrays before retaining sampled output.

## Numerical constraints

Preserve physical LI defaults, FV1 contact_hll/velocity_extrapolation, FP64,
CFL bounds, wet/dry tolerances, reference arrays and restart identity guards.
Source-compatible profile choices are explicit. Never replace operators with
LISFLOOD code or tune roughness to improve comparison scores. Treat native
comparisons as independent implementations, not physical truth.

Both SGS modes remain experimental with visible warnings and their historical
physics. Do not apply physical-LI friction/boundary updates to SGS as cleanup.
Inclined SGS, semi-implicit and routing research are excluded. No campaigns,
training, external-model coupling or prepared-data workflows belong here.

Record simultaneous fields separately from maximum-depth envelopes. Never
sum cellwise event maxima as storage. Report failed gates and missing coverage.
The v0.1 software release is authorised under AGPL-3.0-only. Preserve the
Patrick Durney / Lincoln Agritech Ltd attribution and CC BY 4.0 elevation
credits. The owner controls repository visibility. Research-grade release does
not imply universal hydraulic qualification.

## Default sparse uniform FV1 execution

FV1 defaults to exact-zero scheduling with 8×8 tiles and dense storage. Native
`FullSWESolver`, `create_overland_solver(solver_method='fv1', ...)` and
`create_model(solver_method='fv1', ...)` accept `execution_mode='dense'` for
the retained reference path. Use `solver.configure_execution(mode='dense')`
or `solver.configure_execution(mode='sparse', tile_size=8)` to switch an
existing uniform solver. The sparse path uses dense indexing when its work
set reaches 65% of all raster tiles; explicit dense mode also omits scheduling.
`solver.execution_diagnostics()` performs an explicit host readback of work
counts; the step loop keeps flags and lists on device. Sources, full-domain CFL,
numerical settings and checkpoint identity are unchanged. Activity is rebuilt
after every source update. Scheduling caches are reconstructed after restart.
This mode can be slower at high occupancy; explicit dense mode remains available.
See [design and stage gates](SPARSE_FV1_DESIGN.md) and
[measured qualification](../validation/sparse_review/REPORT.md).

The [matched 72-hour replay](../validation/sparse_review/FULL_EVENT.md) now
retains exact hourly dense-state agreement and measured 1.161× hydraulic
speedup. Widespread-runoff windows remain slower. The validated sparse default
is included in v0.1.1; the fixed v0.1.0 baseline is preserved. See the
[promotion checks](../validation/sparse_review/DEFAULT_PROMOTION.md).

On dev 0.2.0.dev0, subsequent measured changes clear scratch only in released
tiles and fuse the unchanged source update with post-source activity detection.
The full CFL scan still precedes sources. Interleaved saved-state controls
show about 7–8% less wet-phase hydraulic time than released sparse execution,
with exact field/step agreement; the short dry case is slightly slower.
Forcing earlier dense indexing was rejected. See the
[performance evidence](../validation/sparse_perf_review/REPORT.md).
These changes are not in v0.1.1.

The modified sparse FV1 path has now completed the matched 72-hour Orton Bradley event in **1512.372 s hydraulic time**, versus **1641.957 s** for released sparse and **1906.106 s** dense. All 72 hourly numerical-field checks and 1,570,053 accepted steps agree; final native checkpoint arrays directly match released sparse. This is one candidate run against retained same-device controls, not interleaved event repetitions. See [complete results](../validation/sparse_modified_event_review/REPORT.md).

## Experimental fixed block FV1

`floodwarp.static_fv1.StaticBlockFV1Solver.from_solver` constructs an opt-in
two-level, fixed 2:1 forest from a configured native FV1 solver. Supply the
block size, protected-feature mask and explicit surface/momentum/geometry/source
tolerances. Fine geometry and source history remain persistent; shared fine
subfaces and a single donor budget evolve nonoverlapping leaf volumes.
Use the native solver's complete checkpoint methods for this experimental path.

This initial representation requires fully wet representable or flat dry coarse
states and homogeneous source/roughness fields. It rejects unsupported transfers
and source changes. A later partly dry sloping coarse cell invalidates the run;
restore its last complete checkpoint. Public face arrays represent resolved
subfaces; transport internal to a coarse cell is unresolved.

The [static controls and stage decision](../validation/static_review/REPORT.md)
retain conservative/equilibrium/restart evidence and a 17–40% runtime regression
on bounded controls. Static execution remains experimental; dynamic adaptation
and sparse allocation are deferred at their measured stage gates. No sparse/AMR
speedups are multiplied. Static/adaptive research does not change the stable
v0.1.1 tag/main; dev uses the distinct version 0.2.0.dev0.

The subsequent [reconstruction investigation](../validation/static_reconstruction_review/REPORT.md)
adds 512×512 controls: broad coarsenable water gives 1.12–1.44× speedup, while
the protected narrow channel remains slightly slower. The original failed
64×64 results are preserved. A smooth curved-lake control remains in equilibrium
when an explicit small affine-fit tolerance admits it. Saved Orton Bradley
states show that constant-surface reconstruction and the current source/history
vetoes still prevent useful qualified coarsening. An affine-surface diagnostic
improves representability, but requires additional degrees of freedom and
separate connectivity/source/interface validation. It is not implemented in
the static solver. Dynamic AMR remains deferred; no production mask uses these
offline screens or future inundation fields.

## Versioned LI review

The additive [LI parity and engineering review](../validation/li_review_v1/REPORT.md)
keeps the original strict protocol and reference fixtures frozen. In the pinned
Python environment, `python -m validation.li_parity` exports the two recorded
baseline commits into fresh ignored output directories and replays matched LI
configurations plus actual numerical defaults. It reads the sibling Overlander
repository without modifying it. `python -m validation.li_controls` runs explicit
source/friction attribution controls and the public default; solver dry depths
remain unchanged. `python -m validation.li_assess` checks bitwise parity first,
then writes all six inundation thresholds, depth percentiles, hydrographs and
versioned engineering classifications. `python -m validation.li_audit` verifies
frozen artifact hashes and records manifests/test evidence. Inspect each guard
for paths, output roots and log destinations before rerunning; outputs must be
fresh. No independent LISFLOOD outputs are generated by these commands.

`python -m validation.li_engineering_v2` evaluates practical hydraulic
significance from the retained v1 replay arrays without running hydraulics.
Its separate [v2 criteria](../validation/li_review_v2/ENGINEERING_CRITERIA.md)
and [report](../validation/li_review_v2/REPORT.md) supersede v1 engineering
verdicts only; all original strict and numerical integrity evidence is preserved.
The output distinguishes equivalent-boundary agreement, unmatched-boundary
diagnostics, same-boundary timestep sensitivity, and recharge-specific concerns.
Run `python -m validation.li_engineering_audit` after validation to bind v2
outputs and test logs to preserved v1, strict benchmark and runtime hashes.
Inspect its guard for the test-log destination before reproducing the audit.

## v0.1.2 local interface and packaging

High-level factories now default to FV1; LI/SGS examples must select LI
explicitly. Use `python -m floodwarp.ui` after installing `.[ui]` for the local
rainfall interface. See [UI guide](UI.md). `floodwarp.sampling.sample_targets`
shares exact-final-time output targets with the small public examples; the
standalone full-domain replay retains its own identical helper.
The release baseline's modified sparse kernels and numerical controls are
unchanged. See [current evidence index](../validation/README.md).

## Spatial local UI on dev

The [current spatial UI](UI_SPATIAL.md) uses ordinary FV1 with aligned roughness
and infiltration maps, explicit face CSVs and a faithful full Orton Bradley
preset. Run `python src/floodwarp/ui.py` in the editable UI environment.
Leaflet is vendored; only optional basemap tiles require internet. Display
reprojection never alters hydraulic inputs or outputs. Keep the matched
benchmark profile explicit and preserve v0.1.2 artifacts. Focused controls are
`python -m pytest -q tests/test_ui_spatial_inputs.py tests/test_ui_runner.py tests/test_ui_server.py`.
