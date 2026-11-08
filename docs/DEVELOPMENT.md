# Development

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

## External numerical behaviour reference

`validation/lisflood_fp_free_boundary.py` contains a validation-only NumPy
re-expression of LISFLOOD-FP 8.3's GPLv3 FREE1 boundary recurrence.
Production `LocalInertialSolver` exposes no LISFLOOD-specific float
arithmetic modes and no reference-only Warp kernels. The mathematical
`lisflood_fp_acc` validation profile remains distinct and available.
The reference module is excluded from the production import graph.

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

## Sparse uniform FV1 in v0.1.1

Current main now uses the verified modified scheduler by default, clearing
released tiles and fusing post-source activity with the unchanged source update.
The [matched 72-hour event](../validation/sparse_modified_event_review/REPORT.md)
retains all recorded fields and steps at 1512.372 s hydraulic runtime. The
v0.1.1 tag retains the previous scheduler. Static AMR remains separate on dev.

Native `FullSWESolver`, the solver factory and `create_model` select exact-zero
sparse execution with `tile_size=8` and `dense_fraction=0.65` for FV1. Use
`execution_mode='dense'` at construction or
`solver.configure_execution(mode='dense')` for the retained reference.
`solver.configure_execution(mode='sparse', tile_size=8)` restores sparse
scheduling. Both retain dense state, unchanged sources, full-domain CFL and
portable checkpoints. Activity is rebuilt after every source update. At high
work-tile occupancy the scheduler uses dense indexing; explicit dense mode
also removes scheduling overhead. `execution_diagnostics()` explicitly reads
device work counts; the step loop does not read them back.

The [completed event](../validation/sparse_review/FULL_EVENT.md) retained every
hourly field and 1,570,053 steps while using 1641.957 s hydraulic time versus
1906.106 s for the retained same-device dense run. This is one full candidate
run, not repeated interleaved events. High-occupancy windows can be slower.
Static/dynamic AMR, wavelets, sparse allocation, mixed-precision FV1 and new
performance experiments belong on dev and are excluded from v0.1.1. Neither
the v0.1.0 tag nor its immutable datasets are changed.

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

## Spatial local UI on main (v0.1.3)

The [current spatial UI](UI_SPATIAL.md) uses ordinary FV1 with aligned roughness
and infiltration maps, explicit face CSVs and a faithful full Orton Bradley
preset. Run `python src/floodwarp/ui.py` in the editable UI environment.
Leaflet is vendored; only optional basemap tiles require internet. Display
reprojection never alters hydraulic inputs or outputs. Keep the matched
benchmark profile explicit and preserve v0.1.2 artifacts. Focused controls are
`python -m pytest -q tests/test_ui_spatial_inputs.py tests/test_ui_runner.py tests/test_ui_server.py`.
