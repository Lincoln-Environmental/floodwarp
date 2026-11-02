# FloodWarp evidence index

The v0.1.2 candidate starts from release baseline
`bce98b103beca066cd0c9afdeec0fc330576182a` on main. The owner's subsequent
README/COPYRIGHT commits through `731cf6d1c3a69eb1267875c90f67965e9f72c193`
are also preserved; they do not change hydraulics. It retains FP64 uniform-grid
FV1 with verified modified exact-sparse scheduling and explicit dense execution.
The candidate commit and actual distribution hashes are recorded in
[release verification](release_v0.1.2/REPORT.md). Default high-level construction
now selects FV1. Explicit LI and experimental SGS remain available.
No static-mesh or adaptive runtime is part of this release.

| Evidence | Scope |
|---|---|
| [Frozen acceptance protocol](ACCEPTANCE.md) | Meaningful depth, extent, discharge and conservation gates; unchanged |
| Source `tests/test_full_swe.py`, `test_still_water.py`, `test_cfl.py`, `test_rotation.py`, source/infiltration tests | Analytical and invariant controls, accepted-step/CFL and conservation regressions |
| [Orton Bradley completed comparison](../examples/orton_bradley/COMPARISON.md) | Independent FloodWarp FV1 / LISFLOOD-FP FV1, matched 5 m full-domain 72-hour event |
| [Numerical assessment](repair_review/FINAL.md) | Bounded independent comparisons and analytical/refinement controls, including their limitations |
| [Original sparse qualification](sparse_review/REPORT.md), [full event](sparse_review/FULL_EVENT.md) | Exact sparse/dense state, maxima, ledger, graph and restart parity; one completed event and separate repeated short controls |
| [Sparse performance investigation](sparse_perf_review/REPORT.md) | Historical interleaved saved-state timing, allocation and profiling with frozen source snapshots |
| [Modified sparse completed event](sparse_modified_event_review/REPORT.md) | Later evidence superseding the investigation's pending-event conclusion; 72 hourly comparisons and final checkpoint equality |
| `tests/test_direct_inputs.py`, `test_sparse_fv1.py`, `test_benchmark_restart.py`, checkpoint regressions | Direct/native, soil/forcing, sparse/dense, graph and atomic disk restart identity |
| [v0.1.2 checks](release_v0.1.2/REPORT.md) | New factory defaults, exact output timing, headless UI runner, actual browser actions and built-artifact installation |
| [LI v2 review](li_review_v2/REPORT.md) | Versioned practical engineering assessment; numerical integrity and original strict v1 results retained |
| [v0.1.0 release](release_v0.1.0/REPORT.md), [v0.1.1 release](release_v0.1.1/REPORT.md), [publication check](release_v0.1.1/publication_verification.json) | Historical releases, datasets, assets and checksums; not retests of this candidate |

The full-domain demonstration is **Orton Bradley Park**. Historical
`examples.run_purau` is a 200 m Orton Bradley installation crop. The actual
Purau catchment is a separate case. File names do not establish site identity.

Independent numerical agreement is not observed-event validation, calibration
or grid convergence. The completed comparison uses matched benchmark stage
momentum settings (discharge extrapolation), distinct from ordinary FV1's
velocity-extrapolation default. Both settings remain explicit and unchanged.
Final-state metrics and cellwise event maxima are different measurements.
Single full-event runtime measurements are not repeated performance surveys;
short saved-state repetitions are separately identified. Sparse execution uses
dense storage and is not a sparse-memory claim. CPU/GPU portability evidence
does not qualify every hydraulic regime or hardware configuration.

Traditional and enhanced SGS remain experimental with visible warnings.
Historical strict failures and superseded assessments retain their results;
links identify later evidence rather than rewriting measurements. Existing
observed-event, boundary matching, finer-terrain and convergence gaps remain.
The UI adds no new calibration claim or independent hydraulic implementation.
No additional 72-hour event is required for these orchestration, metadata and
packaging changes: kernel bodies, precision, CFL, source integration, boundary
closures and checkpoint serializers are unchanged, and focused accepted-time,
forcing, parity and restart controls cover the new scheduling path.

## Development spatial UI

The separate [spatial UI review](ui_spatial_review/REPORT.md) records current
dev checks for spatial roughness and constant/exponential infiltration,
explicit constant-stage domain faces, original-input Orton Bradley preparation,
finite-support rainfall continuation and OpenStreetMap display overlays.
It includes actual browser/live-tile checks and a bounded 2 s full-domain
original-input comparison, not a new 72-hour event or qualification claim.
Prepared v0.1.2 artifacts and their recorded release evidence remain unchanged.
