# Completed numerical repairs and GPU qualification

All ten benchmark/control cases completed. Each full event reached exactly
259200 s with validated finite, nonnegative fields and conservative accounting.
This establishes measured performance and bounded accuracy, not universal
hydraulic qualification. Historical strict/v1/v2 evidence remains unchanged.

## Matched 72-hour Purau event

One measured run per solver, after warm-up, on the same RTX 4070 Ti SUPER.
The 1123×843, 5 m domain contains 559985 active cells and 212 coastal ports.
Initial depth, DEM, mask, rainfall integral/spatial multipliers, exponential
soil parameters, roughness, gravity 9.80665 m/s² and nominal CFL 0.4 match.
Maximum dt is 1 s; minimum 1e-6 s is a failure guard, never a timestep floor.
The comparison selects FV1 discharge-extrapolated stage ghosts explicitly;
the library default remains velocity extrapolation. Coastal stage is 0 m.
The terrain's unresolved absolute datum lineage remains a release limitation.

| Solver | Hydraulic s | Publication wall s | Hydraulic ratio to external | Steps | Mean dt s | Simulated s/wall s |
|---|---:|---:|---:|---:|---:|---:|
| LISFLOOD-FP FV1 GPU | 1403.559 | 1493.188 | 1.000 | 816514 | 0.317447 | 184.67 |
| FloodWarp FV1 GPU | 1906.106 | 2015.523 | 1.358 | 1570053 | 0.165090 | 135.98 |
| FloodWarp LI GPU | 1019.460 | 1136.464 | 0.726 | 694389 | 0.373278 | 254.25 |

Publication wall includes preprocessing and output/checksum work, from fresh
run-directory creation to final record write. Shared input loading (median
0.1255 s) and final directory fsync are outside it. Native hourly checkpoint
continuations launch a new process; this cost is included in wall time and
excluded from hydraulics. Warm compilation caches were used; no cold-JIT claim
is made. Small cases, both one-hour cases and 36-hour controls have three
measured repetitions; ranges and all timing phases are in
[the measurements](qualified_measurements/measurements.json).

| Device memory, MiB | External FV1 | FloodWarp FV1 | FloodWarp LI |
|---|---:|---:|---:|
| Peak live allocated | 182.173 | 234.859 | 614.152 |
| Persistent solver arrays | 182.173 | 234.858 | 303.465 |

Native explicit-allocation counters and CUDA default-pool allocated high
watermarks measure live allocation; they exclude contexts and reserved pools.
LI's peak includes checkpoint-state temporaries. The separate process-resident
probe is retained and is not substituted for either row above.

## Independent hydraulic comparison

Scalar discharge is sampled every 60 s; it is interval-mean net export,
not instantaneous discharge. Pointwise event maxima are not storage volumes.

| Versus external FV1 | FloodWarp FV1 | FloodWarp LI |
|---|---:|---:|
| Final depth RMSE, m | 0.00024050 | 0.00361645 |
| Wet-union final depth RMSE, m | 0.00404242 | 0.06039035 |
| Maximum-envelope RMSE, m | 0.00143481 | 0.25637980 |
| Largest maximum-envelope error, m | 0.119908 | 6.243813 |
| Inundation IoU, 0.01 / 0.10 m | 0.99798 / 0.99838 | 0.97759 / 0.97284 |
| Hydrograph NRMSE | 0.0002804 | 0.0188316 |
| Peak discharge, m³/s (reference 161.483) | 161.173 | 182.167 |
| Peak error / timing difference | −0.192% / +60 s | +12.809% / −240 s |
| Cumulative export difference, m³ | +12.750 | −836.525 |
| Final storage difference, m³ | +3.989 | −62.560 |
| Infiltration difference, m³ | −16.740 | +899.085 |
| Final balance residual, m³ | 4.48e-9 | 3.11e-7 |

Reference final balance residual is 2.28e-8 m³, against 3.809 million m³ rain.
Every sampled normalized balance satisfies 1e-8. Percentiles, signed bias,
boundary/interior errors, full ledgers and maximum-map relative errors are
retained in the JSON. Wet union is reported because whole-domain percentiles
are dominated by dry cells. The 1e-5 m indifference band excludes only cells
near the practical inundation thresholds; no event cells were excluded here.

FV1 passes the frozen final-depth criterion. LI fails it, with a 13.61%
relative L2 error, and its large transient maxima are materially different.
The whole-event hydrograph NRMSE does not excuse its 12.8% peak error.
Differences in advection, momentum/source treatment and CFL conventions are
documented formulations, not reasons to relax the accuracy assessment.

The 36-hour coastal 5 m control has LI peak error −11.95% and 0.10 m IoU
0.9074. Halving CFL changes these little; repeat-cell terrain refinement to
2.5 m reduces peak error to −6.56% and raises IoU to 0.9583. This is a bounded
crop refinement, not an independent high-resolution DEM or full-domain 2.5 m
event. FV1 peak errors are −0.686%, −0.516% at CFL 0.2 and −0.761% at 2.5 m.
The dry first-hour cases have no meaningful wet-field/hydrograph coverage.
[Event maps and hydrographs](qualified_measurements/event_72h_comparison.png).

## Repairs, restart and controller decision

Completed checkpoint flags now advance before forcing prediction/CFL; accepted
steps use a common forcing interval. Rising stages use a local head-variation
bound and consistent trial heads, with constant-stage behaviour preserved.
All four masked orientations, multiple grid spacings, CPU, CUDA windows,
graphs and supported SGS modes have regressions and timestep-refined controls.
Earlier repairs cover FREE pressure continuity, Green–Ampt safeguarded
convergence, atomic checkpoints, native-stage preservation, pre-mutation stage
validation and public/LTS graph selection/invalidation. FV1 recession after a
finite hyetograph now continues with zero rain, including nonzero final nodes.

The benchmark restores exact hydraulic/soil state, clock, cumulative ledger
and completed samples. Atomic manifests reference immutable checkpoint
generations; interrupted publication preserves the previous restart. Source,
configuration and checksum drift are rejected. Non-multiple durations end at
the requested time. CPU/CUDA graph/window interrupted-versus-continuous tests
cover LI/FV1; the adapted native executable is tested with a moving wet front.
Hourly native stop/read continuations also completed the actual 72-hour event.
Mixed-precision CUDA capture preloads supported LI kernels explicitly rather
than compiling unrelated unsupported SGS modules; both device and LTS captures
use this path. The failed pre-fix log and passing regressions are retained.

Keep enhanced SGS `storage` as default and both SGS modes experimental.
`provisional` removes the infiltrated-rain bottleneck: 300 versus 1 steps,
46–58× faster on dry GPU controls with zero surface depth and exact accounting.
However it is 72% slower on the wet control, and unrefined coastal/steep tests
fail the 1 mm-or-2% criterion. Storage itself fails its minimum-step guard in
the ratio-2 coastal control; this is retained as a limitation. Ratios 2/4,
partial infiltration, ponding, wet/dry, disconnected and irregular native
geometry conserve component groups to normalized residual below 9.2e-14.
No independent evidence qualifies SGS as a full-cell FV1 replacement.

## Measured bottlenecks and optimisations

1. FV1 accepts 1.92× as many steps as native on this event. Its unsplit 2D CFL
   and native directional CFL are distinct; changing the equations or stability
   rule to make speeds agree is outside this repair. Optimised graph profiles
   attribute 47.3% of kernel time to flux faces and 22.1% to state update.
   Faces are FP64 compute bound (about 85% compute, 14–18% DRAM utilisation).
2. Exact-zero reconstructed-depth/inactive face bypasses reduce the controlled
   wet FV1 runtime by 6.9% (467 unchanged steps), and dry first-hour time by
   about 21%. No positive depth is approximated as zero.
3. LI's scalar forcing scans previously used about 12% of kernel time.
   Binary search reduces a controlled wet workload by 10.7% (199 unchanged
   steps/accounting). Completed event time is 4.85% lower than the old hourly
   protocol, with minute samples now adding 0.27% more steps. That event
   comparison includes the changed sampling cadence.
4. SGS coefficient construction, sources and wave bounds account for roughly
   21%, 21%, 11% of profiled kernel time. Many remaining launches last about
   2 μs; graphs reduce launch overhead. Provisional wet work remains additional
   work, not a universally faster controller.
5. Full-event diagnostics/checkpoint time is about 105 s for LI and 102 s for
   FV1. LI checkpoint cloning raises its live allocation peak to 614 MiB.
   Native diagnostics/output sum to 44 s plus 22 s process initialisation.

Profilers ran separately from timing. [Before/after records and CSV profiles](profiles/optimisations.json),
[compiler/profiler versions](toolchain.json), [reference revision](reference_revision.json),
[adapter diff](reference_operator_adapter.patch), [build provenance](reference_build_report.json),
and [lossless completed records/fields](completed_runs/manifest.json) bind the
measurements. Source snapshots bind actual benchmark code; subsequent mixed
storage casts are identities in FP64, and scoped mixed preloading leaves the
FP64 capture path and hydraulic operators unchanged. Original Overlander data/source were
read only. No physical parameters, defaults or engineering gates were tuned
to manufacture agreement; the repository remains private and unqualified for
unrestricted release.
