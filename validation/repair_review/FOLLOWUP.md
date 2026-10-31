# Follow-up repair and qualification

Status: all ten benchmark/control cases have completed, including the three
full-domain 72-hour runs and bounded 2.5 m/CFL controls. Hydraulic repairs
pass the pinned CPU/CUDA suites. Completed measurements, physical disagreements
and experimental SGS limitations are reported separately in
[the final assessment](FINAL.md); completion does not imply hydraulic agreement.

## Controller and forcing repairs

Completed output flags are consumed before the next cap, forcing prediction
or CFL calculation. The regression uses the requested dry 1 m grid, 0–72 mm/h
rain ramp from 30–60 s and a 30 s checkpoint. Traced/untraced accepted steps,
source-aware CFL and water accounting agree for coarse, traditional and
enhanced execution on CPU/CUDA, queued windows and automatic graphs.

Changing stages use conservative trial heads during CFL selection and actual
midpoint stages during the accepted hydraulic update. A local head-change
bound responds to the external slope and the interior flux-induced head rate.
Its 2.5% increment resolves changes in quadratic pressure forcing to about 5%;
the 0.1 mm regularisation is in metres relative to the sill, independent of
vertical datum. Hydraulically inactive below-sill stages can advance to wetting.
Constant series add no variation restriction. A short remainder to a forcing
node is kept separate from the uncapped stability limit, avoiding false
minimum-timestep failures.

Triangular and rapid pulses cover all four masked-stage orientations, 1/5/20 m
grids and both SGS modes, with 0.005/0.002 s refined controls. Tests distinguish
peak-depth/volume/hydrograph response from convergence of late reflected waves.
The JUnit properties retain unrefined final-depth errors as diagnostics;
first-order reflected-wave errors are not presented as zero. The 18.5 ms wet
tidal pulse explicitly permits sub-ms steps for that test; production minimum
timesteps are unchanged. Strict CPU/CUDA comparison tolerances are retained.

FV1 now permits zero-rain recession after the final hyetograph node, including
a nonzero terminal rate. Rainfall to that node retains its exact piecewise
linear integral. Regressions cover continuous, segmented and single-step
recession with strict water accounting. This fix is pushed as `195ecb8`.

## Tests and preserved evidence

- Final full opt-in CUDA suite: **1,535 passed, 24 skipped**; includes the external
  reference's continuous/restarted equivalence test and CPU/CUDA window/graph
  benchmark restart tests. [Final JUnit evidence](final_integrated_cuda.xml).
- Full CPU-only suite: **919 passed, 603 skipped**. CUDA skips are not CPU
  coverage. Numerical defaults, historical fixtures and strict/v1/v2 artifacts
  remain preserved. Earlier failed follow-up logs are retained in
  [checksummed compressed archives](failures/manifest.json).
- Analytical timestep/grid convergence and historical review reproductions:
  [analytical controls](analytical_convergence.json), [sensitivity](sensitivity.json)
  and the existing strict review results. The seven retained strict discrepancies
  remain visible; engineering acceptance does not require bitwise equality.
- Additional CUDA precision controls: FP32 **68 passed, 3 skipped**;
  conditioned mixed **39 passed, 32 skipped**. Mixed SGS remains unsupported.

## SGS controller decision

Keep `storage` as the production default at this stage; `provisional` and
`wave` remain explicit options. All SGS remains experimental. No `minimum_dt`
was lowered in controller qualification.

The expanded bounded qualification covers fully infiltrating sloping rain,
ponding onset, wet/dry recession, disconnected native regions, irregular
geometry, coastal stages, steep terrain and coarse/native ratios 2 and 4.
It checks finite positive states, component stage/volume reconstruction,
connected-group conservation and refinement at 5/0.05/0.01 s timestep caps.
These are internal convergence controls, not independent hydraulic evidence.

The source-consistent provisional controller removes the fully infiltrated
dry-state restriction. However, the 5 s cap produced a 1.34 mm / 3.61% coastal
depth discrepancy and a 1.64 mm / 7.19% steep-case discrepancy against the
0.01 s control at ratio 2. These fail the frozen 1 mm-or-2% depth gate. The
storage controller also reaches its configured minimum-step failure threshold
in the ratio-2 coastal case at all three caps; that limitation is retained.
Neither controller is declared generally qualified from speed or completion.

## Historical recovered 72-hour physical results

All three solvers reached 259,200 s on the same RTX 4070 Ti SUPER, 5 m terrain,
domain, rainfall, exponential soil fields, roughness and coastal ports. The
FV1 comparison explicitly uses discharge-extrapolated stage ghosts. Native
directional and FloodWarp unsplit CFL conventions remain distinct even at the
same nominal CFL 0.4. No physical parameter was tuned for agreement.

| Quantity | FloodWarp FV1 | FloodWarp LI |
|---|---:|---:|
| Final-depth RMSE versus LISFLOOD-FP FV1 | 0.000241 m | 0.003617 m |
| Maximum-envelope RMSE | 0.001435 m | 0.256351 m |
| Inundation IoU at 0.01 m | 0.99798 | 0.97759 |
| Inundation IoU at 0.10 m | 0.99838 | 0.97284 |
| Final water-balance residual | 2.75e-8 m³ | 2.77e-6 m³ |

[Checksummed retained comparison](retained_72h_comparison.json). LI maximum
depth discrepancies are material. Missing advection, differing momentum/source
splitting and timestep/grid effects require assessment; they are not excused
as rounding. FV1's pre-interruption scalar samples are incomplete, so no
whole-event FV1 peak-discharge/timing claim is made from this recovered run.
LI's historical hourly hydrograph is identified as hourly evidence only.
The fresh minute-sampled results in FINAL.md supersede these values for the
primary performance and hydrograph comparison. Historical results remain intact.

## New benchmark protocol

`python -m validation.build_reference` copies the retained source/build into
ignored workspace outputs, adapts host output cadence and timing, and verifies
unchanged hydraulic object hashes. It never modifies Overlander.

`python -m validation.run_qualification` runs small analytical cases, the 40×40
coastal crop and the full-domain one-hour case serially. Set its main-block
scope selection for long, grid and CFL controls. Short cases warm up and use
three measured repetitions. Scalar samples are 1 s for analytical cases and
60 s for Purau; full fields/checkpoints use hourly cadence. The exact final
target is included for non-multiple durations and actual solver time is checked.

Both solvers restore validated state and cumulative water accounting, scalar
samples and timings from an atomic progress manifest referencing an immutable
checkpoint generation. The reference uses its existing checkpoint-read path.
Configuration/source, executable, driver and output checksums reject drift;
interrupted publications preserve the last committed generation. Completed
cases are integrity-checked and reused. Logs/heartbeat detect crashes or stalls.

Peak allocated device memory and persistent array capacity are separate.
FloodWarp resets CUDA's allocation high watermark for each run and includes
checkpoint temporaries; native FV1 records explicit CUDA allocations. Both
exclude context/reserved-pool memory. Complete diagnostics/output timing is
separate from hydraulic execution; no persistent/peak comparison is labelled
equivalent. The one-hour case is entirely infiltrated and has no meaningful
wet-field or outlet-hydrograph coverage.

## Measured optimisations and progress comparisons

The FV1 face kernels originally consumed about 51% of GPU kernel time. Nsight
Compute measured approximately 85% compute utilisation and 14–18% DRAM
bandwidth utilisation. Exactly inactive or zero reconstructed-depth pairs now
clear both flux/pressure outputs and skip the identically zero Riemann solve.
Boundary ghosts are constructed first; no nonzero depth is approximated as dry.
The matched wet FV1 workload improved from 0.826361 to 0.769421 s (6.9%, three
repetitions, identical 467 steps and water accounting). The full dry first hour
improved from 3.812543 to 3.003228 s, about 21%.

LI's three scalar forcing kernels spent roughly 12% of kernel time scanning
long series. Binary searches preserve segment selection and the existing
interpolation rounding barrier. The repeated LI workload improved from
0.356095 to 0.318122 s (10.7%, identical 199 steps/accounting). These are
within-method comparisons from separate saved wet states; their runtimes must
not be compared across solvers as equivalent hydraulic workloads.

`python -m validation.compare_progress` compares common completed event
prefixes directly from checkpoints, checking the physical inputs and controls.
[Saved prefix comparison](progress_speed_comparison.json) records the measured
gain and the changed hourly/minute output cadence, including extra accepted
steps. The completed LI event used 1019.460 s versus 1071.396 s previously
(4.85% less hydraulic time). This is an observed whole-event change, with
minute instead of hourly scalar output and 694389 versus 692533 accepted
steps; the controlled 10.7% workload comparison above isolates the optimisation.
The old FV1 recovered timing includes a separately resumed tail and is not
used as a clean before/after whole-event speed measurement.

The [measurement tables](qualified_measurements/MEASUREMENTS.md) and their JSON
record source snapshots, input signatures, repetition ranges and publication
wall intervals. Component timers omit some Python checksum/manifest overhead;
fresh-directory birth through final record data-write captures that overhead
separately. Shared input loading and final directory fsync remain outside that
interval. Native end-to-end time includes hourly process initialisation for
its existing stop/checkpoint/read mechanism; hydraulic timing excludes it.

Nsight Systems 2022.4 produced an incomplete LI trace because it could not
decode a newer CUDA API. That trace is retained as failed evidence. The
[official NVIDIA CLI](https://developer.nvidia.com/nsight-systems/get-started)
2026.5.1 produced complete LI, FV1 graph and SGS traces. Profiling ran separately
from timing measurements. Install the optional `profile` extra for the pinned
NVTX package; core numerical dependency pins are unchanged.

The [reference revision check](reference_revision.json) binds the retained
upstream revision and verifies matching HLL sources. The accompanying
[operator/adapter diff](reference_operator_adapter.patch) shows the explicit
coastal ghost and diagnostic hooks; upstream hydraulic objects were reused.

Additional CPU precision checks cover FP32 forcing/stage regressions and
conditioned mixed precision. The storage inversion needed its FP32 input cast
to the bed precision before subtraction; the cast is an identity in FP64.
Test volume tolerances now include state-precision roundoff scaled by supplied
m³, while retaining the original FP64 bound. Mixed SGS remains unsupported by
the existing constructors and is explicitly skipped rather than simulated.
Conditioned mixed CUDA graphs now preload their LI kernel module explicitly,
avoiding global eager compilation of imported, unsupported SGS modules. This
also applies to spatial-LTS capture. Five coarse graph tests originally failed
with an SGS type error; the same focused suite now passes 39 tests with 32
unsupported-mode skips. The failure log is retained beside the passing evidence.

[Original completed records](completed_runs/manifest.json) retain each short
repetition and all minute scalar samples, source signatures and timing statistics
as lossless gzip archives. Full-event final-depth and maximum-depth arrays are
also archived byte-for-byte. Unpack a run's record and fields into their original
names to reproduce its recorded checksums; large restart generations remain in
the ignored local output directories.
