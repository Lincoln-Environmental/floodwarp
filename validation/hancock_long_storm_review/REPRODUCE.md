# Reproducing the Hancock investigation

Use the repository's pinned Python 3.14 environment and one otherwise idle CUDA
device. The timings here used an RTX 4070 Ti SUPER. Run modules from the
repository root. The tests can also run on CPU. The new scientific inputs use
the fixed seed and SI parameters in `cases.json`.

## Original baseline

At commit `018032761146a915bd6089f858a97500715e3758`, run the retained driver
in the unmodified checkout. Give the output a fresh path. The following call
adds the two historical sparse Hancock modes to the original seven-repeat
protocol.

```python
from pathlib import Path
from validation.hancock_default_profile import measure

if __name__ == "__main__":
    output = Path("outputs/hancock_unmodified_baseline.json")
    cases = ("stoker", "dry_patch")
    methods = (
        ("contact_hll", "sparse"), ("hancock_adaptive", "dense"),
        ("hancock_adaptive", "sparse"), ("hancock_adaptive", "dynamic"),
    )
    measure(output=output, cases=cases, methods=methods, length=1024., width=640.,
            spacing=1., cap=1/32, duration=2., repeats=7, device="cuda:0")
```

## Numerical and GPU controls

```sh
.venv/bin/python -m validation.hancock_long_storm_review.performance
.venv/bin/python -m validation.hancock_long_storm_review.regression
.venv/bin/python -m validation.hancock_long_storm_review.analytical
.venv/bin/python -m validation.hancock_long_storm_review.timestep
OVERLANDER_TEST_FULL_SWE_CUDA=1 FLOODWARP_TEST_RENDER=1 .venv/bin/python -m pytest -q -rs
```

The archived qualification replay compares numerical trajectories at their
historical caps. The separate analytical driver evaluates 1, 5 and 10 m grids,
both timestep policies, space/time manufactured refinements and 57 short
normal-policy controls per policy. The sloping travelling-front diagnostic
examines each trial on CPU, including rejected ones.

## Complete events and references

```sh
.venv/bin/python -m validation.hancock_long_storm_review.terrain_inputs
.venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=large .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=terrain_large .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=remaining_refinements .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=fine_check .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=resolution_check .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=moving .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=directional .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=reference .venv/bin/python -m validation.hancock_long_storm_review.storms
FLOODWARP_STORM_SUITE=refinement .venv/bin/python -m validation.hancock_long_storm_review.storms
.venv/bin/python -m validation.hancock_long_storm_review.sheet_cycle
.venv/bin/python -m validation.hancock_long_storm_review.structures
FLOODWARP_STRUCTURE_DEVICE=cpu .venv/bin/python -m validation.hancock_long_storm_review.structures
```

The storm driver publishes one JSON per complete run. The event and field directories can be set with
`FLOODWARP_STORM_RESULTS` and `FLOODWARP_STORM_FIELDS`. Set
`FLOODWARP_STORM_DEVICE` to select a device. Read `suites` in `cases.json`
for solver groups, domain sizes, hydraulic and DEM spacings, output cadence
and simulation durations. Missing field archives regenerate from the published settings. The driver checks
the hydraulic totals and preserves the published timing. Choose fresh event
and field directories to repeat timings.
Use one directory per configuration revision. A field archive under
`outputs/hancock_long_storm_review/fields` contains the depth at every sample,
the final depth, accepted-event maximum, first wet sample and last wet sample.
Large-domain records retain hydrographs, ledgers, statistics and timings with
no grid-sized output files. The `moving` control additionally records SHA-256
hashes of the final hydraulic fields.

## Attribution and figures

```sh
.venv/bin/python -m validation.hancock_long_storm_review.registers
.venv/bin/python -m validation.hancock_long_storm_review.compact
.venv/bin/python -m validation.hancock_long_storm_review.compact_events
.venv/bin/python -m validation.hancock_long_storm_review.strip_activity
.venv/bin/python -m validation.hancock_long_storm_review.fallback
.venv/bin/python -m validation.hancock_long_storm_review.switching
.venv/bin/python -m validation.hancock_long_storm_review.capacity
.venv/bin/python -m validation.hancock_long_storm_review.source_schedule
.venv/bin/python -m validation.hancock_long_storm_review.wet_profile
.venv/bin/python -m validation.hancock_long_storm_review.kernel_events
.venv/bin/python -m validation.hancock_long_storm_review.graph_profile
.venv/bin/python -m validation.hancock_long_storm_review.retry_growth
.venv/bin/python -m validation.hancock_long_storm_review.storm_cfl
.venv/bin/python -m validation.hancock_long_storm_review.accuracy
.venv/bin/python -m validation.hancock_long_storm_review.interpret
.venv/bin/python -m validation.hancock_long_storm_review.summarize
.venv/bin/python -m validation.hancock_long_storm_review.results_text
.venv/bin/python -m validation.hancock_long_storm_review.figures
.venv/bin/python -m validation.hancock_long_storm_review.verify
```

Full-event kernel profiles are separate uncaptured replays. The main runtime
table uses production CUDA graphs, each method's own accepted timesteps and
synchronised output boundaries. Running other GPU workloads concurrently will
affect these numbers.

## Integrated source operator

The scheduler and structure implementation is available at `2914077`. Its
storm and timing files are under `events/`. The exact constant-capacity
source specialisation entered the numerical implementation at `e63c936`.
Use the final checkout for the complete validation drivers. The production
repetitions write to `integrated_events/` and `integrated_fields/`. Each
suite publishes a receipt after comparing its sampled FP64 fields, accepted
step statistics, hydrographs and budgets against the scheduler revision.
The tabulation scripts select new measurements only for verified suites and
label the timing source in the exported event table.

```sh
.venv/bin/python -m validation.hancock_long_storm_review.integrated
.venv/bin/python -m validation.hancock_long_storm_review.step_history
.venv/bin/python -m validation.hancock_long_storm_review.kernel_tables
FLOODWARP_PROFILE_FILE=integrated_profile.json FLOODWARP_PROFILE_ONLY=1 .venv/bin/python -m validation.hancock_long_storm_review.performance
FLOODWARP_REGRESSION_RESULTS=validation/hancock_long_storm_review/integrated_regression .venv/bin/python -m validation.hancock_long_storm_review.regression
FLOODWARP_ANALYTICAL_RESULTS=validation/hancock_long_storm_review/integrated_analytical .venv/bin/python -m validation.hancock_long_storm_review.analytical
OVERLANDER_TEST_FULL_SWE_CUDA=1 FLOODWARP_TEST_RENDER=1 .venv/bin/python -m pytest -q -rs > /tmp/floodwarp_hancock_integrated_tests.log
.venv/bin/python -m validation.hancock_long_storm_review.test_evidence
.venv/bin/python -m validation.hancock_long_storm_review.accuracy
.venv/bin/python -m validation.hancock_long_storm_review.interpret
.venv/bin/python -m validation.hancock_long_storm_review.summarize
.venv/bin/python -m validation.hancock_long_storm_review.results_text
.venv/bin/python -m validation.hancock_long_storm_review.figures
.venv/bin/python -m validation.hancock_long_storm_review.verify
.venv/bin/python -m validation.hancock_long_storm_review.decision
```

The graph profiler uses two additional timestamp kernels bracketing each
launch. Their overhead changes timings. The full-event table groups donor
budgets and continuity with flux/update. In the interval table these kernels
are included in control/other. Fused FV1 operations share one measured time.


## Historical checkpoint

The `checkpoint/` directory contains a public-API checkpoint created at
`0180327`, its continuing simulation's 28 arrays and their checksum record.
Run `.venv/bin/python -m validation.hancock_long_storm_review.checkpoint_audit`
to compare the four current schedulers with that continuation. To independently
regenerate the fixture, use the old checkout's `src` as the working directory,
set `FLOODWARP_HISTORICAL_FIXTURE=1` and provide a fresh absolute directory
in `FLOODWARP_CHECKPOINT_AUDIT`. Execute the current audit script on Python's
standard input so imports resolve from that checkout. The driver checks the
imported implementation's Git revision before creating the reference.
