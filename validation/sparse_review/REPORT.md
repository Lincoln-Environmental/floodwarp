# Exact uniform-grid execution: first milestone

Remote refs fetched on 2026-10-31: origin/dev and origin/main at `183c884`.
Work continues on dev; pre-existing `.idea/` files are untouched. No external
solver implementation was copied or imported. Dense FP64 FV1 remains default.

The implementation retains dense state/geometry/history and uses device tile
flags, compact lists and batched grid-stride launches for faces, draining,
limiting and update. Activity is exact nonzero post-source state plus wet stage
ghosts, with the complete neighbouring-tile closure described in the design
note. Sources and source/boundary-aware CFL remain full-domain. Face ownership
is unique; clearing released scratch preserves cumulative transport and maxima.
High occupancy selects dense indexing. Explicit dense mode avoids all scheduling
overhead. Checkpoints remain portable across execution modes and tile sizes.

## Measured saved-state results

RTX 4070 Ti SUPER, pinned Python 3.14.7, Warp 1.17.0, FP64, CUDA graphs.
Each entry is median hydraulic seconds for a matched 30 s continuation after
warm-up, with three measured repetitions. Full records include end-to-end
time, accepted steps, per-step time, work counts, memory and all-field errors.

| Event hour | Dense | Sparse 8 | Sparse 16 | Sparse 32 | Dense / sparse 8 |
|---|---:|---:|---:|---:|---:|
| 1, infiltrating/dry | 0.0252 | 0.00745 | 0.00745 | 0.00744 | 3.39 |
| 13, rising | 0.1061 | 0.07966 | 0.08672 | 0.09658 | 1.33 |
| 25, widespread runoff | 0.3587 | 0.38583 | 0.43497 | 0.44825 | 0.93 |
| 48, late rain | 0.3194 | 0.29946 | 0.32527 | 0.40117 | 1.07 |
| 60, recession | 0.2000 | 0.11034 | 0.14456 | 0.20289 | 1.81 |
| 72, recession | 0.1989 | 0.10749 | 0.13973 | 0.19649 | 1.85 |

All compared fields, including momentum, sources, accepted face volumes,
maxima, limiter scratch and timestep histograms, match exactly in these replays.
Accepted step counts and end times also match. At hour 1, the minimum local CFL
candidate is outside the empty hydraulic set; the unchanged full scan retains
it, and the maximum timestep cap controls the accepted step. This is expected.

The profiled high-water kernels expose why sparse execution is slower there:
classification, clearing and indirection add work when almost every physical
cell contains water. Dense-index fallback preserves physics but does not remove
its scheduling overhead. Larger tiles have particularly poor recession halos.
The ordinary dense path remains the preferred high-occupancy control.

Persistent dense allocation is 246,266,984 bytes: conservative state 22,720,536;
geometry/boundaries 41,677,908; hydrology 53,014,584; face/limiter scratch
83,387,272; diagnostics/history 45,457,312; control/forcing 9,372.
Scheduling adds approximately 119.6 kB, 30.7 kB or 8.1 kB for 8/16/32 tiles.
There is no sparse-storage or AMR memory-saving claim.

Separate CUDA event profiles attribute classification, halo closure/compaction,
scratch clearing and hydraulic kernels. They use instrumented host launches,
so their totals must not replace graph throughput timings. Both measurements
include the initial full scratch clear after checkpoint restore.

## Qualification and decisions

`python -m pytest -q`: **1,552 passed, 58 skipped**, 743 existing warnings,
113.44 seconds, pinned Python 3.14. New sparse controls: **51 passed**, CPU and
CUDA, including common safe prescribed steps, ordinary adaptive stepping,
lake equilibrium, moving/oblique momentum, shallow nonzero storage, masked
geometry, prescribed-stage activation in all orientations, rainfall ponding,
fully infiltrating rain, heterogeneous soil, graph execution, deactivation,
reactivation, preserved maxima/transport, disk restart and execution switching.

First milestone passes: working exact sparse execution with measured benefit
on relevant saved workloads and unchanged hydraulics. This is still explicitly
experimental; it is not enabled by default or merged to main. The peak-flow
slowdown and unsuccessful 32-cell configurations are retained. Short-window
ratios are not an event speedup claim. A complete matched event is a separate
qualification task; no sparse/AMR speedup multiplication is valid.

Sparse storage and static multiresolution are separate stage decisions. A
conservative-state-only pool could save at most a small fraction of total
allocation during widespread runoff; face scratch is the more substantial
target. Coarsenability must be measured within wet regions independently of
exact sparsity. Static qualification budgets are recorded before results.

Evidence: `occupancy.json`, `benchmark.json`, `profile.json`,
`tests_cpu_cuda.txt`; retained hourly inputs are read-only and checkpoint hashes
are in the benchmark record. Historical repair/reference evidence is unchanged.
