# Matched 72-hour exact-sparse FV1 event

Completed 72 h with **1,570,053 accepted timesteps**. All 72 hourly comparisons have zero numerical field difference against retained dense FP64 FV1, including conservative state, sources/soil, boundaries, maxima, accepted face volumes and timestep histograms.

Hydraulic time: **1641.957 s**, retained dense **1906.106 s** (**1.161×**). Active-process time including validation/IO: 1742.796 s. Observed publication wall time: 5378.5 s, including the deliberate pause for the v0.1.0 release. Do not describe that pause as solver compute time.

| Event interval [h] | Sparse hydraulic [s] | Retained dense [s] | Dense / sparse | Dense-cost-weighted endpoint cell fraction |
|---|---:|---:|---:|---:|
| 0–11 | 9.500 | 33.023 | 3.476 | 0.0000 |
| 11–20 | 139.533 | 153.649 | 1.101 | 0.4373 |
| 20–31 | 473.205 | 439.989 | 0.930 | 0.6407 |
| 31–48 | 687.646 | 695.366 | 1.011 | 0.5239 |
| 48–72 | 332.074 | 584.079 | 1.759 | 0.1764 |

Weights use actual retained dense-window cost. Occupancy is sampled at minute ends and must not be claimed as an exact per-step work integral. The full-event comparison is one candidate run against the retained matched control; the saved-state short windows separately have warm-up and three repetitions. Profilers ran outside timed replay.

Steady allocation remains dense: 246,386,556 bytes, including 119,572 scheduling bytes. There is no memory-saving claim. Saved-state peak allocations are recorded in benchmark.json; whole-run allocation peaks were not instrumented, and checkpoint recovery temporarily constructs a second solver.

Sources and full source/boundary-aware CFL remain active over the physical domain; no shallow water is deleted. Dry and recession workloads benefit, while widespread runoff retains almost all physical hydraulic work and can be slower. Dense execution remains the default/fallback. No sparse-storage or AMR speedup is inferred or multiplied into these results.

The uninterrupted physical trajectory survives the authorised release-work interruption. The native checkpoint hash, exact recovery continuation, complete minute records and 72 hourly field checks are retained. The source snapshot binds the actual timed implementation.
