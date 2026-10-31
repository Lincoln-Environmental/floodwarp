# Sparse FV1 performance investigation on dev

Version 0.2.0.dev0 only. Stable v0.1.1 and all historical event/data products remain unchanged.

Retained improvements: clear scratch only in released work tiles; combine the unchanged source update with the unchanged post-source activity predicate. Both use dense storage and the identical hydraulic work set. Full source/boundary-aware CFL remains before sources, fine soil/history remains persistent, and each physical face keeps its existing unique owner. Retained tiles overwrite their complete face/pressure/limiter scratch before consumption. Restore invalidates scheduling caches; leaving dense indexing clears all released scratch. No positive water threshold or state deletion is introduced.

## Independent, interleaved timing controls

Six saved full-domain states; 30 s continuations; one warm-up per variant; three measured repetitions with rotated variant order. Graph throughput timing is separate from CUDA-event profiling. Reference scheduling is frozen FloodWarp v0.1.1, not external solver code. A wrapper changes only the reference kernel import; original released files and the actual timed candidate/driver snapshots are retained.

| Hour | Dense [s] | Released sparse [s] | Clearing only [s] | Fusion only [s] | Combined [s] | Forced dense indexing [s] | Combined reduction vs released |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.025066 | 0.007452 | 0.007738 | 0.007241 | 0.007523 | 0.007525 | -0.96% |
| 13 | 0.106372 | 0.079816 | 0.074915 | 0.077470 | 0.073861 | 0.120102 | 7.46% |
| 25 | 0.358598 | 0.385844 | 0.359541 | 0.378949 | 0.356345 | 0.395535 | 7.65% |
| 48 | 0.320507 | 0.300184 | 0.279455 | 0.293744 | 0.276805 | 0.355277 | 7.79% |
| 60 | 0.199530 | 0.109348 | 0.101113 | 0.106659 | 0.100620 | 0.232470 | 7.98% |
| 72 | 0.198353 | 0.107230 | 0.099393 | 0.104409 | 0.098529 | 0.231059 | 8.11% |

The combined path reduces wet-phase runtime by about 7–8%; the short dry case regresses about 1%. At peak it approaches explicit dense throughput; the small difference from dense is not a universal speedup claim. The forced 0.01 fallback fraction is rejected: extra full-raster work and scheduling are slower than the retained 0.65 policy, particularly in recession. Independent gains are not multiplied; combined timing is measured directly.

All compared conservative, source/soil, face scratch/pressure, accepted face transport, maxima, timestep histogram and ledger arrays agree exactly with the dense control on every measured repetition. Exact final time and accepted step count agree. Full CPU/CUDA regressions: **1638 passed, 25 skipped**, with the existing 743 warnings. Additional controls cover partial tile release while another wet region remains active, followed by reactivation; ordinary dense/sparse/graph/restart tests remain unchanged.

Persistent candidate allocation: **246,506,124 B**, including **239,140 B** scheduling, versus released sparse **246,386,556 B**. Extra device tile caches increase memory; this is not sparse storage. A separate one-solver-at-a-time probe at the peak checkpoint measures live allocator peaks of **246,506,124 B** for the candidate and **246,386,556 B** for released sparse execution. Memory.json records every variant. These peaks exclude CUDA context/code/reserved memory; complete-event/checkpoint-process peaks are not measured. The throughput benchmark's concurrent-control high-water marks are not isolated candidate peaks.

Profile.json separates source/activity, scratch clearing, compaction/dependency closure and hydraulic kernels. Halo handling is included in compaction; fused source/activity cannot be split within its kernel. First-step cache invalidation can make released-tile clearing expensive in short dry replay windows. Profiling uses instrumented host launches and is not substituted for graph timings.

Operator_audit.json verifies the original source and activity bodies are unchanged after extraction to shared indexed functions; every other original full-SWE function is AST-identical. Sources and full CFL are unavoidable work. New savings are redundant scratch writes, duplicate state reads and a launch, not fewer hydraulic cells or a different discretisation.

## Stage decision and limits

Retain the measured combined improvement on dev; keep explicit dense execution and the 0.65 fallback. No new complete 72-hour event has been run for this candidate. The published 1641.957 s sparse event timing continues to describe the frozen v0.1.1 implementation, not this investigation. A complete matched event and isolated peak-memory audit remain gates before a subsequent stable release. CPU speedup and other GPU models are not demonstrated. Next bottlenecks are full-domain CFL/sources, initial invalidation clearing, and high-occupancy scheduling/indirection. Source/runtime snapshots and unsuccessful configurations are preserved.

Subsequent matched 72-hour verification is now recorded in the [full-event comparison](../sparse_modified_event_review/REPORT.md); the original measurements and scope above are preserved.
