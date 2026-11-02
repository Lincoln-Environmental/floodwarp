# Current runtime and timestep metrics

Copyright Lincoln Agritech Ltd. Developed by Patrick Durney.

Main v0.1.3 uses the modified exact-sparse uniform FV1 implementation measured
in the [retained 72-hour Orton Bradley event](../validation/sparse_modified_event_review/REPORT.md).
The four native runtime source files are checked against that record's SHA-256
identities; changing interface metadata or attribution does not relabel the
measurement as a new event.

| Retained modified-sparse metric | Value |
|---|---:|
| Source commit of measured event | `d34acab3c33f3be92bd94d49f8d23d859422da1b` |
| Actual final time | 259200 s |
| Accepted steps | 1570053 |
| Hydraulic runtime | 1512.3722854680382 s |
| Recorded component runtime | 1599.1639685999835 s |
| Minimum accepted dt | 0.00006483119796030223 s |
| Mean accepted dt | 0.16508996830043318 s |
| Maximum accepted dt | 1 s |

These values are recomputed from `complete_record.json.gz`, not estimated from
an older README table. The **1906.106 s** result is the retained **dense** FV1
control. The **1641.957 s** result is released sparse v0.1.1. Both have the same
accepted step count and recorded hydraulic fields. The retained independent
LISFLOOD-FP run took 1403.559 s hydraulic time and 816514 steps; it uses its own
independent discretisation and accepted timestep sequence.

Matched benchmark settings include 5 m terrain, original rainfall/soil/roughness,
212 explicit stage contacts, FP64/contact HLL, CFL 0.4, g 9.80665, dt bounds
1e-6–1 s and discharge-extrapolated prescribed-stage momentum. Ordinary FV1
uses velocity extrapolation. The runtime claim does not apply indiscriminately
to different closures, grids, scenarios or hardware.

New UI runs record their own counters and timing scopes in `metrics.json` /
`runtime.csv` and display the step count and dt min/mean/max. See the
[UI metrics explanation](UI_SPATIAL.md#runtime-and-timestep-statistics).
Worker wall time includes a different workload from the historical component
timer, and excludes process startup/browser validation. Timing from a small
installation control is not a full-event benchmark or an extrapolated speedup.

No fresh 72-hour event is part of this update, as requested. Another GPU
campaign was active during development; it was not interrupted, and contended
GPU timings are not presented as a clean benchmark. New checks use bounded CPU
runs to verify reporting and parity, while the retained event records remain
unchanged. Numerical agreement, observed-event validation and grid convergence
remain distinct. Original tags, assets and measurements are preserved.

The machine-readable source/record audit and current reporting checks are in
[the metrics review](../validation/run_metrics_review/REPORT.md).
