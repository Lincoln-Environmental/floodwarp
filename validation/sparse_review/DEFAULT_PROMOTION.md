# Exact-sparse FV1 default on dev

The user authorised promoting the validated exact-sparse configuration after
the complete matched Orton Bradley Park event. Native `FullSWESolver` and both
public factories now default to `execution_mode='sparse'`, `tile_size=8`,
`dense_fraction=0.65`. LI selection and numerical defaults are unchanged.

This removes only demonstrably zero uniform-grid hydraulic calculations.
Storage remains dense FP64; sources, soil history, full-domain source/boundary
CFL, Riemann solver, boundary closures, limiting, accepted transport and maxima
are unchanged. Dense indexing is used at high work-tile occupancy. Explicit
`execution_mode='dense'` retains the original execution without scheduling
overhead. Execution settings remain outside numerical checkpoint identity.

The [retained complete-event evidence](FULL_EVENT.md) records 1,570,053 accepted
steps and zero numerical field differences at all 72 hourly comparisons.
Hydraulic time was 1641.957 s versus 1906.106 s dense on RTX 4070 Ti SUPER:
13.9% less runtime. This is the existing completed replay, not a new timing
run after promotion. Dry/recession windows benefit; widespread-runoff windows
can be slower. There is no sparse-storage or AMR performance claim.

The promotion adds CPU/CUDA controls for native construction, solver factory,
direct-model factory, actual work omission, agreement with explicit dense
execution and disk restart into dense execution. Existing controls retain
source activation, wet boundaries, tiny positive water, complete donor budgets,
deactivation/reactivation, ordinary CFL and graph equivalence. Static mesh
construction explicitly selects its separate engine; timing drivers explicitly
retain dense reference selection.

Validation uses pinned Python 3.14.7, Warp 1.17.0 and RTX 4070 Ti SUPER:

- Focused sparse/static/direct-input controls: **98 passed**.
- `OVERLANDER_TEST_FULL_SWE_CUDA=1 python -m pytest -q`:
  **1624 passed, 25 skipped**, 743 existing warnings, 112.26 s.
- `pip check`: no broken requirements. Edited benchmark drivers compile.

Current source hashes are in `docs/numerical_integrity.json`. Original event,
static-control and release source snapshots and their measurements are
preserved. Static multiresolution remains experimental after its failed runtime
gate; dynamic AMR is not enabled. The fixed v0.1.0 tag, release assets and main
remain unchanged. This promotion is committed on dev.
