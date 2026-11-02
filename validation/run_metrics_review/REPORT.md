# Per-run metrics and retained benchmark audit

Copyright Lincoln Agritech Ltd. Developed by Patrick Durney.

Main source tested and built: `6a67851189d377aa894ec05304f62065a072fb23`
(v0.1.3). Dev CPU source: `41f86a024c5313556eaab3b1da6005d0c0530a2b`
(v0.2.0.dev0). This evidence-only follow-up does not change that runtime source.
Static-mesh research remains on dev. No hydraulic operators, advance targets,
precision, CFL policy, forcing, boundary closures or checkpoint serializers
changed. No fresh 72-hour event was requested or run.

## New reporting checks

| Check | Result |
|---|---|
| Main full source suite, CUDA hidden | 1,029 passed, 634 skipped, 350 warnings |
| Dev full source suite, CUDA hidden | 1,048 passed, 642 skipped, 350 warnings |
| Main full source suite, devices visible | 1,641 passed, 59 skipped, 743 warnings |
| Dev devices-visible working implementation | 1,668 passed, 59 skipped, 743 warnings |
| Focused reporting/input/server/replay/product controls | 65 passed, 1 skipped |
| Actual Firefox Run action, main checkout | Completed exactly at 120 s; native counters displayed and files downloadable |
| Actual Firefox Run action, installed v0.1.3 wheel outside checkout | Completed exactly at 120 s; stamped source, native counters, output files and attribution checked |
| Fresh core-only wheel installation | Default FV1 and explicit LI complete; optional UI dependencies absent; pip check passes |
| Installation with wheel UI extra | pip check passes; installed interface and worker exercised on port 8766 |
| Wheel/source metadata and contents | Pins preserved, UI optional, metrics included; no static-mesh runtime, local caches or sensitive paths found |

CPU-only suites use `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q` in pinned
Python 3.14.7 environments. Devices-visible suites use ordinary `python -m pytest -q`;
the latter dev run began before committing the working implementation, so its
total is not represented as an exact-main suite. The CPU suites and main suite
have the source identities above. Skips retain optional CUDA/external/opt-in
conditions; warnings include retained experimental SGS warnings. No tolerances
were changed. These are functional regressions, not uncontended GPU timing
benchmarks; another GPU campaign was active and was not stopped. No dedicated
performance run or additional opt-in CUDA campaign was started.

The focused controls check weighted timestep aggregation, direct-API equality,
uniform/all-ones/spatial rainfall, invalid inputs, output georeferencing,
exact final times (2, 20, 22 and 23 s) and incomplete metrics on failure with
the last checkpoint retained. Reporting consumes the native `advance` result
and checks its accumulated step count against the solver's counter. Timing
scopes, completed-advance coverage on failure, compilation, graph capture and
worker overhead are explicit in the files and [guide](../../docs/UI_SPATIAL.md#runtime-and-timestep-statistics).

The installed browser control uses the packaged tiny synthetic DEM/domain,
uniform rainfall, Closed boundaries, CPU and ordinary FV1 defaults. It records
12 accepted steps across three diagnostic intervals, dt min/mean/max
0.5125645804 / 10 / 17.4874354196 s, hydraulic time 0.0089203429 s and worker
wall time 0.6349901720 s. This warm-cache installation control is not a full-event
benchmark or an extrapolated speedup. Its complete results remain locally at:

`outputs/run-metrics-review/floodwarp-20261102T013347Z-ykfbdhtw/`

[Installed browser record](browser-wheel.json), [screenshot](browser-wheel.png),
[source browser record](browser-source.json) and [verification identities](verification.json).
The browser exercised validation and the actual Run button; runner tests are
reported separately. The installed server used `floodwarp.ui.serve` on 8766
because the owner's existing UI occupies 8765. The documented
`python -m floodwarp.ui` launch remains unchanged from prior verified evidence;
the owner's server was not restarted or stopped.

## Retained event evidence

`python -m validation.run_metrics_audit` recomputes the retained event totals,
verifies the lossless-record checksum and checks the four native runtime file
identities against the original source signature. The [machine-readable audit](retained_metrics.json)
identifies the original measured commit, not the current UI version as a new
measurement. Retained modified-sparse FV1: 259,200 s simulated, 1,570,053 accepted
steps, 1,512.3722854680382 s hydraulic and 1,599.1639685999835 s recorded component
time. The earlier dense and released-sparse measurements remain unchanged and
are labelled by implementation in the README. Historical analytical,
independent comparison, graph/restart and event records remain intact.

## Updated local artifacts

Fresh stamped distributions from the tested main source are in
`outputs/release/v0.1.3-metrics/dist/`. Earlier prepared v0.1.2/v0.1.3 files,
tags, data archives and assets are preserved. Use this new directory for the
metrics update; do not mix its checksums with older v0.1.3 candidates.

| Artifact | SHA-256 |
|---|---|
| `floodwarp-0.1.3-py3-none-any.whl` | `52464a4c88b98cc12f5746a1c896b57d79e42818a6edd094aa389ca960803754` |
| `floodwarp-0.1.3.tar.gz` | `56e601e9b5422e621a2689b615e0823eb188373494bdd5a0c04b6a1e18b6de44` |

[Artifact audit](artifact-audit.json). The build exports the committed source,
stamps its identity and builds the wheel through the source distribution.
Installation and browser checks use site-packages outside the checkout, with
no editable installation or source-path injection. The numerical dependency
pins remain unchanged. The path/content inventory found no accidental local
files or credentials; this is not a new audit of all historical Git objects.

No release, asset upload or repository visibility change accompanies these
branch updates. After the owner makes publication changes, verify anonymous
repository/data downloads and checksums, install the chosen wheel with `[ui]`
in a fresh environment outside the checkout, run `pip check`, launch
`python -m floodwarp.ui`, and complete the small synthetic example before
publishing the chosen tag/release and matching assets.
