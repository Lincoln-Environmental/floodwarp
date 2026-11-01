# v0.1.2 local publication candidate

Release hydraulic baseline: `bce98b103beca066cd0c9afdeec0fc330576182a`.
Live main also includes owner commits through
`731cf6d1c3a69eb1267875c90f67965e9f72c193`, changing only README/COPYRIGHT.
Both commits are preserved in candidate history; the updated holder is
Lincoln Agritech Ltd, with Patrick Durney development attribution retained.
This release adds the optional local rainfall interface, deliberate FV1 factory
default, exact-final-time example sampling, package/CI checks and public guides.
The full-SWE and sparse kernels/solvers are byte-identical to that baseline.
FP64, CFL, sources, boundary closures and checkpoint serializers are unchanged.
No static-mesh research was merged from dev.

The candidate is prepared locally. Visibility, releases, assets and remote
branches have not been changed. Historical tags, evidence and existing release
assets are retained. The rights holders' authorised AGPL-3.0-only terms and
separate CC BY 4.0 data credits are unchanged.

## Verification records

The build driver `python -m validation.build_release` exports a clean committed
candidate, stamps its commit into both distributions, builds wheel/sdist and
writes `outputs/release/v0.1.2/build.json` and `SHA256SUMS`.
The final handoff record `verification.json` binds the actual distributions,
exact source commit, source-suite logs and installed-package provenance.
An evidence-only follow-up commit can contain that record without changing the
candidate payload. Distinguish the artifact source commit from the final local
branch tip; do not call a development-suite total an exact release-tree run.

Pre-freeze focused checks cover default FV1/explicit LI, sparse/dense selection,
formulation/multiplier restart rejection, 2/22/23/20 s sampling at 5 s, UI/direct
API agreement, uniform/all-ones equivalence, spatially varying multipliers,
masking, polygon CRS/reprojection, invalid CSV/raster/empty-domain rejection,
output georeferencing/units and completed/failing execution. The deliberate
second-sample failure preserves a loadable first-sample checkpoint.
`tests_focused.txt` is labelled by its associated verification record; final
suite results and any unavailable checks are recorded separately.

`browser.json` and [the screenshot](../../docs/ui.png) describe actual Firefox
Validate and Run clicks, terrain/rainfall/maximum-depth previews, file selector,
refresh without restart and an invalid-input message. Chromium completed the
run but its compositor stalled during capture in this environment; Firefox
provided the completed screenshot. These interface checks are distinct from
headless runner tests. Browser tooling is a development-only dependency. A deliberately terminated
installed worker also verifies failed status/log access and retention of its
complete checkpoint. Incomplete atomic-write temporary archives are excluded
from result links; the checkpoint serializer itself remains unchanged.
The reproducible browser driver is `validation/ui_browser.py` (install
Playwright and its Firefox binary separately). It uses real control clicks;
forced clicks bypass this host's stalled animation-frame stability check.

Fresh core and UI environments test the actual built wheel from outside the
checkout, without editable installation or source-path injection. The core
smoke asserts default FV1 and explicit LI completion and site-packages origin.
The optional UI smoke launches `python -m floodwarp.ui`, validates the installed
example inputs and runs a real worker to 73 s. `pip check`, artifact metadata,
packaged input availability and secret/cache/unrelated-file inspection belong
to the final verification record. CI retains the source regression suite and
adds both fresh-wheel environments.

## Evidence transferred from dev

Copied the three sparse performance/profile/memory drivers and missing files
under `validation/sparse_perf_review/`, including frozen reference/trial runtime
and driver snapshots, profiles, memory measurements and logs, together with
`validation/release_v0.1.1/publication_verification.json`. Source dev was
`90f5b9893afc1ac87e725788ec0431f38adb7834`. Existing main benchmark/operator
records were retained. The report's historical pending-event assessment now
links to the subsequently completed-event evidence already on main.
No development-wide status report or static-mesh runtime/evidence package was
copied. Reproduction drivers require the separately retained large full-domain
checkpoints at their editable paths, not hidden wheel data.

## Scope and remaining scientific limits

No additional 72-hour event was run for interface/metadata/output scheduling.
Focused forcing, final-time, restart and direct-API parity checks address those
changes; the baseline's completed-event evidence remains applicable to its
unchanged hydraulic implementation. Hardware scope is Linux x86-64, Python
3.14.7 and RTX 4070 Ti SUPER. CUDA checks, skips and unavailable independent
reference reruns must be reported as actually executed.

The UI is rainfall-driven FV1 only, with uniform constant infiltration and
FREE/Closed presets. It does not expose stages, river inflow, SGS, LI, arbitrary
roughness or a GIS editor. Large grids still require dense state and checkpoint
IO. One active run per server is enforced; automatic restart/server-restart
job discovery is outside scope. Numerical agreement, observed-event validation
and grid convergence remain different; the UI adds no calibration claim.

## Publication sequence (owner)

1. Review local main/candidate hashes and `verification.json`; preserve earlier
   tags/assets. Push the intended branches yourself after review.
2. Change visibility yourself, tag the recorded candidate source as v0.1.2 and
   publish the wheel, source archive and matching SHA256SUMS from the recorded
   artifact directory. Do not rebuild under the same checksums.
3. From an unauthenticated browser/shell, confirm repository/source and all
   release downloads work. Download and verify SHA-256 values anonymously.
4. Install the downloaded wheel core-only in a fresh Python 3.14 environment
   outside the checkout; run pip check and FV1/LI controls. Repeat with `[ui]`,
   launch the documented command and complete the installed tiny example.
5. Confirm old v0.1.0/v0.1.1 assets and data links remain anonymously accessible.

Anonymous download checks can only be completed after the owner publishes;
they are not a blocker to local preparation and are not claimed as passed.
