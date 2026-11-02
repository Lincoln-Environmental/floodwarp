# FloodWarp v0.1.3 — local rainfall UI and spatial inputs

**Released 2 November 2026 as Latest within the private repository.**
[v0.1.3](https://github.com/Lincoln-Environmental/floodwarp/releases/tag/v0.1.3)
points to `6a67851189d377aa894ec05304f62065a072fb23`, matching the prepared
wheel and source archive. Main includes later evidence/documentation commits.
Historical v0.1.1 is preserved; v0.1.2 was an intermediate local candidate.
[Release status](RELEASE_STATUS.md) separates branch, tag and artifact identities.

Developed by **Patrick Durney**. Copyright **Lincoln Agritech Ltd**.
Software remains AGPL-3.0-only; original LINZ elevation/data credits retain CC BY 4.0.

## Changes since published v0.1.1

- Add the optional local browser interface, launched with `python -m floodwarp.ui`
  at `http://127.0.0.1:8765`. Real FV1 runs execute in a separate worker.
- Accept projected-metre DEMs, aligned binary or declared-CRS polygon domains,
  rainfall hyetographs and uniform or spatial rainfall multipliers.
- Support uniform/spatial Manning roughness, constant/exponential infiltration,
  Closed/FREE presets and explicitly selected constant-stage domain faces.
- Load the original full-domain Orton Bradley setup from its existing data
  archive, with original spatial inputs and explicit coastal contacts. Loading
  previews the setup and waits for Run. Include a separate small synthetic example.
- Add OpenStreetMap context with terrain/domain and depth overlays. Display
  reprojection leaves hydraulic grids unchanged.
- Make FV1 the high-level default; retain explicit LI, exact-sparse execution
  and the dense override. SGS remains experimental.
- Correct final-time sampling and save final diagnostics, fields and checkpoints.
- Report each run's accepted steps, timestep statistics, hydraulic runtime and
  worker wall time in the UI, `metrics.json` and `runtime.csv`.

The primary method remains FP64 uniform-grid FV1. Static-mesh/adaptive methods,
wavelets, mixed precision, surrogate and coupled-model research are excluded.
The UI uses existing hydraulic operators and records resolved inputs/settings,
source identity, georeferencing, units and restart data.

## Installation and use

From the current main checkout:

```bash
python -m pip install '.[ui]'
python -m floodwarp.ui
```

Core-only installation remains available with `python -m pip install .`.
Use Python 3.14 and the existing pinned numerical dependencies. The
[README](../README.md), [spatial UI guide](UI_SPATIAL.md) and [input guide](INPUTS.md)
describe inputs, examples, defaults and limitations.

## Evidence and release files

The [metrics review](../validation/run_metrics_review/REPORT.md) records prior
source-bound CPU/CUDA checks, actual browser actions and installed-wheel checks.
It also identifies the released distributions and SHA-256 checksums, retained under
`outputs/release/v0.1.3-metrics/dist/`, built from `6a67851189d377aa894ec05304f62065a072fb23`.
The wheel, source archive and SHA256SUMS are assets on the private v0.1.3 release.
Later evidence/documentation commits do not relabel that artifact source.

The retained modified-sparse full event measured 1,512.372 s hydraulic runtime
and 1,570,053 accepted steps; published v0.1.1 sparse measured 1,641.957 s.
These are historical measurements with separate source identities, not new
v0.1.3 timings. No new simulation or test accompanies this documentation tidy-up.
Numerical agreement does not establish observed-event validation or grid convergence.

Historical release tags, datasets, assets and numerical records remain intact.
The owner authorised publishing v0.1.3 within the private repository using the
existing prepared files. No models, tests or builds were run for that step.
The [publication handoff](../validation/run_metrics_review/REPORT.md#updated-local-artifacts)
retains the earlier preparation record. Anonymous download/install checks
remain for the owner after making the repository public.
