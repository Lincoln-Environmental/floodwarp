# FloodWarp v0.1.1 release verification

Release branch is based on `main@45a374c`, with only the validated exact-sparse uniform FV1 port, required tests/evidence, disclaimer, packaging and replay fixes. No static mesh, dynamic AMR, wavelets, mixed-precision FV1 or additional performance experiment is included.

FV1 defaults: `execution_mode="sparse"`, `tile_size=8`, `dense_fraction=0.65`. Native construction and both public factories are tested. Explicit dense override and sparse→dense/dense→sparse disk restart preserve numerical identity and exact final times. Dense uniform FP64 state, sources/soil, full source/boundary CFL, HLL/contact fluxes, donor limiting, accepted face volumes, maxima and ledgers remain unchanged. Very shallow positive water is retained.

## Tests and actual distributions

- Pinned Python 3.14.7 / Warp 1.17.0; RTX 4070 Ti SUPER.
- Full CPU-only suite: **965 passed, 632 skipped**; 350 pre-existing warnings.
- Full CPU/CUDA suite with optional FV1 controls: **1609 passed, 25 skipped**; 743 pre-existing warnings.
- Focused release controls: **75 passed**; the same **75 passed** against the installed wheel outside the checkout.
- Actual v0.1.1 wheel installed non-editably in a fresh external environment. Synthetic LI/FV1 and 200 m crop examples completed; pip check passed. Runtime module hashes match the checkout.
- Wheel/source archive contain DISCLAIMER.md and AGPL-3.0-only metadata. Static modules are absent. Generated example GIS files are excluded from the source archive; immutable data remain release assets.
- Standalone FV1/LI replay controls include shorter-than-interval and non-divisible durations, exact final time, final fields/checkpoint/ledger and refusal to overwrite a previous run. No automatic resume is claimed.

## Current saved-state controls

Each case uses a 30 s continuation, warm-up and three measured repetitions. These measurements are separate from profiling and the retained full-event timing.

| Saved hour | Dense hydraulic [s] | Sparse hydraulic [s] | Dense / sparse |
|---|---:|---:|---:|
| 1 | 0.025819 | 0.007454 | 3.464 |
| 13 | 0.106257 | 0.079988 | 1.328 |
| 25 | 0.358661 | 0.385803 | 0.930 |
| 48 | 0.319313 | 0.299208 | 1.067 |
| 60 | 0.200020 | 0.110035 | 1.818 |
| 72 | 0.198946 | 0.107507 | 1.851 |

Every compared field, accepted step count and final time agrees exactly for these saved-state controls. The 25-hour widespread-runoff case remains slower. Dense indexing fallback and explicit dense execution are retained.

## Preserved event and GIS evidence

The complete 72-hour Orton Bradley event used 1641.957 s sparse hydraulic time versus 1906.106 s dense and 1403.559 s external LISFLOOD-FP FV1. Sparse reduced FloodWarp hydraulic runtime by 13.9%, with 1,570,053 accepted steps and zero differences at all 72 recorded hourly comparisons. This is one candidate run against a retained same-device dense benchmark, not interleaved full-event repetitions. Original timed sources and interrupted/restarted event evidence are preserved in `validation/sparse_review`.

All 25 assets listed by the original release checksum file (plus the checksum asset itself) were downloaded through authenticated GitHub access. They match the published v0.1.0 asset audit; all 50 files inside the archive also match. The original v0.1.0 tag, assets, numerical arrays, provenance and snapshots were not modified.

The downloaded archive was extracted outside the repository and reproduced with the fresh environment, without hydraulics. Comparison JSON is exactly equal. Regenerated depth/DEM/difference/extent arrays match original FP64 data, exact 1123×843 grid, EPSG:2193, NZVD2016, mask and 559,985 active cells. All 4,320 actual 60 s hydrograph means and cumulative exports match the retained records. See product_verification.json and baseline_assets.json.

## Disclaimer and publication boundary

DISCLAIMER.md is the developer disclaimer. It is linked near the README top, included in both distributions, linked from release notes and the dataset documentation, and accompanied by figure captions and OUTPUT_NOTICE.md for the unchanged GIS products. No AGPL restriction or runtime acceptance dialogue was added. Following developer clarification, the private release proceeds without claiming independent institutional/legal approval.

The repository is private and visibility remains owner-controlled. Unauthenticated repository and DEM download checks returned HTTP 404. Anonymous installation/reproduction therefore cannot be claimed; the remaining public-access action is for the owner to change visibility and repeat anonymous download/install/reproduction. Authenticated download and independent local reproduction passed.

## Development separation

Release fixes will be merged forward into dev without rewriting its research history, using version 0.2.0.dev0. Static FV1 remains experimental with its retained failed runtime gate; dynamic AMR is not promoted. Subsequent measured performance and reconstruction work belongs only on dev. The final task report records both branch SHAs and the research outcome.
