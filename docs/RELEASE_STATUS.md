# FloodWarp v0.1 candidate: release status

Private standalone extraction from `Lincoln-Environmental/Overlander`, branch
`dev`, baseline `ba503b8bde51d3f212199e81efc7038d9fffecc8`.
The new repository contains no original Git history. It remains a
`0.1.0.dev0` review candidate, not a qualified public v0.1 release.

## Retained functionality

43 runtime modules: public LI/FV1 classes and factory; their frozen kernels,
reference operators, boundaries, numerical profiles, rainfall, infiltration,
water accounting, structures, traditional SGS hypsometry, enhanced SGS
geometry/reduction/preprocessing/stepping, complete checkpoints, and the new
direct input/model helpers. [Extraction manifest](extraction.json) lists every
retained baseline module and source hash. [Import graph](import_graph.json)
includes lazy imports. [Numerical audit](numerical_integrity.json) preserves
the extraction hashes and identifies the subsequent numerical repairs. The
original extraction was baseline-equivalent after namespace/source-channel
renaming; the repaired implementation deliberately differs at documented defects.

The LI factory still resolves ordinary omission to physical_candidate; SGS
omission retains default. FV1 still uses contact_hll and velocity_extrapolation
with reflecting walls and endpoint speeds. SGS constructors warn visibly and
diagnostics identify the selected formulation. No inclined SGS is included.

## Integration and repair qualification

The original dev review commits are integrated without history rewriting.
The repair adds forcing-aware LI step selection, symmetric masked stages,
continuous FV1 FREE pressure, safeguarded Green-Ampt convergence, atomic
direct checkpoints, enhanced-SGS state/rebuild safeguards, automatic public
CUDA graph selection and LTS graph dependency tracking. Numerical defaults,
reference fixtures and historical strict/v1/v2 evidence remain preserved.

The pinned Python 3.14 suite passed 1,421 tests with 24 skips using opt-in
FV1 CUDA controls; the separate CPU-only run passed 853 with 555 skips.
Additional focused checks and fresh validation evidence are recorded in
`validation/repair_review`. Both frozen v1/v2 analyses reproduce exactly.
The repaired retained comparison cases pass 25/32 strict gates; the seven
discrepancies remain visible and are assessed separately for engineering
significance. The full-domain 72-hour independent FV1 GPU comparison and
performance profiling are still in progress at this repair commit.

## Removed dependencies and interfaces

No FloodFM prepared-model loading, GIS conditioning/catchment delineation,
DarcyWarp/RichardsWarp/SHANZ/HydroOne integration, groundwater adapters,
nested exchange hooks, external exchange actuators, external solver binaries,
kinematic/regional routing, semi-implicit solvers, neural surrogates, ensemble
training, campaigns or archives. Source-channel names are standalone; ordinary
surface sources, infiltration soil caps and their water ledgers are retained.
NumPy/SciPy/Numba/Warp are the pinned hydraulic dependencies. Rasterio and
Matplotlib are optional. No other HydroOne installation is required.

## Acceptance evidence

| Requirement | Outcome |
|---|---|
| Clean installation | Fresh isolated Python 3.14 venv; wheel built and installed with independently resolved dependencies; pip check passes |
| Independent CPU suite | **800 passed, 515 skipped**, CUDA hidden; skips are not claimed as CPU coverage |
| CPU/CUDA regression suite | **1328 passed, 24 skipped**, including opt-in FV1 CUDA and retained graph/fused/restart controls; original numerical tolerances retained |
| Direct inputs | Array, raster alignment/orientation/nodata, partial stage opening, time-varying stages, mask interfaces and checkpoint drift tests pass |
| SGS | Both modes operational with visible warnings, legacy physics and retained numerical regressions |
| Purau | Real 40×40, 5 m crop plus 1 m underlying terrain; both solvers run in the clean CPU environment without prepared-model imports |
| Synthetic installation example | LI and FV1 both run in clean CPU environment |
| Independent comparisons | **25/32 applicable gates pass**; seven LI discrepancies retained; incomplete FV1 temporal/grid coverage reported |
| Mass/restart/wet-dry/defaults | Covered by passing retained regressions and independent reference checks; no tolerance relaxation |
| Original repository | Original HEAD remains the baseline on dev; tracked worktree and index unchanged; pre-existing untracked research work retained |

Purau final residuals: LI 6.36824e-12 m³;
FV1 -3.55271e-15 m³. Outflow volumes:
LI 0.00266582 m³;
FV1 0.00205645 m³. These are a short rainfall
illustration, not a calibrated flood or full-event reference.

[Validation report](../validation/REPORT.md) records all verdicts, native binary
identities, boundary/interior diagnostics, grid and timestep sensitivity.
[Verification summaries](evidence/) record installation, tests and example runs.
The pure CPU installation and the validated research environment were tested
separately; the research environment was not upgraded.

## Outstanding release blockers and limits

1. **Software redistribution rights/licence remain unconfirmed.** The source
   had no top-level grant. LICENSE intentionally records pending status rather
   than inventing MIT/BSD permissions. Keep the repository private.
2. **Purau vertical-datum/per-pixel lineage remains incomplete.** All listed
   upstream LINZ surveys are verified CC BY 4.0, and attribution is included.
   The existing conditioned derivative does not record its datum or precise
   survey per pixel. Absolute elevations are unchanged; external absolute-stage
   use requires verification. No invented terrain or single-survey claim is made.
3. **Independent validation is not universal parity.** Six physical-LI and one
   ACC-compatible case fail frozen gates, principally wet/dry classification.
   FV1 field/export gates pass on 16 controls, but independent FV1 grid refinement,
   resolved temporal depths and instantaneous outlet hydrographs remain missing.
4. SGS remains explicitly experimental. FV1's finite-velocity ghost closure
   still has the documented supercritical/incoming-momentum limitations.
   Time-varying FV1 stages use host stepping; constant-stage graphs are retained.

No public release tag or distribution upload has been created. The owner's
requested GitHub push is a private review snapshot with these blockers visible.

## Additive LI engineering review

The [versioned LI review](../validation/li_review_v1/REPORT.md) establishes
bitwise CPU FP64 extraction parity in 48 matched configurations and eight
public-default controls. The original 25/32 strict result above is unchanged;
none of its seven LI discrepancies is attributable to extraction in these runs.
The 5 m ACC-compatible final extent miss is caused by micrometre-scale depth
differences around 1 mm. ACC-compatible LI passes the separate bounded
engineering screen in all eight retained scenarios.

Physical LI retains material transient/partition differences, including a
5 m transient point error of 11.38 mm and steep late-recession export difference
of 24.19 m³. Its actual dry-exterior-head default conserves mass and matches
Overlander, but cannot inherit qualification from slope-FREE references. Its
steep-case default timestep sensitivity also remains a concern. Physical LI is
not granted unrestricted hydraulic qualification by this review. No operator,
solver activation threshold or default changed. Licensing, terrain provenance,
FV1 coverage/limitations and experimental SGS status above remain unchanged.

## Current LI engineering interpretation (v2)

The [practical v2 assessment](../validation/li_review_v2/REPORT.md) supersedes
the engineering interpretation in the preceding v1 section. Both LI profiles
meet practical inundation/runoff budgets on all eight equivalent slope-FREE
cases with the tested timestep caps. The strict 25/32 score and all numerical
integrity requirements are unchanged. Marginal shallow-water classification,
intentional infiltration timing and isolated centimetre transient differences
no longer independently veto that bounded physical-LI use.

Dry-exterior defaults versus slope-FREE references are explicitly non-equivalent,
without hydraulic agreement verdicts. A same-boundary default timestep concern
remains on steep terrain (23.56% hydrograph NRMSE, 46.26% peak difference versus
the capped control), so unrestricted default temporal accuracy is not qualified.
Recharge has a separate concern: the infiltration difference is 29.24% of
reference infiltration despite only 0.823% of rainfall. Recharge-specific
qualification needs independent source-law evidence. Licensing, terrain
provenance, FV1 coverage/limitations and experimental SGS status remain unchanged.
