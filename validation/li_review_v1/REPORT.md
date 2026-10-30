# FloodWarp v0.1 LI parity and engineering validation review

## Release decision

**Extraction parity is verified for the tested configurations. The seven original
LISFLOOD discrepancies were not introduced by extraction.** The two repositories
produce bitwise-identical CPU FP64 hydraulic fields, sampled source diagnostics,
water ledgers and accepted timestep histories in 48 matched replays, plus eight
controls comparing the actual FloodWarp public defaults with equivalent source
constructors. No hydraulic correction is indicated or made.

**ACC-compatible LI meets the proposed engineering screen on all eight retained
scenarios. Physical LI does not receive unrestricted hydraulic qualification.**
Its final millimetre-threshold differences are mostly shallow-film effects, but
transient depth/extent differences and water partitioning cannot all be dismissed
as binary classification artefacts. The shipped default is numerically sound in
these controls, but independent validation of its dry-exterior boundary remains
incomplete. Neither formulation is established as physical truth.

Public v0.1 remains blocked by the existing licence/provenance requirements and
incomplete hydraulic qualification. No public release, default change, physics
change or independent-reference regeneration is part of this review.

## Baselines, execution and preservation

* Source: Overlander `dev`, `ba503b8bde51d3f212199e81efc7038d9fffecc8`.
* Actual starting FloodWarp `dev` HEAD: `77c794a95db7aad8d79fa0574420a13e79273cb8`.
  This is also the requested initial baseline; its working tree was clean.
* Both commits were exported with `git archive` to isolated directories. Source
  imports came from the source export; target imports came from its exported
  `src` directory. Overlander checkout/index/history were not modified.
* Existing `darcywarp`: Python 3.14.7, NumPy 2.3.5, Warp 1.17.0; FP64 CPU.
  No environment upgrades. Retained CUDA tests used the RTX 4070 Ti SUPER.
* Eight one-hour analytic fixtures, maximum physical grid 40×40, 5/10/20 m.
  Same bed, roughness, all-active mask, zero initial depth/discharge, rainfall,
  infiltration, all-perimeter FREE openings and 60 s observation boundaries.
  Matched controls use gravity 9.80665, CFL 0.4, fixture caps 1/0.5/0.25 s,
  minimum step 1e-8 s. Full timestep arrays are retained locally.
* The repeated 60 s output boundaries introduce a tiny terminal subdivision in
  these runs (3601/7201/14401 accepted steps). This occurs identically in both
  repositories. The native fixture's own step sequence need not be identical.
* Original [strict report](../REPORT.md), [criteria](../ACCEPTANCE.md),
  [results](../results.json), [sensitivity](../sensitivity.json), reference
  fixtures and checksums remain unchanged. All fixture hashes were rechecked.
  Review depth arrays exactly reproduce all 16 original LI comparison arrays.
* [Provenance](provenance.json) records preservation and environment details.
  New engineering criteria and outputs are versioned here, separately.

## A. Numerical integrity and extraction parity

[Parity results](parity.json) contain exact array hashes for each of the 48
pairs. Both named profiles and omitted-profile construction were run with
explicit slope-FREE and omitted closure. For physical/omitted profiles the
latter resolves to dry-exterior-head; ACC omission resolves to slope-FREE.
Thus the matrix includes intentional duplicate default-resolution controls.

At each observation time the comparison includes depth, bed-plus-depth surface
elevation, qx, qy, applied rain/infiltration/other-source arrays, all cumulative
ledger entries and mass residual. Every accepted timestep is compared too.
The numerical identity, gravity, CFL, timestep bounds, dtype and device agree.
All compared values are bitwise identical: maximum absolute and relative
numerical differences are zero; there is no first divergent timestep or
operation to investigate. This is deterministic CPU evidence, not a claim that
all devices or all possible input states are bitwise interchangeable.

The source's `groundwater_sources` ledger slot was renamed `additional_sources`
on extraction; corresponding values are compared in matching order. It is zero
in these rainfall/infiltration fixtures. That name change is not a hydraulic
difference. Applied-source arrays describe sampled committed steps; cumulative
ledgers include all steps. No unobserved per-cell source trajectory is claimed.

[Default parity](default_parity.json) adds eight exact matches between public
`create_model` with all numerical controls omitted and the equivalent source
native constructor. Both use physical_candidate, dry_exterior_head, FP64,
CFL 0.4, minimum_dt 0.001, maximum_dt 60 and growth factor 1.5. Identical
sampling times and physical inputs are retained.

All new runs finish with finite nonnegative depths and pass the unchanged mass
criterion. Maximum sampled residual over the 48 target replays is about
4.11e-9 m³. The retained regression suite passed **1328 tests, with 24 skipped**,
including opt-in FV1 CUDA checks and existing CFL, analytical, source,
CPU/CUDA, graph and restart controls. Four additional assessment-metric tests
pass; the final combined rerun reports **1332 passed, 24 skipped**. Skips do not become passes; external-binary/unsupported-path coverage
limits remain. CPU/CUDA and restart tolerances have not been relaxed. This
review does not add cross-repository GPU parity or a new FP32/mixed claim.

## Original strict benchmark and engineering protocol

The original result remains **25/32 strict comparisons passing**: six physical
LI failures, one ACC-compatible LI failure, and 16 passing FV1 controls. The
strict criteria and verdicts are preserved, not recalculated under new rules.

[Engineering criteria v1](ENGINEERING_CRITERIA.md) were written after the parity
gate and before computing new classifications. They are an explicitly proposed
screen for these shallow-runoff/inundation applications, not a universal
engineering standard or a blind acceptance study. They keep the original
1 mm/2% depth scale, add P95/maximum safeguards, apply extent assessment at
10/50/100 mm with a declared ±1 mm uncertainty band, retain the 5% hydrograph
criterion, and assess export/storage against 2% of event supply. The raw
0.1/1/5 mm classification metrics are always retained.

The screen applies at every sample, not only at final time or to a pooled
space-time RMSE. Consequently it can be stricter than the original benchmark.
The ACC-compatible profile passes 8/8; slope-FREE physical passes 1/8 (flat).
The previously strict-passing physical 20 m case fails the engineering transient
extent screen. This is not an effort to turn all original failures into passes.

| Physical slope-FREE case | Final RMSE, mm | Largest sampled RMSE, mm | Largest point error over samples, mm | Engineering concerns |
|---|---:|---:|---:|---|
| Flat 10 m | 0 | 0.025 | 0.025 | None |
| Steep 10 m | 0.654 | 0.654 | 1.352 | Export/storage partition |
| Complex 10 m | 0.536 | 0.939 | 4.606 | Transient 10/50 mm extent |
| Heterogeneous 10 m | 0.598 | 1.198 | 5.919 | Transient depth and extent |
| Complex 5 m | 0.663 | 1.966 | 11.380 | Transient depth, P95, maximum and extent |
| Complex 20 m | 0.461 | 0.709 | 2.551 | Transient 10 mm extent |
| Infiltration, cap 0.5 s | 0.660 | 1.172 | 5.810 | Transient depth and extent |
| Infiltration, cap 0.25 s | 0.661 | 1.173 | 5.810 | Transient depth and extent |

The [machine-readable results](engineering_results.json) list each failed gate,
final errors, numerical configuration, budgets and raw final-threshold metrics.
Attribution experiments and unmatched-default sensitivity results are clearly
separate groups; they are not extra independent validation successes.

## Wetting/drying sensitivity

[wet_thresholds.csv](wet_thresholds.csv) reports IoU, precision, recall, wet-cell
disagreement count/area, false wet/dry count/area and near-threshold diagnostics
at **every retained time and all six thresholds**. [depth_errors.csv](depth_errors.csv)
reports threshold-independent RMSE, MAE, maximum, relative L2 and absolute-error
P50/P90/P95/P99. Counts/areas in each row are for that time, not sums over time.
All cells are active. Wet means strictly `depth > threshold`, as in the strict
benchmark. Empty unions have IoU 1 by convention, flagged explicitly, with
undefined precision/recall stored as null; they are not flood-extent evidence.

Final physical slope-FREE IoU:

| Case | 0.1 mm | 1 mm | 5 mm | 10 mm | 50 mm | 100 mm |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 1 | 1 | 1 | 1 | empty | empty |
| Steep | 0.750 | empty | empty | empty | empty | empty |
| Complex 10 m | 0.975 | 0.922 | 0.864 | 1 | 1 | empty |
| Heterogeneous | 0.975 | 0.932 | 0.972 | 1 | 1 | empty |
| Complex 5 m | 0.953 | 0.887 | 0.918 | 0.937 | 1 | empty |
| Complex 20 m | 1 | 0.964 | 0.857 | 1 | 1 | empty |
| Infiltration, either cap | 0.810 | 0.928 | 0.971 | 0.943 | 1 | empty |

The final ACC-compatible 5 m case has depth RMSE **0.002502 mm**, IoU
**0.94375** at 1 mm, precision/recall **0.971061**, and **36 disagreements =
900 m²** (18 false wet and 18 false dry, 450 m² each). Their median absolute
depth difference is **9.380 µm**, maximum **29.385 µm**. Thirty of 36 have both
depths within 10 µm of 1 mm; all 36 are within 100 µm. This is a micrometre-scale
classification failure, not a materially different final hydraulic solution.
Its other five thresholds have IoU 1; the 100 mm union is empty.

For physical LI, final 1 mm disagreements are also overwhelmingly near the
threshold: all except two of the 78 disagreeing 5 m cells have both depths
within 1 mm of it; all other final 1 mm disagreements meet that band. This
supports a shallow-film explanation for **final** binary extent differences.
It does not explain the full time-dependent comparison. At 1800 s in the
physical 5 m case, 50 mm IoU is 0.794872: 16 false-wet cells cover 400 m², with
median depth difference **7.618 mm** and maximum **8.133 mm** among those cells.
Only two are within the ±1 mm band. These differences occur well above the
solver activation threshold and require a formulation-level assessment.

No solver dry-depth setting was altered: physical remains 0.1 mm, ACC remains
1 mm. Evaluation thresholds are post-processing only.

## Steep-terrain discharge

The original matched-slope physical run exports **1064.440206 m³**, compared
with independent ACC **1040.254465 m³**: **+24.185741 m³**, or **2.325%** of
reference export and **2.239%** of the 1080 m³ rainfall. It leaves **15.559794 m³**
versus **39.745535 m³**. That is a large relative difference in the small residual
storage (~60.9%), despite sub-millimetre domain depth RMSE. It narrowly exceeds
the proposed 2% supplied-volume budget. Sampled outlet NRMSE remains **1.662%**.

| Time, s | Physical minus native outlet, m³/s | Cumulative export difference, m³ |
|---:|---:|---:|
| 600 | -0.045002 | +0.294862 |
| 1200 | -0.002288 | -0.037946 |
| 1800 | +0.001890 | +0.100605 |
| 2400 | +0.017488 | +3.263428 |
| 3000 | +0.016050 | +17.105742 |
| 3600 | +0.008618 | +24.185741 |

Complete signed sampled hydrographs and cumulative differences are in
[hydrographs.csv](hydrographs.csv); these are instantaneous sampled discharges,
not interval means or an integration of the sparse samples.

**Verified controls:** both original runs use slope-FREE. Changing physical
face roughness from resistance RMS to arithmetic or disabling boundary
transverse friction leaves this uniform-roughness, effectively one-directional
steep result identical. Moving sources after momentum changes excess export
only from 24.185741 to 24.172953 m³. Therefore those mechanisms are not principal
causes of this discrepancy. Choosing dry_exterior_head instead changes excess
export to 16.759506 m³ at the matched cap, but this is a different boundary
problem and does not explain the original same-closure discrepancy.

**Strong numerical inference:** shallow-flow shutoff during recession is the
principal cause. Native final cell depths range 0.938–1.000 mm and its export
stops near 2460 s; physical LI continues draining, ending at about 0.100–0.855 mm.
The near-zero difference during the main hydrograph grows chiefly after native
shutoff. No dry-threshold ablation was run, as required; this is not a uniquely
isolated causal percentage. A diagnostic `require_cell_depth_above_dry` profile
field toggle was inert: the constructor does not consume that field, so its
identical run is **not** evidence of an independent activation test. The existing
field is not repaired or repurposed in this review.

## Infiltration and source availability

At the 0.5 s cap, physical LI infiltrates **39.304789 m³**, compared with
ACC-compatible **30.412917 m³** and an effectively identical native result.
The difference **8.891873 m³** is **29.237%** of native infiltration,
**22.623%** of physical infiltration, but **0.8233%** of total rainfall.
This is small in the whole-event budget but potentially important if infiltration
or soil recharge is the intended output.

**Verified implementation:** physical constant-capacity infiltration uses water
after rainfall/other sources and caps extraction by available water. Sources
precede momentum. ACC source mode allows infiltration only when pre-existing
cell depth exceeds its 1 mm cutoff; sources follow momentum and extraction is
also capped to protect already committed outgoing fluxes. Physical infiltration
is thus internally consistent with its intended rain-available capacity law.
That establishes consistency, not soil-process accuracy or superiority to ACC.

Explicit controls preserve the physical 0.1 mm dry threshold:

| Control | Infiltration, m³ |
|---|---:|
| Physical baseline | 39.304789 |
| Sources after momentum only | 39.302231 |
| ACC availability rule, retaining physical cutoff | 38.110306 |
| ACC availability and source ordering, physical cutoff | 38.101556 |
| No boundary transverse friction | 39.206060 |
| Arithmetic face resistance | 39.304784 |

Ordering alone is negligible here. Availability/censoring accounts for some of
the difference, with a substantial remainder consistent with the profile's
unchanged 0.1 versus 1 mm threshold and different hydraulic residence time.
These controls are not an additive factorial decomposition; interactions and
the no-dry-depth-change constraint prevent assigning unique causal percentages.
At 0.25 s physical infiltration is 39.304802 m³, whereas ACC-compatible is
30.433347 m³. The discrepancy persists with timestep refinement.

Boundary vector friction also affects the heterogeneous transient hydrograph:
disabling it reduces outlet NRMSE from 2.717% to 0.215% in the infiltration case.
This is evidence of an intentional hydraulic discretisation difference. It is
not a recommendation to remove vector friction or to tune toward ACC.

An additional **5 m complex-terrain control** isolates the same boundary
friction option. Disabling exterior transverse reconstruction, while retaining
the physical dry threshold and all other settings, reduces the largest sampled
point error from **11.380 to 1.871 mm**, the largest sampled RMSE from
**1.966 to 0.589 mm**, and outlet NRMSE from **2.399% to 0.366%**. It passes
the proposed screen. Arithmetic resistance has no effect on this homogeneous
roughness case; source ordering does not remove the transient discrepancy.
Thus the principal 5 m transient difference is attributable to the intentional
exterior vector-friction treatment, not merely a 1 mm classification threshold.
This explains the implementation difference without establishing which boundary
resistance is more physically appropriate. Production friction is unchanged.

## Grid and timestep sensitivity

The unchanged [original sensitivity results](../sensitivity.json) remain part
of this assessment. Restricting finer simultaneous final depths by area means,
physical 20→10 m and 10→5 m differences are **1.435 and 1.053 mm**. ACC-compatible
values are **1.434 and 0.916 mm**, nearly matching native. Within-method grid
sensitivity decreases, but physical versus native final RMSE **increases** from
0.461 to 0.536 to 0.663 mm as resolution improves. Thus these data do not show
convergence of the two formulations toward each other. The physical 5 m
transient maximum error reaches 11.38 mm.

Halving infiltration-case caps 0.5→0.25 s changes final physical depths by only
**0.000658 mm RMSE** and ACC-compatible by **0.000787 mm**. Physical versus native
RMSE stays near 0.660–0.661 mm. This supports persistent formulation differences
rather than a discrepancy removable by this timestep refinement. Two caps and
three grids do not establish an asymptotic convergence order.

## Actual shipped default and boundary recommendation

The public-default runs omit profile, FREE closure, CFL and timestep overrides.
They resolve to physical_candidate/dry_exterior_head with maximum_dt 60 s and
minimum_dt 0.001 s. All eight complete, conserve mass and reproduce equivalent
Overlander defaults bitwise. Against the existing **slope-FREE** reference they
pass **0/8** engineering screens. This is explicitly **unmatched-boundary
sensitivity**, not eight demonstrated hydraulic bugs.

The strongest distinction is flat terrain. Slope-FREE with no prescribed
slope exports zero, whereas the public dry-exterior default exports
**751.075149 m³** of 1080 m³ rainfall; its matched 1 s cap control exports
750.429104 m³. The closure difference dominates timestep effects there.
On the steep case the public default exports 36.191934 m³ more than native,
versus 16.759506 m³ for dry-exterior with the 1 s cap, and has sampled outlet
NRMSE 21.635%. Default timestep sensitivity is therefore material in that case
and is not resolved merely by CFL compliance.

* Use slope-FREE when the intended outlet is continued terrain/local or a
  supplied normal slope, including matched native ACC comparisons. It does
  not drain a flat zero-slope outlet by inventing an exterior head drop.
* Use dry_exterior_head only when a dry exterior at the boundary sill is the
  intended head condition. It is not interchangeable with slope-FREE and is
  not a generic normal-depth outlet. Its independent hydraulic qualification
  for an equivalent dry-exterior problem remains missing here.
* Use prescribed-stage boundaries when an external water level is known,
  within the existing documented solver limitations.

No default was silently changed. Passing numerical integrity does not establish
that the shipped default meets an independent hydraulic agreement claim for
arbitrary FREE boundaries. Default boundary choice and timestep sensitivity
remain v0.1 qualification concerns for unrestricted physical-LI use.

## Remaining concerns and v0.1 categories

**A — strict numerical integrity:** passes the executed baseline/default parity
and retained regression coverage. No extraction regression demonstrated; no
operator fix required. Missing device/path coverage is explicitly not a pass.

**B — independent hydraulic agreement:** ACC-compatible LI is acceptable for
bounded v0.1 use under the tested FP64 conditions, with the original 5 m strict
threshold discrepancy retained. Physical LI can be retained as a characterised
candidate, but this review does not certify it for unrestricted engineering use
or certify its shipped FREE default. The remaining transient 5 m differences,
late-recession partition and default timestep/boundary behaviour require an
application-specific acceptance decision or additional independent matched
controls. Their existence does not prove the physical formulation is incorrect.

**C — known formulation differences:** dry-depth activation, rain availability,
source ordering, resistance averaging, transverse friction and FREE closures
are intentional, baseline-preserved distinctions. Do not count them as extraction
bugs. Do not claim all consequences are engineering-negligible.

The micrometre-scale ACC 1 mm extent miss alone should not block the bounded
ACC-compatible formulation. Remaining physical/default qualification concerns
and the previously documented FV1 temporal/grid/outlet gaps remain visible.
SGS remains experimental. Software redistribution rights, Purau datum/per-pixel
lineage, FV1 boundary limitations and other [release blockers](../../docs/RELEASE_STATUS.md)
are unchanged. No licence, provenance or broader release blocker was waived.

## Reproduction and evidence limits

Run in the pinned environment from the repository root:

```bash
python -m validation.li_parity
python -m validation.li_controls
python -m validation.li_assess
python -m pytest -q
```

Inspect execution guards first: fresh output directories are required, and
`li_assess` points to the baseline replay directories. The first audit used
`/tmp/floodwarp_li_review_v1`; the completed end-to-end reproduction uses
`outputs/li_review_v1/baselines`. Both gave exact parity. `li_audit` additionally
records preservation and test logs; its log paths are editable in its guard. Raw state archives are local generated
outputs, not new independent fixtures. Committed per-field hashes, numerical
manifests, per-sample metrics and hydrographs preserve the review evidence.
No original external solver was rerun or replaced. The review adds no catchment,
ensemble, training, FV1-operator or SGS-physics work.
