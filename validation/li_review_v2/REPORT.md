# Practical LI engineering validation, v2

## Decision and scope

Both physical_candidate and lisflood_fp_acc meet the revised **bounded hydraulic
screening requirements on all eight equivalent slope-FREE cases**. Physical LI
is acceptable for the tested inundation/runoff screening use with the explicit
matched boundary and timestep controls. The strict result remains **25/32**;
the seven original discrepancies remain recorded. Numerical integrity and the
56 bitwise extraction/default controls retain their original acceptance rules.

This supersedes v1's engineering interpretation, including its use of isolated
transient maxima and 1 mm depth-equivalence budgets as release vetoes. It does
not establish field accuracy or universally qualify every physical-LI default.
The actual default has a meaningful same-boundary timestep-sensitivity concern
on steep terrain. Public distribution also retains the existing licensing,
provenance and broader qualification limitations.

## Criteria and rationale

[Criteria v2](ENGINEERING_CRITERIA.md) were recorded before executing v2
classification, with prior results known. They are requirements for grid-scale
inundation/runoff screening, not a calibrated regulatory or design standard.
The rationale follows output significance rather than ACC's activation depth:

| Quantity | Screening requirement | Physical interpretation |
|---|---|---|
| Meaningful depths | Union of depths >=50 mm; pooled RMSE <=10 mm or relative L2 <=5%; weighted P95 scaled error <=1 | A centimetre absolute budget plus 5% depth scale; protects widespread errors while reporting isolated maxima |
| Flood extent | 50/100 mm raw metrics; <=5% resolved disagreement in area-time and >10% resolved disagreement for <=10% event duration | Separates marginal threshold crossings from persistent extent differences |
| Threshold ambiguity | Both depths within 10 mm of the classification threshold | Same centimetre budget as depths; raw disagreements remain available |
| Hydrograph | NRMSE <=5%, peak magnitude error <=10%, peak timing <=one output interval or 5% duration | Retains routing shape, peak and timing sensitivity |
| Event allocation | Export and simultaneous storage differences each <=5% of supplied/initial water | Modest event-scale partition budget, separate from strict solver mass conservation |
| Local consequence | Any point error >50 mm requires explicit review even if pooled metrics pass | Prevents a small consequential feature disappearing into a domain statistic |
| Recharge | Separate >10% reference infiltration disagreement flag; independent soil evidence required | Avoids treating routing agreement as recharge validation |

The 50 mm scale is the retained substantial-inundation diagnostic; 100 mm
provides conventional flood-extent context. A centimetre is 20% of the lower
assessment depth and 10% of 100 mm. Smaller depth-sensitive applications need
stricter requirements. The 5% event budget permits 50 m³ per 1000 m³ supplied;
it cannot certify precise drainage storage, abstraction or recharge accounting.
The revised extent rules still reject sustained displaced flood patches,
false inundation and large local errors. They are not a rule that all differences
below a spatial pixel are acceptable. Tests explicitly exercise these failures.

Purpose-specific uncertainty and retaining unthresholded maps are supported by
[Environment Agency guidance](https://www.gov.uk/guidance/using-modelling-for-flood-risk-assessments)
and [SEPA guidance](https://www.sepa.org.uk/media/219653/flood_model_guidance_v2.pdf).
The precise numerical budgets here are project choices, not values prescribed
by those sources. Raw differences remain necessary for regulatory flood-risk
change assessments; these allowances cannot be subtracted from those changes.

## Results at equivalent boundaries

| Physical slope-FREE case | Meaningful-depth pooled RMSE, mm | Hydrograph NRMSE, % | Peak error, % | Peak-time difference, s | Outcome |
|---|---:|---:|---:|---:|---|
| Flat 10 m | No >=50 mm coverage | Zero flow | n/a | n/a | Within applicable budgets |
| Steep 10 m | No >=50 mm coverage | 1.662 | 2.081 | 0 | Within applicable budgets |
| Complex 10 m | 0.552 | 1.602 | 2.478 | 0 | Within budgets |
| Heterogeneous 10 m | 1.461 | 2.610 | 3.243 | 60 | Within budgets |
| Complex 5 m | 3.350 | 2.399 | 5.313 | 60 | Within budgets |
| Complex 20 m | 0.280 | 1.444 | 1.749 | 0 | Within budgets |
| Infiltration cap 0.5 s | 1.405 | 2.717 | 3.444 | 60 | Within budgets |
| Infiltration cap 0.25 s | 1.406 | 2.741 | 3.445 | 60 | Within budgets |

All eight ACC-compatible cases also meet the applicable budgets. No matched
case has a >50 mm point error requiring local review. The largest physical
scaled P95 is 0.695 of the local depth budget, on the 5 m complex case. All
covered 50/100 mm extent comparisons have zero **resolved** disagreement under
the declared 10 mm band. Raw extent disagreement has not vanished or been
rewritten: [wet_thresholds.csv](wet_thresholds.csv) retains all six thresholds,
precision, recall, IoU, false wet/dry counts/areas, uncertainty-band counts,
and simultaneous and sampled-envelope diagnostics.

No depth/extent coverage is claimed when a union is empty. In particular the
flat/steep controls support source, discharge and conservation behaviour rather
than substantial flood-depth accuracy; the 20 m complex case has no 100 mm
coverage. Other retained transient samples do cover 100 mm, despite all final
maps being below that threshold. This is bounded small-grid evidence, not a
large-depth or high-velocity qualification.

Temporal weighting represents each stored snapshot over its preceding output
interval (60 s here), including the first interval from zero. Thus pooled and
persistence metrics are sampled approximations; they do not resolve behaviour
between outputs. The discharge series uses instantaneous samples. The sampled
maximum envelope is not an every-timestep maximum or simultaneous storage.
Raw threshold-independent errors and full signed hydrographs remain in the
[v1 depth table](../li_review_v1/depth_errors.csv) and
[v1 hydrograph table](../li_review_v1/hydrographs.csv), unchanged.

## Intentional formulation differences and practical meaning

The ACC-compatible 5 m final 1 mm miss remains 36 cells (900 m²), with median
9.38 µm and maximum 29.38 µm depth differences. This is negligible for the
stated inundation screening outputs; the strict benchmark continues to fail.

Physical LI's 5 m maximum transient difference is still **11.38 mm**, and raw
50 mm extent IoU can fall to 0.795 at 1800 s. At that time the 16 mismatched
cells have median difference **7.62 mm**, within the practical depth budget.
The v1 attribution experiment associated most of the larger transient depth
error with exterior transverse friction: disabling that explicit diagnostic
option reduced the maximum to 1.87 mm. No production option is changed.
This is a characterised model-formulation difference within the bounded output
budget, not a numerical-integrity failure or independent proof of either law.

The steep-case **+24.19 m³** export remains: **2.24% of rainfall**, **2.32% of
native export**, with a ~61% reduction in the small final residual storage.
It occurs chiefly during late recession as physical LI drains below native's
1 mm activation scale. The raw storage difference matters for residual-water
applications, but does not independently block the stated flood/runoff screening
use when event export, storage and hydrograph budgets are satisfied. It is not
silently deleted as 'dry noise' and does not change the strict mass criterion.

Physical infiltration is **39.30 m³** versus native/ACC **30.41 m³** at the 0.5 s
cap. The **8.89 m³** difference is **29.24% of reference infiltration**, **22.62%
of physical infiltration** and **0.823% of rainfall**. Physical LI permits
rain-available infiltration; ACC censors infiltration by pre-existing ponding
and applies sources after momentum, protecting committed outgoing fluxes.
Those intentional laws need not coincide to qualify inundation routing.
The separate recharge agreement budget is exceeded. **Recharge-specific use
remains unqualified** without soil/recharge evidence matching the intended
source law. Neither formulation becomes physically correct by matching ACC.

The original grid/timestep findings also remain: physical/native final RMSE
increases 0.461→0.536→0.663 mm at 20→10→5 m while within-method grid differences
decrease; halving 0.5→0.25 s caps barely changes physical/native disagreement.
This supports persistent formulation differences, not extraction error or a
claim that both solvers converge to the same discretised solution. v2 budgets
are not a substitute for convergence evidence.

## Actual default: equivalent boundaries and retained concern

The 16 dry-exterior-head comparisons (eight fixture-cap controls and eight
public-default runs) are marked **non_equivalent_boundary** against native's
slope-FREE fixtures. They receive no independent hydraulic agreement pass/fail.
Their raw error metrics remain diagnostic. The v1 phrase '0/8 defaults pass'
should not be interpreted as eight hydraulically defective simulations.

Both closures remain appropriate only for their intended physical boundary:
slope-FREE for continued/prescribed normal slope, dry-exterior-head for drainage
to a dry exterior at the sill, prescribed-stage for known external water levels.
The large flat-case difference reflects distinct boundary problems. No change
to a default or boundary closure is made.

There is nevertheless a **same-boundary** concern. Comparing public defaults
against fixture-cap dry-exterior-head controls, the steep case has hydrograph
NRMSE **23.56%**, peak difference **46.26%**, peak-time displacement **240 s**,
and **19.43 m³** additional export. This exceeds the practical hydrograph budgets
with the boundary held equivalent. The other seven same-boundary hydrograph
controls meet the budgets. These are timestep-sensitivity controls, not new
independent references. CFL compliance alone does not establish temporal
accuracy. The evidence supports bounded physical-LI use with the tested timestep
caps; it does not qualify the ordinary 60 s cap for this steep regime. A broader
default claim needs separate timestep qualification and an independent matched
dry-exterior reference. Both remain visible release-scope concerns.

## Release conclusion

* Both LI profiles are acceptable for the **tested FP64, explicit-cap,
  equivalent-boundary inundation/runoff screening scope**. Known shallow-film,
  source-ordering and friction differences within these budgets do not
  independently block that use. The former v1 transient/2% allocation vetoes
  no longer justify a blanket physical-LI rejection.
* The shipped default retains numerical integrity and bitwise Overlander
  parity. Unrestricted default hydraulic qualification is still incomplete;
  the steep same-boundary timestep concern is meaningful and retained.
* Recharge use, deeper/rapid flooding, field calibration and arbitrary
  boundary regimes do not acquire qualification from this assessment.
* Software redistribution rights, Purau datum/per-pixel provenance, FV1
  temporal/grid/outlet gaps, existing FV1 boundary limitations and experimental
  SGS status remain as recorded in [release status](../../docs/RELEASE_STATUS.md).
  This engineering revision does not authorise public distribution.

## Reproduce and preservation

Run `python -m validation.li_engineering_v2` in the pinned environment with the
retained local v1 replay archives. Its execution guard holds all v2 budgets and
paths. It verifies archived array hashes and independent fixture checksums,
reads the existing parity evidence, and runs no hydraulic solver. If local
archives are missing, use the documented v1 replay entry points with fresh
output roots; never regenerate independent fixtures from FloodWarp.

[results.json](results.json) contains every gate, coverage state, unmatched
boundary diagnostic, same-boundary timestep control and recharge flag.
[provenance.json](provenance.json) binds the new evidence and preservation audit.
All v1 review files, strict acceptance/results and solver runtime files remain
byte-identical to the starting commit. No reference output, test tolerance,
solver dry threshold, physics or default configuration changed.

Validation: **1337 passed, 24 skipped** in the unchanged pinned environment,
including the retained opt-in FV1 CUDA checks. Five additional criteria tests
cover false substantial inundation, empty coverage, persistent versus isolated
extent errors, threshold ambiguity and zero-reference discharge. The preservation
audit is `python -m validation.li_engineering_audit`; inspect its test-log path
before reproducing it. Existing skips remain coverage limits.
