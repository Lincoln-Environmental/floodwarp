# LI practical hydraulic assessment v2

This protocol replaces v1 engineering conclusions only. The frozen strict
benchmark, numerical integrity criteria, parity and v1 evidence are historical
records and remain unchanged. Criteria below were written before running v2
classification; prior case results are known, so this is not a blind study.

## Purpose and basis

Scope is comparative qualification for grid-scale shallow rainfall/runoff and
inundation screening on the retained 5–20 m terrains. These budgets are proposed
project requirements, not published universal standards, observational accuracy
claims or tolerances for regulatory no-increase tests, recharge estimation,
property thresholds or structural design. Those applications need their own
error budget and validation. Source formulations are independent hypotheses.

[Environment Agency guidance](https://www.gov.uk/guidance/using-modelling-for-flood-risk-assessments)
requires uncertainty, sensitivity and boundary influence to be assessed for the
application. [SEPA guidance](https://www.sepa.org.uk/media/219653/flood_model_guidance_v2.pdf)
discusses 0.1 m thresholded mapping alongside unthresholded outputs. These
support purpose-specific reporting and retaining shallow-water diagnostics;
neither source endorses the numerical budgets below. Regulatory assessment of
changes to flood risk must still use raw differences, not subtract these budgets.

## A. Mandatory numerical integrity

All original mass, finite/nonnegative depth, CFL, CPU/CUDA, graph, restart,
analytical and extraction-parity criteria remain mandatory and unchanged.
Cross-solver differences in infiltration/export/storage are not mass residuals.

## B. Hydraulic significance at equivalent boundaries

Only slope-FREE versus slope-FREE independent fixtures receive hydraulic
agreement verdicts. Dry-exterior-head versus slope-FREE is labelled
non-equivalent, without an agreement pass/fail. Diagnostics remain visible.

* Depth: use the union where either solver reaches 50 mm at each observation,
  including false inundation, with elapsed-time weighting. Require pooled RMSE
  <= 10 mm OR relative L2 <= 5%, and weighted P95 of
  |depth difference|/(10 mm + 5% of reference depth) <= 1.
  The absolute budget is one fifth of the lowest assessed substantial depth;
  the relative term allows depth-scale growth without accepting systematic
  centimetre errors at shallow depths. P95 limits widespread errors, while
  maxima and the fraction exceeding the local budget are reported. An isolated
  maximum is not an automatic failure. Any error exceeding 50 mm requires
  explicit review, including a small affected feature that a percentile misses.
  Meaningful-depth absence is 'no coverage', not evidence of depth agreement.
* Extent: report all six existing thresholds unchanged. Only 50 and 100 mm
  enter the release screen; 10 mm remains a low-depth sensitivity diagnostic.
  At each threshold retain raw IoU/precision/recall/false wet/dry areas. For
  significance, disregard a mismatch only if BOTH depths are within 10 mm
  of that threshold, consistent with the depth budget. Require resolved
  disagreement <= 5% of union area-time, AND per-sample resolved disagreement
  >10% of wet union for no more than 10% of event duration. This protects both
  event-wide extent and persistent small-feature errors, without a single
  transient raster classification deciding the verdict. Report final and sampled
  maximum-envelope extent too; those are diagnostics, not extra vetoes. An
  envelope is formed from retained simultaneous samples, not every timestep.
  Empty unions carry no inundation coverage and undefined precision/recall.
* Discharge: retain sampled hydrograph NRMSE <=5% of reference peak; require
  peak magnitude difference <=10% and peak-time displacement <=max(one output
  interval, 5% of event duration). These represent modest routing magnitude and
  timing differences for screening, not a drain-capacity design criterion.
  Zero-reference flow requires zero actual flow. Report complete signed
  hydrographs and cumulative export, including late recession. Sparse 60 s
  samples cannot certify faster peak/arrival dynamics.
* Water allocation: final export and simultaneous storage differences must
  each be <=5% of initial plus supplied reference water. This independent
  event-output budget permits modest intentional partition differences while
  retaining deviations of order 1/20 of the event as concerns. Also report
  errors relative to reference export/storage, where tiny residual storage can
  have a large relative difference. No tolerance is added to numerical mass
  conservation. Zero supply requires equality. The 5% allocation budget is
  unsuitable for applications requiring more precise storage/export estimates.

All applicable gates must pass for 'within screening budgets'. A >50 mm local
error requires review regardless of pooled gates. Uncovered depths remain
explicit coverage limitations. Boundary mismatch never yields a pass.

## C. Infiltration and intended use

Rain-available physical infiltration and ACC's ponding/cutoff/source ordering
are intentional different laws. Always report infiltration difference relative
to reference infiltration, physical infiltration and rainfall; do not require
infiltration equivalence to qualify inundation routing if hydraulic budgets pass.
For recharge use, separately flag >10% difference in event infiltration relative
to reference (zero reference with nonzero actual is flagged). This is a proposed
recharge agreement budget, not evidence either source law is correct. Different
source laws require independent soil/recharge evidence before recharge-specific
qualification, even if their agreement happens to meet this budget.

Explained formulation differences within these budgets do not independently
block bounded solver use. Unexplained or out-of-budget results remain concerns.
A pass cannot supply missing matched-default, deeper-flood, field, velocity or
high-frequency temporal validation. No solver parameters or activation depths
are changed during this assessment.
