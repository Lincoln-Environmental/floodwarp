# LI engineering screening protocol v1

Declared after baseline extraction parity and before computing engineering
classifications. This is a proposed application-specific screen, not an industry
standard or a replacement for frozen ACCEPTANCE.md. No original gate changes.
The available evidence already informed the review request; this is not a blind
validation or an independently calibrated error budget.

Scope: shallow rainfall/runoff and coarse inundation screening on the retained
5–20 m analytic terrains. Not structure design, hazard certification, calibrated
catchments, high-velocity SWE flows, or sub-grid property flooding.

A. Numerical integrity is mandatory with existing tolerances: conservation,
finite nonnegative depths, accepted CFL steps, analytical regressions, CPU/CUDA,
graph and restart controls, and extraction parity. Skips are coverage limits.

B. Hydraulic screening (apply at final time and at every retained sample):

* Depth RMSE <= 1 mm OR relative L2 <= 2%, retaining the frozen field scale.
  Additionally absolute-error P95 <= 5 mm and maximum <= 10 mm. The latter
  guard against domain averaging hiding errors comparable to the lowest
  engineering inundation threshold. These are screening resolution budgets,
  not claims of surveyed elevation or model accuracy.
* Inundation: report all six thresholds (0.1, 1, 5, 10, 50, 100 mm), never
  substitute them for solver activation thresholds. At 10, 50 and 100 mm,
  require IoU >= 0.95, or every disagreeing cell to have BOTH depths within
  1 mm of the threshold. One millimetre is the retained absolute depth
  comparison budget, 10% of the smallest engineering threshold, and 1% at
  100 mm. Publish raw precision/recall/counts alongside the uncertainty-band
  diagnostic. Empty unions are not flood-extent evidence; mark them empty.
  No automatic pass based on coarse-cell area or vanishing wet fraction.
* Outlet sampled NRMSE <= 5% of reference peak, unchanged. If reference flow
  is identically zero, only zero actual flow passes; normalisation is otherwise
  undefined. Report signed time series and cumulative volume differences.
* Cumulative export and simultaneous storage error each <= 2% of initial
  plus reference rainfall plus boundary inflow, at the end of the event.
  This is an event water-allocation budget (20 m3 per 1000 m3 supplied),
  independent of small remaining storage. Unlike the original mixed 5 m3/2%
  rule it has no domain-size-independent absolute allowance. For zero supply,
  require equality. Report errors relative to both storage/export and supply;
  important partitioning differences cannot disappear into inundation scores.

C. Source, friction and boundary distinctions are assessed separately. An
engineering-screen failure is a qualification concern, not proof either solver
is wrong. A pass only supports the bounded conditions represented. No default
boundary is certified by comparison to a different closure. Report matched
slope-FREE agreement separately from dry-exterior-head sensitivity.

All criteria must pass for an unqualified screen pass. Preserve failed gates,
not merely a single composite verdict. Attribution experiments change explicit
research profile objects only; production operators/defaults remain frozen.
