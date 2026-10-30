# Frozen extraction acceptance protocol

Set before running floodwarp comparisons. No roughness, boundaries, solver
operators or tolerances may be adjusted to obtain passes. All physical grids
are at most 100 by 100 cells; external halos are excluded from physical metrics.

For matched independent reference fields, require depth RMSE <= 0.001 m OR
relative L2 <= 0.02. Require final storage difference <= 5 m3 + 2% of the
larger storage, wet Jaccard >= 0.95 OR wet mismatch <= 0.0005 of domain
(using the common 0.001 m wet threshold). When a sampled hydrograph is
available, require outlet NRMSE <= 0.05 (normalised by reference peak), and
cumulative outlet volume error <= 5 m3 + 2% of reference export.

For each run require finite nonnegative depth, successful CFL-controlled
completion, and mass residual <= 1e-8 * max(1 m3, initial plus supplied volume).
CPU/CUDA, graph, restart and analytical regressions retain their source-test
tolerances without relaxation. Failed comparison gates remain failures.

Compare physical LI and ACC-compatible LI independently. Deliberate source,
friction, dry-depth and boundary-operator differences must be recorded. FV1
uses contact_hll/velocity_extrapolation with reflecting walls. Native FV1 uses
its independent operators; agreement is not an instruction to replace ours.
Report final boundary-band (three cells) and interior depth errors separately.

Grid and timestep sensitivity require independently supplied matching input
and reference fixtures. Missing temporal fields, hydrographs, sensitivity
controls or source/boundary matches are explicitly incomplete coverage,
never inferred passes. Historical campaign scores are not new release tests.
