# Independent sparse FV1 development

The reference fetched on 2026-10-31 is `183c884` on origin/dev and origin/main.
The pre-existing untracked `.idea/` directory is preserved. All implementation
is derived from FloodWarp's discrete operator; external solvers supply no code.

## First stage: execution, with dense storage

Retain FP64 conservative state, geometry, fine source calculations, soil history,
maxima and transport histories. Keep the full-domain source-aware/boundary-aware
CFL reduction before sources. Construct activity after every source update from
exact nonzero state and wet prescribed stages. No positive depth threshold defines
activity. Execution options are independent of the numerical/restart identity.

The retained hourly checkpoints show zero surface water for hours 1–11, nonzero
water throughout all 559,985 physical cells at hours 20–30, and 2,112 nonzero
cells at 72 hours. The retained wet profile attributes 47.3% of GPU kernel time to
faces, 22.1% to update, 9.0% to draining, 8.2% to face limiting, 7.0% to CFL and
5.9% to sources. These are diagnostic bounds, not an event speedup prediction.
`validation.sparse_fv1_audit` records all hourly tile occupancies, weighted by
the actual preceding-hour hydraulic runtime; endpoint sampling is explicitly
not an exact per-step cost integral.

A seed tile contains any nonzero conservative component or a physical cell
touching a prescribed stage above its bed. Include all eight adjacent tiles.
Each face belongs to the cell on its east/north side, with exterior east/north
faces assigned to the last column/row. Every face incident to a nonzero state
or wet ghost is therefore evaluated once. Every possible receiving cell is in
the work set. All four faces of every possible nonzero donor are present before
draining is calculated; limiting then uses that donor's single water budget.
The outer fringe contains only zero states and zero ghosts: omitted faces have
zero flux and pressure. This proof depends on the current first-order face
operator and must be revisited for any reconstructed/higher-order stencil.

Clear previously evaluated face and pressure scratch before rebuilding lists;
reset released limiter values to one. Never clear cumulative volumes, maxima,
soil or source ledgers. Device flags and compact lists are rebuilt every step
and reconstructed after restart. Batched kernels consume these device lists.
At high occupancy they use dense indexing; the ordinary dense execution option
also remains available without sparse scheduling overhead.

Unavoidable work includes the full CFL reduction and all physical-cell sources,
activity classification, dependency closure, list compaction and scratch clearing.
The first implementation adds scheduling memory; it makes no sparse-storage claim.

## Stage gates and controls

1. Compare common safe prescribed steps, then the adaptive controller, across
   lake equilibrium, moving/oblique fronts, rain/infiltration, wet boundaries,
   masking, shallow nonzero storage, deactivation/reactivation and restart.
   Require unchanged state, sources, accepted face histories and maxima with
   FP64 tolerances. Do not relax tolerances to hide omitted transport.
2. Replay saved Purau dry, rising, peak and recession states with warm-up and
   at least three timed repetitions. Profile separately. Record scheduling,
   work counts, memory and complete-field differences. Dense fallback remains
   the default until measured benefit and qualification justify a change.
3. Sparse allocation requires useful measured occupancy and memory savings;
   pool exhaustion, deallocation and restart need independent controls.
4. Static multiresolution requires independent accuracy budgets, conservative
   subface transport and donor budgets, equilibrium-preserving transfers,
   fine-geometry/connectivity vetoes and explicit fine-source aggregation.
   Assess active-region coarsenability even if exact sparsity is unprofitable.
5. Dynamic adaptation follows passing static controls; its overhead must be
   measured. No subcycling, precision change, new CFL or Riemann law is part
   of these stages. Unqualified methods remain experimental.

No completed maximum-depth or future inundation field may determine a production
activity/refinement mask. Retained checkpoints are only initial replay states
and measurement inputs.

## Default promotion on dev

After the matched 72-hour replay and explicit user authorisation, the uniform
FV1 default is now 8×8 sparse execution with dense fallback at 65% tile
occupancy. This fulfils the first execution gate; it does not qualify sparse
storage or AMR. The original gate and reports above describe the staged
investigation. The [promotion record](../validation/sparse_review/DEFAULT_PROMOTION.md)
records current defaults and regression checks; original timed source snapshots
and the fixed v0.1.0 release are preserved.
