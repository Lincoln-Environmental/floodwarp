# Conservative static block FV1 — experimental stage result

**Numerical foundation implemented; production runtime gate failed. Dynamic adaptation is deferred.** This is a real leaf-volume FV1 evolution with conservative shared subfaces, not a compressed depth map.

The forest has regular blocks, two spatial levels (2:1), fixed topology and one global timestep. Each fine physical cell belongs to exactly one evolved leaf. Each represented fine subface is solved once with FloodWarp’s existing HLL operator. Its accepted mass flux enters the two neighbouring leaves with opposite signs. A coarse donor shares one volume budget across all of its subfaces; hydrostatic pressure corrections remain side-specific.

Finest geometry is immutable. Coarse bed is the area mean used only to define the parent common free surface; fine subface beds are retained. Restriction preserves area-integrated water and momentum. Prolongation preserves those totals and equilibrium where representable. Mixed wet/dry, non-affine/connectivity-changing geometry, mask/boundary interfaces, protected features and source/roughness heterogeneity veto coarsening. Unrepresentable transfers stay fine. A later unrepresentable sloping coarse wet/dry state fails the experimental run; it is not silently clipped into an accepted solution.

Sources and soil/history arrays stay at fine resolution. Surface availability beneath a wet parent comes from its common free surface and velocity; accepted fine source increments are restricted to that parent. That availability/reconstruction is a documented spatial approximation, not parameter averaging. Source setters cannot bypass heterogeneity vetoes. The existing finest source/boundary-aware global CFL scan is retained.

## Accuracy and runtime

Pinned FP64 environment, RTX 4070 Ti SUPER. Each matched 6 s control uses warm-up and three graph-timed repetitions on a 64×64 physical grid. Median hydraulic seconds are below. Topology setup, end-to-end loop time, timesteps, per-step time, per-level leaf counts and subface interface counts are in benchmark.json. Setup/JIT are outside hydraulic timing.

| Control | Dense [s] | Static B=4 [s] | Static B=8 [s] | B=4 leaves | B=4 fine subfaces | B=4 depth RMSE [m] | B=4 relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lake | 0.021678 | 0.026063 | 0.025457 | 2944 | 6784 | 1.298148e-17 | 2.058205e-17 |
| front | 0.018070 | 0.022431 | 0.021884 | 2944 | 6784 | 0.003892699 | 0.01513073 |
| ponding | 0.021227 | 0.025547 | 0.025430 | 2368 | 6016 | 0 | 0 |
| infiltrating | 0.010149 | 0.014168 | 0.014055 | 2368 | 6016 | 0 | dry |
| heterogeneous | 0.019771 | 0.023747 | 0.023696 | 4096 | 8320 | 0 | 0 |
| channel | 0.017392 | 0.021078 | 0.020382 | 2944 | 6784 | 0 | 0 |

Measured dense/static ratios range from 0.716 to 0.853: **17.2–39.6% slower**. The B=8 lake/front/channel controls have zero eligible coarse roots; they measure the all-fine bookkeeping cost and must not be presented as multiresolution accuracy successes. B=4 does evolve coarse wet leaves in lake, ponding and moving-front controls.

The coarsened moving front has approximately 3.89 mm domain depth RMSE and 1.51% relative L2 error, within the predeclared relative budget. The runtime rejection is substantive, not a rejection of insignificant floating-point differences. Lake, ponding, infiltration and protected-channel controls reproduce the uniform result to FP64 tolerance. Closed controls have no applicable outlet hydrograph; the channel outlet comparison records NRMSE, peak magnitude and timing separately.

Separate instrumented profiles identify restriction, two fine projections, raw-outflow aggregation, subface acceptance/scatter, clearing and additional launches as overhead. Fine sources/CFL are not removed. Bounded one-step R(A_f(U)) versus A_c(R(U)) diagnostics are retained; there is no production fine-grid shadow evolution.

## Memory

| Control | Mode/block | Persistent bytes | Peak additional live pool bytes |
|---|---|---:|---:|
| lake | dense | 1073308 | 1073324 |
| lake | static/4 | 1364124 | 1364124 |
| lake | static/8 | 1467036 | 1467036 |
| front | dense | 1073308 | 1073324 |
| front | static/4 | 1364124 | 1364124 |
| front | static/8 | 1467036 | 1467036 |
| ponding | dense | 1073308 | 1073324 |
| ponding | static/4 | 1312668 | 1312668 |
| ponding | static/8 | 1398428 | 1398428 |
| infiltrating | dense | 1073308 | 1073324 |
| infiltrating | static/4 | 1312668 | 1312668 |
| infiltrating | static/8 | 1398428 | 1398428 |
| heterogeneous | dense | 1073308 | 1073324 |
| heterogeneous | static/4 | 1467036 | 1467036 |
| heterogeneous | static/8 | 1467036 | 1467036 |
| channel | dense | 1073308 | 1073324 |
| channel | static/4 | 1364124 | 1364124 |
| channel | static/8 | 1467036 | 1467036 |

Static construction retains a resident uniform source model. The separate memory probe reports its bytes and whole-process peak as well as the additional static allocation. CUDA context/code/reserved memory is excluded. Dense fine projection/face/history arrays remain allocated and leaf fields add memory; there is no sparse-storage saving.

## Allocation and adaptation stage decisions

The sampled ideal high-water state-plus-face pool bounds for Purau save at most 38,166,432 bytes (15.50% of 246,266,984 bytes), before pool growth slack, maps and padding. A conservative-state-only pool saves only about 3.3% of total allocation. The current model has no device-capacity bottleneck on a 16 GiB GPU. Block-sparse storage is therefore deferred at its occupancy/memory gate; no pool exhaustion/deallocation/restart implementation is claimed.

Coarsenability is measured inside active regions independently of exact sparsity. At 25 h there are 139,576 fully wet 2×2 physical parents, 19 pass the simple hydraulic indicator, and none pass the initial affine-geometry safeguard. Across all six sampled phases, 4/8/16-cell block choices admit no coarse Purau roots. These are limitations of this conservative initial representation and thresholds, not proof that all AMR is unprofitable. Completed-event fields serve diagnostics only.

Static numerical controls pass, but the measured runtime gate fails and no admissible full Purau candidate exists under the initial safeguards. Dynamic refinement/coarsening, Haar indicator comparisons, sparse-storage integration and combined speedups remain unimplemented. No static/adaptive 72-hour Purau simulation is launched. A lower-overhead leaf/source/geometry representation needs evidence before advancing that stage.

## Verification and scope

Full suite: **1585 passed, 58 skipped**. The focused static/sparse log contains **80 passing controls**: stepped/affine lake equilibrium, positivity, common safe steps, ordinary CFL, moving and oblique fronts, physical crossings in all four orientations, rain ponding, fully infiltrating rain, heterogeneous sources, disconnected regions, protected narrow channel/barriers, source-change vetoes, shared donor budgets, masked boundaries, 32 repeated conservative transfers, CPU/CUDA disk restart and exact sparse history/activation controls. Tiny positive water at a high datum is retained by stable bed-difference arithmetic. Repeated transfer testing is not a claim of dynamic adaptation.

Both research methods remain opt-in on dev. The published v0.1.0 tag/main remain fixed at 45a374c and contain neither research implementation. Dense FP64 FV1 remains the reference/fallback. No Riemann law, source order, boundary closure, precision, finest CFL restriction, reference data, external solver implementation or Overlander checkout was changed by this stage.
