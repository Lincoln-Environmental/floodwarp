# Static multiresolution reconstruction investigation

Development research only; no change to StaticBlockFV1Solver, its geometry/source vetoes, or either stable release. Original failed 64×64 controls and snapshots remain intact. No dynamic AMR, wavelets, subcycling or mixed precision is introduced.

## Larger genuinely coarsenable controls

512×512 cells, one global timestep, immutable finest geometry and fine source/history, warm-up plus three repetitions. The physical domain is larger than the old 64×64 controls; this is not a spatial grid-convergence study. Options and the pre-existing physical accuracy budgets are recorded in large_controls.json.

| Case | Block | Fine leaves | Coarse leaves | Fine subfaces | Dense/static runtime | Depth RMSE [m] | Water residual [m³] |
|---|---:|---:|---:|---:|---:|---:|---:|
| lake | 4 | 24064 | 59520 | 287232 | 1.443 | 4.23544e-17 | 0 |
| lake | 8 | 47104 | 53760 | 310272 | 1.386 | 4.025e-17 | 0 |
| front | 4 | 24064 | 59520 | 287232 | 1.141 | 3.08915e-05 | 7.28e-12 |
| front | 8 | 47104 | 53760 | 310272 | 1.117 | 1.36687e-09 | 7.28e-12 |
| ponding | 4 | 16128 | 61504 | 279296 | 1.392 | 0 | 7.96e-13 |
| ponding | 8 | 31744 | 57600 | 294912 | 1.375 | 0 | 7.96e-13 |
| channel | 4 | 24064 | 59520 | 287232 | 0.987 | 0 | 0 |
| channel | 8 | 47104 | 53760 | 310272 | 0.969 | 0 | 0 |

Lake, front and rainfall ponding now show 1.12–1.44× speedup relative to dense on these broad coarsenable controls. The protected narrow-channel case remains slightly slower: its active channel stays fine, even though dry terrain can be compressed. Mesh compression outside the hydraulic workload is not a demonstrated hydraulic saving. All 262,144 fine source cells remain evaluated. Persistent fine scratch plus leaf arrays use more memory; per-case persistent totals and separate transfer profiles are retained. No sparse/static speedups are multiplied.

## Smooth curved terrain and dynamical error

128×128 fully wet, known connected synthetic terrain; the fine DEM is never changed. Strict affine geometry tolerance admits no roots. A bounded 0.001 m affine-fit tolerance admits 196 roots on this smooth curved bed; this experiment is not a general connectivity veto. Lake depth RMSE is approximately 8.3e-17 m and momentum remains below 4.3e-16 m²/s. The moving-surface control has approximately 2.90e-5 m depth RMSE after 0.3 s with nonnegative depth and a zero recorded mass residual. Initial transfer error, maximum component error and the bounded one-step R(A_f(U)) versus A_c(R(U)) discrepancy are recorded independently in curved_controls.json. These controls show why exact-affine terrain is too restrictive for some representable equilibria; they do not authorise coarsening barriers, sills or disconnected water.

## Active-region Orton Bradley reconstruction

Saved-state analysis uses current depth/momentum, immutable bed, geometry and fine source/history only, never future inundation. The constant-surface representation often cannot reconstruct positive thin runoff on sloping fine terrain, even before the geometry veto. At hour 25, of 139,576 fully wet 2×2 physical parents, only 87 satisfy positive reconstruction and a 0.001 m depth-error screen; none also satisfy the existing exact source/history-homogeneity screen. Relaxing affine-bed tolerance alone therefore does not produce a qualified model.

| Hour | Constant-surface parents at 0.001 m | Affine-surface parents at 0.001 m |
|---|---:|---:|
| 1 | 0 | 0 |
| 13 | 0 | 241 |
| 25 | 87 | 2496 |
| 48 | 129 | 988 |
| 60 | 105 | 105 |
| 72 | 96 | 96 |

An affine free-surface diagnostic preserves parent volume, nonnegative representable states and a lake over non-affine bed, and exactly reconstructs the analytical thin film on a slope to roundoff. This requires additional hydraulic degrees of freedom and is not implemented in the static solver. Its larger screen counts do not establish eligible blocks, connectivity, source availability, momentum accuracy, interface transport or dynamical error. Source/history constraints are not waived and no production refinement mask is built. Parent transfer roundoff and analytic errors are retained in affine_surface_screens.json.

## Stage decision and measured bottlenecks

Static research remains experimental. Larger broad-water controls pass their bounded accuracy/runtime budgets, while narrow-channel runtime and real-terrain representability/source screens remain substantive limitations. Fine restriction/projection and fine source work are separately profiled; leaf-native hydraulics with persistent fine soil/history is the next representation task. Further work needs topology/connectivity safeguards, positivity-preserving surface-gradient transfers and matching coarse/fine subface and donor-budget controls. Fine source availability beneath a coarse state is a separate numerical choice, not parameter averaging. No full-domain static event or dynamic adaptation is promoted. Sparse storage, mixed precision and temporal subcycling remain separate gates.
