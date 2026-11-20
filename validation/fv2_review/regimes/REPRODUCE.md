# Reproducing the FV2 terrain-regime study

Run the modules from the FloodWarp repository root in its pinned Python 3.14 environment. The GPU was a GeForce RTX 4070 Ti SUPER. The base tree is `85ffd69`, the speed-reuse branch head reviewed was `03d39c8`. The archived HIRDS inputs retain their Lincoln Agritech and data-provider credits. Read the repository's data licences before distributing inputs.

## Analytical and synthetic runs

```sh
.venv/bin/python -m validation.fv2_jump_review
FV2_JUMP_METHODS=adaptive FV2_STUDY_REVISION=halo .venv/bin/python -m validation.fv2_jump_review
FV2_JUMP_METHODS=adaptive FV2_STUDY_REVISION=marked .venv/bin/python -m validation.fv2_regime_jump_history
.venv/bin/python -m validation.fv2_jump_residual
.venv/bin/python -m validation.fv2_regime_verify
.venv/bin/python -m validation.fv2_regime_planes
.venv/bin/python -m validation.fv2_lowland
FV2_FINE_REFERENCE=1 .venv/bin/python -m validation.fv2_lowland
.venv/bin/python -m validation.fv2_regime_cost
.venv/bin/python -m validation.fv2_regime_figures
.venv/bin/python -m validation.fv2_regime_tables
```

The default jump output label is `baseline`. For an actual baseline, use the audited solver source in a separate checkout as below. Choose a different `FV2_STUDY_REVISION` for new timings or jump records. Run other modules in a disposable checkout to keep their checked-in result files intact. Source parameters, domain sizes, sample times and thresholds appear in each module's execution guard.

The original 0.625 m lowland numerical-reference matrix includes a simultaneous grid and terrain change from 1.25 m. The refined study preserves the native 0.625 m terrain while subdividing the hydraulic grid to 0.3125 m. The fixed-geometry branch subdivides the aggregated 5 m raster. Storage, physical input and connectivity changes and the two fine-formulation differences are separate columns. The cost matrix uses errors from the original 0.625 m study. Cross-method spreads are sensitivity measurements, without a rigorous bound on error.

## GPU and storm replay

```sh
git worktree add --detach /tmp/floodwarp-regime-baseline 85ffd69
FV2_STUDY_REVISION=baseline PYTHONPATH=/tmp/floodwarp-regime-baseline/src .venv/bin/python -m validation.fv2_regime_profile
FV2_STUDY_REVISION=final FV2_STUDY_METHODS=contact_hll,mc,minmod,adaptive .venv/bin/python -m validation.fv2_regime_profile
.venv/bin/python -m validation.fv2_regime_sparse
FV2_PARITY_ROLE=baseline PYTHONPATH=/tmp/floodwarp-regime-baseline/src .venv/bin/python -m validation.fv2_regime_parity
.venv/bin/python -m validation.fv2_regime_parity
FV2_STUDY_REVISION=baseline PYTHONPATH=/tmp/floodwarp-regime-baseline/src .venv/bin/python -m validation.fv2_regime_orton
FV2_STUDY_REVISION=final FV2_ORTON_METHODS=contact_hll,fv2_mc,fv2_adaptive .venv/bin/python -m validation.fv2_regime_orton
FV2_EVENT_REVISION=baseline FV2_EVENT_METHOD=fv2_mc FV2_EVENT_STUDY=storm PYTHONPATH=/tmp/floodwarp-regime-baseline/src .venv/bin/python -m validation.fv2_regime_event
FV2_EVENT_REVISION=final FV2_EVENT_METHOD=fv2_adaptive FV2_EVENT_STUDY=storm .venv/bin/python -m validation.fv2_regime_event
FV2_EVENT_REVISION=final FV2_EVENT_METHOD=contact_hll FV2_EVENT_STUDY=storm .venv/bin/python -m validation.fv2_regime_event
FV2_EVENT_REVISION=halo FV2_EVENT_STUDY=continuation .venv/bin/python -m validation.fv2_regime_event
```

Storm replay requires the native archive identified in `validation/fv1_riemann_review/corrected/catchment.json`. The runner checks its SHA-256 and the original HIRDS time and rate hashes. Use an empty `outputs/fv2_regime_hirds48/<revision>` to time a fresh storm. With existing checkpoints it resumes the matching method. Runs from different limiter formulations must use different output directories.

Use Nsight Compute with the uncaptured probe. Record original and final code in separate files.

```sh
ncu --csv --kernel-name 'regex:(faces_|slopes_|split_sources_|euler_)' --launch-count 5 --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second .venv/bin/python -m validation.fv2_counter_probe
```

Allocation reports distinguish solver-owned arrays, the Warp allocator pool and its process high-water mark. Per-kernel figures aggregate both RK stages and any rejected trials per accepted step. Source and CFL checks remain enabled throughout the hydraulic timings. Capture/preload fields measure driver setup in that process. A warmed disk cache can make preload much shorter than a first-ever JIT.

## Version and outcome labels

- `jump_baseline`, `baseline_mms`, `profile_baseline`, `orton_baseline` and `baseline/20m_fv2_mc_dense` use the audited dev mathematics.
- `jump_adaptive` tests a strict characteristic crossing. `jump_adaptive_band`, `jump_history.json`, `20m_event.json` and `wet_5m.json` evaluate intermediate sensor designs. The long stationary-shock test rejected the central-stencil-only sensor at 1.25 m.
- Commit `d104751` introduces the central compressive sensor, primitive flux and dynamic face experiment. Commit `4a65e32` adds the inverse-depth and source changes. Commit `60c6464` adds one neighbouring stencil and a distinct checkpoint sensor identity. The final implementation marks the identical directional stencil from its fast incident cell.
- `jump_halo`, `jump_history_marked`, `profile_marked`, `orton_marked`, `sparse`, `halo/wet_5m` and the `final/` storm records qualify the selected algorithm. `reuse`, `source`, `warp`, `traces`, `primitive`, `inv_cache`, `fused` and `decay` identify successive ablation runs. The discarded full-warp shuffle reduction used five shuffle-down minima and a leader atomic for full warps, with the original per-lane atomic on partial warps.
- Whole-catchment peaks are inter-model comparisons. An observed storm hydrograph was unavailable. The 5 m adaptive continuation starts from the audited FV1 field at hour 25. The 20 m complete events each start dry. The full 5 m adaptive event did not pass the native performance gate.
- `regression.json` records the full test outcome. The independent numerical references, fallback sources and data licences in older investigations remain intact.
