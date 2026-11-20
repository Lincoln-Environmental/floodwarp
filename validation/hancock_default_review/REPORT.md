# Adaptive MUSCL–Hancock as the development default

New full shallow-water models on `dev` select `fv1_hancock`, the adaptive MC/minmod limiter, contact-upwind HLL and FP64. Conditional CUDA graphs are automatic where supported. This promotes the fused predictor qualified at `7a078fe`. The change covers both Python factories, the local UI and its worker, the editable direct-input example and the synthetic example. The hydraulic kernel code is identical to the qualification revision. Tagged v0.1.3 retains its FV1 default. [Original numerical investigation](../fv1_diffusion_review/REPORT.md)

## Analytical and software qualification

The public factory was called with the method, limiter and flux omitted. The analytical driver retained the historical timestep caps, terrain, initial states and observation windows. Across 48 matched case-and-grid records the stored hydraulic results, including accepted section hydrographs where recorded, matched explicit adaptive Hancock exactly. The comparison also checks stored timestep statistics and graph use. The largest absolute volume-balance residual across those cases was 7.28 × 10⁻¹² m³. [Comparison counts](verified_counts.json) · [Executable comparison](../hancock_default_qualification.py)

| Control | New measurement |
|---|---:|
| Partly dry lake, 2.5 m | Depth RMSE 1.31 × 10⁻¹⁷ m |
| Ritter dry-bed dam break, 2.5 m | Depth RMSE 0.00374 m |
| Stoker wet-bed dam break, 2.5 m | Depth RMSE 0.00669 m |
| Thacker basin with moving shoreline, 2.5 m | Depth RMSE 0.00207 m |
| Radial oscillating basin, 10 / 5 / 2.5 m | Depth RMSE 0.1274 / 0.05065 / 0.01716 m |
| Late transcritical jump, 5 / 2.5 / 1.25 m | Section discharge RMSE 0.001830 / 0.000473 / 0.000148 m²/s |

The remaining controls cover shear transport, aligned and offset jumps, sloping dry-bed flow, subcritical and transcritical flow over a bump, gradually varied Manning flow and two rain-fed moving equilibria. [Wave metrics](analytical.json) · [Hydraulic controls](controls.json) · [Manning flow](manning.json) · [Rain-fed equilibria](rain.json) · [Radial hydrographs](radial.json) · [Late jump histories](shock_histories.json)

The forced two-dimensional manufactured test uses the Hancock driver with the adaptive limiter and analytically prescribed forcing. Coupled refinement from 64 to 128 cells per axis gave orders **2.02, 1.98 and 1.99** for depth and the two momentum components. Timestep refinement on the fixed 16 × 16 grid gave **1.05, 1.05 and 1.05** against the small-timestep numerical limit. Applications needing demonstrated second-order fixed-grid time convergence should use `solver_method="fv2", limiter="adaptive"`. [Manufactured results](manufactured.json) · [SSP-RK2 control](../fv1_diffusion_review/manufactured.json)

The complete pinned Python 3.14 suite with CUDA validation enabled passed **1,854 tests**, with **24 skipped**. Tests cover CPU and CUDA, conditional graph execution, constant and varying stages, Manning drag, rain, constant and cumulative exponential infiltration, dry fronts, positivity, mass ledgers, accepted face volumes, restarts and explicit FV1/FV2 controls. Skips comprise 18 conditional-graph/debug configurations, four precision-specific configurations and two optional external LISFLOOD-FP checks. The fresh installed development wheel also passed the public CPU default and LI smoke checks. [Selection and replay tests](../../tests/test_release_defaults.py) · [UI and graph tests](../../tests/test_ui_runner.py) · [Numerical parity tests](../../tests/test_fv2.py)

## GPU cost and solver choice

Seven warmed, synchronised repeats on one RTX 4070 Ti SUPER used the same 1 m terrain, FP64, 1,024 × 640 cells and 2 s duration. Both methods accepted 64 steps of 0.03125 s with zero retries. The table uses each method’s default scheduler, sparse FV1 and dense Hancock. It reports median hydraulic wall time excluding initial graph capture. Device-array sizes for these schedules were 162.8 MiB and 282.6 MiB respectively. [Timings, memory and kernel measurements](large_grid_timing.json)

| Synthetic case | FV1 | Adaptive Hancock | Hancock / FV1 |
|---|---:|---:|---:|
| Fully wet Stoker dam break | 153.56 ms | 147.03 ms | 0.96× |
| Mostly dry domain with a wet strip | 31.79 ms | 90.71 ms | 2.85× |

Across the earlier 57 paired small-grid runs the median ratio was **1.48×** under common caps and **1.69×** under individual timestep policies. The latter ranged from 1.21× to 3.47×. For the 5 m shallow frictional front on a 0.05% grade, adaptive Hancock reduced depth RMSE from 20.06 to 3.05 mm and section discharge RMSE from 0.02917 to 0.00510 m²/s. Cost depends on wet area, scheduling and accepted timesteps. [Full accuracy and cost tables](../fv1_diffusion_review/TABLES.md)

Use adaptive Hancock for general full-SWE runs on dev, especially where front smearing, transverse transport or discharge attenuation matters. Use explicit FV1 contact-HLL when its measured depth and discharge errors meet the application tolerance, for sparse dry domains where cost dominates, or for supported full-SWE culverts. FV2 adaptive provides the SSP-RK2 temporal control. MUSCL–Euler remains selectable as a spatial reconstruction control. Select LI through Python when independent checks justify neglecting nonlinear advection. Neither bed slope alone nor agreement between solvers establishes that approximation.

## Replays and feature compatibility

- New Python models may use `solver_method="hancock"` or omit the method. Both record the canonical identity `fv1_hancock`. Explicit FV2 and MUSCL–Euler keep their previous MC limiter unless configured otherwise. The UI offers all four full-SWE methods. Its ordinary form receives the default from the server. The matched Orton preset explicitly selects FV1.
- Replay saved Python options through `floodwarp.model.create_model_from_config(terrain=terrain, options=options)`. A saved mapping from the FV1-default era with no method selects FV1. A historical explicit `fv1_hancock` request with no limiter selects MC. Save the explicit method and limiter for new Python replays. For pre-FV1-default LI scripts specify `li`. Passing an unlabelled dictionary as keyword arguments to the new-model constructor has new-model semantics.
- Load historical UI options with `RunConfig.from_saved(options=...)`. Both the HTTP request reader and worker apply that policy to methodless inputs. A normal browser submission carries an explicit method. The worker reports the selected solver in its run configuration, metrics and checkpoint.
- Build the matching model before calling `load_checkpoint`. Checkpoints include the actual method, limiter and integration. A first-order checkpoint records a null slope limiter and forward Euler. Older files continue to use their original configuration and input identity checks. CUDA checkpoints written at `7a078fe` for FV1, FV2 MC and Hancock MC reproduced all five original continuation hashes per model, covering the conservative state, maximum depth, cumulative infiltration and accepted x/y volumes. [Cross-version hashes](continuations.json)
- Hancock rejects culverts. Select `fv1` for supported conservative full-SWE structure exchange. Traditional and enhanced SGS require explicit `li` and retain their experimental status. FV1-only fluxes and sparse tile tuning require explicit FV1. Hancock supports dense, fixed-mask sparse and experimental dynamic execution. Its predictor fallback map is available through `predictor_fallback_cells()`. The two-stage reconstruction diagnostic belongs to FV2. Low-Froude HLL remains opt-in.

## Reproduction and release scope

Run in the pinned environment from the repository root with CUDA enabled. The analytical driver compares numerical results with the existing checked-in qualification. Performance records are separate from those exact comparisons.

```sh
.venv/bin/python -m validation.hancock_default_qualification
.venv/bin/python -m validation.hancock_default_profile
OVERLANDER_TEST_FULL_SWE_CUDA=1 FLOODWARP_TEST_RENDER=1 FLOODWARP_REFERENCE_BINARY="$PWD/outputs/qualified_reference/build/lisflood" .venv/bin/python -m pytest -q -rs
```

To verify historical files, export the prior source to a temporary directory. Use the same device and environment for both processes. The second process restores the checkpoints and compares its hashes with the first.

```sh
prior_source=$(mktemp -d)
export FLOODWARP_RESTART_DIRECTORY=$(mktemp -d)
git archive 7a078fe src | tar -x -C "$prior_source"
PYTHONPATH="$prior_source/src" FLOODWARP_WRITE_BASELINE=1 .venv/bin/python validation/hancock_default_restart.py
.venv/bin/python validation/hancock_default_restart.py
```

This is development-branch qualification. The larger-grid measurements cover two synthetic hydraulic workloads on one GPU. Representative event-scale cost, memory and accuracy on native catchment terrain remain open for a separate operational review before public release.
