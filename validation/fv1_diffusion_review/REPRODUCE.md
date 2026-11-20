# Reproduction and evidence status

Use FloodWarp's pinned Python 3.14 environment with Warp 1.17.0, FP64 and CUDA. The measurement GPU was an NVIDIA GeForce RTX 4070 Ti SUPER with 16 GiB and driver 13.2. Source base and reference checkout were dev `7d9a6fe`. Run from the repository root. The study writes beneath `validation/fv1_diffusion_review`. A separate worktree preserves checked-in observations when repeating timings. [Experiment manifest](domains.json)

```sh
.venv/bin/python -m validation.fv1_diffusion_truth
.venv/bin/python -m validation.fv1_diffusion_time
.venv/bin/python -m validation.fv1_diffusion_controls
.venv/bin/python -m validation.fv1_diffusion_profile
.venv/bin/python -m validation.fv1_diffusion_matrix
.venv/bin/python -m validation.fv1_diffusion_history
.venv/bin/python -m validation.fv1_diffusion_fallback
.venv/bin/python -m validation.fv1_diffusion_calibrate
.venv/bin/python -m validation.fv1_diffusion_zones
.venv/bin/python -m validation.fv1_calibration_evidence
.venv/bin/python -m validation.fv1_diffusion_walls
.venv/bin/python -m validation.fv1_diffusion_figures
.venv/bin/python -m validation.fv1_diffusion_tables
.venv/bin/python -m validation.fv1_diffusion_archive
```

The primary matrix checkpointed one file per domain. Four additional physical controls were timed after the original fifteen, using the same settings and hardware. The final matrix driver includes all nineteen. A replay of the three slower moving-front cases tightened spatial error support to stay beyond reflected signals. Replays checked accepted section errors and step counts before updating spatial metrics. The current driver applies that support directly. `fv1_diffusion_support` reproduces the optional audit. For the expanded-domain sensitivity study, keep the observation support, initial wave and gauge coordinates fixed. The raw matrix stores the complete numerical and analytical hydrographs, checks, thresholds, minima, budgets and three timing replicates for each setup. The CSV contains the primary comparison columns.

The performance experiment uses one untimed warm trajectory and seven timed repeats per isolated profile. A hash comparison checks fused and unfused final fields. The Nsight probe uses one uncaptured step on the same 98,304-cell Stoker input for both counter captures. The profiler can change GPU clock settings while sampling. Run it separately from hydraulic timings.

```sh
FV_PROFILE_METHOD=hancock_mc FV_PROFILE_FUSE=1 ncu --csv --kernel-name 'regex:(faces_|slopes_predict_|split_sources_|euler_)' --launch-count 5 --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second .venv/bin/python -m validation.fv2_counter_probe
FV_PROFILE_METHOD=hancock_mc FV_PROFILE_FUSE=0 ncu --csv --kernel-name 'regex:(slopes_|predict_)' --launch-count 3 --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second .venv/bin/python -m validation.fv2_counter_probe
```

## Regression, parity and qualification

```sh
git worktree add --detach /tmp/floodwarp-diffusion-baseline 7d9a6fe
PYTHONPATH=/tmp/floodwarp-diffusion-baseline/src FV_PARITY_OUTPUT=/tmp/floodwarp-base.json .venv/bin/python -m validation.fv1_diffusion_parity
FV_PARITY_OUTPUT=/tmp/floodwarp-final.json .venv/bin/python -m validation.fv1_diffusion_parity
cmp /tmp/floodwarp-base.json /tmp/floodwarp-final.json
OVERLANDER_TEST_FULL_SWE_CUDA=1 FLOODWARP_TEST_RENDER=1 FLOODWARP_REFERENCE_BINARY="$PWD/outputs/qualified_reference/build/lisflood" .venv/bin/python -m pytest -q -rs
```

The suite passed **1,843 tests**, with **24 skipped** and 743 warnings. The skip record identifies optional mixed precision, external validation and device-mode combinations. The CPU/Warp tests cover heterogeneous wet/dry cells, arbitrary bed, exponential and constant infiltration, rainfall, both flux variants, limiters, checkpoint identity, reflective boundaries, large elevation datums and accepted flux accounting. The paired clean-checkout run compared steps and SHA-256 hashes of all conservative state components, maxima, soil and both accepted-volume arrays for FV1 and FV2, each dense and sparse. All matched. [Base fingerprints](parity_base.json) · [Final fingerprints](parity_final.json) · [Regression record](regression.json)

| Status | Evidence |
|---|---|
| Completed | Coupled smooth convergence, fixed-grid timestep isolation, positive-friction source balances and accepted volumes, wet/dry fronts and terrain controls, 2D exact basin, low-Froude gravity pulse, shear, LI reduced-equation controls, 1/5/10 m matrix, common-cap and own-CFL timing, fusion and sparse tests, roughness calibration and held-out transfer. |
| Failed universal qualification | MC Hancock falls short of general second-order fixed-grid time refinement on the limited manufactured solution. Unrestricted MC rings at a fine stationary jump. The universal 1.5× runtime target and universal roughness compensation fail. The reported failed rows are retained. |
| Inconclusive | Millimetre-level depth rankings that reverse on the enlarged damped-shear domain. Small runtime gaps on tiny grids may depend on output cadence and hardware. The exact pulse and grade matrix cannot identify a universal slope or Froude threshold. |
| Outside this study | Catchment reruns, observations, calibration on field data, terrain coarsening, new infiltration tuning and hydraulically coupled calibration zones. The earlier terrain and catchment evidence remains separately traceable. |
