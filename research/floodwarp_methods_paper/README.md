# Reproducing the FloodWarp numerical methods study

The manually executed master script runs from the repository root. The protocol fixes the experiments, physical controls, grouped splits and tolerance rules. The hydraulic development default is adaptive Hancock. The source and experiment identifiers in the receipts select the implementation actually measured. [Study protocol](PROTOCOL.md) · [Methods](METHODS.md) · [Results](RESULTS.md) · [Discussion](DISCUSSION.md) · [Unqualified investigations](LIMITATIONS.md)

```bash
./scripts/run_floodwarp_paper.sh
./scripts/run_floodwarp_paper.sh check
./scripts/run_floodwarp_paper.sh all --dry-run
./scripts/run_floodwarp_paper.sh all
```

Use Python 3.14, a compatible NVIDIA CUDA driver and a machine with enough memory for the selected grids. The recorded reference GPU has 16 GiB. On a new checkout `check` creates the local environment and runs the repository tests. The bootstrap pins the hydraulic stack through the package metadata. The version inventory in the supplement records the experiment environment. A different GPU or Warp version generates distinct timing identities. Model runtime predictions are hardware specific. [Package dependencies](../../pyproject.toml) · [Environment inventory](supplementary/environment.json)

| Stage | Experiment or output |
|---|---|
| check | Environment, CUDA and full repository tests |
| hydrograph-fidelity | Signed, reference-normalised magnitudes, discharge passage, hydrograph spreading and cost through the measured recession. [Experiments and definitions](fidelity/README.md) |
| fidelity-figures | Rebuild hydrograph plots from completed records. |
| analytical | Nineteen primary exact domains plus waves, controls, lakes, jumps and equilibria |
| convergence | Matched geometric refinements, fixed-grid temporal and manufactured convergence |
| fixed-dt-space | Smooth exact gravity and shear at two common fixed timesteps across four spacings, included by convergence |
| synthetic | Generate and independently repeat 24 seeded landforms at three grid spacings |
| long-storms | Matched production forecasts and two reference refinements per storm |
| resolution | Fixed and resampled terrain on matched domains |
| terrain-initial-controls | Added paired fan runs with matched initial channel water at both spacings, included by resolution |
| calibration | Physical roughness controls, bounded training and held-out transfer |
| performance | Repeated operational forecasts and common-cap GPU controls |
| million-terrain | Generate full-grid descriptors and georeferenced native and represented control windows |
| million-pilots | Check allocations, timestep behaviour and complete physical-input identities on the reference GPU |
| million-storms | Full six-hour rainfall and recession forecasts at fixed grid counts and fixed physical extent |
| million-auto | Measure the frozen preflight policy on the same complete events |
| large-references | Same-domain refinements and fast hydrograph controls |
| large-refinements | Additional reference grids at 1 m or coarser and matched wet initial-state mappings |
| heldout-references | Fine-grid hydraulic sensitivity of the reserved plain and mixed catchments |
| million-profiles | Separate complete graph-instrumented replays of sparse, wet and mixed events, with a fresh process for each GPU pool high-water mark |
| million-repeats | Repeat sparse plain and wet floodplain controls twice on the same GPU, in addition to each original timing |
| recession-controls | Follow the dry plain for 48 hours and test widespread downstream rainfall on the same DEM for 24 hours |
| execution-policy | Reconstruct measured dense, tiled and prospective automatic execution comparisons |
| pareto | Reconstruct observable-specific measured frontiers and reference eligibility |
| terrain-features | Read the simulation archives, derive hydraulic errors, compare grids, map terrain and compute Pareto sets |
| advisor-train | Train on development groups, select model parameters on validation groups and freeze |
| advisor-validate | Assess the frozen advisor and three rules on whole held-out groups |
| tables | Generate Markdown, CSV and LaTeX from archived metrics |
| figures | Render PDF, SVG and 350 dpi PNG from archived data |
| manuscript | Populate results and discussion from processed records |
| verify | Check raw checksums, matched forcing, storm durations, accepted-step histories, scheduler equivalence, SQL and publication outputs |

### Run individual stages

```bash
./scripts/run_floodwarp_paper.sh analytical --case front_steep --method hancock_adaptive
./scripts/run_floodwarp_paper.sh long-storms --case 'plain-18217-*' --dry-run
./scripts/run_floodwarp_paper.sh long-storms
./scripts/run_floodwarp_paper.sh resolution
./scripts/run_floodwarp_paper.sh calibration
./scripts/run_floodwarp_paper.sh performance
./scripts/run_floodwarp_paper.sh million-terrain
./scripts/run_floodwarp_paper.sh million-pilots
./scripts/run_floodwarp_paper.sh million-storms --case 'plain-701-*' --method hancock_dense --dx 10
./scripts/run_floodwarp_paper.sh million-storms
./scripts/run_floodwarp_paper.sh million-auto
./scripts/run_floodwarp_paper.sh large-references
./scripts/run_floodwarp_paper.sh large-refinements
./scripts/run_floodwarp_paper.sh heldout-references
./scripts/run_floodwarp_paper.sh million-profiles
./scripts/run_floodwarp_paper.sh million-repeats
./scripts/run_floodwarp_paper.sh recession-controls
./scripts/run_floodwarp_paper.sh execution-policy
./scripts/run_floodwarp_paper.sh pareto
./scripts/run_floodwarp_paper.sh terrain-features
./scripts/run_floodwarp_paper.sh advisor-train
./scripts/run_floodwarp_paper.sh advisor-validate
./scripts/run_floodwarp_paper.sh figures
./scripts/run_floodwarp_paper.sh tables
./scripts/run_floodwarp_paper.sh manuscript
./scripts/run_floodwarp_paper.sh verify
```

Use `--dx` and `--dem` to select a hydraulic spacing and represented DEM spacing. Numerical commands take an advisory lock for the selected device in this checkout. Pass `--force` to create a fresh attempt. Each attempt retains its fields, log, initial metadata, timings, status and checksums. An interrupted run retains its partial directory. A new attempt supersedes the current manifest pointer. Resume by issuing the same stage command or by passing `--resume` explicitly. Configuration and measured source identities must match for a numerical result to be reused. The publication loader requires a common measured implementation, configuration and timing environment. If several GPUs share an archive, `FLOODWARP_PAPER_RESULT_GPU` selects the recorded GPU string. The attempt catalogue retains timing exclusions and superseded trials. Set `FLOODWARP_PAPER_CONFIG` to an alternative study index, `FLOODWARP_PAPER_DEVICE` to a Warp CUDA device, and `FLOODWARP_PAPER_PYTHON` to override the Python executable. Figures and tables run from archives, including on a machine with the analysis dependencies and no CUDA device.

### Trace an observation

Start with the physical case, representation and observable in `processed/pareto/storms.csv`. Its experiment ID identifies a row in the SQLite `experiments` table with a numerical specification, source SHA, device and raw directory. Within the raw directory, `result.json` contains the accepted-step totals, hydrographs, forcing fingerprints, signed volumes and timings. The compressed NumPy files hold physical fields and the per-step history. A parallel `manifests` entry and terminal receipt record artifact digests. The current manifest selects the archived attempt for reproduction. The earlier large-grid timings and bounded analytical fits retain their original provenance. The corrective full-event forecasts have separate receipts under the million-cell and reference stages. They store accepted counts and timestep extrema per output interval, compressed common-support fields, native state hashes and interval face transports. [Results schema](manifests/SCHEMA.md) · [Archival qualification](../../validation/hancock_long_storm_review/DECISION.md)

### Apply the advisor

```bash
.venv/bin/python -m floodwarp.solver_advisor \
  --dem research/floodwarp_methods_paper/terrain/surfaces/plain/elevation.tif \
  --resolution 10 --objective discharge --tolerance 0.01 \
  --compare-resolutions 5 --output outputs/advisor
```

The Python API is `floodwarp.solver_advisor.recommend_dem(dem=path, resolution=10, objective="discharge", output=folder)`. Use `floodwarp.terrain_advisor.analyse_dem(dem=path, resolution=10, output=folder)` for diagnostics alone. For a preprocessed grid, `analyse_array(elevation=z, native_resolution=1, resolution=10)` returns summary statistics plus native and model-grid maps. The CLI writes GeoTIFFs, a map, a human-readable assessment and JSON evidence for supported candidates. Optional `--hydraulic-json` accepts `rain_mm`, `peak_mm_h`, `rain_duration_h`, `forecast_duration_h`, `rain_centroid_h`, `spatial_mean_multiplier`, `manning_n`, `initial_mean_m`, `infiltration_mean_mm_h` and `downstream_stage_m`. An externally specified `expected_flow_regime` can restrict neighbours to subcritical, mixed or supercritical observations. The held-out evaluation uses pre-run inputs without access to the event’s realised flow regime. Specify `--memory-mib` and `--hardware` to constrain timing and memory. Compare candidate grids by supplying `--compare-resolutions`. The result can request a finer DEM or return insufficient evidence. The tool performs inference without starting a hydraulic simulation.

The six standard hydraulic objectives are final depth, maximum depth, inundation, discharge, peak discharge and peak timing. Their CLI keys and default tolerances are in the advisor configuration. Choosing `runtime` selects by measured operational cost subject to supported hardware and optional memory constraints. [Tolerance definitions](configs/advisor/design.json) · [Held-out performance](tables/10_advisor.md)

The exact measured numerical sources are available together at `894a375762c9cbcbf24a4b612532ca2628ea7a01`. The completed paper commit adds all archived data and regeneration tools. The manifest verifies the numerical source fingerprints against Git blob contents. [Numerical tree proof](manifests/numerical_source_tree.json)


### Operational qualification

The 120 × 80 m controls contain 96 cells at 10 m. Use the corrective observations to assess operational performance. The design identifies eight 1,024 × 1,024 landforms, three 2,048 × 1,024 events, and matched 1,000 × 1,000 m and 10,240 × 10,240 m physical-domain studies. Cells, spacing, storm duration and initial state are explicit for every forecast. [Event design](qualification/design.json) · [Corrective audit](CORRECTIVE_AUDIT.md)

For an experimental automatic schedule, construct the Hancock model and set its initial state, rainfall multiplier and boundary configuration first. Call `configure_hancock_execution(solver=model.solver, execution_mode="auto")` from `floodwarp.solver_advisor.execution` before the first advance. Keep its returned preflight record with the configuration. The decision is made once. Explicit dense, fixed-mask sparse and tiled execution remain available. [Calibration data and thresholds](../../src/floodwarp/solver_advisor/execution_policy.json)

Use `processed/million/reference_checks.csv` to identify resolved observables. The strict empirical Pareto column describes dominance of the measured points. The sensitivity column requires the competitor's upper error to be below the candidate's lower error, or an exact endpoint tie with a lower runtime. Each interval uses the measured reference changes. These envelopes describe sensitivity to the tested references. [Measured configurations](processed/million/pareto.csv)

The ten corrective numerical stages have a separate committed-tree proof. Verification extracts the recorded Git revision, recomputes each numerical source fingerprint and checks it against the selected simulation receipts. [Corrective numerical source proof](manifests/corrective_source_tree.json)

The corrective storm study uses 1 m as its finest hydraulic resolution, matching the available terrain data. The two planned 0.5 m forecasts were withdrawn. The interrupted attempt has no hydraulic result. Matched 1 m Hancock and FV2 runs can inform comparisons at coarser resolutions. The 1 m configurations themselves have no finer independent reference in this study. [Recorded scope amendment](qualification/resolution_scope.json)

### Boundary repair and paired-grid advisor qualification

Run `./scripts/run_floodwarp_paper.sh outlet-qualification --dry-run` to inspect the ordered suite. `outlet-storms` starts 14 matched trajectories on the incised catchment, with gauges fixed at 5,120 and 10,240 m. The 400 m extension continues the native metre terrain, channel and soil beyond the eastern gauge. It receives zero rainfall. `outlet-results` extracts interval peaks, water balances and wet-union depth disagreements from completed trajectories.

On a fresh checkout, `native-input-audit` reconstructs the original bed, water depth, momentum, rain footprint and soil capacity from the 1 m source, checks the two levels of area averaging and verifies the file hashes. `outlet-states --dry-run` counts available and missing checkpoint windows. `outlet-states --resume` verifies local fields and replays missing windows from the physical start on the reference hardware. A recovered checkpoint must match every archived field hash and the primary interval water measurements before it is installed. The full qualification calls these stages. Tables, figures and manuscript generation read the archived products.

Run `./scripts/run_floodwarp_paper.sh outlet-qualification --resume` on the reference GPU to continue the complete qualification. Use `native-status` to inspect completed storm windows, failed attempts and matched recession receipts. The workflow freezes the advisor after fit and validation, checks the installed CLI with that model, runs the held-out terrains and then executes the external 48-hour Orton Bradley transfer. The transfer uses historical HIRDS forcing. Observational prediction error remains unmeasured.

`native-storms --dry-run` lists the selected physical cases, grid dimensions, formulations and CFL controls. It also lists the primary run IDs from the current solver build across the wider archive, their common recession endpoints when available, and the number of older-build windows retained. Resume validates the full input and hardware contract before accepting an archived run.

The [24 November 2026 qualification inventory](fidelity/verification/qualification_inventory_20261124.txt) records the manual driver's dry run with 15 initial windows available. It includes the incised checkpoints, grid dimensions, reference controls, training and test partitions, and nine planned 48-hour real-terrain transfers. Run the same inventory command above for the current completion counts.

`./scripts/run_floodwarp_paper.sh post-repair-exact` verifies and joins the archived analytical controls from the audited and repaired solvers. It compares exact references, sampled model fields, gauge series and ordered accepted timesteps. The full qualification also builds this comparison after numerical convergence. [Paired numerical results](fidelity/analytical_revision/comparisons.csv)

`advisor-design` checks paired reference and terrain coverage in development, validation and final test. The empirical model requires two independent development neighbours. The additional hill development terrain uses seed 8500, the first seed checked from 8500 upward whose opening and mapped channel path survived area averaging at both 5 and 10 m. Selection used terrain alone, before hydraulic simulation. The measured runtime of a terrain with a lost control remains available. Its hydraulic error is excluded from the advisor's training labels for affected objectives.

`orton-transfer --dry-run` lists source-specific run IDs, dimensions and completed checkpoint windows. For example, `./scripts/run_floodwarp_paper.sh orton-transfer --method hancock --dx 5 --resume` continues just the native Hancock transfer after the synthetic qualification.

The transfer records the two gauge northings in the native projected CRS. Each launch retains its hardware, elapsed time and completion or failure receipt. After each finished grid, `fidelity/orton_transfer/historical_transfer.csv` compares the measured outlet series with the first 48 hours of the 5 m release FV1, LISFLOOD-FP FV1 and local-inertia controls. The comparison verifies every minute against the recorded cumulative net export and reports peak, timing, volume and hydrograph differences. Comparisons at 10 and 20 m include changes in represented terrain and outlet geometry. All three historical controls are numerical forecasts under design forcing.

`native-timing-repeats` repeats alternatives within 5% of the cheapest measured execution policy or cheapest tolerance-feasible configuration, twice more at the primary endpoint. The 5% band decides which GPU costs to remeasure. It has no hydraulic interpretation. The selection is recorded before repeating trajectories. `native-products` rebuilds the timing distribution, including all execution IDs, before applying the median to cost comparisons. Repeats share the same terrain and storm in statistical analyses.

To inventory only the independent dense Hancock timings selected for the 10 m plain, run:

```bash
./scripts/run_floodwarp_paper.sh native-timing-repeats --case 'plain-709-*' --method hancock_dense --dx 10 --dry-run
```

Omit `--dry-run` to measure those configurations. A later full invocation reuses each completed timing at the same physical endpoint. Overlapping selections contribute one observation per execution identity.


`native-step-history` replays dense, tiled and automatic Hancock on the matched 1 km² spatial-storm plain at 1 and 10 m after the training-partition storms. Each replay must reproduce the measured transport at both gauges every reporting interval and the primary final field hashes. The ordered FP64 timestep arrays and their hashes are saved separately from uninstrumented runtime measurements. Original outputs that lack an ordered history keep their unrecorded status.

`advisor-design` checks the planned and measured support by physical terrain realisation. `advisor-cli-check` invokes the installed Python package on actual GeoTIFFs. Pass `--control-json` to the advisor with `window_m` set to west, south, width and height in the analysed metric CRS, `axis` set to `west_east` or `south_north`, and a nonnegative `spill_head_m` above the native spill. Bounds must align with the native and requested grids. A missing control window leaves opening connectivity unassessed.

The qualification driver fits on development and validation terrains, freezes the model, then runs the independent test partition. The final `orton-transfer` stage compares the established full-domain 48-hour Orton Bradley HIRDS design event on 20, 10 and 5 m grids. Design-event transfer has no observed discharge reference. Historical experiment receipts retain their recorded implementation identities. Tables and figures read these archives independently of the hydraulic stages.

The frozen grid selector records the physical areas at which two independent validation terrains supported the same objective and spacing pair. A proposed grid change must fall within the range of those validated areas. The design inventory records the cell counts at both candidate spacings and at the finer reference spacing. A million-cell reference and a million-cell candidate support different operational claims.
