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
| large-refinements | Additional reference grids and matched wet initial-state mappings |
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

Use `--dx` and `--dem` to select a hydraulic spacing and represented DEM spacing. Numerical commands take an advisory lock for the selected device in this checkout. Pass `--force` to create a fresh attempt. Each attempt retains its fields, log, initial metadata, timings, status and checksums. An interrupted run retains its partial directory. A new attempt supersedes the current manifest pointer. Resume by issuing the same stage command. Configuration and measured source identities must match for a numerical result to be reused. The publication loader requires a common measured implementation, configuration and timing environment. If several GPUs share an archive, `FLOODWARP_PAPER_RESULT_GPU` selects the recorded GPU string. The attempt catalogue retains timing exclusions and superseded trials. Set `FLOODWARP_PAPER_CONFIG` to an alternative study index, `FLOODWARP_PAPER_DEVICE` to a Warp CUDA device, and `FLOODWARP_PAPER_PYTHON` to override the Python executable. Figures and tables run from archives, including on a machine with the analysis dependencies and no CUDA device.

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
