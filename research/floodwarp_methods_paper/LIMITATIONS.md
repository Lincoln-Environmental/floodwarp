# Unresolved and unqualified investigations

A refinement screen applies to a physical event, terrain representation and hydraulic observable. The complete per-event record retains both grid identities and the magnitude of the change.

| Terrain representation | Observable | Qualified | Unresolved |
|---|---|---:|---:|
| sampled | depth | 25 | 23 |
| sampled | maximum_depth | 41 | 7 |
| sampled | inundation | 14 | 34 |
| sampled | discharge | 36 | 12 |
| sampled | peak_discharge | 32 | 16 |
| sampled | peak_timing | 47 | 1 |
| fixed | depth | 1 | 2 |
| fixed | maximum_depth | 3 | 0 |
| fixed | inundation | 1 | 2 |
| fixed | discharge | 2 | 1 |
| fixed | peak_discharge | 2 | 1 |
| fixed | peak_timing | 3 | 0 |

- Reference error on general catchments is estimated from numerical refinements. Independently exact storm-cycle evidence is limited to analytical geometries and forcing. Two formulations can share a topographic or outlet error.
- Fixed and resampled terrain at 1 m and 5 m were compared on three development realisations. The remaining families lack a prospective matched 1 m operational matrix. The larger scaling runs quantify cost on the original qualification geometries.
- Analytical timestep and spatial convergence rates depend on smoothness, source splitting, limiter activity and reference precision. Values near roundoff have limited order interpretation.
- The 15-minute outlet interval bounds timing resolution and smooths subinterval peaks. Area averages to 10 m can obscure very narrow wet connections.
- Roughness transfer uses one prospective fitted terrain and event. The original bounded analytical study covers additional training physics. Repeated objectives, bounds and timings share terrain and forcing.
- The advisor is supported only within its measured grid, area, terrain and hardware range. Empirical prediction intervals can under-cover. Baseline rules and abstentions remain part of the reported assessment.
- Native FREE outlet closures differ across formulations. Quasi-steady structures have separate split-operator CPU and GPU qualifications. Surveyed structure calibration and additional hydraulic controls require independent work.
- No real-catchment qualification or multi-GPU architecture comparison was included. Public-release qualification remains open.

[Reference changes and case IDs](processed/metrics/reference_uncertainty.csv) · [Held-out decisions](processed/advisor/heldout.csv) · [Structure qualifications](../../docs/HANCOCK_STRUCTURES.md)
