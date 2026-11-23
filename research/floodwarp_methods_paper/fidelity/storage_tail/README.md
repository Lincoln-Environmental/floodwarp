# Residual storage after 48 hours

The 10 m mixed catchment received six hours of spatial rainfall. Both million-cell models first passed the two-hour transport check at 20 hours. The extended integrations below also passed the 12-hour check at the internal section and outlet.

| Method | Hydraulic time (s) | Accepted steps | Export since first check (m³) | Final outward volume (m³) | Last 12 h share | Infiltration (m³) | Remaining storage (m³) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FV1 sparse | 106.19 | 184,106 | 2,225.7 | 928,239.4 | 0.052% | 322,058.7 | 22,316.3 |
| Hancock tiled | 116.19 | 259,216 | 10,490.0 | 801,291.4 | 0.303% | 346,472.9 | 124,850.1 |

Tiled Hancock took 1.094× FV1 at the longer endpoint. The two forecasts differ in both infiltration and retained surface storage. A difference in total export alone cannot establish numerical diffusion.

[Complete rain, discharge, wet fraction and cost curves](figures/mixed-708-spatial-10240x10240_cost.pdf) · [Water accounts and timings](runtime_and_storage.csv) · [Two-station tail test](late_export.csv)
