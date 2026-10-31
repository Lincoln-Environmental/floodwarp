# Completed GPU measurements

Status: partial. Pending cases: purau_event_72h.

Median hydraulic runtime uses three short-case repetitions and one 72-hour event. Ratios use external LISFLOOD-FP FV1 GPU.

| Case | Solver | Hydraulic s | End-to-end s | Ratio | Steps | dt mean s | Peak allocated MiB | Persistent arrays MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| boundary_outflow | reference | 0.0160113 | 0.242012 | 1 | 60 | 1 | 0.078 | 0.078 |
| boundary_outflow | li | 0.00819748 | 0.130007 | 0.512 | 60 | 1 | 0.159 | 0.078 |
| boundary_outflow | fv1 | 0.00934572 | 0.0952227 | 0.5837 | 120 | 0.5 | 0.066 | 0.066 |
| dam_break | reference | 0.0136728 | 0.242074 | 1 | 60 | 1 | 0.132 | 0.132 |
| dam_break | li | 0.00822949 | 0.132237 | 0.6019 | 60 | 1 | 0.203 | 0.100 |
| dam_break | fv1 | 0.00975864 | 0.0960047 | 0.7137 | 120 | 0.5 | 0.085 | 0.085 |
| lake | reference | 0.0159813 | 0.242379 | 1 | 60 | 1 | 0.103 | 0.103 |
| lake | li | 0.00815526 | 0.129642 | 0.5103 | 60 | 1 | 0.199 | 0.098 |
| lake | fv1 | 0.0101821 | 0.0972309 | 0.6371 | 120 | 0.5 | 0.083 | 0.083 |
| purau_coast_crop | reference | 0.483068 | 0.744036 | 1 | 3600 | 1 | 0.373 | 0.373 |
| purau_coast_crop | li | 0.156678 | 0.293014 | 0.3243 | 3600 | 1 | 0.974 | 0.481 |
| purau_coast_crop | fv1 | 0.126838 | 0.219011 | 0.2626 | 3600 | 1 | 0.411 | 0.411 |
| purau_coast_crop_cfl02 | reference | 32.3383 | 41.121 | 1 | 188052 | 0.689171 | 0.373 | 0.373 |
| purau_coast_crop_cfl02 | li | 9.45669 | 13.3767 | 0.2924 | 173947 | 0.745055 | 0.974 | 0.481 |
| purau_coast_crop_cfl02 | fv1 | 19.2552 | 21.7581 | 0.5954 | 308905 | 0.419546 | 0.411 | 0.411 |
| purau_coast_crop_grid_control | reference | 21.5581 | 30.1215 | 1 | 131479 | 0.985709 | 0.373 | 0.373 |
| purau_coast_crop_grid_control | li | 6.76162 | 10.6705 | 0.3136 | 129626 | 0.999799 | 0.974 | 0.481 |
| purau_coast_crop_grid_control | fv1 | 10.881 | 13.3766 | 0.5047 | 188098 | 0.689003 | 0.411 | 0.411 |
| purau_coast_crop_refined | reference | 33.4908 | 42.4831 | 1 | 188552 | 0.687344 | 1.379 | 1.379 |
| purau_coast_crop_refined | li | 13.2058 | 17.3579 | 0.3943 | 187513 | 0.691152 | 3.876 | 1.914 |
| purau_coast_crop_refined | fv1 | 18.7539 | 21.5461 | 0.56 | 309391 | 0.418887 | 1.606 | 1.605 |
| purau_event | reference | 4.24057 | 5.88828 | 1 | 3600 | 1 | 182.173 | 182.173 |
| purau_event | li | 4.25102 | 6.09029 | 1.002 | 3600 | 1 | 614.152 | 303.465 |
| purau_event | fv1 | 3.00315 | 4.65022 | 0.7082 | 3600 | 1 | 234.859 | 234.858 |
| rain_infiltration | reference | 0.0137572 | 0.242218 | 1 | 60 | 1 | 0.078 | 0.078 |
| rain_infiltration | li | 0.00685813 | 0.128802 | 0.4985 | 60 | 1 | 0.159 | 0.078 |
| rain_infiltration | fv1 | 0.00329258 | 0.0933919 | 0.2393 | 60 | 1 | 0.066 | 0.066 |

Dry first-hour Purau fields provide timing evidence without meaningful wet or hydrograph coverage.
CUDA allocated high-watermark and native explicit-allocation counters exclude context/reserved memory. Persistent capacity is separate.
Warm compiled caches were used; compilation fields include module loading. External build provenance is recorded separately.

End-to-end table values use fresh-directory birth through final record write, including Python checksum/manifest work missed by component totals. Shared input loading and final directory fsync are outside that interval. Native end-to-end includes its hourly process restarts; hydraulic timing excludes initialization.
