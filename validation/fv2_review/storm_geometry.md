## Separating represented terrain from hydraulic spacing in the coastal-tile storm

Each row uses the same solver throughout. Combined compares the coarse run to its 1 m terrain/1 m hydraulic counterpart. Grid refines the hydraulic spacing to 1 m with every coarse input held fixed. Terrain compares the two terrain/forcing representations at identical 1 m hydraulic spacing. Both groups include the mask and coastal openings belonging to the represented domain. Depth uses the original active 1 m footprint. The stored signed interaction makes the squared-error decomposition exact. The 1 m runs are model comparators with finite truncation error.

| Terrain (m) | Method | End depth: combined (mm) | Grid (mm) | Terrain (mm) | Max depth: combined (mm) | Grid (mm) | Terrain (mm) | Hydrograph: combined (L/s) | Grid (L/s) | Terrain (L/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | fv1 | 72.804 | 1.3608 | 72.92 | 73.392 | 2.899 | 73.384 | 4.4241 | 1.2079 | 3.7527 |
| 10 | fv1 | 99.042 | 4.1675 | 98.966 | 101.79 | 5.7213 | 101.57 | 4.5115 | 1.9853 | 3.6713 |
| 5 | mc | 72.661 | 4.0388 | 72.883 | 73.041 | 5.928 | 73.234 | 4.0118 | 1.0424 | 3.6678 |
| 10 | mc | 97.898 | 11.844 | 98.779 | 100.5 | 13.088 | 101.12 | 4.2752 | 2.1296 | 3.3661 |
