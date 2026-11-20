## Sensitivity to FV formulation and grid spacing in the full Orton storm

Peak change is second minus first. Discharge RMSE uses all 2,880 one-minute intervals. Spatial RMSE uses the original active 5 m footprint and piecewise-constant transfer of coarser predictions. Fixed-grid comparisons share all represented inputs. Changing grid also changes represented terrain, mask, rain distribution and coastal openings. Each value measures disagreement between simulations.

| First run | Second run | Peak change (m³/s) | Hydrograph RMSE (m³/s) | End depth RMSE (mm) | Maximum depth RMSE (mm) |
|---|---|---|---|---|---|
| 5 m contact_hll | 5 m roe | -0.25648 | 0.098771 | 0.50494 | 3.1304 |
| 5 m contact_hll | 5 m hllc | 0.043423 | 0.022913 | 0.26936 | 0.62944 |
| 5 m roe | 5 m hllc | 0.2999 | 0.11926 | 0.56881 | 3.4966 |
| 10 m contact_hll | 10 m roe | -0.38471 | 0.16224 | 0.55259 | 3.5684 |
| 10 m contact_hll | 10 m hllc | 0.12948 | 0.041114 | 0.18475 | 0.8253 |
| 10 m roe | 10 m hllc | 0.5142 | 0.19884 | 0.70108 | 4.176 |
| 20 m contact_hll | 20 m roe | -0.649 | 0.19621 | 0.36269 | 4.0727 |
| 20 m contact_hll | 20 m hllc | 0.21355 | 0.065379 | 0.12481 | 1.2248 |
| 20 m roe | 20 m hllc | 0.86255 | 0.25431 | 0.48568 | 5.0748 |
| 5 m contact_hll | 10 m contact_hll | -10.535 | 1.3372 | 30.502 | 64.637 |
| 5 m roe | 10 m roe | -10.663 | 1.3996 | 30.583 | 64.999 |
| 5 m hllc | 10 m hllc | -10.449 | 1.3228 | 30.488 | 64.585 |
| 5 m contact_hll | 20 m contact_hll | -21.442 | 2.8241 | 65.017 | 114.57 |
| 5 m roe | 20 m roe | -21.835 | 2.9261 | 65.125 | 114.87 |
| 5 m hllc | 20 m hllc | -21.272 | 2.7849 | 64.994 | 114.52 |
| 10 m contact_hll | 20 m contact_hll | -10.908 | 1.5583 | 60.514 | 95.652 |
| 10 m roe | 20 m roe | -11.172 | 1.5996 | 60.615 | 95.8 |
| 10 m hllc | 20 m hllc | -10.824 | 1.5345 | 60.484 | 95.599 |
