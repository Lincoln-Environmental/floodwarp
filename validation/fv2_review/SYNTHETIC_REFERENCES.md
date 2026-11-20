## Fine numerical references for the 5 m synthetic models

All entries are final depth RMSE in millimetres. Grid gaps compare 0.625 and 0.3125 m results on the 0.625 m grid. The inter-method and timestep comparisons use 0.3125 m support. The last four columns conservatively average the fine reference to the 5 m candidate grid. Resampled terrain evaluates the same mathematical surface at cell centres. Fixed terrain preserves every 5 m cell's input, including rain and the initial condition, under subdivision. The gaps measure sensitivity and have no formal a posteriori error-bound interpretation.

| Case | Bed | MC grid gap | Roe/HLL grid gap | Fine inter-method gap | MC half-dt change | FV1 vs MC | FV2 vs MC | FV1 vs Roe/HLL | FV2 vs Roe/HLL |
|---|---|---|---|---|---|---|---|---|---|
| plane | Resampled | 0.45175 | 0.79947 | 1.6599 | 0.0016805 | 5.5413 | 3.5229 | 4.5065 | 2.4243 |
| plane | Fixed 5 m | 0.71997 | 0.88255 | 1.7175 | 0.0034501 | 5.683 | 3.5948 | 4.6155 | 2.3626 |
| irregular | Resampled | 0.22634 | 0.7647 | 1.044 | 0.00070704 | 6.6183 | 3.9569 | 6.0619 | 3.6878 |
| irregular | Fixed 5 m | 0.99154 | 1.0275 | 1.5738 | 0.0018837 | 5.2613 | 5.7508 | 4.4171 | 5.8504 |
| steep | Resampled | 0.57064 | 1.1888 | 1.8578 | 0.0051005 | 11.066 | 5.5416 | 9.7105 | 4.5578 |
| steep | Fixed 5 m | 3.7572 | 3.5193 | 5.0563 | 0.034399 | 10.082 | 6.1103 | 8.7105 | 5.1899 |
| expansion | Resampled | 0.4139 | 0.85757 | 1.2233 | 0.0011484 | 7.9747 | 6.3877 | 7.5131 | 6.1425 |
| expansion | Fixed 5 m | 0.59949 | 0.84304 | 1.4848 | 0.0015506 | 5.3548 | 1.7474 | 4.8166 | 1.1874 |
| rainfall | Resampled | 0.035643 | 0.34602 | 0.41435 | 1.4655e-05 | 2.2071 | 1.0186 | 1.9051 | 0.90213 |
| rainfall | Fixed 5 m | 0.34308 | 0.50081 | 0.49773 | 6.7716e-05 | 2.8909 | 3.148 | 2.6155 | 2.9126 |
| barrier | Resampled | 11.971 | 11.983 | 1.2674 | 0.00043236 | 13.292 | 12.753 | 13.066 | 12.589 |
| barrier | Fixed 5 m | 0.92023 | 0.7127 | 1.4031 | 0.00049038 | 3.9319 | 2.4439 | 3.2303 | 1.6723 |
| oblique_channel | Resampled | 0.17744 | 0.25237 | 1.4732 | 6.1159e-05 | 6.7796 | 6.3813 | 6.4778 | 6.5203 |
| oblique_channel | Fixed 5 m | 0.96917 | 0.6579 | 1.6265 | 0.0011223 | 3.8239 | 2.761 | 3.5874 | 3.0426 |
