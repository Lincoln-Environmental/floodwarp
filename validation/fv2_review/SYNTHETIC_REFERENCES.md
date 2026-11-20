## Fine numerical references for the 5 m synthetic models

All entries are final depth RMSE in millimetres. Grid gaps compare 0.625 and 0.3125 m results on the 0.625 m grid. The inter-method and timestep comparisons use 0.3125 m support. The last four columns conservatively average the fine reference to the 5 m candidate grid. Resampled terrain evaluates the same mathematical surface at cell centres. Fixed terrain preserves every 5 m cell's input, including rain and the initial condition, under subdivision. The gaps measure sensitivity and have no formal a posteriori error-bound interpretation.

| Case | Bed | MC grid gap | Roe/HLL grid gap | Fine inter-method gap | MC half-dt change | FV1 vs MC | FV2 vs MC | FV1 vs Roe/HLL | FV2 vs Roe/HLL |
|---|---|---|---|---|---|---|---|---|---|
| plane | Resampled | 0.45175 | 0.79947 | 1.6599 | 0.0016805 | 5.5413 | 3.5229 | 4.5065 | 2.4243 |
| plane | Fixed 5 m | 0.71997 | 0.88255 | 1.7175 | 0.0034501 | 5.683 | 3.5948 | 4.6155 | 2.3626 |
| irregular | Resampled | 0.22195 | 0.7647 | 1.0541 | 0.00075631 | 6.621 | 3.9021 | 6.0619 | 3.6628 |
| irregular | Fixed 5 m | 0.99154 | 1.0275 | 1.5738 | 0.0018837 | 5.2613 | 5.6644 | 4.4171 | 5.7971 |
| steep | Resampled | 0.56935 | 1.1888 | 1.8603 | 0.0051039 | 11.069 | 5.3983 | 9.7105 | 4.4611 |
| steep | Fixed 5 m | 3.7572 | 3.5193 | 5.0563 | 0.034399 | 10.082 | 6.0568 | 8.7105 | 5.1837 |
| expansion | Resampled | 0.41192 | 0.85757 | 1.2267 | 0.0011073 | 7.976 | 6.3619 | 7.5131 | 6.1232 |
| expansion | Fixed 5 m | 0.59949 | 0.84304 | 1.4848 | 0.0015506 | 5.3548 | 1.6558 | 4.8166 | 1.0991 |
| rainfall | Resampled | 0.025115 | 0.34602 | 0.41887 | 1.5026e-05 | 2.2123 | 0.4201 | 1.9051 | 0.53288 |
| rainfall | Fixed 5 m | 0.34308 | 0.50081 | 0.49773 | 6.7716e-05 | 2.8909 | 3.0521 | 2.6155 | 2.8535 |
| barrier | Resampled | 11.973 | 11.983 | 1.2975 | 0.00046743 | 13.297 | 12.85 | 13.066 | 12.684 |
| barrier | Fixed 5 m | 0.92023 | 0.7127 | 1.4031 | 0.00049038 | 3.9319 | 2.3794 | 3.2303 | 1.6421 |
| oblique_channel | Resampled | 0.17736 | 0.25237 | 1.4902 | 6.4487e-05 | 6.7857 | 6.2764 | 6.4778 | 6.5045 |
| oblique_channel | Fixed 5 m | 0.96917 | 0.6579 | 1.6265 | 0.0011223 | 3.8239 | 2.6184 | 3.5874 | 3.0038 |
