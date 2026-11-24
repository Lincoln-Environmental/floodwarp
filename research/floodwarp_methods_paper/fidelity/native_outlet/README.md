# Free downstream boundary on a steep slope

In the matched eight-hour incised FV2 forecasts, 54 of the 54 common blocks exceeding 5 cm mean depth in either forecast lie in the last 40 m of the catchment. The eastern strip accounts for 99.953% of the full-domain squared depth difference. The open boundary was checked face by face in the saved state. Each station flux integral agrees with its recorded boundary export.

| Grid (m) | Peak (m³/s) | Export (m³) | Storage in last 40 m (m³) | Storage upstream (m³) | Deepest terminal-strip cell at hour 8 (m) |
|---:|---:|---:|---:|---:|---:|
| 10 | 47.475 | 769,061 | 26,402 | 71,519 | 1.671 |
| 5 | 48.330 | 771,947 | 3,541 | 78,702 | 0.338 |

[Native checkpoint strip budget](strip_storage.csv) · [Spatial depth comparison](outlet_depth.pdf)

## Independent steady sheet

Steady uniform full-SWE sheet flow on a plane has discharge Q = Wh⁵ᐟ³√S/n. Here W = 20 m, h = 0.03 m, S = 0.07329 and n = 0.035. Both characteristics travel downstream. The upstream boundary disturbance cannot reach the measuring section within the test window. A pair of runs measures flux at the same physical section, first with a FREE boundary there and then with another 400 m of the same native plane downstream. Mean fluxes in the table cover seconds 30–60.

| Formulation | Timestep cap (s) | 10 m free-face bias | 10 m interior bias |
|---|---:|---:|---:|
| fv2 | 0.1 | -30.72% | +0.12% |
| fv2 | 0.05 | -30.56% | +0.03% |
| hancock | 0.1 | -30.43% | +0.12% |
| hancock | 0.05 | -30.46% | +0.03% |
| fv1 | 0.1 | -72.50% | -46.84% |
| fv1 | 0.05 | -72.45% | -46.82% |

All four outlet orientations are included in the data. The MUSCL reconstruction sets the normal slope to zero at a missing exterior neighbour. FREE copies the interior trace into the ghost state and suppresses inward flux. The normal surface-slope source therefore vanishes in the terminal cell. Moving the boundary restores a two-sided stencil at the measured section. This test establishes sensitivity to boundary placement on a known supercritical sheet. The incised storm also contains lateral convergence, changing storage and friction. Its final depth error against an independent physical reference remains unresolved.

[Analytical discharge controls](../free_plane/discharge.csv) · [Implementation and source identity](../free_plane/identity.json)

## Same physical section in the incised catchment

The section was selected after localising the depth difference. Each run starts with the native metre bed, depth, momentum, rainfall and infiltration fields on an aligned subdomain of the original catchment. The coarser subsets agree cell for cell with the full-domain inputs. The table reports the final observed time interval, in m³/s. The two longer reaches test sensitivity to the new downstream boundary. This reach has no exact analytical solution.

| Grid (m) | 0 m downstream | 100 m downstream | 200 m downstream |
|---:|---:|---:|---:|
| 1 | 3.324711 | 3.324428 | 3.324428 |
| 5 | 3.189217 | 3.324378 | 3.324378 |
| 10 | 2.462526 | 3.324204 | 3.324204 |

[Reach histories and water balance](../free_reach/2ae79b3aa1a6138e2a2bd603/comparison.csv)
