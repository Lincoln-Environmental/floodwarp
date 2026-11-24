# FREE outlet qualification

The archived pre-repair numerical source matches Git commit `0afdd684d2d1`. The corrected hydraulic source and storm driver match commit `46afa3f280a5`. Several trajectories started while the repair was in the working tree. The source hashes identify the committed implementation used by those runs. [Git-tree verification and per-run hardware](revision_evidence.json)

The exact Manning reference is steady uniform flow on a 7.329% plane. Discharge below uses the accepted face volume between 30 and 60 s at the same section, 400 m from the upstream boundary.

| Method | Grid (m) | Cap (s) | Original face bias (%) | Repaired face bias (%) | Buffered section bias (%) |
|---|---:|---:|---:|---:|---:|
| fv2 | 1 | 0.1 | +0.1158 | +0.1158 | +0.1158 |
| fv2 | 5 | 0.1 | -6.9367 | +0.1158 | +0.1158 |
| fv2 | 10 | 0.1 | -30.7219 | +0.1158 | +0.1158 |
| fv2 | 10 | 0.05 | -30.5563 | +0.0290 | +0.0290 |
| hancock | 1 | 0.1 | +0.1158 | +0.1158 | +0.1158 |
| hancock | 5 | 0.1 | -6.6538 | +0.1158 | +0.1158 |
| hancock | 10 | 0.1 | -30.4284 | +0.1158 | +0.1158 |
| hancock | 10 | 0.05 | -30.4567 | +0.0290 | +0.0290 |
| fv1 | 1 | 0.1 | -36.8162 | -36.8162 | -36.8162 |
| fv1 | 5 | 0.1 | -54.5994 | -54.5994 | -44.8927 |
| fv1 | 10 | 0.1 | -72.4977 | -72.4977 | -46.8352 |
| fv1 | 10 | 0.05 | -72.4466 | -72.4466 | -46.8188 |

The terminal slope previously vanished when the reconstruction stencil reached the FREE face. This left a half-cell bed discontinuity at the last internal face and suppressed the surface-gradient momentum source. Hancock and RK2 now extrapolate the bed and surface increments from the wet interior at an explicitly open face, with zero normal velocity gradient and the existing dry and positivity checks. The residual uniform-sheet error is predicted by the discrete source balance. Let U be the exact normal speed, a = gS the bed acceleration, and k = a/U² the quadratic drag coefficient. The steady speed v just before acceleration satisfies k v(v + aΔt) = a because consecutive drag half steps combine into one full drag step. The accepted halfway face speed is v + aΔt/2 = √(U² + (aΔt/2)²). Across both formulations, all measured orientations, the face and the buffered section, this predicts the observed 30–60 s bias to within 6.97e-12 percentage points. [Numerical and predicted residual](uniform_sheet_source_balance.csv) FV1 reconstructs from piecewise-constant cell beds. At this grade, the bed drops 0.07329 m per metre, exceeding the 0.03 m sheet depth even on the finest tested grid. Its hydrostatic cut to the higher cell bed clips the downstream state. At the 0.1 s cap the buffered FV1 deficit drops from 46.84% on 10 m cells to 36.82% on 1 m cells. The extra face deficit falls from 25.66 percentage points to below the displayed precision.

## Fixed incised section at 10,040 m

Mean discharge (m³/s) from 30 to 60 seconds on the same physical 120 m-wide section. Boundary placement is varied independently of the section position. The reach has no exact discharge reference.

| Version | Formulation | Grid (m) | At FREE face | 100 m buffer | 200 m buffer |
|---|---|---:|---:|---:|---:|
| audit | fv2 | 1 | 3.32471 | 3.32443 | 3.32443 |
| audit | fv2 | 5 | 3.18922 | 3.32438 | 3.32438 |
| audit | fv2 | 10 | 2.46253 | 3.32420 | 3.32420 |
| repair | fv1 | 1 | 2.12739 | 2.12699 | 2.12699 |
| repair | fv1 | 5 | 1.65071 | 1.87715 | 1.87715 |
| repair | fv1 | 10 | 1.00802 | 1.83284 | 1.83284 |
| repair | fv2 | 1 | 3.32443 | 3.32443 | 3.32443 |
| repair | fv2 | 5 | 3.32436 | 3.32438 | 3.32438 |
| repair | fv2 | 10 | 3.32413 | 3.32420 | 3.32420 |
| repair | hancock | 1 | 3.32443 | 3.32443 | 3.32443 |
| repair | hancock | 5 | 3.32436 | 3.32438 | 3.32438 |
| repair | hancock | 10 | 3.32413 | 3.32420 | 3.32420 |

[Measurement windows and source runs](incised_reach.csv)

## Full incised storm

The gauge remains at the eastern edge of the original 104.8576 km² catchment. The extension continues the 1 m valley, initial channel state and soil. Its additional rainfall is zero. The source fields on the original footprint retain their historical values.

| Method | Grid (m) | Extension (m) | CFL factor | Pre-fix direct peak (m³/s) | Peak (m³/s) | Minute ending (h) | Fixed 15 min peak (m³/s) | Moving 15 min peak (m³/s) | Export at gauge (m³) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive | 10 | 0 | 1.0 | 66.399 | 47.660 | 4.800 | 47.495 | 47.590 | 857,174.3 |
| fv1 | 10 | 0 | 1.0 | 48.371 | 48.371 | 5.400 | 48.290 | 48.293 | 910,216.6 |
| hancock_adaptive | 10 | 0 | 1.0 | 79.651 | 47.184 | 4.783 | 47.021 | 47.120 | 869,085.7 |
| adaptive | 10 | 400 | 1.0 | — | 47.660 | 4.800 | 47.495 | 47.590 | 857,182.5 |
| fv1 | 10 | 400 | 1.0 | — | 48.319 | 5.400 | 48.236 | 48.241 | 910,412.8 |
| hancock_adaptive | 10 | 400 | 1.0 | — | 47.185 | 4.783 | 47.021 | 47.120 | 869,078.9 |
| adaptive | 5 | 0 | 0.5 | — | 48.123 | 4.817 | 47.982 | 48.039 | 805,082.8 |
| adaptive | 5 | 0 | 1.0 | 48.330 | 48.115 | 4.800 | 47.960 | 48.032 | 822,874.5 |
| fv1 | 5 | 0 | 1.0 | 49.186 | 49.186 | 5.350 | 49.087 | 49.101 | 910,143.2 |
| hancock_adaptive | 5 | 0 | 1.0 | 48.286 | 48.019 | 4.800 | 47.849 | 47.934 | 829,026.1 |
| adaptive | 5 | 400 | 1.0 | — | 48.115 | 4.800 | 47.960 | 48.032 | 822,871.8 |
| hancock_adaptive | 5 | 400 | 1.0 | — | 48.019 | 4.800 | 47.849 | 47.934 | 829,017.1 |

12 of 14 trajectories have completed the historical 16-hour comparison. Each hydrograph is an interval average of accepted transport. On these irregular terrains a solver difference alone does not establish accuracy.

[Water accounts, infiltration inside and outside the rainfall footprint](events.csv) · [Crest and whole-recession last 40 m budgets](last_40m.csv) · [Spatial extent](extent.csv) · [Wet-union depth differences on common spatial support](depth_disagreement.csv) · [Spatial reference qualification](spatial_reference_checks.csv) · [Hydrograph and infiltration reference qualification](reference_checks.csv)

A focused CUDA regression briefly shared the GPU during the direct 5 m half-CFL RK2 run. Keep its recorded time for diagnostics. Use its hydraulics only for timestep sensitivity.

## FV1 minus Hancock on the original catchment

| Grid (m) | Extension (m) | Extra export (m³) | Infiltration difference in rain footprint (m³) | Infiltration difference outside (m³) | Final storage difference (m³) | Unexplained export (m³) |
|---|---|---:|---:|---:|---:|---:|
| 10 | 0 | +41,130.9 | +17,863.3 | -59,098.2 | +104.1 | +1.18e-09 |
| 10 | 400 | +41,333.9 | +17,862.7 | -59,196.6 | +0.0 | +1.27e-09 |
| 5 | 0 | +81,117.1 | +6,210.5 | -87,383.4 | +55.8 | +1.26e-08 |

## Hour-eight inundation and maximum extent

| Model | Grid (m) | 5 cm at 8 h (km²) | 10 cm at 8 h (km²) | Ever above 5 cm by 16 h (km²) | Ever above 10 cm by 16 h (km²) |
|---|---:|---:|---:|---:|---:|
| adaptive | 10 | 0.0000 | 0.0000 | 1.1875 | 0.0000 |
| fv1 | 10 | 0.7233 | 0.0143 | 6.1042 | 2.1934 |
| hancock_adaptive | 10 | 0.0000 | 0.0000 | 1.1838 | 0.0000 |
| adaptive | 5 | 0.0000 | 0.0000 | 0.5205 | 0.0000 |
| fv1 | 5 | 0.5847 | 0.0092 | 7.4040 | 1.8847 |
| hancock_adaptive | 5 | 0.0000 | 0.0000 | 0.5195 | 0.0000 |

The spatial comparison thresholds area-averaged depths on the coarser common support. For maxima, it averages the recorded native-cell maxima. The extent table above counts cells at each native hydraulic spacing. The 5 m trajectories quantify formulation and timestep sensitivity. A finer storm reference is required for absolute 5 m accuracy.

[Water-balance decomposition](water_differences.csv) · [Least measured cost by tolerance](least_cost_fidelity.csv) · [Smallest qualified error within the FV1 budget](equal_cost_fidelity.csv)

Depth and inundation costs run from the initial state to each snapshot. The 8 h comparison uses only the first eight hours of measured hydraulic execution. [Least cost by spatial tolerance](least_cost_depth_extent.csv) · [Smallest spatial error at each FV1 budget](equal_cost_depth_extent.csv) · [Independent timing windows](spatial_timing_windows.csv)

The [regional water and inundation accounts](regional_contrasts.csv) divide the original footprint at the internal gauge and at the start of the final 40 m. Each area is also split by the prescribed rainfall footprint.

On the 10 m moderate-rainfall plain the two CPU policies reproduced all 99,238 ordered timesteps through 40 hours. Their interval face transport and final fields matched the primary trajectory. [Ordered timestep evidence](../native_step_history/39b84d74d143dcf4a64e60f0/verified.json)
