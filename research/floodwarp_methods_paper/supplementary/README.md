# Supporting information

The archive contains individual runs and their physical inputs. All run times are seconds, depths and DEM elevations metres, areas square metres and integrated volumes cubic metres. Storm hydrographs report volumetric flow in m³/s. Analytical 1D gauges report transport per unit width in m²/s. Arrival and recession metrics use the 15-minute sampling interval. Unresolved and unavailable observations stay explicit in the database and tables.

| Output | Source and interpretation |
|---|---|
| Tables 1–2 | Formulation registry, archived domain definitions and independent analytical controls |
| Tables 3–5 | Terrain metrics, analytical errors, adjacent spatial orders and separate temporal records |
| Table 6 | Per-storm metrics with objective-specific reference qualifications |
| Table 7 | Operational event rows, live Warp pool and original 250k–2M-cell qualification |
| Table 8 | Measured equal-cost pairs and least-cost feasible points |
| Table 9 | Nested prospective fits, transferred forecasts and archival exact-solution fits |
| Table 10 | Whole-terrain held-out predictions, observed errors, method and cost regret |
| Figures 1–6 | Protocol, exact cell-average fields and gauges, temporal records, terrain maps and matched grids |
| Figures 7–10 | Qualified Pareto CSVs, frozen training fits and transfer rows, 24-hour hydrograph and interval timings |
| Figures 11–12 | Actual held-out decisions and spatial diagnostics, with grey crosses indicating abstention |
| Supplementary Pareto panels | Individual storm by family and exact benchmark, seven separate error or memory objectives |

Each nondominated point requires a strict improvement in either cost or the objective for another point to dominate it. The step segments mark dominance regions between measured points. The dots are the only measured configurations. Dotted horizontal lines show the change upon refining the numerical reference. Horizontal ranges in the analytical panels show repeated GPU timings. The error interval of the solver advisor is an empirical neighbourhood interval. Its held-out coverage is measured explicitly. The database also retains domain-level uncertainty and the physical units of each objective through its column names. [Pareto results](../processed/pareto/) · [Complete hydraulic errors](../processed/metrics/) · [Archive schema](../manifests/SCHEMA.md)

The LaTeX tables use the longtable package. Wide comparisons render as panels with repeated group identifiers. Compile with a Unicode-aware engine for SI superscripts and en dashes. CSV preserves the original column names and full precision.
