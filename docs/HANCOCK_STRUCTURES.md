# Hydraulic structures in adaptive MUSCL–Hancock

Pass a sequence of structure records to `create_model`, `create_overland_solver`
or the native `HancockSolver`. The Python factories use adaptive Hancock when
the method is omitted. Each record identifies two distinct active cells by
south-first row and column. Separate links may share either endpoint. Inverts
and crests are absolute elevations in the terrain vertical datum.

| Type | Required hydraulic geometry |
|---|---|
| Circular culvert | `type="culvert"`, `shape="circular"`, `diameter_m`, `inlet_invert_m`, `outlet_invert_m` |
| Box culvert | `type="culvert"`, `shape="box"`, `width_m`, `height_m`, both inverts |
| Bridge opening | `type="bridge"`, `opening_width_m`, `opening_height_m`, `opening_invert_m` |
| Weir | `type="weir"`, `weir_type="sharp_crested"` or `"broad_crested"`, `crest_elevation_m`, `crest_length_m`, `discharge_coefficient` |

Openings accept `length_m`, `manning_n`, `entrance_loss_coefficient` and
`exit_loss_coefficient`. Specify them to represent the surveyed opening and
losses. Each defaults to zero. Every type supports `one_way` and the discharge
cap `maximum_discharge_m3s`. The FV1 spelling `max_discharge_m3_s` is accepted.
For example, the following record represents an opening between two cells of
an already configured terrain.

```python
crossing = dict(
    id="road-crossing", type="culvert", shape="box",
    inlet_row=5, inlet_col=8, outlet_row=5, outlet_col=11,
    width_m=1.2, height_m=0.9, length_m=12.0,
    inlet_invert_m=7.30, outlet_invert_m=7.25,
    manning_n=0.015, entrance_loss_coefficient=0.5, exit_loss_coefficient=1.0,
)
model = create_model(terrain=terrain, structures=[crossing], execution_mode="tiles")
```

Hancock completes its conservative surface step and both source half-steps
before applying link discharge. The laws represent quasi-steady open or
submerged conveyance with energy losses, or crest flow with tailwater
correction. They use the model gravity. Deck and road overtopping follows the
surface DEM. Matching a crossing requires terrain that represents the barrier.
Pressure transients, barrel storage, jets and unresolved road barriers require
additional physical modelling.

Each accepted trial evaluates links from one simultaneous snapshot of endpoint
stages. The requested magnitude is bounded by the equal-area pair head
difference. An endpoint budgets concurrent withdrawals against its available
water at every incident sill. Each link debits and credits the same signed
volume. Donor momentum leaves in proportion to the volume removed. The
receiver gains water with zero added resolved momentum. A newly wetted
receiver holds its water and updates its maximum on that step. The next
surface evaluation sees the composed state.

The split link operator is first-order in time. Compare timestep refinements
for fast exchanges, stiff ratings and shared endpoints. For example, the
two-basin synthetic storm compares the coupled solver to independent integration
of the continuous weir rating. [Storm evidence](../validation/hancock_long_storm_review/REPORT.md)

Call `model.solver.get_structure_diagnostics()` for each link's signed current
discharge, transferred volumes by direction, peak absolute accepted discharge,
maximum head difference and limiter count. Cumulative D4 face volumes describe
surface flow. Add any crossing link transport explicitly when computing a
section budget. Checkpoints bind each normalized link definition, endpoint
and the integration version. The numerics support conditional CUDA graphs
and all native Hancock scheduling overrides.
