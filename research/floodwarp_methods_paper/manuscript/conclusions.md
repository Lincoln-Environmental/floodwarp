# Conclusions

Of 48 small events with six hours of rain, 31 required a longer recession. The longest matched simulation reached 76 hours. For the 12 storms with independently qualified normalised outlet shape at 10 m, the Hancock–FV2 difference in absolute shape error exceeded the combined reference sensitivity in 0 cases. That comparison concerns the timing distribution of discharged volume. Peak and arrival-time conclusions have separate reference screens. Among 6 FV1 storms with independently assessable peaks and final depth RMSE within 1 mm, 3 had an absolute peak bias above 10%. [Complete small-storm comparisons](../fidelity/small_storms/README.md) · [Million-cell export after the original stop time](../fidelity/NATIVE_RECESSIONS.md)

Across the original fixed windows of 48 small-domain storms, the faster qualified Hancock execution took a median 1.352 times the FV1 runtime. Tiled execution alone had median 2.220×.

Across the 8 primary million-cell events, the faster measured explicit Hancock policy took a median of 1.69 times the FV1 runtime over the original observation windows. The accepted-step ratios ranged from 1.06 to 1.62, and the corresponding mean cost per accepted step ranged from 1.02 to 1.61 times FV1. These factors multiply within each event. The normal timestep policy therefore matters even where tile scheduling removes most of the dry-domain arithmetic. [Per-event costs](../tables/11_execution.md)

The frozen advisor made a qualified feasible selection in 17.54% of held-out decisions with a measured feasible option. It abstained on 134 of 150 decisions. Terrain loss and reference uncertainty limit recommendations for specific sites and observables. [Held-out comparisons](../tables/10_advisor.md)

Retain adaptive Hancock as the development default. Keep explicit execution overrides and select FV1 when the measured error is acceptable for the intended observable. [Results](results.md) · [Discussion](discussion.md)
