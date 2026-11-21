# Hancock qualification checkpoint

The integrated solver completed the 48 primary storm pairs and the larger plain, coastal and mixed-terrain comparisons on an RTX 4070 Ti SUPER. The fine-grid replays and additional reference refinements are in progress. All primary events include six hours of rain and at least two hours of recession.

| Complete-event comparison | Tiled Hancock / sparse FV1 |
|---|---:|
| Median of the 48 small-domain events | 1.65× |
| Largest small-domain penalty, mixed terrain with spatial rain | 2.58× |
| Spatial rainfall on the plain, 250,000 / 500,000 cells | 1.33× / 1.18× |
| Spatial rainfall on the plain, one / two million cells | 1.12× / 1.06× |
| Uniform rainfall on the plain, two million cells | 1.65× |
| Coastal uniform and mixed spatial rainfall, 250,000 cells | 1.70× / 1.84× |

The mostly dry 655,360-cell common-step control improved from 90.70 ms for historical dense Hancock to 32.56 ms with tiles at the scheduler revision. FV1 took 31.82 ms. The two-million-cell spatial storm subsequently took 16.96 s using integrated tiled Hancock and 15.94 s using FV1. Keep adaptive Hancock as the development default and select tiles explicitly when the expected hydraulic footprint is spatially restricted. [Detailed architecture, validation and limitations](REPORT.md)

The 16 matched 10 m comparisons give median final-depth RMSEs of 3.30 mm for Hancock and 3.61 mm for FV2. Median outlet RMSEs were 0.00380 and 0.00390 m³/s. Successive refinement has not yet qualified every reference hydrograph, especially in the steep incised catchment. [Matched hydraulic comparisons](COMPARISONS.md)

The integrated implementation passed 1,874 tests, with 25 skipped. A checkpoint generated on the original development revision restored and reproduced all 28 saved state arrays under four Hancock execution modes. [Test receipt](test_results.json) · [Historical checkpoint comparison](checkpoint/results.json) · [Verified event records](integrated_events)
