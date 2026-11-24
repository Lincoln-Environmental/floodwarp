# Native matrix run receipts

Each case directory retains its measured initial hydraulic intervals, spatial fields and recession group identities. The initial and recession runs share a physical checkpoint when their hydraulic states agree. A shared restart can carry first-window timing metadata from a different execution policy.

Use the paper driver’s `native-products` stage for complete-event runtimes. It reads each configuration’s own primary timing intervals followed by the observed tail from that configuration. The join checks the time grid, accepted steps and accepted transport at both gauges for every initial interval. The archived group snapshots and `complete_events.csv` preserve the original cursor totals. [Policy-specific measured costs](../native_results/measured_windows.csv) · [Original cursor totals and timing corrections](../native_results/timing_provenance.csv)
