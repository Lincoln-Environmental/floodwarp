"""Independent objective-specific empirical evidence for event hydrograph fidelity."""

import json
from pathlib import Path


METRICS = {
    "relative_peak_discharge": "peak_bias_pct",
    "integrated_runoff": "volume_bias_pct",
    "net_exported_volume": "net_export_bias_pct",
    "cumulative_infiltration": "infiltration_bias_pct",
    "outlet_arrival": "arrival10_delay_pct",
    "median_passage": "t50_delay_pct",
    "relative_peak_timing": "peak_delay_pct",
    "hydrograph_shape": "shape_tv_pct",
    "hydrograph_spreading": "width80_bias_pct",
    "hydrograph_propagation": "wasserstein_pct",
    "spatial_arrival": None,
    "transient_momentum": None,
}


def attach(*, model, supplement):
    """Keep the historical model and its uncertainties untouched for original objectives."""
    training = []
    for item in model["training"]:
        measured = supplement["training"].get(item["run_id"], {})
        forecast = supplement.get("forecasts", {}).get(item["run_id"])
        if "forecasts" in supplement and forecast is None:
            continue
        if forecast:
            item = dict(
                item, runtime_s=forecast["runtime_s"], run_id=forecast["evidence_id"]
            )
        training.append(
            dict(
                item,
                qualified={**item["qualified"], **dict.fromkeys(measured, True)},
                errors={**item["errors"], **measured},
            )
        )
    from floodwarp.solver_advisor.model import scales

    return dict(
        model,
        scales=scales(rows=training)
        if training and "forecasts" in supplement
        else model.get("scales", {}),
        training=training,
        duration_power=0.0,
        tolerances={**model["tolerances"], **supplement["tolerances"]},
        interval_padding={**model.get("interval_padding", {}), **supplement["padding"]},
        fidelity_validation=supplement["validation_support"],
    )


def with_evidence(*, model):
    file = Path(__file__).with_name("fidelity_evidence.json")
    supplement = json.loads(file.read_text()) if file.is_file() else {}
    if supplement.get("window") != "matched_completed_recessions":
        return dict(
            model,
            tolerances={**model["tolerances"], **dict.fromkeys(METRICS, 10.0)},
            fidelity_validation={},
        )
    if (
        sorted(r["run_id"] for r in model["training"])
        != supplement["base_training_ids"]
    ):
        # A custom external model needs its own calibrated supplement.
        supplement = dict(supplement, training={}, validation_support={})
    return attach(model=model, supplement=supplement)
