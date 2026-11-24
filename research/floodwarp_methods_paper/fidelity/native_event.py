"""Use verified metre-derived terrain, initial conserved state and storm forcing for a fresh event."""

from pathlib import Path
import numpy as np
from floodwarp.inputs import boundary_faces, terrain_from_array
from research.floodwarp_methods_paper import events
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def create(*, config, template, spec, device, bundle, receipt):
    inputs = receipt["inputs"]
    dx = spec["dx"]
    if (
        inputs["native_m"] != 1
        or inputs["aggregation"] != "native_1m_block_mean"
        or spec["initial_grid"] != inputs["aggregation"]
        or spec["terrain_aggregation"] != inputs["aggregation"]
        or dx != spec["dem"]
        or dx not in inputs["grids_m"]
        or inputs["dimensions_m"] != spec["dimensions"]
        or inputs["landform"] != spec["landform"]
        or inputs["storm"] != spec["storm"]
        or inputs["parameters"]["initial"] != spec["initial"]
    ):
        raise ValueError(
            "Practical runs require aligned native metre input provenance for this event"
        )
    fields = {
        k: np.load(Path(bundle) / f"{k}_{dx}m.npy", mmap_mode="r")
        for k in receipt["native_sha256"]
    }
    z, h, hu = fields["bed"], fields["depth"], fields["hu"]
    mask, soil = fields["rain"], fields["infiltration"]
    t = np.r_[template["hours"], spec["hours"]] * 3600.0
    rates = np.r_[template["storms_mm_h"][spec["storm"]], 0.0] / 3.6e6
    terrain = terrain_from_array(
        elevation=z, dx=dx, dy=dx, row_order="south_to_north", vertical_datum="local"
    )
    bx, by = boundary_faces(terrain=terrain)
    coastal = spec["landform"] == "coastal"
    bx[:, -1] = 3 if coastal else 2
    method, mode, options = events.numerical(method=spec["method"])
    if method != "li":
        options.update(execution_mode=mode, gravity=config["storms"]["gravity"])
    roughness = config["storms"]["manning_n"] * spec["multiplier"]
    model = events.create_model(
        terrain=terrain,
        solver_method=method,
        boundary_qx=bx,
        boundary_qy=by,
        stage_qx=0.015 if coastal else 0.0,
        stage_datum="local",
        initial_depth=h,
        manning_n=roughness,
        rainfall_series=(t, rates),
        infiltration=soil,
        device=device,
        **options,
    )
    model.solver.set_rainfall_spatial_multiplier(multiplier=mask)
    if method == "li":
        qx = np.zeros(bx.shape)
        qx[:, 1:-1] = 0.5 * (hu[:, 1:] + hu[:, :-1])
        qx[:, -1] = hu[:, -1]
        model.solver.set_face_fluxes(qx=qx, qy=np.zeros(by.shape))
    else:
        model.solver.set_momentum(hu=hu, hv=np.zeros_like(hu))
    model.identity["native_inputs"] = dict(
        native_sha256=receipt["native_sha256"], aggregation=inputs["aggregation"]
    )
    provenance = dict(
        terrain_sha256=array_sha(z),
        rain_mask_sha256=array_sha(mask),
        initial_depth_sha256=array_sha(h),
        initial_hu_sha256=array_sha(hu),
        infiltration_sha256=array_sha(soil),
        boundary_x_sha256=array_sha(bx),
        boundary_y_sha256=array_sha(by),
        stage_m=0.015 if coastal else None,
        roughness=roughness,
        shape=list(z.shape),
        parameters=inputs["parameters"],
        numerical=model.solver.numerical_configuration(),
        rain_times_s=t.tolist(),
        rain_rates_ms=rates.tolist(),
        native_inputs=model.identity["native_inputs"],
    )
    return model, float(np.trapezoid(rates, t) * mask.sum() * dx**2), provenance
