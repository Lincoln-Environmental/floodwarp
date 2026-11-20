"""HIRDS storm controls from native archived inputs, signature-checked at both resolutions."""

from pathlib import Path
import hashlib
import json
import os
import numpy as np
from floodwarp.inputs import terrain_from_array, boundary_faces
from floodwarp.model import create_model, advance
from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_local_inertia import read_state
from validation.fv2_orton_matrix import geometry, forcing_check, replay_solver, dry_initial, simulate


def liquid(*, state, d, device):
    active = d["active"]
    t = terrain_from_array(
        elevation=d["bed"],
        active=active,
        dx=5.0,
        dy=5.0,
        vertical_datum="NZVD2016",
        row_order="south_to_north",
    )
    x, y = boundary_faces(terrain=t)
    y[d["by"] == 3] = 3
    model = create_model(
        solver_method="li",
        terrain=t,
        device=device,
        manning_n=d["manning"],
        initial_depth=d["q"][..., 0],
        boundary_qx=x,
        boundary_qy=y,
        stage_qy=d["sy"],
        stage_datum="NZVD2016",
        explicit_minimum_dt=1.0e-6,
        explicit_maximum_dt=1.0,
        rainfall_series=(state["rain_times"], state["rain_rates"]),
        infiltration_method="exponential",
        infiltration_f0_m_per_s=d["f0"],
        infiltration_fd_m=d["fd"],
        infiltration_k_m_per_s=d["conductivity"],
        adaptive_dt_growth_factor=None,
        track_max_water_depth=True,
    )
    s = model.solver
    s.set_rainfall_spatial_multiplier(multiplier=d["multiplier"])
    s.gravity = type(s.gravity)(state["configuration"]["gravity"])
    s.cfl_number = type(s.cfl_number)(state["configuration"]["cfl"])
    s.dry_depth = type(s.dry_depth)(state["configuration"]["dry_depth"])
    s.cumulative_infiltration.assign(d["soil"])
    # Native staggered discharge at the last accepted FV1 step, including ports.
    raw = state["arrays"]
    s.set_face_fluxes(qx=raw["flux_x"][:, 0].reshape(x.shape), qy=raw["flux_y"][:, 0].reshape(y.shape))
    s.simulation_time = state["time"]
    return model


def continuation(*, state, d, device, methods, duration, interval, output):
    signatures = forcing_check(state=state, data=d, dx=5.0)
    rows = []
    ref = None
    active = d["active"]
    for method in methods:
        model = None
        if method == "li":
            model = liquid(state=state, d=d, device=device)
            s = model.solver
        else:
            s = replay_solver(state=state, data=d, device=device, factor=1, method=method)
            if method == "fv2_adaptive":
                s.configure_execution(mode="sparse")
        start = s.cumulative_accounting()
        initial = float(d["q"][..., 0].sum() * 25)
        previous = start["boundary_outflow"]
        hydro = []
        for end in np.arange(interval, duration + interval / 2, interval):
            stats = (
                advance(model=model, end_time=state["time"] + end)
                if model is not None
                else s.run_device_loop(duration=interval)
            )
            ledger = s.cumulative_accounting()
            export = ledger["boundary_outflow"]
            hydro.append(
                dict(
                    t=state["time"] + end,
                    outflow_m3_s=(export - previous) / interval,
                    steps=stats["steps"],
                    loop_seconds=stats["loop_seconds"],
                )
            )
            previous = export
        field = read_state(solver=s, li=method == "li")
        if ref is None:
            ref = field.copy()
        wet = (field[..., 0] > 0.01) & active
        b = ledger
        change = {k: b[k] - start[k] for k in ("rain", "infiltration", "boundary_inflow", "boundary_outflow")}
        residual = (
            initial
            + change["rain"]
            - change["infiltration"]
            + change["boundary_inflow"]
            - change["boundary_outflow"]
            - field[..., 0].sum() * 25
        )
        rows.append(
            dict(
                method=method,
                hydrograph=hydro,
                forcing=signatures,
                ledger_change=change,
                volume_residual_m3=float(residual),
                q95_speed=float(
                    np.percentile(np.linalg.norm(field[wet, 1:], axis=-1) / field[..., 0][wet], 95)
                ),
                maximum=float(s.get_max_water_depth()[active].max()),
                wet_area_m2=int(wet.sum() * 25),
                vs_contact=errors(result=field, expected=ref, window=active, dx=5.0, wet_depth=0.01),
            )
        )
        report_json(path=output, content=rows)
        print("matched continuation", method, hydro[-1], flush=True)


if __name__ == "__main__":
    provenance = json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())
    checkpoint = Path(provenance["checkpoint"])
    with checkpoint.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != provenance["checkpoint_sha256"]:
            raise ValueError("Archive fingerprint mismatch")
    revision = os.environ.get("FV2_EVENT_REVISION", "halo")
    directory = Path("validation/fv2_review/regimes") / revision
    directory.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    state = checkpoint_solver(scheme="contact_hll", device="cpu", path=checkpoint).capture_state()
    fields, _ = geometry(state=state, factor=1)
    selection = os.environ.get("FV2_EVENT_STUDY", "both")
    if selection in ("both", "continuation"):
        continuation(
            state=state,
            d=fields,
            device=device,
            methods=("contact_hll", "fv2_adaptive", "li"),
            duration=600.0,
            interval=30.0,
            output=directory / "wet_5m.json",
        )
    if selection in ("both", "storm"):
        initial = dry_initial(checkpoint=checkpoint)
        factor = int(os.environ.get("FV2_STORM_FACTOR", "4"))
        fields, info = geometry(state=initial, factor=factor)
        forcing = forcing_check(state=initial, data=fields, dx=5.0 * factor)
        method = os.environ.get("FV2_EVENT_METHOD", "fv2_adaptive")
        mode = os.environ.get("FV2_EVENT_MODE", "dense")
        simulate(
            settings=dict(
                data=fields, state=initial, factor=factor, method=method, device=device, execution_mode=mode
            ),
            folder=Path("outputs/fv2_regime_hirds48") / revision / f"{5 * factor}m_{method}_{mode}",
            public=directory / f"{5 * factor}m_{method}_{mode}.json",
            forcing=forcing,
            info=info,
            sample_s=60,
            checkpoint_hours=6,
            diagnostic_hours=(),
        )
