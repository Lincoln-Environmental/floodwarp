"""Isolate the effect of representing a uniform bed gradient by repeated parent elevations."""

from pathlib import Path
import json
import numpy as np

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model
from research.floodwarp_methods_paper import metrics, runner


def experiment(*, dx, bed_dx, method, settings):
    length, width = settings["domain_m"]
    x = (np.arange(round(length / dx)) + 0.5) * dx
    bed_x = (np.floor(x / bed_dx) + 0.5) * bed_dx
    shape = (round(width / dx), len(x))
    elevation = np.broadcast_to(-settings["grade"] * bed_x, shape).copy()
    terrain = terrain_from_array(
        elevation=elevation,
        dx=dx,
        dy=dx,
        row_order="south_to_north",
        vertical_datum="local",
    )
    h = settings["depth_m"]
    gravity = settings["gravity"]
    n = settings["manning"]
    num = {} if method == "fv1" else {"limiter": "adaptive"}
    model = create_model(
        terrain=terrain,
        solver_method=method,
        initial_depth=np.full(shape, h),
        device="cpu",
        execution_mode="dense",
        manning_n=n,
        explicit_maximum_dt=settings["maximum_dt_s"],
        gravity=gravity,
        **num,
    )
    stats = model.solver.run_device_loop(duration=settings["duration_s"])
    state = model.solver.get_conservative_state()
    a, b = settings["interior_x_m"]
    centre = state[:, (x > a) & (x < b)]
    velocity = centre[..., 1].sum() / centre[..., 0].sum()
    acceleration = gravity * settings["grade"]
    drag = gravity * n**2 / h ** (4 / 3)
    smooth_exact = np.sqrt(acceleration / drag) * np.tanh(
        settings["duration_s"] * np.sqrt(acceleration * drag)
    )
    return dict(
        method=method,
        grid_m=dx,
        represented_bed_m=bed_dx,
        time_s=model.solver.time,
        steps=stats["steps"],
        domain_storage_change_m3=float(
            (state[..., 0].sum() - np.prod(shape) * h) * dx**2
        ),
        interior_mean_h_m=float(centre[..., 0].mean()),
        interior_velocity_ms=float(velocity),
        smooth_plane_exact_ms=float(smooth_exact),
        ratio_to_smooth_plane=float(velocity / smooth_exact),
    )


def run(*, directory, settings):
    rows = [
        experiment(dx=dx, bed_dx=bed_dx, method=method, settings=settings)
        for method in settings["methods"]
        for dx, bed_dx in settings["grids_and_beds_m"]
    ]
    folder = directory / "incised_crest"
    metrics.csv_write(file=folder / "plane_controls.csv", rows=rows)
    runner.write_json(
        path=folder / "plane_controls_inputs.json",
        data=dict(
            settings=settings,
            code=runner.checksum(Path(__file__)),
            solver=runner.source_identity(
                root=directory.parents[2], study=directory.parent, stage="long-storms"
            ),
            device="cpu",
            comparison="Smooth-plane exact velocity applies only to smooth representations.",
        ),
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "plane_settings.json").read_text())
    run(directory=directory, settings=settings)
