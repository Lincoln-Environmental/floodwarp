"""Move a free face within the real synthetic catchment while holding the measuring section fixed."""

import json
from pathlib import Path

import numpy as np
from floodwarp.inputs import boundary_faces, terrain_from_array
from floodwarp.model import create_model, diagnostics
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.terrain.generators import native_inputs
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def measure(*, dx, distance, settings, receipt, bundle, forcing, gravity):
    x = settings["section_x_m"]
    lo, hi = settings["section_y_bounds_m"]
    margin = settings["side_margin_m"]
    west, east = x - settings["upstream_margin_m"], x + distance
    south, north = lo - margin, hi + margin
    inp = receipt["inputs"]
    length, width = inp["dimensions_m"]
    if not (0 <= west < east <= length and 0 <= south < north <= width) or any(
        n % 10 for n in (west, east, south, north, lo, hi, x)
    ):
        raise ValueError(
            "Native reach must be aligned and entirely inside the original catchment"
        )
    source = native_inputs.native_strip(
        x=np.arange(west, east)[None, :] + 0.5,
        y=np.arange(south, north)[:, None] + 0.5,
        dimensions=inp["dimensions_m"],
        params=inp["parameters"],
        family=inp["landform"],
        storm=inp["storm"],
        options=inp["options"],
        channel_core_m=inp["moving_channel_min_depth_m"],
    )
    fields = {}
    for key, value in source.items():
        if dx == 1:
            fields[key] = value
        else:
            five = mean_cells(values=value, factor=5)
            fields[key] = five if dx == 5 else mean_cells(values=five, factor=2)
            stored = np.load(bundle / f"{key}_{dx}m.npy", mmap_mode="r")[
                south // dx : north // dx, west // dx : east // dx
            ]
            np.testing.assert_array_equal(fields[key], stored)
    z = fields["bed"]
    terrain = terrain_from_array(
        elevation=z, dx=dx, dy=dx, row_order="south_to_north", vertical_datum="local"
    )
    bx, by = boundary_faces(terrain=terrain)
    bx[:, -1] = 2
    model = create_model(
        terrain=terrain,
        solver_method="fv2",
        limiter="adaptive",
        execution_mode="dense",
        device="cpu",
        gravity=gravity,
        explicit_maximum_dt=settings["maximum_dt_s"],
        boundary_qx=bx,
        boundary_qy=by,
        initial_depth=fields["depth"],
        manning_n=forcing["roughness"],
        infiltration=fields["infiltration"],
        rainfall_series=(forcing["rain_times_s"], forcing["rain_rates_ms"]),
    )
    solver = model.solver
    solver.set_momentum(hu=fields["hu"], hv=np.zeros_like(z))
    solver.set_rainfall_spatial_multiplier(multiplier=fields["rain"])
    sec = (x - west) // dx
    rows = slice((lo - south) // dx, (hi - south) // dx)
    previous = 0.0
    out = []
    for t in settings["times_s"]:
        start = solver.time
        status = solver.run_device_loop(duration=t - start)
        face = solver._fields["volume_x"].numpy().reshape(bx.shape)[rows, sec].sum()
        h = solver.get_water_depth()[rows, sec - 1]
        out.append(
            dict(
                grid_m=dx,
                extension_m=distance,
                section_x_m=x,
                south_m=lo,
                north_m=hi,
                start_s=start,
                end_s=t,
                accepted_steps=status["steps"],
                section_m3s=(face - previous) / (t - start),
                upstream_cell_mean_depth_m=h.mean(),
                residual_m3=diagnostics(model=model)["residual_m3"],
            )
        )
        previous = face
    return out


def run(*, directory, settings):
    root = directory.parents[2]
    parent = directory / "mean_storms"
    receipt = json.loads((parent / "terrain.json").read_text())
    sha = runner.digest(receipt["inputs"])
    bundle = root / ".fidelity-restarts/native_inputs" / sha
    native_inputs.verify(folder=bundle, identity=receipt["inputs"])
    prefix = next(
        r
        for p in parent.glob("*/result.json")
        if (r := json.loads(p.read_text()))["inputs"]["spec"]["method"] == "adaptive"
    )
    identity = dict(
        source=runner.checksum(Path(__file__)),
        solver=runner.source_identity(
            root=root, study=directory.parent, stage="long-storms"
        ),
        settings=settings,
        physical_sha=runner.digest(receipt),
        device="cpu",
    )
    dest = directory / "free_reach" / runner.digest(identity)[:24]
    dest.mkdir(parents=True, exist_ok=True)
    runner.write_json(path=dest / "inputs.json", data=identity)
    combined = []
    for dx in settings["grids_m"]:
        for distance in settings["downstream_lengths_m"]:
            path = dest / f"grid_{dx}m_extension_{distance}m.csv"
            if not path.is_file():
                rows = measure(
                    dx=dx,
                    distance=distance,
                    settings=settings,
                    receipt=receipt,
                    bundle=bundle,
                    forcing=prefix["data"]["forcing"],
                    gravity=prefix["data"]["numerical"]["gravity"],
                )
                temp = path.with_suffix(".tmp")
                metrics.csv_write(file=temp, rows=rows)
                temp.replace(path)
            combined.extend(metrics.csv_read(file=path))
            print(
                "Native reach",
                dx,
                "m grid,",
                distance,
                "m extension, complete",
                flush=True,
            )
    metrics.csv_write(file=dest / "comparison.csv", rows=combined)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "free_reach_settings.json").read_text())
    run(directory=directory, settings=settings)
