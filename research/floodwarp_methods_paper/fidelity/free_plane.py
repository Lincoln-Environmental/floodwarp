"""Compare coincident interior and free-boundary sections against uniform supercritical sheet flow."""

import json
from pathlib import Path

import numpy as np
from floodwarp.inputs import boundary_faces, terrain_from_array
from floodwarp.model import create_model, diagnostics
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def along_flow(*, field, face):
    array = field.T if face in ("north", "south") else field
    return array[:, ::-1] if face in ("west", "south") else array


def measure(*, settings, dx, extension, method, face, dt, device="cpu", execution_mode="dense"):
    section, width = settings["section_m"], settings["width_m"]
    depth, grade, n = settings["depth_m"], settings["grade"], settings["manning"]
    length = section + extension
    native = np.broadcast_to(-grade * (np.arange(length) + 0.5), (width, length))
    five = mean_cells(values=native, factor=5)
    z = {1: native, 5: five, 10: mean_cells(values=five, factor=2)}[dx]
    if face in ("west", "south"):
        z = z[:, ::-1]
    vertical = face in ("north", "south")
    if vertical:
        z = z.T
    terrain = terrain_from_array(
        elevation=z, dx=dx, dy=dx, row_order="south_to_north", vertical_datum="local"
    )
    bx, by = boundary_faces(terrain=terrain)
    boundary = by if vertical else bx
    along_flow(field=boundary, face=face)[:, -1] = 2
    speed = depth ** (2 / 3) * np.sqrt(grade) / n
    c = np.sqrt(settings["gravity"] * depth)
    if speed <= c or max(settings["times_s"]) * (speed + c) >= section:
        raise ValueError(
            "A supercritical section outside the upstream domain of dependence is required"
        )
    options = {} if method == "fv1" else dict(limiter="adaptive")
    model = create_model(
        terrain=terrain,
        solver_method=method,
        device=device,
        execution_mode=execution_mode,
        manning_n=n,
        gravity=settings["gravity"],
        explicit_maximum_dt=dt,
        boundary_qx=bx,
        boundary_qy=by,
        initial_depth=np.full(z.shape, depth),
        **options,
    )
    solver = model.solver
    sign = -1 if face in ("west", "south") else 1
    velocity = np.full(z.shape, sign * depth * speed)
    zeros = np.zeros_like(velocity)
    solver.set_momentum(
        hu=zeros if vertical else velocity, hv=velocity if vertical else zeros
    )
    exact = width * depth * speed
    rows = []
    k = section // dx
    previous = 0.0
    for t in settings["times_s"]:
        start = solver.time
        solver.run_device_loop(duration=t - start)
        q = solver.get_conservative_state()
        field = "volume_y" if vertical else "volume_x"
        faces = sign * along_flow(
            field=solver._fields[field].numpy().reshape(boundary.shape), face=face
        )
        h = along_flow(field=q[..., 0], face=face)
        volume = faces[:, k].sum()
        flow = (volume - previous) / (t - start)
        rows.append(
            dict(
                method=method,
                device=device,
                execution_mode=execution_mode,
                face=face,
                maximum_dt_s=dt,
                dx_m=dx,
                section_m=section,
                downstream_extension_m=extension,
                time_s=t,
                start_s=start,
                reference_froude=speed / c,
                reference_discharge_m3s=exact,
                section_discharge_m3s=float(flow),
                discharge_bias_pct=float(100 * (flow / exact - 1)),
                mean_upstream_cell_depth_m=float(h[:, k - 1].mean()),
                exact_depth_m=depth,
                internal_face_m3=float(faces[:, k - 1].sum()),
                boundary_or_section_m3=float(volume),
                residual_m3=diagnostics(model=model)["residual_m3"],
            )
        )
        previous = volume
    return rows


def run(*, directory, settings):
    identity = dict(
        settings=settings, source_sha256=runner.checksum(Path(__file__)),
        solver_sha256=runner.source_identity(
            root=directory.parents[2], study=directory.parent, stage="long-storms"
        ),
        reference="Uniform supercritical Manning sheet beyond upstream influence",
        downstream="FREE", upstream_and_lateral_faces="closed", device="cpu",
    )
    folder = directory / "free_plane" / runner.digest(identity)[:24]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "identity.json").is_file() and (folder / "discharge.csv").is_file():
        archived = json.loads((folder / "identity.json").read_text())
        if archived != identity:
            raise ValueError("Analytical control identity mismatch")
        return
    rows = []
    for method in settings["methods"]:
        for dx in settings["grids_m"]:
            for distance in settings["extensions_m"]:
                faces = (
                    settings["cardinal_faces"]
                    if dx == settings["rotation_grid_m"]
                    else ["east"]
                )
                caps = (
                    (settings["maximum_dt_s"], settings["fine_dt_s"])
                    if dx == settings["rotation_grid_m"]
                    else (settings["maximum_dt_s"],)
                )
                for face in faces:
                    for dt in caps:
                        rows.extend(
                            measure(
                                settings=settings,
                                dx=dx,
                                extension=distance,
                                method=method,
                                face=face,
                                dt=dt,
                            )
                        )
    temporary = folder / "discharge.tmp"
    metrics.csv_write(file=temporary, rows=rows)
    temporary.replace(folder / "discharge.csv")
    runner.write_json(path=folder / "identity.json", data=identity)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "free_plane_settings.json").read_text())
    run(directory=directory, settings=settings)
