"""Full 48-hour HIRDS storm on the Orton Bradley domain on 5, 10 and 20 m grids."""

from pathlib import Path
import os
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.fv2_solver import FV2Solver
from validation.sparse_fv1_benchmark import select_checkpoints
from validation.fv1_riemann import errors, report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv1_riemann_transport import InteriorTransport
from validation.fv2_terrain import blocks


def padded(*, a, multiple):
    py, px = (-a.shape[0]) % multiple, (-a.shape[1]) % multiple
    return np.pad(a, [(0, py), (0, px)] + [(0, 0)] * (a.ndim - 2))


def geometry(*, state, factor):
    fields = state["arrays"]
    config = state["configuration"]
    shape = (config["ny"], config["nx"])
    active = fields["active"].reshape(shape).astype(bool)

    def mean(*, array):
        return blocks(value=padded(a=array, multiple=factor), factor=factor)

    fraction = mean(array=active.astype(float))
    mask = fraction > 0.0

    def cell(*, name, weighted):
        a = fields[name].reshape(shape)
        reduced = mean(array=np.where(active, a, 0.0))
        if weighted:
            return np.divide(reduced, fraction, out=np.zeros_like(fraction), where=mask)
        return reduced

    q = mean(array=fields["q"].reshape(*shape, 3))
    y, x = np.where(fields["types_y"].reshape(shape[0] + 1, shape[1]) == 3)
    if np.any(fields["types_x"] == 3) or np.any(fields["types_y"] == 2) or np.any(fields["types_x"] == 2):
        raise ValueError("Orton boundary mapping requires its original north-facing stage outlets")
    if np.any(~active[y - 1, x]) or np.any(active[y, x]):
        raise ValueError("Unexpected stage orientation")
    north = np.pad(mask, ((0, 1), (0, 0))) & ~np.pad(mask[1:], ((0, 2), (0, 0)))
    cy, cx = np.where(north)
    coordinates = np.column_stack(((cy + 1) * factor, (cx + 0.5) * factor))
    tree = cKDTree(coordinates)
    shifts, idx = tree.query(np.column_stack((y, x + 0.5)))
    by = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=np.int32)
    by[cy[idx] + 1, cx[idx]] = 3
    sy = np.zeros_like(by, float)
    level = fields["stages_y"].reshape(shape[0] + 1, shape[1])[y, x]
    if np.ptp(level) > 0.0:
        raise ValueError("Expected constant Orton external stage")
    sy[by == 3] = level[0]
    data = {
        key: cell(name=key, weighted=key in ("bed", "manning"))
        for key in (
            "bed",
            "manning",
            "rain",
            "infil",
            "f0",
            "fd",
            "conductivity",
            "multiplier",
            "soil",
            "rain_sum",
        )
    }
    data["fd"][~mask] = 1.0
    for key in ("inflow", "outflow"):
        data[key] = cell(name=key, weighted=False) * factor**2
    data.update(q=q, active=mask, by=by, sy=sy)
    info = dict(
        fine_stage_faces=len(y),
        stage_faces=int(np.count_nonzero(by)),
        stage_width_m=int(5 * factor * np.count_nonzero(by)),
        max_stage_displacement_m=float(5 * shifts.max()),
        mean_stage_displacement_m=float(5 * shifts.mean()),
        active_area_m2=int(mask.sum() * (5 * factor) ** 2),
        added_footprint_m2=int(mask.sum() * (5 * factor) ** 2 - active.sum() * 25),
    )
    return data, info


def replay_solver(*, data, state, factor, method, device):
    config = state["configuration"].copy()
    mask = data["active"]
    config.update(
        ny=mask.shape[0],
        nx=mask.shape[1],
        dx=5.0 * factor,
        dy=5.0 * factor,
        flux_scheme="contact_hll" if method == "fv2_mc" else method,
    )
    factory = FV2Solver if method == "fv2_mc" else FullSWESolver
    s = factory(
        **config,
        bed_elevation=data["bed"],
        manning_n=data["manning"],
        active_mask=mask,
        device=device,
        execution_mode="dense",
    )
    s.set_boundary_type_qy(boundary_type=data["by"])
    s.set_boundary_stage_qy(stage=data["sy"])
    s.set_rainfall_hyetograph(times_s=state["rain_times"], rates_m_s=state["rain_rates"])
    q = data["q"].copy()
    q[q[..., 0] <= s.dry_depth, 1:] = 0.0
    s.q.assign(q.reshape(-1, 3))
    s._fields["maximum"].assign(q[..., 0].ravel())
    for name in (
        "rain",
        "infil",
        "f0",
        "fd",
        "conductivity",
        "multiplier",
        "soil",
        "rain_sum",
        "inflow",
        "outflow",
    ):
        s._fields[name].assign(data[name].ravel())
    s.initial_volume = state["initial_volume"]
    s.time, s.steps = state["time"], state["steps"]
    return s


def atomic_json(*, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pending")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def dry_initial(*, checkpoint):
    s = checkpoint_solver(scheme="contact_hll", device="cpu", path=checkpoint)
    state = s.capture_state()
    if state["initial_volume"] != 0.0:
        raise ValueError("Archived Orton storm must start dry")
    state["time"], state["steps"] = 0.0, 0
    for key in ("q", "soil", "rain_sum", "inflow", "outflow", "maximum"):
        state["arrays"][key].fill(0.0)
    return state


def release_initial(*, directory):
    """Rebuild the same dry start from the public v0.1.0 ordinary inputs."""
    from examples.orton_bradley.run_model import load_forcing
    from floodwarp.inputs import read_terrain, boundary_faces

    terrain = read_terrain(path=directory / "outputs/orton_bradley_dem_5m.tif", vertical_datum="NZVD2016")
    folder = directory / "inputs"
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in ("inputs_f64.bin", "roughness.asc"):
        with (folder / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["files"][name]:
                raise ValueError(f"Public Orton forcing field checksum mismatch: {name}")
    t, r, fields = load_forcing(path=folder / "inputs_f64.bin")
    mask = terrain.active
    np.testing.assert_array_equal(fields[0].astype(bool), mask)
    rough = np.loadtxt(folder / "roughness.asc", skiprows=6)[::-1]
    s = FullSWESolver(
        ny=mask.shape[0],
        nx=mask.shape[1],
        dx=terrain.dx,
        dy=terrain.dy,
        bed_elevation=terrain.elevation,
        manning_n=rough,
        active_mask=mask,
        device="cpu",
        gravity=9.80665,
        minimum_dt=1.0e-6,
        maximum_dt=1.0,
        dry_depth=0.0001,
        cfl=0.4,
        stage_boundary_momentum="discharge_extrapolation",
    )
    bx, by = boundary_faces(terrain=terrain)
    ports = np.r_[fields[5, 1:], np.zeros_like(fields[5, :1])].astype(bool)
    y, x = np.where(ports)
    by[y + 1, x] = 3
    s.set_boundary_type_qx(boundary_type=bx)
    s.set_boundary_type_qy(boundary_type=by)
    s.set_rainfall_hyetograph(times_s=t, rates_m_s=r)
    s.set_rainfall_spatial_multiplier(multiplier=fields[1])
    s.set_exponential_infiltration(f0=fields[2], fd=fields[3], conductivity=fields[4])
    return s.capture_state()


def forcing_check(*, state, data, dx):
    t, r = state["rain_times"], state["rain_rates"]
    depth = float(np.trapezoid(r, t))
    if (
        hashlib.sha256(r.tobytes()).hexdigest()
        != "f8936d9aa11897c15963c073015342fbbc15f34a0fb064f532e5d8039aab606a"
        or hashlib.sha256(t.tobytes()).hexdigest()
        != "0ec50d187829ab6857a834fbb6224fc289fd34eeb6a4a8559090e6132bf13334"
    ):
        raise ValueError("Archived HIRDS temporal shape changed")
    volume = float(depth * dx**2 * np.sum(data["multiplier"]))
    if (
        t[0] != 0
        or t[-1] != 48 * 3600
        or len(t) != 578
        or not np.isclose(depth, 0.27209564626437216, rtol=1.0e-13)
    ):
        raise ValueError("Archived Orton HIRDS hyetograph identity mismatch")
    if not np.isclose(volume, 3809237.0118338605, rtol=1.0e-12) or np.any(data["rain"]):
        raise ValueError("Storm volume or background rainfall mismatch")
    return dict(
        duration_h=48,
        ari_years=100,
        hirds_version=4,
        climate="historical",
        nodes=len(t),
        rate_nodes_sha256=hashlib.sha256(r.tobytes()).hexdigest(),
        node_times_sha256=hashlib.sha256(t.tobytes()).hexdigest(),
        mean_depth_m=depth,
        integrated_volume_m3=volume,
        peak_time_h=float(t[np.argmax(r)] / 3600),
        peak_rate_mm_h=float(r.max() * 3600000),
    )


def fields_summary(*, solver):
    q = solver.get_conservative_state()
    wet = q[..., 0] > 0.01
    mask = solver._active
    velocity = np.linalg.norm(q[wet, 1:], axis=-1) / q[..., 0][wet] if wet.any() else np.zeros(1)
    return dict(
        depth_max_m=float(q[..., 0].max()),
        peak_depth_m=float(solver.get_max_water_depth().max()),
        wet_area_m2=int(np.count_nonzero(wet) * solver.dx * solver.dy),
        max_envelope_wet_area_m2=int(
            np.count_nonzero(solver.get_max_water_depth() > 0.01) * solver.dx * solver.dy
        ),
        wet_speed_p95_m_s=float(np.percentile(velocity, 95)),
        minimum_depth_m=float(q[..., 0][mask].min()),
    )


def finish(*, history, solver, info, forcing, method, output, origin):
    if solver.time != forcing["duration_h"] * 3600 or not history or history[-1]["t"] != solver.time:
        raise ValueError("The design storm or its outlet hydrograph has not finished")
    budget = solver.cumulative_accounting()
    np.testing.assert_allclose(budget["rain"], forcing["integrated_volume_m3"], rtol=1.0e-10, atol=1.0e-7)
    if abs(budget["residual"]) > 1.0e-8 * budget["rain"]:
        raise ArithmeticError("Storm balance failed")
    rates = np.array([r["outflow_m3_s"] for r in history])
    peak = int(np.argmax(rates))
    row = dict(
        method=method,
        dx=solver.dx,
        time_s=solver.time,
        origin=origin,
        geometry=info,
        forcing=forcing,
        steps=int(solver.steps),
        peak_outflow_m3_s=float(rates[peak]),
        peak_interval_end_s=history[peak]["t"],
        ledger=budget,
        hydrograph=history,
        **fields_summary(solver=solver),
    )
    atomic_json(path=output, value=row)


def archive_baseline(*, directory, work, output, info, forcing, device):
    record = json.loads((directory / "record.json").read_text())
    path = select_checkpoints(directory=directory, hours=(48,))[0]
    with np.load(path) as a:
        config = json.loads(str(a["metadata"]))["configuration"]
        shape = config["ny"], config["nx"]
        s = FullSWESolver(
            **config,
            bed_elevation=a["bed"].reshape(shape),
            manning_n=a["manning"].reshape(shape),
            active_mask=a["active"].reshape(shape),
            device=device,
        )
    s.load_checkpoint(path=path)
    if config["flux_scheme"] != "contact_hll" or s.time != 48 * 3600:
        raise ValueError("Incorrect historical baseline")
    history = []
    previous = 0.0
    for ledger, stats in zip(record["ledger"], record["statistics"], strict=True):
        if ledger["seconds"] > s.time:
            break
        history.append(
            dict(
                t=ledger["seconds"],
                outflow_m3_s=(ledger["outflow_m3"] - previous) / 60.0,
                steps=stats["steps"],
                seconds=stats["loop_seconds"],
            )
        )
        previous = ledger["outflow_m3"]
    finish(
        history=history,
        solver=s,
        info=info,
        forcing=forcing,
        method="contact_hll",
        output=output,
        origin=str(path),
    )
    return path


def simulate(*, settings, folder, public, forcing, info, sample_s, checkpoint_hours, diagnostic_hours):
    s = replay_solver(**settings)
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "hour-48.npz"
    history_path = folder / "hydrograph.json"
    files = sorted(folder.glob("hour-*.npz"))
    history = []
    if files:
        s.load_checkpoint(path=files[-1])
        history = [r for r in json.loads(history_path.read_text()) if r["t"] <= s.time]
    previous = float(s._fields["outflow"].numpy().sum())
    total = 48 * 3600
    first = int(round(s.time / sample_s)) + 1
    for index in range(first, round(total / sample_s) + 1):
        target = index * sample_s
        method = settings["method"]
        track = target in diagnostic_hours and method in ("roe", "hllc")
        if track:
            probe = InteriorTransport(
                solver=s, edges=(0.001, 0.01, 0.1, 1.0), wet=0.01, channel_depth=0.1, channel_discharge=0.05
            )
        stats = s.run_device_loop(duration=target - s.time, track_flux=track)
        exported = float(s._fields["outflow"].numpy().sum())
        entry = dict(
            t=target,
            outflow_m3_s=(exported - previous) / sample_s,
            steps=stats["steps"],
            seconds=stats["loop_seconds"],
            dt_min=stats["dt_min_s"],
            dt_mean=stats["dt_mean_s"],
        )
        if track:
            entry["face_counts"] = stats["flux_outcomes"]
            entry["interior_transport"] = probe.results()
            del probe
        previous = exported
        history.append(entry)
        if target % 3600 == 0:
            hour = target // 3600
            atomic_json(path=history_path, value=history)
            if hour % checkpoint_hours == 0 or hour == 48:
                checkpoint = folder / f"hour-{hour:02d}.npz"
                s.save_checkpoint(path=checkpoint)
            print(f"{s.dx:g} m {method} hour {hour} of 48, export {exported:.1f} m3", flush=True)
    finish(
        history=history,
        solver=s,
        info=info,
        forcing=forcing,
        method=settings["method"],
        output=public,
        origin=str(completed),
    )
    return completed


def storm(
    *,
    checkpoint,
    release,
    work,
    directory,
    methods,
    factors,
    device,
    sample_s,
    checkpoint_hours,
    diagnostic_hours,
):
    archived = checkpoint.is_file()
    state = dry_initial(checkpoint=checkpoint) if archived else release_initial(directory=release)
    paths = {}
    for factor in factors:
        data, info = geometry(state=state, factor=factor)
        forcing = forcing_check(state=state, data=data, dx=5 * factor)
        for method in methods:
            name = f"{5 * factor}m_{method}"
            output = directory / (name + ".json")
            if archived and factor == 1 and method == "contact_hll":
                path = archive_baseline(
                    directory=checkpoint.parent,
                    work=work,
                    output=output,
                    info=info,
                    forcing=forcing,
                    device=device,
                )
            else:
                settings = dict(data=data, state=state, factor=factor, method=method, device=device)
                path = simulate(
                    settings=settings,
                    folder=work / name,
                    public=output,
                    forcing=forcing,
                    info=info,
                    sample_s=sample_s,
                    checkpoint_hours=checkpoint_hours,
                    diagnostic_hours=diagnostic_hours,
                )
            paths[factor, method] = path
    rows = []
    for (factor, method), path in paths.items():
        with np.load(path) as a:
            c = json.loads(str(a["metadata"]))["configuration"]
            q = a["q"].reshape(c["ny"], c["nx"], 3)
            maximum = a["maximum"].reshape(q.shape[:2])
        row = dict(dx=5 * factor, method=method)
        for key, pair in (
            ("vs_5m_contact", (1, "contact_hll")),
            ("vs_same_grid_contact", (factor, "contact_hll")),
            ("vs_same_method_5m", (1, method)),
        ):
            with np.load(paths[pair]) as a:
                c = json.loads(str(a["metadata"]))["configuration"]
                ratio = factor // pair[0]
                base = blocks(
                    value=padded(a=a["q"].reshape(c["ny"], c["nx"], 3), multiple=ratio), factor=ratio
                )
                maxbase = blocks(
                    value=padded(a=a["maximum"].reshape(c["ny"], c["nx"]), multiple=ratio), factor=ratio
                )
            row[key] = errors(result=q, expected=base, dx=5 * factor, wet_depth=0.01)
            row[key]["maximum_envelope_rmse_m"] = float(np.sqrt(np.mean((maximum - maxbase) ** 2)))
        rows.append(row)
    report_json(path=directory / "comparison.json", content=rows)


if __name__ == "__main__":
    work = Path("outputs/fv2_hirds48_v1")
    directory = Path("validation/fv2_review/hirds48")
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    release = Path("outputs/release/orton_bradley_v0.1.0")
    methods = ("contact_hll", "roe", "hllc", "fv2_mc")
    factors = (4, 2, 1)
    sample_s = 60
    checkpoint_hours = 2
    diagnostic_hours = {hour * 3600 for hour in (12, 24, 36, 48)}
    storm(
        checkpoint=checkpoint,
        release=release,
        work=work,
        directory=directory,
        methods=methods,
        factors=factors,
        device="cuda:0",
        sample_s=sample_s,
        checkpoint_hours=checkpoint_hours,
        diagnostic_hours=diagnostic_hours,
    )
