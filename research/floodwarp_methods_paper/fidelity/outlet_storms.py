"""Paired incised storm trajectories with a fixed physical gauge and explicit regional budgets."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from floodwarp.inputs import boundary_faces, terrain_from_array
from floodwarp.model import advance, create_model, load_checkpoint, save_checkpoint
from research.floodwarp_methods_paper import events, runner
from research.floodwarp_methods_paper.fidelity import recession
from research.floodwarp_methods_paper.terrain.generators import native_inputs
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def archived(*, directory):
    controls = {}
    for path in sorted((directory / "mean_storms").glob("*/result.json")):
        record = json.loads(path.read_text())
        spec = record["inputs"]["spec"]
        key = spec["method"], spec["dx"]
        if key in controls:
            raise ValueError(f"Ambiguous physical control for {key}")
        controls[key] = record
    return controls


def load_fields(*, directory, receipt, dx, extension):
    inp = receipt["inputs"]
    folder = directory.parents[2] / ".fidelity-restarts/native_inputs" / runner.digest(inp)
    native_inputs.verify(folder=folder, identity=inp)
    fields = {}
    for name in receipt["native_sha256"]:
        five = np.load(folder / f"{name}_5m.npy", mmap_mode="r")
        coarse = np.load(folder / f"{name}_10m.npy", mmap_mode="r")
        np.testing.assert_array_equal(coarse, mean_cells(values=five, factor=2))
        fields[name] = five if dx == 5 else coarse
    if extension:
        length, width = inp["dimensions_m"]
        # Continue the identical native valley and its initial moving channel. The
        # archived spatial storm has zero rainfall here. Soil capacity continues.
        tail = native_inputs.native_strip(
            x=np.arange(length, length + extension)[None, :] + .5,
            y=np.arange(width)[:, None] + .5, dimensions=inp["dimensions_m"],
            params=inp["parameters"], family=inp["landform"], storm=inp["storm"],
            options=inp["options"], channel_core_m=inp["moving_channel_min_depth_m"],
        )
        if np.any(tail["rain"]):
            raise ValueError("Downstream extension would add storm rainfall")
        for name, field in fields.items():
            five = mean_cells(values=tail[name], factor=5)
            extra = five if dx == 5 else mean_cells(values=five, factor=2)
            fields[name] = np.concatenate((field, extra), axis=1)
    return fields


def create(*, control, fields, dx, factor, device, horizon):
    data = control["data"]
    forcing = data["forcing"]
    method, mode, options = events.numerical(method=control["inputs"]["spec"]["method"])
    terrain = terrain_from_array(elevation=fields["bed"], dx=dx, dy=dx,
                                 row_order="south_to_north", vertical_datum="local")
    bx, by = boundary_faces(terrain=terrain)
    bx[:, -1] = 2
    times = list(forcing["rain_times_s"])
    rates = list(forcing["rain_rates_ms"])
    if times[-1] < horizon:
        if rates[-1] != 0:
            raise ValueError("Historical rainfall does not end at zero")
        times.append(horizon)
        rates.append(0.)
    model = create_model(
        terrain=terrain, solver_method=method, execution_mode=mode, device=device,
        manning_n=forcing["roughness"], gravity=data["numerical"]["gravity"],
        cfl=data["numerical"]["cfl"] * factor, initial_depth=fields["depth"],
        rainfall_series=(times, rates), infiltration=fields["infiltration"],
        boundary_qx=bx, boundary_qy=by, **options,
    )
    model.solver.set_momentum(hu=fields["hu"], hv=np.zeros_like(fields["hu"]))
    model.solver.set_rainfall_spatial_multiplier(multiplier=fields["rain"])
    return model


def observe(*, solver, gauge, tail_cells, mask, spatial):
    shape = solver.ny, solver.nx
    vol = solver._fields["volume_x"].numpy().reshape(shape[0], shape[1] + 1)
    columns = [gauge // 2, gauge - tail_cells, gauge, solver.nx]
    section = vol[:, columns].sum(axis=0)
    data = dict(zip(("interior", "tail_in", "gauge", "boundary"), section.tolist(), strict=True))
    h = solver.get_water_depth()
    soil = solver._fields["soil"].numpy().reshape(shape)
    rain = solver._fields["rain_sum"].numpy().reshape(shape)
    cell_area = solver.dx * solver.dy
    for name, region in (("footprint", slice(0, gauge)), ("extension", slice(gauge, None)),
                         ("last40m", slice(gauge-tail_cells, gauge))):
        for field, values in (("storage", h), ("infiltration", soil), ("rain", rain)):
            data[f"{name}_{field}_m3"] = float(values[:, region].sum() * cell_area)
    if spatial:
        orig = h[:, :gauge]
        maximum = solver.get_max_water_depth()[:, :gauge]
        if not np.isfinite(h).all() or h.min() < 0:
            raise ArithmeticError("Invalid water depth")
        data["infiltration_inside_rain_m3"] = float(soil[:, :gauge][mask].sum() * cell_area)
        data["infiltration_outside_rain_m3"] = float(soil[:, :gauge][~mask].sum() * cell_area)
        for cm in (5, 10):
            data[f"area_above_{cm}cm_m2"] = int((orig > cm/100).sum()) * cell_area
            data[f"ever_above_{cm}cm_m2"] = int((maximum > cm/100).sum()) * cell_area
    return data


def close_budgets(*, initial, row):
    for region, incoming, outgoing in (("footprint", None, "gauge"), ("extension", "gauge", "boundary"),
                                       ("last40m", "tail_in", "gauge")):
        rain = row[region + "_rain_m3"]
        supply = initial[region + "_storage_m3"] + rain
        residual = supply - row[region + "_storage_m3"] - row[region + "_infiltration_m3"] - row[outgoing]
        if incoming:
            residual += row[incoming]
        row[region + "_residual_m3"] = residual
        if abs(residual) > 1e-8 * max(1., supply, abs(row[outgoing])):
            raise ArithmeticError(f"{region} budget residual {residual} m3")


def trajectory(*, directory, control, dx, factor, extension, end, settings, stopping, run_dir, identity):
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dir = directory.parents[2] / ".fidelity-restarts/outlet_storms" / run_dir.name
    state_dir.mkdir(parents=True, exist_ok=True)
    fields = load_fields(directory=directory, receipt=control["inputs"]["terrain"], dx=dx, extension=extension)
    gauge = control["inputs"]["spec"]["dimensions"][0] // dx
    device = settings["device"]
    model = create(control=control, fields=fields, dx=dx, factor=factor, device=device,
                   horizon=stopping["forcing_hours"] * 3600)
    solver = model.solver
    kwargs = dict(solver=solver, gauge=gauge, tail_cells=settings["strip_m"]//dx,
                  mask=fields["rain"][:, :gauge] > 0)
    initial = observe(**kwargs, spatial=True)
    manifest = run_dir / "inputs.json"
    if not manifest.exists():
        recession.atomic_json(path=manifest, payload=dict(
            identity=identity, initial=initial, numerical=solver.numerical_configuration(),
            fields={k: array_sha(v) for k,v in fields.items()},
            hardware=runner.machine(root=directory.parents[2]),
        ))
    cadence = settings["interval_s"]
    window = stopping["checkpoint_hours"] * 3600
    past = []
    start = 0
    for t in range(window, end+1, window):
        part = run_dir / f"{t}.json"
        if not part.exists():
            break
        record = json.loads(part.read_text())
        if record["start_s"] != start:
            raise ValueError("Discontinuous trajectory archive")
        past.extend(record["frames"])
        start = t
    if start:
        load_checkpoint(model=model, path=state_dir / f"{start}.npz")
    previous = past[-1] if past else initial
    for stop in range(start + window, end+1, window):
        tick = perf_counter()
        frames = []
        for t in range(start + cadence, stop+1, cadence):
            wp.synchronize_device(device=device)
            step_start = perf_counter()
            stats = advance(model=model, end_time=float(t))
            wp.synchronize_device(device=device)
            wall = perf_counter()-step_start
            row = observe(**kwargs, spatial=t % settings["spatial_s"] == 0)
            capture, compile_s = stats["graph_capture_seconds"], stats.get("compilation_seconds", 0.)
            row.update(t_s=t, steps=stats["steps"], rejected=stats.get("rejected_cfl_trials", 0),
                       hydraulic_s=wall-capture-compile_s, graph_capture_s=capture,
                       compilation_s=compile_s, dt_min_s=stats["dt_min_s"], dt_max_s=stats["dt_max_s"],
                       outlet_m3s=(row["gauge"]-previous["gauge"])/cadence,
                       section_m3s=(row["interior"]-previous["interior"])/cadence)
            close_budgets(initial=initial, row=row)
            frames.append(row)
            previous = row
        save_checkpoint(model=model, path=state_dir / f"{stop}.npz")
        record = dict(start_s=start, end_s=stop, frames=frames, window_wall_s=perf_counter()-tick)
        recession.atomic_json(path=run_dir / f"{stop}.json", payload=record)
        print(run_dir.name, dx, factor, extension, stop/3600, "h, gauge export m3", row["gauge"], flush=True)
        past.extend(frames)
        start = stop
    passed, details = recession.settled(frames=past, storage=past[-1]["footprint_storage_m3"],
                                         supply=past[-1]["footprint_rain_m3"]+initial["footprint_storage_m3"],
                                         config=stopping)
    result = dict(end_s=end, recession_passed=passed, stations=details, frames=len(past),
                  hydraulic_s=sum(f["hydraulic_s"] for f in past), steps=sum(f["steps"] for f in past),
                  peak_m3s=max(f["outlet_m3s"] for f in past), final=past[-1])
    recession.atomic_json(path=run_dir / f"summary_{end}.json", payload=result)
    return result


def run(*, directory, settings, stopping, dry):
    controls = archived(directory=directory)
    root = directory.parents[2]
    cohort = directory / "outlet_storms"
    solver = runner.source_identity(root=root, study=directory.parent, stage="long-storms")
    cases = []
    for dx in settings["grids_m"]:
        for extension in settings["extensions_m"]:
            for method in settings["methods"]:
                record = controls[method, dx]
                for factor in ([1., .5] if dx == 5 and method == "adaptive" else [1.]):
                    identity = dict(settings=settings, stopping=stopping, numerical=solver, source=runner.checksum(Path(__file__)),
                                    dx=dx, extension_m=extension, cfl_factor=factor, method=method,
                                    terrain=runner.digest(record["inputs"]["terrain"]), forcing=record["data"]["forcing"],
                                    original_control=runner.digest(record["inputs"]))
                    dest = cohort / runner.digest(identity)[:24]
                    cases.append(dict(directory=directory, control=record, dx=dx, factor=factor, extension=extension,
                                      settings=settings, stopping=stopping, run_dir=dest, identity=identity))
                    print(method, dx, "m,", (record["inputs"]["spec"]["dimensions"][0]+extension)//dx,
                          "x", record["inputs"]["spec"]["dimensions"][1]//dx, "cells, CFL factor", factor,
                          "extension", extension, "m, archived windows", len(list(dest.glob("[0-9]*.json"))), flush=True)
    if dry:
        return
    for hour in range(settings["minimum_h"], stopping["forcing_hours"]+1, stopping["checkpoint_hours"]):
        results = []
        for case in cases:
            path = case["run_dir"] / f"summary_{hour*3600}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
            else:
                results.append(trajectory(**case, end=hour*3600))
        if all(r["recession_passed"] for r in results):
            return
    raise RuntimeError("Unresolved recession at the common forcing horizon")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "outlet_storm_settings.json").read_text())
    stopping = json.loads((directory / "recession_settings.json").read_text())
    options = set(sys.argv[1:])
    if options.difference({"--dry-run", "--resume"}):
        raise ValueError(f"Unknown options {options}")
    run(directory=directory, settings=settings, stopping=stopping, dry="--dry-run" in options)
