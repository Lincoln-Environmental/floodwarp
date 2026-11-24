"""Manually run the archived Orton Bradley 48 h HIRDS storm with matched physical sections."""

import fnmatch
import getopt
import json
from contextlib import contextmanager
from math import ceil
from pathlib import Path
import sys
from time import perf_counter, time_ns
import traceback

import numpy as np
import warp as wp

from floodwarp.fv2_solver import FV2Solver, HancockSolver
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import orton_controls, recession
from validation import fv2_orton_matrix as archive


def discharge_volumes(*, solver, gauges):
    field = solver._fields["volume_y"].numpy().reshape(solver.ny + 1, solver.nx)
    positions = np.array(gauges) / solver.dy
    if not np.array_equal(positions, np.rint(positions)) or positions.max() >= len(
        field
    ):
        raise ValueError("Gauges must lie on represented faces")
    return field[positions.astype(int)].sum(axis=1)


def observation_checksums(*, folder, hours, checkpoint_h):
    paths = [folder / "inputs.json"]
    paths.extend(folder / f"hour_{h}.json" for h in range(checkpoint_h, hours + 1, checkpoint_h))
    return {p.name: runner.checksum(p) for p in paths}


def completed_attempt(*, folder):
    return (folder / "complete.json").is_file() and any(folder.glob("attempt_*/completed.json"))


def read_complete(*, folder, identity):
    result = folder / "complete.json"
    entry = json.loads(result.read_text())
    if (entry["id"], entry["dx_m"], entry["method"]) != (
        runner.digest(identity)[:24], identity["dx"], identity["method"]
    ) or json.loads((folder / "inputs.json").read_text())["identity"] != identity:
        raise ValueError("Transfer result differs from source and physical contract")
    actual = observation_checksums(folder=folder, hours=entry["duration_h"],
                                   checkpoint_h=identity["settings"]["checkpoint_h"])
    if runner.digest(actual) != entry["observation_sha256"]:
        raise ValueError("Changed transfer observations or input manifest")
    if not completed_attempt(folder=folder):
        raise ValueError("Transfer summary has no completed execution receipt")
    for receipt in folder.glob("attempt_*/completed.json"):
        started = json.loads(receipt.with_name("started.json").read_text())
        if started["id"] != entry["id"]:
            raise ValueError("Completed transfer attempt belongs to a different forecast")
        saved = json.loads(receipt.read_text())
        if saved["summary_sha256"] != runner.checksum(result):
            raise ValueError("Changed completed transfer summary")
    return entry


def verify_index(*, directory):
    table = directory / "transfer.csv"
    if not table.exists():
        return 0
    seen = set()
    for row in metrics.csv_read(file=table):
        ident = row["id"]
        if ident in seen:
            raise ValueError("Duplicate forecast in transfer index")
        folder = directory / ident
        identity = json.loads((folder / "inputs.json").read_text())["identity"]
        entry = read_complete(folder=folder, identity=identity)
        if row != {key: str(value) if value is not None else "" for key, value in entry.items()}:
            raise ValueError("Transfer index differs from completed forecast " + ident)
        seen.add(ident)
    return len(seen)


def simulate(
    *, directory, settings, state, data, geometry, forcing, dx, method, identity
):
    dest = directory / "orton_transfer" / runner.digest(identity)[:24]
    dest.mkdir(parents=True, exist_ok=True)
    cache = directory.parents[2] / ".fidelity-restarts/orton_transfer" / dest.name
    cache.mkdir(parents=True, exist_ok=True)
    result = dest / "complete.json"
    if completed_attempt(folder=dest):
        return read_complete(folder=dest, identity=identity)
    names = {
        "fv1": ("contact_hll", "sparse"),
        "hancock": ("fv2_adaptive", "tiles"),
        "rk2": ("fv2_adaptive", "dense"),
    }
    label, mode = names[method]
    setup_start = perf_counter()
    s = archive.replay_solver(
        data=data,
        state=state,
        factor=dx // 5,
        method=label,
        device=settings["device"],
        execution_mode=mode,
        fv2_class=HancockSolver if method == "hancock" else FV2Solver,
    )
    actual = s.numerical_configuration()
    if (
        actual["solver_method"]
        != {"fv1": "fv1", "hancock": "fv1_hancock", "rk2": "fv2"}[method]
    ):
        raise ValueError("Orton transfer formulation mismatch")
    manifest = dest / "inputs.json"
    if not manifest.exists():
        recession.atomic_json(
            path=manifest,
            payload=dict(
                identity=identity,
                numerical=actual,
                policy=mode,
                hardware=runner.machine(root=directory.parents[2]),
                forcing=forcing,
                geometry=geometry,
                gauges_above_southern_origin_m=settings["gauges_m"],
                setup_s=perf_counter() - setup_start,
            ),
        )
    interval = settings["cadence_s"]
    records = []
    previous_in = previous_out = 0.0
    initial = discharge_volumes(solver=s, gauges=settings["gauges_m"])
    for hour in range(
        settings["checkpoint_h"], forcing["duration_h"] + 1, settings["checkpoint_h"]
    ):
        path = dest / f"hour_{hour}.json"
        checkpoint = cache / f"hour_{hour}.npz"
        if path.exists():
            record = json.loads(path.read_text())
            if runner.checksum(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("Orton transfer checkpoint changed")
            s.load_checkpoint(path=checkpoint)
            records.extend(record["frames"])
            previous_in, previous_out = (
                records[-1]["boundary_in_m3"],
                records[-1]["boundary_out_m3"],
            )
            initial = discharge_volumes(solver=s, gauges=settings["gauges_m"])
            continue
        new = []
        tick = perf_counter()
        for target in range(int(s.time) + interval, hour * 3600 + 1, interval):
            wp.synchronize_device(device=s.device)
            start = perf_counter()
            info = s.run_device_loop(duration=target - s.time)
            wp.synchronize_device(device=s.device)
            elapsed = perf_counter() - start
            boundary_out = float(s._fields["outflow"].numpy().sum())
            boundary_in = float(s._fields["inflow"].numpy().sum())
            net = boundary_out - boundary_in
            current = discharge_volumes(solver=s, gauges=settings["gauges_m"])
            capture = info["graph_capture_seconds"]
            compile_s = info.get("compilation_seconds", 0.0)
            new.append(
                dict(
                    t_s=target,
                    net_outlet_m3s=(net - previous_out + previous_in) / interval,
                    boundary_net_m3=net,
                    outward_m3s=(boundary_out - previous_out) / interval,
                    inward_m3s=(boundary_in - previous_in) / interval,
                    boundary_in_m3=boundary_in,
                    boundary_out_m3=boundary_out,
                    gauge_m3s=((current - initial) / interval).tolist(),
                    steps=info["steps"],
                    retries=info.get("rejected_cfl_trials", 0),
                    dt_min_s=info["dt_min_s"],
                    hydraulic_s=elapsed - capture - compile_s,
                    graph_capture_s=capture,
                    compilation_s=compile_s,
                    pool_mib=wp.get_mempool_used_mem_current(device=s.device) / 2**20,
                )
            )
            previous_out, previous_in, initial = boundary_out, boundary_in, current
        s.save_checkpoint(path=checkpoint)
        recession.atomic_json(
            path=path,
            payload=dict(
                frames=new,
                checkpoint_sha256=runner.checksum(checkpoint),
                window_wall_s=perf_counter() - tick,
            ),
        )
        records.extend(new)
        print(
            "Orton Bradley",
            dx,
            "m",
            method,
            hour,
            "of",
            forcing["duration_h"],
            "h",
            flush=True,
        )
    ledger = s.cumulative_accounting()
    h = s.get_water_depth()
    maximum = s.get_max_water_depth()
    np.testing.assert_allclose(
        ledger["rain"], forcing["integrated_volume_m3"], rtol=1e-9
    )
    if (
        h.min() < 0
        or not np.isfinite(h).all()
        or abs(ledger["residual"]) > 1e-8 * ledger["rain"]
    ):
        raise ArithmeticError("Orton transfer failed conservation or positivity")
    peak = max(records, key=peak_flow)
    times = np.array([r["t_s"] for r in records])
    np.testing.assert_array_equal(
        times, np.arange(interval, forcing["duration_h"] * 3600 + 1, interval)
    )
    window_times = [
        json.loads(p.read_text())["window_wall_s"] for p in dest.glob("hour_*.json")
    ]
    elapsed = json.loads(manifest.read_text())["setup_s"] + sum(window_times)
    answer = dict(
        id=dest.name,
        method=method,
        dx_m=dx,
        policy=mode,
        peak_m3s=peak["net_outlet_m3s"],
        peak_end_s=peak["t_s"],
        storage_m3=float(h.sum() * dx**2),
        steps=sum(r["steps"] for r in records),
        retries=sum(r["retries"] for r in records),
        end_to_end_solver_s=elapsed,
        initial_m3=float(s.initial_volume),
        duration_h=forcing["duration_h"],
        interval_s=interval,
        minimum_dt_s=min(r["dt_min_s"] for r in records),
        mean_dt_s=times[-1] / sum(r["steps"] for r in records),
        hydraulic_s=sum(r["hydraulic_s"] for r in records),
        sampled_peak_pool_mib=max(r["pool_mib"] for r in records),
        **ledger,
    )
    answer["mean_dt_s"] = float(answer["mean_dt_s"])
    for index, position in enumerate(settings["gauges_m"]):
        series = np.array([r["gauge_m3s"][index] for r in records])
        top = int(np.argmax(series))
        answer[f"gauge_{position}m_peak_m3s"] = float(series[top])
        answer[f"gauge_{position}m_peak_end_s"] = int(times[top])
        answer[f"gauge_{position}m_net_volume_m3"] = float(series.sum() * interval)
    for cm in settings["thresholds_cm"]:
        answer[f"final_area_{cm}cm_m2"] = int((h > cm / 100).sum()) * dx**2
        answer[f"maximum_area_{cm}cm_m2"] = int((maximum > cm / 100).sum()) * dx**2
    answer["observation_sha256"] = runner.digest(observation_checksums(
        folder=dest, hours=forcing["duration_h"], checkpoint_h=settings["checkpoint_h"],
    ))
    if result.exists():
        if answer != json.loads(result.read_text()):
            raise ValueError("Restored transfer observations disagree with the unconfirmed summary")
    else:
        recession.atomic_json(path=result, payload=answer)
    return answer


def peak_flow(frame):
    return frame["net_outlet_m3s"]


@contextmanager
def measured_attempt(*, folder, machine, ident):
    attempt = folder / f"attempt_{time_ns()}"
    attempt.mkdir(parents=True)
    resume = any(folder.glob("hour_*.json"))
    restoring_summary = (folder / "complete.json").is_file()
    start = perf_counter()
    runner.write_json(path=attempt / "started.json", data=dict(id=ident, hardware=machine, resumed=resume, restoring_summary=restoring_summary))
    try:
        yield
    except BaseException:
        runner.write_json(path=attempt / "failed.json",
                          data=dict(seconds=perf_counter() - start, traceback=traceback.format_exc()))
        raise
    runner.write_json(path=attempt / "completed.json", data=dict(
        seconds=perf_counter() - start, uninterrupted_from_start=not resume,
        restored_unconfirmed_summary=restoring_summary,
        summary_sha256=runner.checksum(folder / "complete.json"),
    ))


def inventory(*, directory, identity, settings, native_shape, method, spacing, case):
    rows = []
    for dx in settings["grids_m"]:
        for candidate in settings["methods"]:
            physical_case = "orton-bradley-48h-hirds"
            if (not fnmatch.fnmatchcase(physical_case, case)
                    or method not in (None, candidate) or spacing not in (None, dx)):
                continue
            contract = dict(identity, dx=dx, method=candidate)
            dest = directory / "orton_transfer" / runner.digest(contract)[:24]
            rows.append(dict(
                identity=contract, case=physical_case, grid_m=dx, method=candidate,
                nx=ceil(native_shape[1] / (dx // 5)), ny=ceil(native_shape[0] / (dx // 5)),
                complete=completed_attempt(folder=dest),
                archived_windows=len(list(dest.glob("hour_*.json"))), id=dest.name,
            ))
    return rows


def run(*, directory, settings, arguments):
    import rasterio

    flags, rest = getopt.getopt(arguments, "", ["dry-run", "resume", "case=", "method=", "dx="])
    if rest:
        raise ValueError(f"Unknown transfer arguments {rest}")
    options = dict(flags)
    release = directory.parents[2] / settings["release"]
    identity = dict(
        source=runner.checksum(Path(__file__)),
        forcing_reader=runner.checksum(directory.parents[2] / "examples/orton_bradley/run_model.py"),
        inputs=runner.checksum(release / "inputs/manifest.json"),
        dem=runner.checksum(release / "outputs/orton_bradley_dem_5m.tif"),
        settings=settings,
        numerical=runner.source_identity(
            root=directory.parents[2], study=directory.parent, stage="long-storms"
        ),
    )
    with rasterio.open(release / "outputs/orton_bradley_dem_5m.tif") as dem:
        if (not dem.crs or not dem.crs.is_projected or dem.crs.linear_units != "metre"
                or (dem.transform.a, dem.transform.b, dem.transform.d, dem.transform.e) != (5, 0, 0, -5)):
            raise ValueError("Orton gauges require the projected north-up native 5 m DEM")
        identity["spatial_reference"] = dict(crs=dem.crs.to_string(), west_m=dem.bounds.left,
                                             south_m=dem.bounds.bottom, gauge_northings_m=[
                                                 dem.bounds.bottom + g for g in settings["gauges_m"]
                                             ])
        plans = inventory(
            directory=directory, identity=identity, settings=settings, native_shape=dem.shape,
            method=options.get("--method"), spacing=float(options["--dx"]) if "--dx" in options else None,
            case=options.get("--case", "*"),
        )
    if not plans:
        raise ValueError("No Orton Bradley transfer configurations match the selection")
    print(f"{len(plans)} configurations, 48 h HIRDS design forcing, fixed physical gauges, external transfer", flush=True)
    for plan in plans:
        print(plan["id"], plan["method"], plan["grid_m"], "m,", plan["nx"], "×", plan["ny"],
              "cells,", plan["archived_windows"], "checkpoint windows, complete", plan["complete"], flush=True)
    if "--dry-run" in options:
        return
    machine = runner.machine(root=directory.parents[2])
    state = archive.release_initial(directory=release)
    for dx in dict.fromkeys(p["grid_m"] for p in plans):
        data, geometry = archive.geometry(state=state, factor=dx // 5)
        forcing = archive.forcing_check(state=state, data=data, dx=dx)
        for plan in plans:
            if plan["grid_m"] != dx or plan["complete"]:
                continue
            folder = directory / "orton_transfer" / plan["id"]
            if (folder / "inputs.json").is_file():
                saved = json.loads((folder / "inputs.json").read_text())
                if saved["identity"] != plan["identity"] or any(
                    machine[key] != saved["hardware"][key] for key in ("gpu", "warp", "python", "precision", "cpu_model")
                ):
                    raise ValueError("Transfer restart provenance differs from its original launch")
            with measured_attempt(folder=folder, machine=machine, ident=plan["id"]):
                simulate(directory=directory, settings=settings, state=state, data=data, geometry=geometry,
                         forcing=forcing, dx=dx, method=plan["method"], identity=plan["identity"])
        # Recreate the index from every completed matching contract, including
        # peers outside the current manual selection.
        root = directory / "orton_transfer"
        rows = []
        for grid in settings["grids_m"]:
            for candidate in settings["methods"]:
                contract = dict(identity, dx=grid, method=candidate)
                key = runner.digest(contract)[:24]
                result = root / key / "complete.json"
                if completed_attempt(folder=result.parent):
                    rows.append(read_complete(folder=result.parent, identity=contract))
        metrics.csv_write(file=root / "transfer.csv", rows=rows)
        orton_controls.compare(release=release, destination=root, settings=settings, rows=rows)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "orton_transfer_settings.json").read_text())
    run(directory=directory, settings=settings, arguments=sys.argv[1:])
