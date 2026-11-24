"""Independent CPU hydrographs on the same dry plain under widespread six-hour rainfall."""

import json
from pathlib import Path

import numpy as np
from floodwarp.model import diagnostics, load_checkpoint, save_checkpoint
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_matrix, recession
from research.floodwarp_methods_paper.plan import read_config


def forecast(*, directory, spec, design, config, settings, stopping):
    inputs, bundle, terrain = native_matrix.physical_inputs(
        directory=directory, base=spec, config=config, design=design
    )
    identity = dict(
        spec=spec,
        terrain=runner.digest(terrain),
        settings=settings,
        stopping=stopping,
        numerical=runner.source_identity(
            root=directory.parents[2], study=directory.parent, stage="long-storms"
        ),
        source={
            p.name: runner.checksum(p)
            for p in (
                Path(__file__),
                Path(native_matrix.__file__),
                Path(native_matrix.native_event.__file__),
                Path(recession.__file__),
                Path(native_matrix.hydraulics.__file__),
            )
        },
    )
    folder = directory / "strong_storms" / runner.digest(identity)[:24]
    folder.mkdir(parents=True, exist_ok=True)
    model, prescribed, forcing = native_matrix.prepare(
        spec=dict(spec, hours=stopping["forcing_hours"]),
        bundle=bundle,
        receipt=terrain,
        template=inputs["template"],
        config=config,
        device=settings["device"],
    )
    contract = folder / "inputs.json"
    if contract.is_file():
        if json.loads(contract.read_text())["identity"] != identity:
            raise ValueError("Changed strong-storm inputs")
    else:
        runner.write_json(
            path=contract,
            data=dict(
                identity=identity,
                prescribed_m3=prescribed,
                hardware=runner.machine(root=directory.parents[2]),
                physical_fields=terrain["native_sha256"],
                forcing=forcing,
            ),
        )
    return dict(
        model=model,
        options=inputs["options"],
        spec=spec,
        folder=folder,
        prescribed=prescribed,
        frames=[],
        start=0,
        budget=None,
    )


def state_signature(*, path):
    with np.load(path, allow_pickle=False) as state:
        return json.loads(state["_direct_manifest"].item())["sha256"]


def advance_window(*, forecast, end, directory, settings):
    run = forecast
    folder, start, model = run["folder"], run["start"], run["model"]
    archive = folder / f"{end}.json"
    cache = directory.parents[2] / ".fidelity-restarts/strong_storms" / folder.name
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"{end}.npz"
    if archive.exists() and checkpoint.is_file():
        record = json.loads(archive.read_text())
        if (
            record["start_s"] != start
            or state_signature(path=checkpoint) != record["checkpoint_fields"]
        ):
            raise ValueError("CPU storm restart or parent window changed")
        load_checkpoint(model=model, path=checkpoint)
    else:
        frames = recession.advance_frames(
            model=model,
            spec=run["spec"],
            start=start,
            end=end,
            options=run["options"],
            device=settings["device"],
        )
        budget = diagnostics(model=model)
        if abs(budget["residual_m3"]) > 1e-8 * max(1.0, run["prescribed"]):
            raise ArithmeticError("CPU storm failed water balance")
        np.testing.assert_allclose(
            budget["rain"], run["prescribed"], rtol=1e-9, atol=1e-8
        )
        save_checkpoint(model=model, path=checkpoint)
        record = dict(
            start_s=start,
            end_s=end,
            frames=frames,
            budget=budget,
            checkpoint_fields=state_signature(path=checkpoint),
        )
        if archive.exists():
            previous = json.loads(archive.read_text())
            recession.verify_prefix(old=previous, new=frames)
            if previous["checkpoint_fields"] != record["checkpoint_fields"]:
                raise ValueError(
                    "Portable CPU replay changed the archived endpoint state"
                )
            for field in (
                "storage_m3",
                "infiltration",
                "boundary_outflow",
                "residual_m3",
            ):
                np.testing.assert_allclose(
                    budget[field], previous["budget"][field], rtol=1e-10, atol=1e-7
                )
        else:
            runner.write_json(path=archive, data=record)
    if model.solver.time != end:
        raise ArithmeticError("CPU storm stopped before the observation time")
    run["start"] = end
    run["frames"].extend(record["frames"])
    run["budget"] = record["budget"]


def run(*, directory, settings, config, design, stopping):
    output = []
    for event in settings["events"]:
        specs = [
            s
            for s in native_matrix.specifications(
                study=directory.parent, config=config, design=design
            )
            if s["landform"] == settings["family"]
            and s["seed"] == settings["seed"]
            and s["dx"] == settings["dx_m"]
            and s["storm"] == event
            and s["requested_method"] in settings["methods"]
            and s["factor"] == 1
        ]
        if {s["requested_method"] for s in specs} != set(settings["methods"]):
            raise ValueError("Missing CPU control configuration")
        peers = [
            forecast(
                directory=directory,
                spec=s,
                design=design,
                config=config,
                settings=settings,
                stopping=stopping,
            )
            for s in specs
        ]
        for h in range(
            settings["first_check_h"],
            stopping["forcing_hours"] + 1,
            stopping["checkpoint_hours"],
        ):
            checks = []
            rows = []
            for peer in peers:
                advance_window(
                    forecast=peer, end=h * 3600, directory=directory, settings=settings
                )
                d = peer["budget"]
                ok, stations = recession.settled(
                    frames=peer["frames"],
                    storage=d["storage_m3"],
                    supply=peer["prescribed"],
                    config=stopping,
                )
                checks.append(ok)
                q = np.array([f["outlet_m3s"] for f in peer["frames"]])
                times = np.array([f["t_s"] for f in peer["frames"]])
                peak = int(np.argmax(q))
                volume = float(q @ np.diff(times, prepend=0))
                np.testing.assert_allclose(
                    volume,
                    d["boundary_outflow"] - d["boundary_inflow"],
                    atol=1e-7,
                    rtol=1e-10,
                )
                rows.append(
                    dict(
                        case=peer["spec"]["case"],
                        storm=event,
                        method=peer["spec"]["requested_method"],
                        dx_m=settings["dx_m"],
                        device=settings["device"],
                        id=peer["folder"].name,
                        end_h=h,
                        rain_m3=d["rain"],
                        infiltration_m3=d["infiltration"],
                        storage_m3=d["storage_m3"],
                        outlet_m3=volume,
                        runoff_ratio=volume / d["rain"],
                        residual_m3=d["residual_m3"],
                        peak_m3s=float(q[peak]),
                        peak_interval_start_s=times[peak] - peer["spec"]["interval"],
                        peak_interval_end_s=times[peak],
                        both_stations_passed=ok,
                    )
                )
                print(
                    event,
                    peer["spec"]["method"],
                    h,
                    "h, outlet peak",
                    float(q[peak]),
                    "m3/s, both stations passed",
                    ok,
                    flush=True,
                )
            if all(checks):
                output.extend(rows)
                break
        else:
            raise ValueError(
                "Strong storm still has unresolved discharge at maximum horizon"
            )
        metrics.csv_write(file=directory / "strong_storms/completed.csv", rows=output)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "strong_storm_settings.json").read_text())
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    stopping = json.loads((directory / "recession_settings.json").read_text())
    run(
        directory=directory,
        settings=settings,
        config=config,
        design=design,
        stopping=stopping,
    )
