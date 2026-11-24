"""Replay selected native storms with ordered accepted timesteps outside the timing sample."""

import json
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import history, metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    native_matrix,
    parity,
    receipts,
    recession,
)
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def trace(*, model, frames, fields, spec, end, options, device):
    steps = sum(r["steps"] for r in frames)
    with history.history(solver=model.solver, capacity=steps) as values:
        actual = recession.advance_frames(
            model=model, spec=spec, start=0, end=end, options=options, device=device
        )
        times = values()
    history.checked(times=times, count=steps, duration=end)
    for key in ("t_s", "steps", "section_m3s", "outlet_m3s"):
        np.testing.assert_array_equal(
            [r[key] for r in actual], [r[key] for r in frames], err_msg=key
        )
    fresh = dict(
        state=array_sha(model.solver.get_conservative_state()),
        maximum=array_sha(model.solver.get_max_water_depth()),
    )
    for key in ("rain_sum", "soil", "volume_x", "volume_y", "dt_histogram"):
        fresh[key] = array_sha(model.solver._fields[key].numpy())
    for key in ("state", "maximum", "rain_sum", "soil", "volume_x", "volume_y"):
        if fields[key] != fresh[key]:
            raise ValueError("Ordered timestep replay changed final " + key)
    if "dt_histogram" in fields and fields["dt_histogram"] != fresh["dt_histogram"]:
        raise ValueError("Changed accepted timestep histogram")
    return times


def replay(*, directory, entry, design, stopping, config):
    primary = directory / entry["prefix_dir"]
    source = primary / "result.json"
    raw = json.loads(source.read_text())
    spec = raw["inputs"]["spec"]
    end = round(float(entry["end_h"]) * 3600)
    ident = entry["id"]
    numerical = runner.source_identity(
        root=directory.parents[2], study=directory.parent, stage="long-storms"
    )
    if raw["inputs"]["solver"] != numerical or spec.get("timing_repeat"):
        raise ValueError("Ordered timesteps require the current primary run")
    data = raw["data"]
    if ident != entry["parent_id"]:
        data = receipts.read(root=directory / "extended_runs", ident=ident, end=end)
    frames = data["frames"]
    fields = parity.fingerprints(
        directory=directory,
        item=dict(s=dict(hours=end / 3600), d=data, id=ident, path=primary),
    )
    if "q" in fields:
        fields.pop("q")
    definition = dict(
        experiment_id=ident,
        end_s=end,
        numerical=numerical,
        primary=runner.checksum(source),
        frames=runner.digest(
            [
                [r[k] for k in ("t_s", "steps", "outlet_m3s", "section_m3s")]
                for r in frames
            ]
        ),
        fields=fields,
        driver=runner.checksum(Path(__file__)),
        recorder=runner.checksum(Path(history.__file__)),
    )
    folder = directory / "native_step_history" / runner.digest(definition)[:24]
    proof = folder / "verified.json"
    file = folder / "accepted_dt.npz"
    if proof.is_file():
        result = json.loads(proof.read_text())
        if (
            result["definition"] != definition
            or runner.checksum(file) != result["sha256"]
        ):
            raise ValueError("Changed ordered timestep archive")
        return result
    inputs, bundle, terrain = native_matrix.physical_inputs(
        directory=directory, base=spec, config=config, design=design
    )
    if runner.digest(terrain) != entry["native_sha"]:
        raise ValueError("Different metre-grid input for ordered timestep replay")
    model, _, _ = native_matrix.prepare(
        spec=dict(spec, hours=stopping["forcing_hours"]),
        config=config,
        template=inputs["template"],
        device=stopping["device"],
        bundle=bundle,
        receipt=terrain,
    )
    times = trace(
        model=model,
        frames=frames,
        fields=fields,
        spec=spec,
        end=end,
        options=inputs["options"],
        device=stopping["device"],
    )
    folder.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(file, dt_s=times)
    result = dict(
        definition=definition,
        sha256=runner.checksum(file),
        accepted_dt_sha256=array_sha(times),
        accepted_steps=len(times),
        run_id=ident,
        policy=entry["method"],
        dx_m=int(entry["dx_m"]),
        end_s=end,
        replay_id=folder.name,
    )
    runner.write_json(path=proof, data=result)
    return result


def cpu_control(*, directory, config, design, storm, policies):
    archive = directory / "strong_storms"
    (primary,) = [
        r
        for r in metrics.csv_read(file=archive / "completed.csv")
        if r["storm"] == storm and r["method"] == "hancock_dense"
    ]
    folder = archive / primary["id"]
    original = json.loads((folder / "inputs.json").read_text())["identity"]
    spec, stopping = original["spec"], original["stopping"]
    numerical = runner.source_identity(
        root=directory.parents[2], study=directory.parent, stage="long-storms"
    )
    if original["numerical"] != numerical:
        raise ValueError("CPU control requires a corrected primary trajectory")
    end = round(float(primary["end_h"]) * 3600)
    windows = [
        json.loads(path.read_text())
        for path in sorted(folder.glob("[0-9]*.json"), key=window_end)
        if int(path.stem) <= end
    ]
    frames = [r for w in windows for r in w["frames"]]
    fields = dict(
        windows[-1]["checkpoint_fields"], state=windows[-1]["checkpoint_fields"]["q"]
    )
    source = runner.digest(
        dict(
            primary=primary,
            hashes=fields,
            driver=runner.checksum(Path(__file__)),
            recorder=runner.checksum(Path(history.__file__)),
        )
    )
    target = directory / "native_step_history" / source[:24]
    target.mkdir(parents=True, exist_ok=True)
    inputs, bundle, terrain = native_matrix.physical_inputs(
        directory=directory, base=spec, config=config, design=design
    )
    if runner.digest(terrain) != original["terrain"]:
        raise ValueError("CPU replay changed physical inputs")
    result = []
    for name in policies:
        # Use the existing CPU control's native input factory without publishing
        # a new primary measurement or altering the original storm receipt.
        effective = dict(
            spec, method=name, requested_method=name, hours=stopping["forcing_hours"]
        )
        model, _, _ = native_matrix.prepare(
            spec=effective,
            config=config,
            template=inputs["template"],
            device="cpu",
            bundle=bundle,
            receipt=terrain,
        )
        times = trace(
            model=model,
            frames=frames,
            fields=fields,
            spec=spec,
            end=end,
            options=inputs["options"],
            device="cpu",
        )
        path = target / (name + ".npz")
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["dt_s"], times)
        else:
            np.savez_compressed(path, dt_s=times)
        result.append(
            dict(
                primary=primary["id"],
                method=name,
                end_s=end,
                steps=len(times),
                accepted_dt_sha256=array_sha(times),
                sha256=runner.checksum(path),
            )
        )
        print(
            name, len(times), "accepted timesteps through", end / 3600, "h", flush=True
        )
    if len({r["accepted_dt_sha256"] for r in result}) != 1:
        raise ArithmeticError(
            "Different ordered CPU timesteps across execution policies"
        )
    runner.write_json(
        path=target / "verified.json", data=dict(definition=source, methods=result)
    )


def window_end(path):
    return int(path.stem)


def run(*, directory, config, design, stopping, case, spacings, methods, dry):
    if dry:
        for dx in spacings:
            print(
                case,
                dx,
                "m",
                methods,
                "ordered accepted timestep replay at matched recession endpoint",
            )
        return
    table = directory / "native_results/recession_execution_parity.csv"
    pairs = [
        r
        for r in metrics.csv_read(file=table)
        if r["cohort"] == "complete-native-metre"
        and r["case"] == case
        and int(r["dx"]) in spacings
    ]
    registered = {(int(r["dx"]), r["other_method"]) for r in pairs}
    if registered != {
        (dx, m) for dx in spacings for m in methods if m != "hancock_dense"
    }:
        raise ValueError("Missing complete hydraulic peers for ordered timestep replay")
    peers = {ident for r in pairs for ident in (r["dense_id"], r["other_id"])}
    runs = {}
    for table in sorted(
        (directory / "native_matrix" / case).glob("*/complete_events.csv")
    ):
        for row in metrics.csv_read(file=table):
            if row["id"] in peers:
                runs[row["id"]] = row
    if runs.keys() != peers:
        raise ValueError("Missing primary archive for ordered timestep replay")
    verified = {
        ident: replay(
            directory=directory,
            entry=row,
            config=config,
            design=design,
            stopping=stopping,
        )
        for ident, row in runs.items()
    }
    results = []
    for pair in pairs:
        a, b = verified[pair["dense_id"]], verified[pair["other_id"]]
        if a["accepted_dt_sha256"] != b["accepted_dt_sha256"]:
            raise ArithmeticError(
                "Different ordered timesteps between dense and alternate Hancock"
            )
        results.append(
            dict(
                case=case,
                dx_m=pair["dx"],
                end_h=pair["end_h"],
                policy=pair["other_method"],
                dense_id=a["run_id"],
                peer_id=b["run_id"],
                accepted_steps=a["accepted_steps"],
                dense_replay=a["replay_id"],
                peer_replay=b["replay_id"],
                accepted_dt_sha256=a["accepted_dt_sha256"],
                accepted_timestep_history_equal=True,
                final_state_equal=True,
                interval_transport_equal=True,
            )
        )
    metrics.csv_write(file=directory / "native_step_history/parity.csv", rows=results)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    stopping = json.loads((directory / "recession_settings.json").read_text())
    case = "plain-709-spatial-1000x1000"
    spacings = (1, 10)
    methods = ("hancock_dense", "hancock_adaptive", "hancock_auto")
    options = set(sys.argv[1:])
    if options.difference({"--dry-run", "--resume", "--cpu-control"}):
        raise ValueError(f"Unknown history options {options}")
    if "--cpu-control" in options:
        if "--dry-run" in options:
            print(
                "CPU dense/tiled accepted timestep comparison on the completed 10 m moderate plain"
            )
        else:
            cpu_control(
                directory=directory,
                config=config,
                design=design,
                storm="moderate",
                policies=("hancock_dense", "hancock_adaptive"),
            )
    else:
        run(
            directory=directory,
            config=config,
            design=design,
            stopping=stopping,
            case=case,
            spacings=spacings,
            methods=methods,
            dry="--dry-run" in options,
        )
