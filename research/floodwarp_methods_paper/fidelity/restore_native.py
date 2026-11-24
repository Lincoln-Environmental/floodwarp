"""Reconstruct workstation-only incised checkpoints from verified native inputs when absent."""

from functools import partial
from pathlib import Path
import json
from unittest.mock import patch

import numpy as np
from floodwarp.model import diagnostics, load_checkpoint, save_checkpoint

from research.floodwarp_methods_paper import events, metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    continuation,
    native_event,
    native_products,
    recession,
)
from research.floodwarp_methods_paper.terrain.generators import native_inputs
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def prefix_index(*, root):
    paths = {}
    for file in (root / "prefixes").glob("*/state.npz"):
        with np.load(file) as state:
            keys = json.loads(state["_direct_manifest"].item())["sha256"]
            paths[keys.get("q"), keys.get("volume_x"), keys.get("soil")] = file
    return paths


def verify_fields(*, solver, expected):
    fields = dict(
        state=solver.get_conservative_state(), maximum=solver.get_max_water_depth()
    )
    for key in ("volume_x", "volume_y", "soil", "rain_sum"):
        fields[key] = solver._fields[key].numpy()
    for key, value in fields.items():
        if array_sha(value) != expected[key]:
            raise ValueError("Reconstructed checkpoint differs in " + key)


def rebuild(*, directory, prefix, series, root, settings):
    spec = prefix["s"]
    archived = json.loads((prefix["path"] / "result.json").read_text())["inputs"]
    current = runner.source_identity(
        root=directory.parents[2], study=directory.parent, stage="long-storms"
    )
    if (
        current != archived["solver"]
        or runner.checksum(Path(native_event.__file__)) != archived["factory"]
    ):
        raise ValueError(
            "Reconstruct native checkpoints using the source revision in the original receipt"
        )
    design = json.loads((directory / "mean_settings.json").read_text())
    physical = archived["physical"]
    folder, receipt = native_inputs.generate(
        spec=spec,
        inputs=physical,
        cache_root=root / "native_inputs",
        strip_rows=design["strip_rows"],
        include_native=1 in archived["terrain"]["inputs"]["grids_m"],
        channel_core_m=design["channel_core_min_depth_m"],
    )
    if receipt != archived["terrain"]:
        raise ValueError("Native fields differ from the measured forecast")
    factory = partial(native_event.create, bundle=folder, receipt=receipt)
    with patch.object(events, "create_event", factory):
        _, model, _, _, options = recession.build(
            it=prefix, study=directory.parent, config=physical["cfg"], settings=settings
        )
    end = round(spec["hours"] * 3600)
    start_file = (
        continuation.seed_path(directory=directory, model=model, end=end) / "state.npz"
    )
    final = root / series["id"] / f"{series['end_s']}.npz"
    if start_file.is_file():
        load_checkpoint(model=model, path=start_file)
        verify_fields(solver=model.solver, expected=prefix["d"]["hashes"])
    else:
        frames = recession.advance_frames(
            model=model,
            spec=spec,
            start=0,
            end=end,
            options=options,
            device=settings["device"],
        )
        recession.verify_prefix(old=prefix["d"], new=frames)
        verify_fields(solver=model.solver, expected=prefix["d"]["hashes"])
        start_file.parent.mkdir(exist_ok=True, parents=True)
        save_checkpoint(model=model, path=start_file)
    frame_path = start_file.with_name("prefix.json")
    if not frame_path.is_file():
        runner.write_json(
            path=frame_path,
            data=dict(
                frames=prefix["d"]["frames"],
                extended_zero_rain_coverage_h=settings["forcing_hours"],
            ),
        )
    if not final.is_file():
        frames = recession.advance_frames(
            model=model,
            spec=spec,
            start=end,
            end=series["end_s"],
            options=options,
            device=settings["device"],
        )
        recession.verify_prefix(
            old=dict(frames=[f for f in series["frames"] if f["t_s"] > end]), new=frames
        )
        actual = diagnostics(model=model)
        for key, name in (
            ("storage_m3", "storage_m3"),
            ("rain", "rain_m3"),
            ("infiltration", "infiltration_m3"),
        ):
            np.testing.assert_allclose(actual[key], series[name], rtol=2e-11, atol=1e-7)
        final.parent.mkdir(exist_ok=True, parents=True)
        save_checkpoint(model=model, path=final)
    return final


def run(*, directory):
    root = directory.parents[2] / ".fidelity-restarts"
    paths = prefix_index(root=root)
    settings = json.loads((directory / "recession_settings.json").read_text())
    prefixes = native_products.prefixes(directory=directory)
    outputs = metrics.csv_read(file=directory / "mean_storms/complete_events.csv")
    if not outputs:
        raise ValueError(
            "Complete the matched area-mean storm before rebuilding its checkpoints"
        )
    for row in outputs:
        ident = row["series_id"]
        series = json.loads(
            (directory / "extended_runs" / ident / "series.json").read_text()
        )
        prefix = prefixes[series["parent_id"]]
        h = prefix["d"]["hashes"]
        final = root / ident / f"{series['end_s']}.npz"
        if (h["state"], h["volume_x"], h["soil"]) not in paths or not final.is_file():
            rebuild(
                directory=directory,
                prefix=prefix,
                series=series,
                root=root,
                settings=settings,
            )
            print("Reconstructed", ident, flush=True)
        else:
            print("Present", ident, flush=True)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
