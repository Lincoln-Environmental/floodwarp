"""Run incised resolution controls on bottom-up area means of the native metre terrain."""

import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from research.floodwarp_methods_paper import events, metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    continuation,
    extended_analysis,
    native_event,
    recession,
)
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics, matrix
from research.floodwarp_methods_paper.terrain.generators import mean_dem, native_inputs


def base_inputs(*, directory, design):
    study = directory.parent
    cfg = read_config(path=study / "configs/study.json")
    root = directory.parents[2]
    base = next(
        s
        for s in matrix.cases(stage="million-storms", config=cfg, root=root)
        if s["case"] == design["case"] and s["method"] == "fv1"
    )
    inputs = dict(
        cfg=cfg,
        template=json.loads(
            (root / "validation/hancock_long_storm_review/cases.json").read_text()
        ),
        options=matrix.design(study=study),
    )
    return base, inputs


def reference_inputs(*, directory, design):
    """Elevation-only diagnostic for the archived reach replay."""
    base, inputs = base_inputs(directory=directory, design=design)
    root = directory.parents[2]
    folder, desc = mean_dem.generate(
        spec=base,
        inputs=inputs,
        cache_root=root / ".fidelity-restarts/mean_dems",
        strip_rows=design["strip_rows"],
    )
    return base, inputs, folder, desc


def run(*, directory, design, settings):
    root = directory.parents[2]
    base, inputs = base_inputs(directory=directory, design=design)
    if design["native_m"] != 1:
        raise ValueError("The native physical source must be one metre")
    folder, terrain = native_inputs.generate(
        spec=base,
        inputs=inputs,
        cache_root=root / ".fidelity-restarts/native_inputs",
        strip_rows=design["strip_rows"],
        channel_core_m=design["channel_core_min_depth_m"],
        include_native=any(dx == 1 for dx, _ in design["controls"]),
    )
    output = directory / "mean_storms"
    output.mkdir(exist_ok=True)
    runner.write_json(path=output / "terrain.json", data=terrain)
    machine = runner.machine(root=root)
    key = ("area-mean", base["case"], "sampled")

    def prefixes():
        items = []
        for dx, method in design["controls"]:
            spec = dict(
                base,
                dx=dx,
                dem=dx,
                method=method,
                initial_grid=terrain["inputs"]["aggregation"],
                terrain_aggregation=terrain["inputs"]["aggregation"],
            )
            identity = dict(
                spec=spec,
                terrain=terrain,
                physical=inputs,
                design=design,
                recession=settings,
                code=runner.checksum(Path(__file__)),
                factory=runner.checksum(Path(native_event.__file__)),
                capture=runner.checksum(Path(continuation.__file__)),
                hydraulic_driver=runner.checksum(Path(hydraulics.__file__)),
                solver=runner.source_identity(
                    root=root, study=directory.parent, stage="long-storms"
                ),
                hardware=[machine["gpu"], machine["warp"]],
            )
            ident = runner.digest(identity)[:24]
            dest = output / ident
            dest.mkdir(exist_ok=True)
            result = dest / "result.json"
            if not result.is_file():
                data = hydraulics.run_event(
                    spec=spec,
                    config=inputs["cfg"],
                    template=inputs["template"],
                    options=inputs["options"],
                    directory=dest,
                    device=settings["device"],
                )
                recession.atomic_json(
                    path=result,
                    payload=dict(inputs=identity, hardware=machine, data=data),
                )
            else:
                data = json.loads(result.read_text())["data"]
            print(
                "Area-mean DEM storm",
                dx,
                method,
                data["hours_completed"],
                "h",
                data["hydraulic_s"],
                "hydraulic seconds",
                flush=True,
            )
            items.append(
                assessment.item(
                    spec=spec, data=data, ident=ident, path=dest, cohort=key[0]
                )
            )
        return items

    fresh = partial(native_event.create, bundle=folder, receipt=terrain)
    with patch.object(events, "create_event", fresh):
        items = continuation.capture(
            directory=directory,
            settings=settings,
            stage="area-mean-storms",
            experiment=prefixes,
        )

        def selected(**kwargs):
            return {key: items}

        with patch.object(assessment, "archive", selected):
            continuation.extend(
                directory=directory, settings=settings, cohorts=(key[0],)
            )
    complete = extended_analysis.attach(
        directory=directory, groups={key: items}, target=output
    )
    rows = []
    for item in complete["complete-area-mean", base["case"], "sampled"]:
        data = item["d"]
        t, q = assessment.curve(it=item, station="outlet")
        rows.append(
            dict(
                case=base["case"],
                dx_m=item["s"]["dx"],
                method=item["s"]["method"],
                series_id=item["id"],
                end_h=t[-1] / 3600,
                outlet_peak_m3s=q.max(),
                net_outlet_m3=q @ np.diff(t),
                storage_m3=data["final_storage_m3"],
                infiltration_m3=data["infiltration_m3"],
                hydraulic_s=data["hydraulic_s"],
                residual_m3=data["residual_m3"],
            )
        )
    metrics.csv_write(file=output / "complete_events.csv", rows=rows)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    design = json.loads((directory / "mean_settings.json").read_text())
    settings = json.loads((directory / "recession_settings.json").read_text())
    run(directory=directory, design=design, settings=settings)
