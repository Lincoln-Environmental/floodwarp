"""Preserve the end state of new reference runs for their measured extended recessions."""

from pathlib import Path
import json
import sys
from unittest.mock import patch
import numpy as np
from research.floodwarp_methods_paper.fidelity.spatial_capture import Recorder

from floodwarp.model import load_checkpoint, save_checkpoint
from research.floodwarp_methods_paper import events, runner
from research.floodwarp_methods_paper.fidelity import (
    receipts,
    recession,
    rolling_reference,
    storm_controls,
)
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics


def seed_path(*, directory, model, end):
    from floodwarp.model import checkpoint_identity

    key = runner.digest(dict(identity=checkpoint_identity(model=model), end_s=end))
    return directory.parents[2] / ".fidelity-restarts" / "prefixes" / key


def recorder(*, directory):
    settings = json.loads((directory / "spatial_settings.json").read_text())
    return Recorder(
        directory=directory,
        studied=settings["support_m"],
        interval=settings["interval_s"],
    )


def record_transport(*, spatial, original):
    def volumes(*, solver, li):
        result = original(solver=solver, li=li)
        spatial.snapshot(solver=solver, li=li)
        return result

    return volumes


def capture(*, directory, settings, stage):
    spatial = recorder(directory=directory)
    original_run = hydraulics.run_event
    original_model = events.create_model

    def record(**kw):
        end = round(kw["spec"]["hours"] * 3600)
        retained = []

        def build(**options):
            times, rates = options["rainfall_series"]
            if rates[-1] != 0 or rates[-2] != 0:
                raise ValueError("The final rain segment must already be dry")
            times = times.copy()
            times[-1] = settings["forcing_hours"] * 3600.0
            m = original_model(**dict(options, rainfall_series=(times, rates)))
            spatial.register(model=m, spec=kw["spec"])
            retained.append(m)
            return m

        with (
            patch.object(events, "create_model", build),
            patch.object(
                hydraulics,
                "volumes",
                record_transport(spatial=spatial, original=hydraulics.volumes),
            ),
        ):
            result = original_run(**kw)
        (model,) = retained
        folder = seed_path(directory=directory, model=model, end=end)
        folder.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model=model, path=folder / "state.npz")
        runner.write_json(
            path=folder / "prefix.json",
            data=dict(
                frames=result["frames"],
                extended_zero_rain_coverage_h=settings["forcing_hours"],
            ),
        )
        return result

    with patch.object(hydraulics, "run_event", record):
        if stage == "storm-controls":
            cfg = read_config(path=directory.parent / "configs/study.json")
            storm_controls.run(
                directory=directory,
                cfg=cfg,
                extra=("mixed-708-spatial-10240x10240",),
                rates=settings["double_mm_h"],
            )
        elif stage == "rolling-reference":
            rolling_reference.run(
                directory=directory,
                case="rolling-704-soil-5120x5120",
                refinements=(
                    (4, "adaptive", 1.0),
                    (3.2, "adaptive", 1.0),
                    (3.2, "hancock_dense", 1.0),
                    (3.2, "adaptive", 0.5),
                ),
                device=settings["device"],
                support=80,
            )
        else:
            raise ValueError(stage)


def extend(*, directory, settings, cohorts):
    spatial = recorder(directory=directory)
    original_build = recession.build

    def build(**kwargs):
        spec, model, expected, forcing, options = original_build(**kwargs)
        spatial.register(model=model, spec=spec)
        return spec, model, expected, forcing, options

    original = recession.advance_frames
    advance = recession.advance
    transport = hydraulics.volumes
    cadence = json.loads((directory / "spatial_settings.json").read_text())[
        "interval_s"
    ]

    def resume(**kwargs):
        if kwargs["start"] == 0:
            folder = seed_path(
                directory=directory, model=kwargs["model"], end=kwargs["end"]
            )
            if (folder / "prefix.json").exists() and (folder / "state.npz").exists():
                load_checkpoint(model=kwargs["model"], path=folder / "state.npz")
                return json.loads((folder / "prefix.json").read_text())["frames"]
        stats_by_time = {}
        wet_by_time = {}

        def accepted(**inputs):
            result = advance(**inputs)
            stats_by_time[round(inputs["end_time"])] = result
            return result

        def sample(*, solver, li):
            result = transport(solver=solver, li=li)
            spatial.snapshot(solver=solver, li=li)
            t = round(solver.time)
            if t % cadence == 0:
                depth = solver.get_water_depth()
                if not np.isfinite(depth).all() or depth.min() < 0:
                    raise ArithmeticError("Invalid recession depth")
                wet_by_time[t] = dict(
                    wet_fraction=float((depth > 0).mean()),
                    cm_fraction=float((depth >= 0.01).mean()),
                )
            return result

        with (
            patch.object(recession, "advance", accepted),
            patch.object(hydraulics, "volumes", sample),
        ):
            frames = original(**kwargs)
        for frame in frames:
            ts = frame["t_s"]
            stats = stats_by_time[ts]
            frame.update(
                dt_min_s=stats.get("dt_min_s"),
                dt_max_s=stats.get("dt_max_s"),
                rejected=stats.get("rejected_cfl_trials"),
                graph_capture_s=stats.get("graph_capture_seconds"),
                compilation_s=stats.get("compilation_seconds"),
                **wet_by_time.get(ts, {}),
            )
        return frames

    with (
        patch.object(recession, "advance_frames", resume),
        patch.object(recession, "build", build),
        patch.object(
            recession, "atomic_json", receipts.preserve(original=recession.atomic_json)
        ),
    ):
        recession.run(directory=directory, settings=settings, cohorts=cohorts)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "recession_settings.json").read_text())
    stage = sys.argv[1]
    if stage == "operational-preview":
        scan = recession.assessment.archive

        def preview(**kwargs):
            all_groups = scan(**kwargs)
            key = ("expanded", "mixed-708-spatial-10240x10240", "sampled")
            return {
                key: [
                    i
                    for i in all_groups[key]
                    if i["s"]["dx"] == 10
                    and i["s"]["method"] in ("fv1", "hancock_adaptive")
                ]
            }

        with patch.object(recession.assessment, "archive", preview):
            extend(directory=directory, settings=settings, cohorts=("expanded",))
    elif stage == "recession":
        cohorts = tuple(sys.argv[2:]) or ("legacy", "expanded", "double")
        extend(directory=directory, settings=settings, cohorts=cohorts)
    else:
        capture(directory=directory, settings=settings, stage=stage)
