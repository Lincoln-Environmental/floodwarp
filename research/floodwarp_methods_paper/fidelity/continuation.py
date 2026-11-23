"""Preserve the end state of new reference runs for their measured extended recessions."""

from pathlib import Path
import json
import sys
from time import perf_counter, time_ns
import warp as wp
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
    adapter_at_start = runner.checksum(Path(__file__))
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
        result["recession_capture"] = dict(
            adapter=adapter_at_start,
            extended_zero_rain_coverage_h=settings["forcing_hours"],
            actual_forcing=model.identity["forcing"],
        )
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


def extend(*, directory, settings, cohorts, tail_only=False):
    spatial = recorder(directory=directory)
    original_build = recession.build

    def build(**kwargs):
        spec, model, expected, forcing, options = original_build(**kwargs)
        spatial.register(model=model, spec=spec)
        return spec, model, expected, forcing, options

    original = recession.advance_frames
    advance = recession.advance
    transport = hydraulics.volumes
    spatial_settings = json.loads((directory / "spatial_settings.json").read_text())
    cadence = spatial_settings["interval_s"]

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
        extraction = {}

        def accepted(**inputs):
            result = advance(**inputs)
            stats_by_time[round(inputs["end_time"])] = result
            return result

        def sample(*, solver, li):
            tick = perf_counter()
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
                    live_pool_mib=wp.get_mempool_used_mem_current(
                        device=settings["device"]
                    )
                    / 2**20,
                )
                if not li and t > 0:
                    work = solver.execution_diagnostics()
                    cells = work.get("evaluated_cells")
                    if cells is None and work.get("execution") == "dense_uniform":
                        cells = depth.size
                    if cells is not None:
                        wet_by_time[t]["active_fraction"] = cells / depth.size
                    wet_by_time[t]["work_tiles"] = work.get("work_tiles")
            extraction[t] = perf_counter() - tick
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
                extraction_s=extraction[ts],
                **wet_by_time.get(ts, {}),
            )
        return frames

    from research.floodwarp_methods_paper.fidelity import endpoints

    prolong = recession.prolong

    def measured(**kwargs):
        writes = []

        def timer(call, name):
            def record(**args):
                tick = perf_counter()
                output = call(**args)
                writes.append((name, perf_counter() - tick))
                return output

            return record

        clock = perf_counter()
        with (
            patch.object(recession, "build", timer(build, "setup")),
            patch.object(
                recession, "save_checkpoint", timer(save_checkpoint, "checkpoint_write")
            ),
            patch.object(
                recession, "load_checkpoint", timer(load_checkpoint, "checkpoint_read")
            ),
        ):
            out = prolong(**kwargs)
        endpoints.capture(
            directory=directory, data=out, supports=spatial_settings["support_m"]
        )
        if writes:
            stamp = time_ns()
            runner.write_json(
                path=directory / "extension_timings" / out["id"] / f"{stamp}.json",
                data=dict(
                    case=out["spec"]["case"],
                    method=out["spec"]["method"],
                    id=out["id"],
                    floor_s=kwargs["floor_s"],
                    end_s=out["end_s"],
                    elapsed_wall_s=perf_counter() - clock,
                    operating_segments=writes,
                    recorded_ns=stamp,
                ),
            )
        return out

    with (
        patch.object(recession, "prolong", measured),
        patch.object(recession, "advance_frames", resume),
        patch.object(recession, "build", build),
        patch.object(
            recession, "atomic_json", receipts.preserve(original=recession.atomic_json)
        ),
    ):
        if tail_only:
            from research.floodwarp_methods_paper.fidelity import long_tail

            long_tail.run(directory=directory, settings=settings)
        else:
            recession.run(directory=directory, settings=settings, cohorts=cohorts)


def operational_pairs(*, directory, settings, cases):
    """Measure native million-cell end times before the more expensive fine-grid extensions."""
    from research.floodwarp_methods_paper.qualification import matrix

    study = directory.parent
    config = read_config(path=study / "configs/study.json")
    scan = recession.assessment.archive
    native = {
        s["case"]: s["dx"]
        for s in matrix.cases(
            stage="million-storms", config=config, root=study.parents[1]
        )
        if s["method"] == "fv1" and s["case"] in cases
    }

    def selected(**kwargs):
        result = {}
        for key, items in scan(**kwargs).items():
            co, case, rep = key
            if co == "expanded" and rep == "sampled" and case in native:
                result[key] = [
                    i
                    for i in items
                    if i["factor"] == 1
                    and i["s"]["dx"] == native[case]
                    and i["s"]["method"] in ("fv1", "hancock_dense", "hancock_adaptive")
                ]
        return result

    with patch.object(recession.assessment, "archive", selected):
        extend(directory=directory, settings=settings, cohorts=("expanded",))


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
    elif stage == "storage-tail":
        extend(directory=directory, settings=settings, cohorts=(), tail_only=True)
    elif stage in ("recession", "operational-pairs"):
        primary_cases = json.loads(
            (directory / "operational_settings.json").read_text()
        )["primary_cases"]
        cohorts = tuple(sys.argv[2:]) or ("legacy", "expanded", "double")
        if "expanded" in cohorts:
            operational_pairs(
                directory=directory, settings=settings, cases=primary_cases
            )
        if stage == "recession":
            if "expanded" in cohorts:
                extend(
                    directory=directory, settings=settings, cohorts=(), tail_only=True
                )
            extend(directory=directory, settings=settings, cohorts=cohorts)
    else:
        capture(directory=directory, settings=settings, stage=stage)
