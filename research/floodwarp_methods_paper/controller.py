"""Manual stage entry point with archive-only regeneration and offline rendering."""

from contextlib import contextmanager
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch
from time import perf_counter

from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper import (
    runner,
    evidence,
    terrain_controls,
    fixed_dt,
)


def offline_machine(*, root, study):
    # Predictions of runtime retain the hardware used to obtain the observations.
    raw = evidence.records(study=study, stages=("long-storms", "analytical"))
    source = raw[0]["context"] if raw else {}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    hardware = {k: v for k, v in source.items() if k not in ("source_sha256", "device")}
    return dict(
        hardware,
        git_sha=revision,
        analysis_cpu=platform.machine(),
        gpu=source.get("gpu", "unavailable"),
        warp=source.get("warp", "unavailable"),
        python=platform.python_version(),
    )


def missing_output(*, study, stage):
    manifests = runner.archived(study=study, stages=(stage,))
    if not manifests:
        return False
    record = max(manifests, key=receipt_time(study))
    result = json.loads((study / record["directory"] / "result.json").read_text())
    for name, expected in result.get("output_sha256", {}).items():
        path = study / name
        if not path.exists() or runner.checksum(path) != expected:
            return True
    return False


def receipt_time(study):
    def modified(record):
        return (study / record["directory"] / "receipt.json").stat().st_mtime_ns

    return modified


def derived_identity(*, root, study, stage):
    dependencies = [
        study / "manuscript/generate.py",
        study / "configs/analytical/methods.json",
        study / "supplementary/environment.json",
        study / "manifests/timing_exclusions.json",
    ]
    for name in (
        "METHODS.md",
        "manuscript/introduction.md",
        "manuscript/references.bib",
    ):
        dependencies.append(study / name)
    for name in (
        "fv1_diffusion_review/calibration/transfers.json",
        "fv1_diffusion_review/local_inertia.json",
        "hancock_long_storm_review/event_summary.csv",
    ):
        dependencies.append(root / "validation" / name)
    return runner.digest(
        dict(
            geometry_inputs=[
                r["attempt"]
                for r in evidence.records(
                    study=study, stages=(terrain_controls.STAGE, fixed_dt.STAGE)
                )
            ],
            base=original_identity(root=root, study=study, stage=stage),
            dependencies={
                str(p.relative_to(root)): runner.checksum(p)
                for p in dependencies
                if p.exists()
            },
        )
    )


original_identity = runner.source_identity


@contextmanager
def device_slot(*, study, stage, device, dry):
    if dry or stage not in runner.SIMULATIONS or stage == "synthetic":
        yield
        return
    locks = study / "manifests/devices"
    locks.mkdir(exist_ok=True, parents=True)
    key = "cuda:0" if device == "cuda" else device
    with (locks / (runner.digest(key)[:12] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def grid_filter(*, arguments):
    other = []
    grid = {}
    values = iter(arguments)
    for token in values:
        if token in ("--dx", "--dem"):
            grid[token[2:]] = float(next(values))
        else:
            other.append(token)
    return other, grid


def select_specs(*, grid, source):
    def generator(*, stage, config, root):
        for s in source(stage=stage, config=config, root=root):
            if all(s.get(k) == v for k, v in grid.items()):
                yield s

    return generator


def run(*, root, study, config_path, arguments):
    stage = arguments[0]
    selected = runner.STAGES if stage == "all" else (stage,)
    extensions = {"convergence": fixed_dt, "resolution": terrain_controls}
    supplemental = {module.STAGE: module for module in extensions.values()}
    selected = tuple(
        name
        for base in selected
        for name in ((base, extensions[base].STAGE) if base in extensions else (base,))
    )
    if set(selected) - set((*runner.STAGES, *supplemental)):
        raise ValueError("Unknown study stage " + stage)
    arguments, grid = grid_filter(arguments=arguments[1:])
    opts = runner.options(args=arguments)
    config = read_config(path=config_path)
    started = perf_counter()
    for group in selected:
        flags = opts.copy()
        if group in supplemental:
            module = supplemental[group]
            select = select_specs(grid=grid, source=module.cases)
            matches = [
                s
                for s in select(stage=group, config=config, root=root)
                if fnmatch.fnmatchcase(s["case"], flags["case"])
                and flags["method"] in ("*", s["method"])
            ]
            if not matches and stage != group:
                continue
            system = {} if flags["dry"] else runner.machine(root=root)
            with device_slot(
                study=study,
                stage="resolution",
                device=flags["device"],
                dry=flags["dry"],
            ):
                module.execute(
                    root=root,
                    study=study,
                    config=config,
                    system=system,
                    flags=flags,
                    selector=select,
                )
            continue
        system = {}
        if not flags["dry"]:
            if group in runner.SIMULATIONS:
                system = runner.machine(root=root)
            else:
                system = offline_machine(root=root, study=study)
                if missing_output(study=study, stage=group):
                    flags["force"] = True
        identity = (
            original_identity if group in runner.SIMULATIONS else derived_identity
        )
        select = select_specs(grid=grid, source=runner.specs)
        with (
            device_slot(
                study=study, stage=group, device=flags["device"], dry=flags["dry"]
            ),
            patch.object(runner, "source_identity", identity),
            patch.object(runner, "specs", select),
        ):
            runner.run_stage(
                stage=group,
                opts=flags,
                root=root,
                study=study,
                config=config,
                system=system,
            )
        if (
            not flags["dry"]
            and group == "analytical"
            and not grid
            and flags["case"] == "*"
            and flags["method"] == "*"
        ):
            from research.floodwarp_methods_paper.comparisons import exact_check

            print("Analytical qualification", exact_check(study=study), flush=True)
    print("total_experiment_wall_s", round(perf_counter() - started, 3))


if __name__ == "__main__":
    study = Path(__file__).resolve().parent
    root = study.parents[1]
    config_path = Path(
        os.environ.get("FLOODWARP_PAPER_CONFIG", study / "configs/study.json")
    )
    arguments = sys.argv[1:]
    run(root=root, study=study, config_path=config_path, arguments=arguments)
