"""Manual stage entry point with archive-only regeneration and offline rendering."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch
from time import perf_counter

from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper import runner


def offline_machine(*, root, study):
    # Predictions of runtime retain the hardware used to obtain the observations.
    raw = runner.archived(study=study, stages=("long-storms", "analytical"))
    source = raw[0]["context"] if raw else {}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return dict(
        source,
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
            base=original_identity(root=root, study=study, stage=stage),
            dependencies={
                str(p.relative_to(root)): runner.checksum(p)
                for p in dependencies
                if p.exists()
            },
        )
    )


original_identity = runner.source_identity


def run(*, root, study, config_path, arguments):
    stage = arguments[0]
    selected = runner.STAGES if stage == "all" else (stage,)
    if set(selected) - set(runner.STAGES):
        raise ValueError("Unknown study stage " + stage)
    opts = runner.options(args=arguments[1:])
    config = read_config(path=config_path)
    started = perf_counter()
    for group in selected:
        flags = opts.copy()
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
        with patch.object(runner, "source_identity", identity):
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
