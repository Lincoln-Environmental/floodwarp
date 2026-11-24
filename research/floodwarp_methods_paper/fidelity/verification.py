"""Run the repository numerical and checkpoint regressions on the reference device."""

import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from research.floodwarp_methods_paper import runner


def run(*, directory, environment):
    root = directory.parents[2]
    inputs = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("tests/**/*.py"),
            *directory.glob("*.py"),
            root / "pyproject.toml",
        }
    )
    source = {str(p.relative_to(root)): runner.checksum(p) for p in inputs}
    hardware = runner.machine(root=root)
    key = runner.digest(
        dict(
            source=source,
            overrides=environment,
            gpu=hardware["gpu"],
            warp=hardware["warp"],
            python=hardware["python"],
        )
    )
    folder = directory / "verification" / key[:18]
    receipt = folder / "passed.json"
    if receipt.is_file():
        print("Verified numerical suite", key[:18], flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    i = 1
    while (folder / f"attempt_{i}.log").exists():
        i += 1
    log = folder / f"attempt_{i}.log"
    args = [sys.executable, "-m", "pytest", "-q"]
    start = perf_counter()
    with log.open("x") as stream:
        result = subprocess.run(
            args,
            cwd=root,
            env={**os.environ, **environment},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    info = dict(
        source=source,
        hardware=hardware,
        command=args,
        environment=environment,
        wall_s=perf_counter() - start,
        returncode=result.returncode,
        log=log.name,
        sha256=runner.checksum(log),
        summary=log.read_text().strip().splitlines()[-1],
    )
    runner.write_json(
        path=receipt if result.returncode == 0 else folder / f"failed_{i}.json",
        data=info,
    )
    print(info["summary"], flush=True)
    result.check_returncode()


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    environment = {
        "MPLBACKEND": "Agg",
        "PROJ_NETWORK": "OFF",
        "OVERLANDER_TEST_FULL_SWE_CUDA": "1",
        "FLOODWARP_TEST_RENDER": "1",
    }
    options = set(sys.argv[1:])
    if options.difference({"--resume", "--cpu-only", "--dry-run"}):
        raise ValueError(f"Unknown verification options {options}")
    if "--cpu-only" in options:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.pop("OVERLANDER_TEST_FULL_SWE_CUDA")
    if "--dry-run" in options:
        print(sys.executable, "-m pytest -q", environment)
        sys.exit(0)
    run(directory=directory, environment=environment)
