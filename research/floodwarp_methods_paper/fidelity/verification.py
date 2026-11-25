"""Run the repository numerical and checkpoint regressions on the reference device."""

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from research.floodwarp_methods_paper import runner


def suite_sources(*, root, directory):
    study = directory.parent
    paths = {root / "pyproject.toml", study / "supplementary/constraints.txt"}
    for tree in (root / "src", root / "tests", root / "validation", study):
        paths.update(tree.rglob("*.py"))
    paths.update((study / "configs").rglob("*.json"))
    paths.update((root / "src/floodwarp").rglob("*.json"))
    paths.update(directory.glob("*_settings.json"))
    paths.update((root / "scripts").glob("*.sh"))
    return {str(p.relative_to(root)): runner.checksum(path=p) for p in sorted(paths)}


def verify_pass(*, path, source, environment, hardware):
    saved = json.loads(path.read_text())
    basename = saved["log"]
    if Path(basename).name != basename:
        raise ValueError("Test log must be inside its own verification directory")
    log = path.parent / basename
    if (saved["returncode"] != 0 or saved["source"] != source or saved["environment"] != environment
            or saved["command"][1:] != ["-m", "pytest", "-q"]
            or any(saved["hardware"].get(k) != hardware.get(k) for k in ("gpu", "cpu_model", "python", "warp", "numpy"))
            or not log.is_file() or runner.checksum(path=log) != saved["sha256"]):
        raise ValueError(f"Verification receipt disagrees with its source, environment or test log at {path}")
    lines = log.read_text().splitlines()
    if not lines or saved["summary"] != lines[-1] or " passed" not in saved["summary"]:
        raise ValueError(f"Verification summary differs from the passing test log at {path}")


def run(*, directory, environment, force=False):
    root = directory.parents[2]
    source = suite_sources(root=root, directory=directory)
    hardware = runner.machine(root=root)
    key = runner.digest(dict(source=source, overrides=environment,
                             hardware={k: hardware[k] for k in ("gpu", "cpu_model", "python", "warp", "numpy")}))
    folder = directory / "verification" / key[:18]
    receipt = folder / "passed.json"
    if receipt.is_file():
        verify_pass(path=receipt, source=source, environment=environment, hardware=hardware)
        if not force:
            print("Verified numerical suite", key[:18], flush=True)
            return
    folder.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        log = folder / f"attempt_{i}.log"
        try:
            stream = log.open("x")
            break
        except FileExistsError:
            i += 1
    args = [sys.executable, "-m", "pytest", "-q"]
    start = perf_counter()
    run_env = {**os.environ, **environment}
    with stream:
        # A masked device can leave CUDA-marked regressions skipped on a host
        # where nvidia-smi still identifies a GPU. Exercise Warp in the same
        # subprocess environment used for pytest before accepting a CUDA suite.
        if environment.get("OVERLANDER_TEST_FULL_SWE_CUDA") == "1":
            probe = subprocess.run([sys.executable, "-c", "import warp as wp; raise SystemExit(0 if wp.get_cuda_device_count() else 1)"],
                                   cwd=root, env=run_env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if probe.returncode:
                runner.write_json(path=folder / f"failed_{i}.json",
                                  data=dict(source=source, hardware=hardware, environment=environment,
                                            returncode=probe.returncode, log=log.name,
                                            error="Warp did not find an accessible CUDA device for the requested suite"))
                raise RuntimeError("CUDA verification requires a visible Warp CUDA device")
        result = subprocess.run(
            args,
            cwd=root,
            env=run_env,
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
        summary=next(iter(reversed(log.read_text().splitlines())), ""),
    )
    if result.returncode:
        runner.write_json(path=folder / f"failed_{i}.json", data=info)
        result.check_returncode()
    success = folder / f"passed_{i}.json"
    runner.write_json(path=success, data=info)
    verify_pass(path=success, source=source, environment=environment, hardware=hardware)
    try:
        os.link(success, receipt)
    except FileExistsError:
        verify_pass(path=receipt, source=source, environment=environment, hardware=hardware)
    print(info["summary"], flush=True)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    environment = {
        "MPLBACKEND": "Agg",
        "PROJ_NETWORK": "OFF",
        "OVERLANDER_TEST_FULL_SWE_CUDA": "1",
        "FLOODWARP_TEST_RENDER": "1",
        "PYTEST_ADDOPTS": "",
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    options = set(sys.argv[1:])
    if options.difference({"--resume", "--force", "--cpu-only", "--dry-run"}):
        raise ValueError(f"Unknown verification options {options}")
    if "--cpu-only" in options:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.pop("OVERLANDER_TEST_FULL_SWE_CUDA")
    if "--dry-run" in options:
        print(sys.executable, "-m pytest -q", environment)
        sys.exit(0)
    run(directory=directory, environment=environment, force="--force" in options)
