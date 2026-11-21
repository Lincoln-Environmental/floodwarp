"""Isolate each complete CUDA graph replay so pool high-water marks have one owner."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification import launch


def source(*, root, study, stage):
    return runner.digest(
        dict(
            base=launch.identity_original(root=root, study=study, stage=stage),
            instrumentation=runner.checksum(study / "qualification/profiled.py"),
        )
    )


def calculate(*, stage, spec, study, root, config, context, directory):
    with TemporaryDirectory() as temporary:
        parameters = Path(temporary) / "input.json"
        output = Path(temporary) / "result.json"
        parameters.write_text(
            json.dumps(
                dict(
                    stage=stage,
                    spec=spec,
                    study=str(study),
                    root=str(root),
                    config=config,
                    context=context,
                    directory=str(directory),
                )
            )
        )
        subprocess.run(
            [sys.executable, "-m", __name__, str(parameters), str(output)],
            cwd=root,
            check=True,
        )
        return json.loads(output.read_text())


def profile(*, parameters, output):
    import warp as wp

    inputs = json.loads(parameters.read_text())
    for key in ("study", "root", "directory"):
        inputs[key] = Path(inputs[key])
    result = launch.calculate(**inputs)
    result["isolated_peak_pool_mib"] = (
        wp.get_mempool_used_mem_high(device=inputs["context"]["device"]) / 2**20
    )
    runner.write_json(path=output, data=result)


if __name__ == "__main__":
    parameters = Path(sys.argv[1])
    output = Path(sys.argv[2])
    profile(parameters=parameters, output=output)
