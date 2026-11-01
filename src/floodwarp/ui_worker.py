"""Private process entry point; launched by the local interface."""
import json
from pathlib import Path
import sys

from floodwarp.ui_runner import RunConfig, execute_run


def run_request(*, request_path):
    request = json.loads(request_path.read_text())
    execute_run(config=RunConfig(**request['config']), output=request_path.parent,
                expected_inputs=request['inputs'])


if __name__ == "__main__":
    request_path = Path(sys.argv[1])
    run_request(request_path=request_path)
