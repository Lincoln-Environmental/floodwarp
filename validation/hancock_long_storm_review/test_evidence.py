"""Retain the numerical regression outcomes and the reasons for skipped controls."""

import hashlib
import json
import os
from pathlib import Path
import re


def extract(*, log):
    data = log.read_bytes()
    lines = data.decode().splitlines()
    match = re.search(r"([0-9]+) passed, ([0-9]+) skipped.* in ([0-9.]+)s", lines[-1])
    if match is None or any(term in lines[-1] for term in ("failed", "error")):
        raise ValueError("A completed pytest run with zero failures is required")
    passed, skipped, duration = match.groups()
    return dict(
        passed=int(passed),
        skipped=int(skipped),
        duration_s=float(duration),
        summary=lines[-1],
        skipped_controls=[line for line in lines if line.startswith("SKIPPED [")],
        command="OVERLANDER_TEST_FULL_SWE_CUDA=1 FLOODWARP_TEST_RENDER=1 .venv/bin/python -m pytest -q -rs",
        stdout_sha256=hashlib.sha256(data).hexdigest(),
    )


if __name__ == "__main__":
    review = Path(__file__).parent
    log = Path(
        os.environ.get(
            "FLOODWARP_PYTEST_LOG", "/tmp/floodwarp_hancock_integrated_tests.log"
        )
    )
    output = review / os.environ.get("FLOODWARP_TEST_RECEIPT", "test_results.json")
    output.write_text(json.dumps(extract(log=log), indent=2) + "\n")
