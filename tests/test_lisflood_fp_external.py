"""Opt-in checks for the separately installed LISFLOOD-FP executable."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def sha256_file(file_path: Path) -> str:
    """Return the SHA-256 checksum of an external executable."""
    digest = hashlib.sha256()
    with file_path.open("rb") as source_file:
        chunk = source_file.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = source_file.read(1024 * 1024)
    return digest.hexdigest()


def test_lisflood_fp_cpu_manifest_matches_executable() -> None:
    """Validate an installed external executable without rebuilding it."""
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / "external" / "lisflood_fp" / "provenance.json"
    executable_path = project_root / ".external" / "lisflood_fp" / "bin" / "lisflood_cpu"
    if not executable_path.is_file():
        pytest.skip("external LISFLOOD-FP CPU executable is not installed")
    provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert provenance["cpu_executable_sha256"] == sha256_file(file_path=executable_path)


def test_lisflood_fp_acc_smoke_test_when_enabled() -> None:
    """Run the external ACC test only when explicitly requested by the environment."""
    if os.environ.get("RUN_EXTERNAL_HYDRAULIC_TESTS") != "1":
        pytest.skip("set RUN_EXTERNAL_HYDRAULIC_TESTS=1 to run external hydraulic tests")
    project_root = Path(__file__).resolve().parents[2]
    smoke_test_script = project_root / "tools" / "run_lisflood_fp_smoke_test.py"
    completed = subprocess.run(
        [sys.executable, str(smoke_test_script)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
