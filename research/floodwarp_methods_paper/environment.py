"""Check the recorded scientific package versions before numerical verification."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path


def mismatches(*, inventory):
    expected = json.loads(inventory.read_text())["packages"]
    differences = []
    for package, requested in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != requested:
            differences.append(f"{package} {actual} (study requires {requested})")
    if importlib.util.find_spec("floodwarp") is None:
        differences.append("FloodWarp package missing")
    return differences


if __name__ == "__main__":
    inventory = Path(__file__).resolve().parent / "supplementary/environment.json"
    differences = mismatches(inventory=inventory)
    for difference in differences:
        print(difference)
    raise SystemExit(int(bool(differences)))
