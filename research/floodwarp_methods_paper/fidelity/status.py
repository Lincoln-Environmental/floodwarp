"""Read-only progress of the manually started hydrograph qualification."""

from collections import Counter
import json
from pathlib import Path


def status(*, directory):
    pulse = directory / "pulse_runs"
    prints = []
    if (pulse / "current.txt").is_file():
        latest = pulse / (pulse / "current.txt").read_text().strip()
        prints.append(
            f"Exact channel runs archived: {sum(1 for _ in latest.glob('*.npz'))}"
        )
    for key in ("long_pulses", "oblique_runs"):
        folder = directory / key
        print(
            f"{key}: {sum(1 for file in folder.glob('*/*.npz') if not file.name.endswith('.tmp.npz'))} completed runs"
        )
    for item in ("legacy", "expanded", "double"):
        done = 0
        for file in (directory / "storm_runs").glob("*/result.json"):
            if json.loads(file.read_text())["cohort"] == item:
                done += 1
        prints.append(f"Additional {item} reference/control forecasts: {done}")
    by_group = Counter()
    for file in (directory / "extended_runs").glob("group_*.json"):
        group = json.loads(file.read_text())
        by_group[group["cohort"], group["end_s"] / 3600] += 1
    for (cohort, hours), count in sorted(by_group.items()):
        prints.append(
            f"{cohort} groups with all compared runs checked at {hours:g} h: {count}"
        )
    files = list((directory / "extended_runs").glob("*/series.json"))
    if files:
        latest = max(files, key=modified)
        run = json.loads(latest.read_text())
        prints.append(
            f"Latest archived continuation: {run['spec']['case']} {run['spec']['method']} {run['spec']['dx']} m at {run['end_s'] / 3600:g} h. Individual tail check: {run['settled']}."
        )
    print("\n".join(prints))


def modified(path):
    return path.stat().st_mtime_ns


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    status(directory=directory)
