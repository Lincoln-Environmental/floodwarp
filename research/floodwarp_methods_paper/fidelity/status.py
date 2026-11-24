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
    counts = Counter(
        json.loads(file.read_text())["cohort"]
        for file in (directory / "storm_runs").glob("*/result.json")
    )
    for cohort in ("legacy", "expanded", "double"):
        prints.append(
            f"Additional {cohort} original-window controls completed: {counts[cohort]}"
        )
    incised = json.loads((directory / "incised_settings.json").read_text())
    cases = list((directory / "incised_crest").glob("*/result.json"))
    prints.append(
        f"Incised fixed-bed controls through the original window: {len(cases)}/{len(incised['controls'])}."
    )
    for file in sorted(cases, key=modified):
        result = json.loads(file.read_text())
        inputs = result["inputs"]
        spec = inputs["spec"]
        observed = result["data"]
        prints.append(
            f"  {spec['method']} at {spec['dx']:g} m, CFL {0.4 * inputs['factor']:g}: "
            f"{observed['hours_completed']:g} h simulated, {observed['steps']:,} accepted steps."
        )
    by_group = Counter()
    primary_cases = json.loads((directory / "operational_settings.json").read_text())[
        "primary_cases"
    ]
    comparison_ends = {}
    for file in (directory / "extended_runs").glob("group_*.json"):
        group = json.loads(file.read_text())
        by_group[group["cohort"], group["end_s"] / 3600] += 1
        if (
            group["cohort"] == "expanded"
            and group["representation"] == "sampled"
            and group["case"] in primary_cases
        ):
            previous = comparison_ends.get(group["case"], (0, 0))
            if len(group["sources"]) >= previous[0]:
                comparison_ends[group["case"]] = (
                    len(group["sources"]),
                    group["end_s"] / 3600,
                )
    prints.append(
        f"Native or larger matched groups with a verified recession: {len(comparison_ends)}/{len(primary_cases)} primary million-cell families."
    )
    for case in primary_cases:
        if case in comparison_ends:
            n, end = comparison_ends[case]
            prints.append(f"  {case}: {end:g} h verified for {n} configurations.")
    for file in sorted((directory / "storage_tail").glob("*.json")):
        record = json.loads(file.read_text())
        prints.append(
            f"Residual-storage comparison: {record['common_end_s'] / 3600:g} h, both stations passed the longer export test."
        )
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
    snapshots = [
        f for f in (directory / "transient").glob("*/*.npz") if f.stem.isdecimal()
    ]
    if snapshots:
        file = max(snapshots, key=modified)
        spec = json.loads((file.parent / "spec.json").read_text())
        prints.append(
            f"Latest spatial snapshot: {spec['case']}, {spec['method']} at {spec['dx']:g} m, CFL {spec['cfl']:g}, {int(file.stem) / 3600:g} simulated hours."
        )
    failed = list((directory / "verification/failed_reference_attempts").glob("*.json"))
    if failed:
        prints.append(
            f"Archived failed reference attempts excluded from hydraulic statistics: {len(failed)}."
        )
    for file in sorted((directory / "verification").glob("*/passed.json")):
        prints.append(json.loads(file.read_text())["summary"])
    print("\n".join(prints))


def modified(path):
    return path.stat().st_mtime_ns


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    status(directory=directory)
