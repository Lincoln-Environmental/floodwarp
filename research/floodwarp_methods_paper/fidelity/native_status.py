"""Read current corrected storm receipts, including individual recession checkpoints."""

from collections import defaultdict
from pathlib import Path
import json
import sys
from research.floodwarp_methods_paper import runner

from research.floodwarp_methods_paper.fidelity import native_matrix, native_products
from research.floodwarp_methods_paper.plan import read_config


def policy_key(*, spec, factor):
    return (
        spec["case"],
        spec["dx"],
        spec.get("requested_method", spec["method"]),
        factor,
    )


def status(*, directory, config, design, inventory=False):
    expected = defaultdict(set)
    for spec in native_matrix.specifications(
        study=directory.parent, config=config, design=design
    ):
        key = policy_key(spec=spec, factor=spec["factor"])
        expected[key[0]].add(key)
    by_id = {}
    done = defaultdict(set)
    current = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    historical = 0
    for ident, record in native_products.prefixes(directory=directory).items():
        spec = record["s"]
        if spec.get("timing_repeat"):
            continue
        if record["solver_source"] != current:
            historical += 1
            continue
        key = policy_key(spec=spec, factor=record["factor"])
        if key in expected.get(key[0], ()):
            by_id[ident] = key
            done[key[0]].add(key)
    matched = {}
    extended = directory / "extended_runs"
    for receipt in extended.glob("group_*.json"):
        r = json.loads(receipt.read_text())
        if r["cohort"] == "native-metre" and set(r["sources"]) <= by_id.keys():
            if {by_id[i] for i in r["sources"]} == expected[r["case"]]:
                matched[r["case"]] = r["end_s"] / 3600
    planned = sum(map(len, expected.values()))
    measured = sum(map(len, done.values()))
    if inventory:
        print(f"Archive inventory across all {len(expected)} domains, solver {current}:")
        print(f"{historical} older-build primary windows retained and excluded from reuse.")
        for ident, key in sorted(by_id.items()):
            print("  Same-build candidate", ident, *key, "common end (h)", matched.get(key[0], "pending"))
        print(f"{measured}/{planned} configurations have a same-build initial window. "
              "Resume additionally checks the physical fields, driver, controls and launch environment.")
        return
    print(
        f"{measured}/{planned} initial windows available; {len(matched)}/{len(expected)} domains at their common recession endpoint."
    )
    for case in expected:
        if done.get(case):
            end = (
                f"{matched[case]:g} h matched"
                if case in matched
                else "common recession pending"
            )
            print(
                f"  {case}  {len(done[case])}/{len(expected[case])} initial windows, {end}."
            )
    attempts = directory / "native_matrix"
    for start in sorted(attempts.glob("*/*/attempt_*/started.json")):
        attempt = start.parent
        if not (attempt / "complete.json").is_file():
            data = json.loads(start.read_text())["inputs"]["spec"]
            state = (
                "failed"
                if (attempt / "failed.json").is_file()
                else "started, no completion receipt"
            )
            print(
                f"  {policy_key(spec=data, factor=data['factor'])}: {state} ({attempt.name})"
            )
    latest = []
    # A matching group receives its receipt only after all members pass at the same time.
    for path in extended.glob("*/series.json"):
        data = json.loads(path.read_text())
        parent = data["parent_id"]
        if parent not in by_id or data["cohort"] != "native-metre":
            continue
        if by_id[parent][0] not in matched:
            latest.append((path.stat().st_mtime_ns, by_id[parent], data))
    for _, key, data in sorted(latest, reverse=True)[:5]:
        seconds = data["end_s"]
        q = data["checks"]["outlet"]
        print(
            f"  {key}: {seconds / 3600:g} h, both stations passed = {data['settled']}, "
            f"outlet {q['terminal_m3s']:.6g} m³/s, measured peak {q['peak_m3s']:.6g} m³/s, "
            f"storage {data['storage_m3']:.5g} m³."
        )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    status(directory=directory, config=config, design=design, inventory="--inventory" in sys.argv[1:])
