"""Read current corrected storm receipts, including individual recession checkpoints."""

from collections import defaultdict
import csv
import fnmatch
import getopt
import gzip
from pathlib import Path
import json
import sys

import numpy as np
from research.floodwarp_methods_paper import runner

from research.floodwarp_methods_paper.fidelity import native_matrix, native_products, recession
from research.floodwarp_methods_paper.plan import read_config


def policy_key(*, spec, factor):
    return (
        spec["case"],
        spec["dx"],
        spec.get("requested_method", spec["method"]),
        factor,
    )


def snapshot_budget(*, data, initial, end_s, interval):
    """Integrate the accepted outlet and internal face transports at one requested time."""
    frames = data["frames"]
    times = np.asarray([r["t_s"] for r in frames], dtype=float)
    if len(times) == 0 or times[-1] != end_s:
        raise ValueError("Requested end differs from observed hydrograph end")
    dt = np.diff(times, prepend=0)
    if not np.all(dt == interval):
        raise ValueError("Missing or irregular accepted reporting intervals")
    q = np.asarray([r["outlet_m3s"] for r in frames])
    qi = np.asarray([r["section_m3s"] for r in frames])
    exported = float(q @ dt)
    storage = data["storage_m3"] if "storage_m3" in data else data["final_storage_m3"]
    rain, soil = data["rain_m3"], data["infiltration_m3"]
    residual = initial + rain - exported - storage - soil
    allowed = max(1., initial + rain + np.maximum(-q, 0) @ dt) * 1e-8
    if not np.all(np.isfinite([q.max(), qi @ dt, residual, data["residual_m3"]])) or max(abs(residual - data["residual_m3"]), abs(residual)) > allowed:
        raise ValueError("Accepted outlet transport disagrees with the archived water balance")
    return dict(initial_m3=initial, rain_m3=rain, infiltration_m3=soil, storage_m3=storage,
                outlet_net_m3=exported, section_net_m3=float(qi @ dt), residual_m3=residual,
                terminal_outlet_m3s=float(q[-1]), peak_outlet_m3s=float(q.max()),
                peak_interval_end_s=float(times[np.argmax(q)]))


def window_report(*, directory, config, design, case_pattern, spacing, method, end_s, output):
    if not np.isfinite(end_s) or end_s <= 0 or not float(end_s).is_integer():
        raise ValueError("Request an archived positive endpoint at whole-second precision")
    specs = native_matrix.specifications(study=directory.parent, config=config, design=design)
    planned = sorted({policy_key(spec=s, factor=s["factor"]) for s in specs})
    keys = [k for k in planned if fnmatch.fnmatchcase(k[0], case_pattern)
            and spacing in (None, k[1]) and method in (None, k[2])]
    if not keys:
        raise ValueError("No scheduled forecasts match the requested terrain, spacing and method")
    records = native_products.prefixes(directory=directory)
    current = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    selected = {}
    for ident, item in records.items():
        key = policy_key(spec=item["s"], factor=item["factor"])
        if item["solver_source"] == current and not item["s"].get("timing_repeat") and key in keys:
            if key in selected:
                raise ValueError("Ambiguous primary observation for the selected policy")
            selected[key] = ident
    series = {}
    for path in (directory / "extended_runs").glob(f"*/horizons/{int(end_s)}.json.gz"):
        with gzip.open(path, "rt") as stream:
            d = json.load(stream)
        parent = d["parent_id"]
        if parent not in selected.values() or d["cohort"] != "native-metre":
            continue
        if (d["end_s"] != end_s or d["spec"] != records[parent]["s"] or d["factor"] != records[parent]["factor"]
                or d["id"] != path.parents[1].name or parent in series):
            raise ValueError("Ambiguous or mismatched horizon receipt")
        series[parent] = d
    settings = json.loads((directory / "recession_settings.json").read_text())
    rows = []
    for key in keys:
        ident = selected.get(key)
        row = dict(case=key[0], dx_m=key[1], method=key[2], cfl_factor=key[3], end_h=end_s / 3600,
                   initial_id=ident, status="no receipt at requested endpoint")
        if ident is not None:
            item = records[ident]
            window = series.get(ident)
            if item["s"]["hours"] * 3600 == end_s:
                window = item["d"]
            if window is not None:
                budget = snapshot_budget(data=window, initial=item["d"]["initial_m3"], end_s=end_s,
                                         interval=item["s"]["interval"])
                passed, _ = recession.settled(frames=window["frames"], storage=budget["storage_m3"],
                                               supply=max(1, budget["rain_m3"] + budget["initial_m3"]), config=settings)
                if "settled" in window and passed != window["settled"]:
                    raise ValueError("Recomputed recession status differs from the measured receipt")
                row.update(status="measured", series_id=window.get("id", ident), both_gauges_receded=passed,
                           interval_s=item["s"]["interval"], native_sha=item["native_sha"], **budget)
        rows.append(row)
    writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


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
    flags, extra = getopt.getopt(sys.argv[1:], "", ["inventory", "resume", "dry-run", "end-h=", "case=", "dx=", "method="])
    options = dict(flags)
    if extra:
        raise ValueError(f"Unrecognised native status arguments {extra}")
    if "--dry-run" in options:
        print("Read native forecast receipts and report the selected measured endpoints.")
    elif "--end-h" in options:
        if "--inventory" in options:
            raise ValueError("Choose an endpoint comparison or an inventory")
        window_report(directory=directory, config=config, design=design, case_pattern=options.get("--case", "*"),
                      spacing=float(options["--dx"]) if "--dx" in options else None, method=options.get("--method"),
                      end_s=float(options["--end-h"]) * 3600, output=sys.stdout)
    elif {"--case", "--method", "--dx"}.intersection(options):
        raise ValueError("Choose --end-h to report selected forecasts on a common observation window")
    else:
        status(directory=directory, config=config, design=design, inventory="--inventory" in options)
