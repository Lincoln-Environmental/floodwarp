"""Independent matched-endpoint timings of close FV1, Hancock and RK2 storm costs."""

from collections import defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import outlet_storms, timing_repeats


def cost_groups(*, rows, fraction):
    groups = defaultdict(dict)
    for r in rows:
        if float(r["cfl_factor"]) == 1:
            groups[int(r["extension_m"]), int(r["dx_m"]), float(r["end_h"])][
                r["run_id"]
            ] = float(r["hydraulic_s"])
    return set().union(
        *(
            timing_repeats.close_options(points=group, relative_gap=fraction)
            for group in groups.values()
        )
    )


def marginal_decisions(*, frontier, tolerances, fraction):
    """Repeat peers whose timing could change a decision on one fixed hydraulic support."""
    groups = defaultdict(list)
    for row in frontier:
        key = (row["case"], row["representation"], row["reference_id"], row["station"],
               row["objective"], row["cap"], float(row["interval_s"]), float(row["end_h"]))
        groups[key].append(row)
    selected = set()
    for group in groups.values():
        options = timing_repeats.marginal_budget_options(rows=group, relative_gap=fraction)
        selected.update(r["experiment_id"] for r in group if (r["method"], int(r["dx"])) in options)
        for tolerance in tolerances.get(group[0]["objective"], []):
            qualified_costs = {r["experiment_id"]: float(r["runtime_s"]) for r in group
                               if r["eligible"] == "True" and r["reference_sensitivity"] != ""
                               and float(r["absolute_error"]) + float(r["reference_sensitivity"]) <= tolerance}
            selected.update(timing_repeats.close_options(points=qualified_costs, relative_gap=fraction))
    return selected


def read_frames(*, path, end):
    data = [
        json.loads(p.read_text())
        for p in sorted(path.glob("[0-9]*.json"), key=window_end)
        if int(p.stem) <= end
    ]
    return [frame for record in data for frame in record["frames"]], sum(
        record["window_wall_s"] for record in data
    )


def window_end(path):
    return int(path.stem)


def checkpoint_hashes(*, directory, ident, end):
    path = (
        directory.parents[2] / ".fidelity-restarts/outlet_storms" / ident / f"{end}.npz"
    )
    with np.load(path) as data:
        return json.loads(str(data["_direct_manifest"]))["sha256"]


def replicate(*, directory, base, end, replicate_id, control):
    record = json.loads((base / "inputs.json").read_text())
    original = record["identity"]
    if runner.digest(original)[:24] != base.name:
        raise ValueError("Incorrect primary storm identity")
    machine = runner.machine(root=directory.parents[2])
    environment = {key: machine[key] for key in ("gpu", "cpu_model", "warp", "python", "precision")}
    if any(record["hardware"][key] != value for key, value in environment.items()):
        raise ValueError("Independent timing requires the primary GPU, driver, CPU and numerical environment")
    identity = dict(
        original,
        timing_repeat=replicate_id,
        primary=base.name,
        timing_driver=runner.checksum(Path(__file__)),
        timing_environment=environment,
    )
    dest = directory / "outlet_timing_repeats" / runner.digest(identity)[:24]
    receipt = dest / f"timing_{end}.json"
    if receipt.is_file():
        return json.loads(receipt.read_text())
    resumed = any(dest.glob("[0-9]*.json"))
    start = perf_counter()
    result = outlet_storms.trajectory(
        directory=directory,
        control=control,
        dx=original["dx"],
        factor=original["cfl_factor"],
        extension=original["extension_m"],
        end=end,
        settings=original["settings"],
        stopping=original["stopping"],
        run_dir=dest,
        identity=identity,
    )
    elapsed = perf_counter() - start
    frames, window_s = read_frames(path=dest, end=end)
    primary, _ = read_frames(path=base, end=end)
    for key in ("t_s", "section_m3s", "outlet_m3s", "steps", "rejected"):
        np.testing.assert_array_equal(
            [r[key] for r in frames], [r[key] for r in primary], err_msg=key
        )
    if checkpoint_hashes(
        directory=directory, ident=base.name, end=end
    ) != checkpoint_hashes(directory=directory, ident=dest.name, end=end):
        raise ValueError("Timed trajectory changed the primary final state")
    measured_machine = json.loads((dest / "inputs.json").read_text())["hardware"]
    if any(measured_machine[key] != value for key, value in environment.items()):
        raise ValueError("Repeat launch hardware differs from its timing contract")
    timing = dict(
        primary=base.name,
        id=dest.name,
        repeat=replicate_id,
        method=original["method"],
        dx_m=original["dx"],
        extension_m=original["extension_m"],
        end_s=end,
        interval_s=original["settings"]["interval_s"],
        **environment,
        solver_sha=original["numerical"],
        steps=result["steps"],
        policy=control["data"]["execution"],
        hydraulic_s=result["hydraulic_s"],
        checkpoint_windows_s=window_s,
        end_to_end_s=None if resumed else elapsed,
    )
    runner.write_json(path=receipt, data=timing)
    return timing


def run(*, directory, fraction, repeats, dry):
    root = directory / "outlet_results"
    products = list(root.glob("end_*/events.csv")) + [root / "events.csv"]
    controls = outlet_storms.archived(directory=directory)
    runs = []
    for file in products:
        if not file.is_file():
            continue
        rows = metrics.csv_read(file=file)
        selected = cost_groups(rows=rows, fraction=fraction)
        for name in ("fidelity", "depth_extent"):
            front = file.parent / ("pareto.csv" if name == "fidelity" else "spatial_pareto.csv")
            limits = defaultdict(set)
            for decision in metrics.csv_read(file=file.parent / f"least_cost_{name}.csv"):
                limits[decision["objective"]].add(float(decision["tolerance"]))
            selected.update(marginal_decisions(frontier=metrics.csv_read(file=front),
                                               tolerances=limits, fraction=fraction))
        for r in rows:
            if r["run_id"] not in selected:
                continue
            base = directory / "outlet_storms" / r["run_id"]
            end = round(float(r["end_h"]) * 3600)
            print(
                base.name,
                r["method"],
                r["dx_m"],
                "m,",
                end // 3600,
                "h,",
                repeats,
                "additional timings",
                flush=True,
            )
            if dry:
                continue
            control = controls[r["method"], int(r["dx_m"])]
            contract = json.loads((base / "inputs.json").read_text())
            _, wall = read_frames(path=base, end=end)
            timings = [
                dict(
                    primary=base.name,
                    id=base.name,
                    repeat=0,
                    method=r["method"],
                    dx_m=int(r["dx_m"]),
                    extension_m=int(r["extension_m"]),
                    end_s=end,
                    interval_s=contract["identity"]["settings"]["interval_s"],
                    gpu=contract["hardware"]["gpu"],
                    cpu_model=contract["hardware"]["cpu_model"],
                    warp=contract["hardware"]["warp"],
                    python=contract["hardware"]["python"],
                    precision=contract["hardware"]["precision"],
                    solver_sha=contract["identity"]["numerical"],
                    steps=int(r["steps"]),
                    policy=control["data"]["execution"],
                    hydraulic_s=float(r["hydraulic_s"]),
                    checkpoint_windows_s=wall,
                    end_to_end_s=None,
                )
            ]
            for index in range(1, repeats + 1):
                timings.append(
                    replicate(
                        directory=directory,
                        base=base,
                        end=end,
                        replicate_id=index,
                        control=control,
                    )
                )
            costs = [row["hydraulic_s"] for row in timings]
            runs.extend(
                dict(
                    row,
                    hydraulic_median_s=float(np.median(costs)),
                    hydraulic_min_s=min(costs),
                    hydraulic_max_s=max(costs),
                )
                for row in timings
            )
    if not dry:
        metrics.csv_write(file=root / "independent_timings.csv", rows=runs)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    fraction = 0.05
    repeats = 2
    dry = "--dry-run" in sys.argv[1:]
    run(directory=directory, fraction=fraction, repeats=repeats, dry=dry)
