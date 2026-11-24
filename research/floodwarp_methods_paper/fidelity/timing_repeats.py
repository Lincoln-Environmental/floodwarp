"""Re-run close measured alternatives through the same recession on the same GPU."""

from collections import defaultdict
import fnmatch
import getopt
import json
from pathlib import Path
import sys

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_matrix
from research.floodwarp_methods_paper.plan import read_config


def close_options(*, points, relative_gap):
    if len(points) < 2:
        return set()
    fastest = min(points.values())
    members = {
        key for key, cost in points.items() if cost <= fastest * (1 + relative_gap)
    }
    return members if len(members) > 1 else set()


def marginal_budget_options(*, rows, relative_gap):
    """Find configurations whose timing could change the best qualified error at an FV1 budget."""
    selected = set()
    for anchor in rows:
        if anchor["method"] != "fv1":
            continue
        budget = float(anchor["runtime_s"])
        lower, upper = budget / (1 + relative_gap), budget * (1 + relative_gap)
        qualified = [r for r in rows if r["eligible"] == "True"]
        fixed_errors = [float(r["absolute_error"]) for r in qualified
                        if r is anchor or float(r["runtime_s"]) < lower]
        best_fixed = min(fixed_errors, default=float("inf"))
        for candidate in qualified:
            if (candidate is not anchor and lower <= float(candidate["runtime_s"]) <= upper
                    and float(candidate["absolute_error"]) < best_fixed):
                selected.add((anchor["method"], int(anchor["dx"])))
                selected.add((candidate["method"], int(candidate["dx"])))
    return selected


def candidates(*, target, tolerances, relative_gap):
    selected = defaultdict(set)
    for row in metrics.csv_read(file=target / "policy_cost_events.csv"):
        if row["cohort"] != "complete-native-metre":
            continue
        costs = {
            (method, int(row["dx_m"])): float(row[field])
            for method, field in (
                ("hancock_dense", "dense_s"),
                ("hancock_adaptive", "tiled_s"),
                ("hancock_auto", "auto_s"),
            )
            if row[field]
        }
        selected[row["case"], float(row["end_h"])].update(
            close_options(points=costs, relative_gap=relative_gap)
        )
    for filename in ("hydrograph_pareto", "recession_depth_pareto"):
        groups = defaultdict(dict)
        budget_groups = defaultdict(list)
        for row in metrics.csv_read(file=target / (filename + ".csv")):
            if row["cohort"] != "complete-native-metre":
                continue
            objective = row["objective"]
            key = (row["case"], float(row["end_h"]), row["representation"], row["station"],
                   float(row["interval_s"]), row["reference_id"], row["cap"], objective)
            budget_groups[key].append(row)
            if row["eligible"] != "True":
                continue
            if "depth" in objective:
                thresholds = [tolerances["depth_tolerance_m"]]
            elif "extent_error" in objective:
                thresholds = [p / 100 for p in tolerances["assessment_percentages"]]
            else:
                thresholds = tolerances["assessment_percentages"]
            upper = float(row["absolute_error"]) + float(row["reference_sensitivity"])
            for tolerance in thresholds:
                if upper <= tolerance:
                    group = (*key, tolerance)
                    groups[group][row["method"], int(row["dx"])] = float(
                        row["runtime_s"]
                    )
        for key, points in groups.items():
            selected[key[:2]].update(
                close_options(points=points, relative_gap=relative_gap)
            )
        for key, rows in budget_groups.items():
            selected[key[:2]].update(marginal_budget_options(rows=rows, relative_gap=relative_gap))
    return {key: values for key, values in selected.items() if values}


def requested_options(*, selected, case, method, dx):
    runs = {}
    for (event, end), options in selected.items():
        if fnmatch.fnmatchcase(event, case):
            options = {(name, spacing) for name, spacing in options
                       if (method is None or method == name) and (dx is None or spacing == dx)}
            if options:
                runs[event, end] = options
    return runs


def run(*, directory, relative_gap, additional_repeats, dry, case="*", method=None, dx=None):
    target = directory / "native_results"
    thresholds = json.loads((directory / "native_advisor_settings.json").read_text())
    selected = candidates(
        target=target, tolerances=thresholds, relative_gap=relative_gap
    )
    cfg = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    settings = json.loads((directory / "recession_settings.json").read_text())
    primary = defaultdict(list)
    for row in metrics.csv_read(file=target / "measured_windows.csv"):
        if row["horizon"] == "complete-native-metre":
            primary[row["case"], float(row["end_h"])].append(row["source_id"])
    for key, ids in primary.items():
        if not fnmatch.fnmatchcase(key[0], case):
            continue
        identity = dict(
            runs=sorted(ids),
            gap=relative_gap,
            thresholds=thresholds,
            repeats=additional_repeats,
            implementation=runner.checksum(Path(__file__)),
        )
        path = (
            directory / "native_timing_plans" / (runner.digest(identity)[:24] + ".json")
        )
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["inputs"] != identity:
                raise ValueError("Timing selection identity changed")
            selected[key] = {tuple(option) for option in saved["options"]}
        elif not dry:
            runner.write_json(
                path=path,
                data=dict(inputs=identity, options=sorted(selected.get(key, set()))),
            )
    selected = requested_options(selected=selected, case=case, method=method, dx=dx)
    specs = list(
        native_matrix.specifications(study=directory.parent, config=cfg, design=design)
    )
    if dry or not selected:
        print(
            f"{len(selected)} completed domains currently have close alternatives. "
            f"Repeat {additional_repeats} independent executions when the gap is at most {100 * relative_gap:g}% of the fastest."
        )
    machine = None if dry or not selected else runner.machine(root=directory.parents[2])
    for (case, end), members in sorted(selected.items()):
        for replicate in range(1, additional_repeats + 1):
            items = [
                dict(s, timing_repeat=replicate)
                for s in specs
                if s["case"] == case
                and s["factor"] == 1
                and (s["requested_method"], s["dx"]) in members
            ]
            print(
                case, end, "h", sorted(members), "timing repeat", replicate, flush=True
            )
            if not dry:
                native_matrix.run_group(
                    directory=directory,
                    specs=items,
                    config=cfg,
                    design=design,
                    settings=dict(settings, timing_repeat_end_s=round(end * 3600)),
                    machine=machine,
                )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    relative_gap = 0.05
    additional_repeats = 2
    flags, extra = getopt.getopt(sys.argv[1:], "", ["dry-run", "resume", "case=", "method=", "dx="])
    if extra:
        raise ValueError(f"Unexpected timing repeat arguments {extra}")
    options = dict(flags)
    run(
        directory=directory,
        relative_gap=relative_gap,
        additional_repeats=additional_repeats,
        dry="--dry-run" in options,
        case=options.get("--case", "*"),
        method=options.get("--method"),
        dx=float(options["--dx"]) if "--dx" in options else None,
    )
