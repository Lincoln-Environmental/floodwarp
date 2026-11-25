"""Replay matched native-cell fields across the scheduled fixed-footprint study."""

from collections import defaultdict
import fnmatch
import getopt
import json
from pathlib import Path
import sys

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_map_comparison, native_matrix
from research.floodwarp_methods_paper.plan import read_config


def inventory(*, specs, windows, heldout_family, partition, case, dx, method):
    if partition not in ("fit", "test", "all"):
        raise ValueError("Unknown native map partition")
    planned = defaultdict(set)
    for spec in specs:
        if spec["experiment_class"] not in ("fixed_physical_domain", "independent_reference"):
            continue
        test = spec["split"] == "test" or spec["landform"] == heldout_family
        if partition != "all" and (partition == "test") != test:
            continue
        if not fnmatch.fnmatchcase(spec["case"], case) or (dx is not None and spec["dx"] != dx):
            continue
        name = spec["requested_method"]
        if name not in ("fv1", "adaptive", "hancock_dense"):
            continue
        baseline = name == "adaptive" and spec["factor"] == 1
        if baseline or method is None or name == method:
            planned[spec["case"], spec["dx"]].add((name, spec["factor"]))
    observed = defaultdict(set)
    for row in windows:
        if row["horizon"] == "original_window" and not int(row["timing_repeat"] or 0):
            observed[row["case"], int(row["dx_m"])].add((row["method"], float(row["factor"])))
    for key, methods in sorted(planned.items()):
        if ("adaptive", 1.0) in methods and len(methods) >= 2:
            yield key, methods - observed[key]


def run(*, directory, case, partition, dx, method, horizons, dry, recompute):
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    thresholds = json.loads((directory / "application_settings.json").read_text())["endpoint_depth"]["thresholds_m"]
    solver = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    windows_file = directory / "native_results/measured_windows.csv"
    windows = [r for r in metrics.csv_read(file=windows_file) if r["solver_source"] == solver] if windows_file.is_file() else []
    grids = list(inventory(specs=native_matrix.specifications(study=directory.parent, config=config, design=design),
                           windows=windows, heldout_family=config["advisor"]["heldout_family"],
                           case=case, dx=dx, partition=partition, method=method))
    print(f"{len(grids)} scheduled event/grid groups, {sum(not missing for _, missing in grids)} with measured initial peers.", flush=True)
    for (event, resolution), missing in grids:
        if missing:
            print(event, resolution, "m awaiting initial observations", sorted(missing), flush=True)
            continue
        for horizon in horizons:
            native_map_comparison.compare(directory=directory, case=event, dx=float(resolution), method=method,
                                          thresholds=[t for t in thresholds if t >= 0.05], dry=dry,
                                          repeat=recompute, horizon=horizon)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    flags, extra = getopt.getopt(sys.argv[1:], "", ["partition=", "case=", "dx=", "method=", "end-h=",
                                                    "recompute", "dry-run", "resume"])
    if extra:
        raise ValueError(f"Unexpected native map selection {extra}")
    options = dict(flags)
    run(directory=directory, case=options.get("--case", "*"), partition=options.get("--partition", "fit"),
        dx=int(options["--dx"]) if "--dx" in options else None, method=options.get("--method"),
        horizons=[options["--end-h"]] if "--end-h" in options else ["initial", "matched"],
        dry="--dry-run" in options, recompute="--recompute" in options)
