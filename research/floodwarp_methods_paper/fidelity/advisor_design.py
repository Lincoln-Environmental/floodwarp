"""Preflight paired-grid advisor opportunities separately from observed hydraulic qualification."""

from collections import defaultdict
import json
from pathlib import Path

from floodwarp.solver_advisor.model import terrain_control_status
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_advisor, native_matrix
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import advisor_data


def inventory(
    *, specs, features, measured, heldout, methods, factor, spacings, objectives
):
    groups = advisor_data.split_groups(
        rows=[dict(s, terrain_id=f"{s['landform']}-{s['seed']}") for s in specs],
        heldout_family=heldout,
    )
    records = []
    for split, runs in groups.items():
        events = defaultdict(list)
        for run in runs:
            events[run["case"]].append(run)
        for case, cases in sorted(events.items()):
            keys = {(s["dx"], s["requested_method"], s["factor"]) for s in cases}
            for candidate in spacings:
                fine = max((dx for dx in spacings if dx < candidate), default=None)
                # The common reference for comparing both practical grids is the same fine grid.
                if fine is not None:
                    fine = min(spacings)
                controls = {(fine, "adaptive", f) for f in (1.0, factor)} | {
                    (fine, "hancock_dense", 1.0)
                }
                required = {(candidate, method, 1.0) for method in methods}
                for objective in objectives:
                    feature = features.get((case, candidate), {})
                    admissible = terrain_control_status(features=feature, objective=objective) == "admissible"
                    measured_methods = {
                        r["method"]
                        for r in measured.get((case, candidate), [])
                        if r["qualified"][objective]
                    }
                    scheduled = (
                        fine is not None and controls <= keys and required <= keys
                    )
                    observable = {**native_advisor.DEPTH, **native_advisor.WAVE}.get(
                        objective
                    ) is not None
                    records.append(
                        dict(
                            split=split,
                            terrain_id=cases[0]["terrain_id"],
                            case=case,
                            dx=candidate,
                            objective=objective,
                            reference_dx=fine,
                            scheduled_controls=scheduled,
                            admissible=admissible,
                            observable_recorded=observable,
                            possible=scheduled and admissible and observable,
                            observed_qualified=scheduled
                            and admissible
                            and observable
                            and set(methods) <= measured_methods,
                        )
                    )
    return records


def measured_schedule(*, windows, numerical_source):
    runs = {}
    for r in windows:
        if (r["horizon"] not in ("original_window", "complete-native-metre")
                or r["solver_source"] != numerical_source or r.get("timing_repeat") not in ("", None, "0")):
            continue
        key = r["case"], float(r["dx_m"]), r["method"], float(r["factor"]), r["horizon"]
        if key in runs and runs[key] != (r["source_id"], float(r["end_h"])):
            raise ValueError("Conflicting native run identities for a scheduled forecast")
        runs[key] = r["source_id"], float(r["end_h"])
    return runs


def run(*, directory, config, design, settings):
    features = {}
    for row in metrics.csv_read(file=directory / "native_features/observations.csv"):
        path = (
            directory
            / "native_features"
            / row["case"]
            / f"{row['dx_m']}m"
            / row["identity"]
            / "pre_run.json"
        )
        features[row["case"], int(row["dx_m"])] = json.loads(path.read_text())[
            "features"
        ]
    measured = defaultdict(list)
    for row in native_advisor.sample_rows(
        directory=directory,
        config=config,
        design=design,
        partitions=("development", "validation", "test"),
        require_full=False,
    ):
        measured[row["case"], row["features"]["resolution"]].append(row)
    objectives = native_advisor.tolerances(
        pct=settings["fit_percent"], depth_m=settings["depth_tolerance_m"]
    )
    specs = list(
        native_matrix.specifications(
            study=directory.parent, config=config, design=design
        )
    )
    spacing = sorted(map(int, design["reference_grid_methods"]))
    rows = inventory(
        specs=specs,
        features=features,
        measured=measured,
        heldout=config["advisor"]["heldout_family"],
        methods=design["reference_grid_methods"][str(max(spacing))],
        factor=design["reference_cfl_factor"],
        spacings=spacing,
        objectives=objectives,
    )
    dest = directory / "advisor_design"
    dest.mkdir(exist_ok=True)
    schedule = []
    numerical = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    runs = measured_schedule(windows=metrics.csv_read(file=directory / "native_results/measured_windows.csv"),
                             numerical_source=numerical)
    grids = {}
    for spec in specs:
        length, width = spec["dimensions"]
        dx = spec["dx"]
        split = "test" if spec["landform"] == config["advisor"]["heldout_family"] else spec["split"]
        key = spec["case"], dx, spec["requested_method"], spec["factor"]
        initial_id, initial_h = runs.get((*key, "original_window"), (None, None))
        final_id, end_h = runs.get((*key, "complete-native-metre"), (None, None))
        grids[spec["case"], dx] = round(length / dx) * round(width / dx)
        schedule.append(dict(case=spec["case"], terrain_id=f"{spec['landform']}-{spec['seed']}", split=split,
                             experiment=spec["experiment_class"], length_m=length, width_m=width, dx_m=dx,
                             nx=round(length/dx), ny=round(width/dx), method=spec["requested_method"],
                             cfl_factor=spec["factor"], cadence_s=spec["interval"],
                             native_floor=dx==min(spacing), initial_window_id=initial_id, initial_end_h=initial_h,
                             matched_measured_ids=final_id, matched_end_h=end_h))
    metrics.csv_write(file=dest/"scheduled.csv",rows=schedule)
    metrics.csv_write(file=dest / "opportunities.csv", rows=rows)
    pairs = []
    decisions = []
    frozen = {}
    pointer = directory / "native_advisor/frozen_model.json"
    if pointer.is_file():
        manifest = json.loads(pointer.read_text())
        model = directory / manifest["path"]
        evidence = model.parent / "validation_decisions.csv"
        seal = json.loads((model.parent / "frozen.json").read_text())
        if runner.checksum(model) != manifest["sha256"]:
            raise ValueError("Changed frozen validation model")
        if evidence.exists():
            if runner.checksum(evidence) != seal["validation_decisions_sha256"]:
                raise ValueError("Changed decision evidence in validation")
            candidates = metrics.csv_read(file=evidence)
            ids = {r["run_id"] for members in measured.values() for r in members}
            if all(r["selected_run"] in ids for r in candidates if r["selected_run"]):
                decisions = candidates
                frozen = json.loads(model.read_text())["native_cross_grid_validation"]
    for split in ("development", "validation", "test"):
        for objective in objectives:
            for i, low in enumerate(spacing):
                for high in spacing[i + 1 :]:
                    paired = {}
                    for field in ("possible", "observed_qualified"):
                        good = defaultdict(set)
                        for r in rows:
                            if (
                                r["split"] == split
                                and r["objective"] == objective
                                and r[field]
                            ):
                                good[r["terrain_id"], r["case"]].add(r["dx"])
                        paired[field] = sorted(
                            {
                                t
                                for (t, _), grids in good.items()
                                if {low, high} <= grids
                            }
                        )
                        if field == "possible":
                            paired["opportunity_domains"] = [
                                dict(terrain_id=t, case=case, candidate_cells=[grids[case, low], grids[case, high]],
                                     reference_dx_m=min(spacing), reference_cells=grids[case, min(spacing)])
                                for (t, case), available in sorted(good.items()) if {low, high} <= available
                            ]
                    pair = dict(split=split, objective=objective, dx=[low, high], **paired)
                    if split == "validation":
                        successes = defaultdict(set)
                        for r in decisions:
                            if r["objective"] == objective and r["feasible"] == "True":
                                successes[r["terrain_id"], r["case"]].add(float(r["dx"]))
                        pair["successful_paired_selections"] = sorted(
                            {t for (t, _), grids in successes.items() if {low, high} <= grids}
                        )
                        pair["frozen_pair_supported"] = [low, high] in frozen.get(objective, []) if decisions else None
                    pairs.append(pair)
    runner.write_json(
        path=dest / "paired.json", data=dict(configurations=len(specs), pairs=pairs)
    )
    for p in pairs:
        if (
            p["dx"] == [5, 10]
            and p["split"] != "development"
            and p["objective"] == "depth"
        ):
            print(p, flush=True)
    if any(
        len(p["possible"]) < settings["validation_groups_per_spacing"]
        for p in pairs
        if p["dx"] == [5, 10]
        and p["split"] in ("development", "validation", "test")
        and {**native_advisor.DEPTH, **native_advisor.WAVE}.get(p["objective"])
        is not None
    ):
        raise ValueError(
            "Design lacks independent admissible development neighbours or validation/test terrains on both practical grids"
        )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    settings = json.loads((directory / "native_advisor_settings.json").read_text())
    run(directory=directory, config=config, design=design, settings=settings)
