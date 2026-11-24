"""Apply the paper's observable-specific reference tests to the original catchment."""

from collections import defaultdict
import json
from statistics import median

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment, budgets, statistics


def window_cost(*, frames, end_s):
    selected = [r for r in frames if r["t_s"] <= end_s]
    if not selected or selected[-1]["t_s"] != end_s:
        raise ValueError("Missing timing at spatial assessment endpoint")
    return sum(r["hydraulic_s"] for r in selected), sum(r["steps"] for r in selected)


def spatial_decisions(*, directory, destination, experiments, checks):
    from research.floodwarp_methods_paper.fidelity import outlet_storms, outlet_timing

    controls = outlet_storms.archived(directory=directory)
    application = json.loads((directory / "application_settings.json").read_text())
    advisor = json.loads((directory / "native_advisor_settings.json").read_text())
    timing_file = directory / "outlet_results/independent_timings.csv"
    repeats = metrics.csv_read(file=timing_file) if timing_file.exists() else []
    observations = []
    costs = []
    for contract, frames, summary, path in experiments:
        identity = contract["identity"]
        if identity["cfl_factor"] != 1:
            continue
        spec = controls[identity["method"], identity["dx"]]["inputs"]["spec"]
        matched = [r for r in checks if r["run_id"] == path.name]
        for hour in sorted({r["hour"] for r in matched}):
            end = round(hour * 3600)
            cost, steps = window_cost(frames=frames, end_s=end)
            samples = {path.name: cost}
            for repeat in repeats:
                if (repeat["primary"] != path.name or repeat["id"] in samples
                        or int(repeat["end_s"]) != summary["end_s"]):
                    continue
                data, _ = outlet_timing.read_frames(
                    path=directory / "outlet_timing_repeats" / repeat["id"], end=end
                )
                seconds, count = window_cost(frames=data, end_s=end)
                if count != steps:
                    raise ValueError("Repeated timing changed spatial forecast step count")
                samples[repeat["id"]] = seconds
            for sample_id, seconds in samples.items():
                costs.append(dict(primary=path.name, timing_id=sample_id, end_h=hour,
                                  hydraulic_s=seconds, accepted_steps=steps))
            data = dict(
                cohort="repaired-outlet", case=spec["case"],
                representation=f"{identity['extension_m']} m downstream extension",
                station="domain", interval_max_s=identity["settings"]["spatial_s"],
                end_h=hour, experiment_id=path.name, method=identity["method"],
                dx=identity["dx"], runtime_s=median(samples.values()), steps=steps,
                **assessment.numerical_identity(method=identity["method"]),
            )
            for row in matched:
                if row["hour"] == hour:
                    metric = row["metric"]
                    data.update(reference_id=row["reference_id"])
                    data[metric] = row["error"]
                    data["qualified_" + metric] = row["qualified"]
                    data[metric + "_sensitivity"] = row["reference_sensitivity"]
            observations.append(data)
    objectives = sorted({row["metric"] for row in checks})
    limits = {
        k: [advisor["depth_tolerance_m"]] if "depth" in k
        else [pct / 100 for pct in application["tolerances_pct"]] for k in objectives
    }
    frontier = statistics.frontiers(rows=observations, objectives=objectives)
    metrics.csv_write(file=destination / "spatial_timing_windows.csv", rows=costs)
    metrics.csv_write(file=destination / "spatial_pareto.csv", rows=frontier)
    budgets.feasible_cost(rows=frontier, target=destination, tolerances=limits,
                         spacings=application["fv1_budget_grids_m"], name="depth_extent")


def event(*, contract, frames, summary, path, spec):
    identity = contract["identity"]
    forcing = identity["forcing"]
    end = frames[-1]["t_s"]
    actual_rain = [list(forcing[key]) for key in ("rain_times_s", "rain_rates_ms")]
    if actual_rain[0][-1] < end:
        if actual_rain[1][-1] != 0:
            raise ValueError("Rainfall does not end before the recession")
        actual_rain[0].append(end)
        actual_rain[1].append(0.0)
    spec = dict(spec, hours=end / 3600)
    data = dict(
        frames=frames,
        spatial=[],
        forcing=forcing,
        recession_capture=dict(actual_forcing=dict(rainfall=actual_rain)),
        initial_m3=contract["initial"]["footprint_storage_m3"],
        final_m3=frames[-1]["footprint_storage_m3"],
        infiltration_m3=frames[-1]["footprint_infiltration_m3"],
        hydraulic_s=summary["hydraulic_s"],
        steps=summary["steps"],
    )
    return assessment.item(
        spec=spec,
        data=data,
        ident=path.name,
        path=path,
        cohort="repaired-outlet",
        factor=identity["cfl_factor"],
    )


def qualify(*, directory, destination, experiments):
    from research.floodwarp_methods_paper.fidelity.outlet_storms import archived

    controls = archived(directory=directory)
    design = json.loads((directory / "design.json").read_text())
    advisor = json.loads((directory / "native_advisor_settings.json").read_text())
    design["reference_limits"].update(advisor["reference_limits"])
    groups = defaultdict(list)
    timing_file = directory / "outlet_results/independent_timings.csv"
    repeat_rows = metrics.csv_read(file=timing_file) if timing_file.is_file() else []
    for contract, frames, summary, path in experiments:
        identity = contract["identity"]
        spec = controls[identity["method"], identity["dx"]]["inputs"]["spec"]
        key = (
            "repaired-outlet",
            spec["case"],
            f"{identity['extension_m']} m downstream extension",
        )
        item = event(
            contract=contract, frames=frames, summary=summary, path=path, spec=spec
        )
        repeats = [
            r
            for r in repeat_rows
            if r["primary"] == path.name and int(r["end_s"]) == summary["end_s"]
        ]
        if repeats:
            if (
                len({r["id"] for r in repeats}) < 3
                or len({r["hydraulic_median_s"] for r in repeats}) != 1
            ):
                raise ValueError("Incomplete or inconsistent timing replicates")
            item["cost_for_comparison_s"] = float(repeats[0]["hydraulic_median_s"])
        groups[key].append(item)
    rows, checks = assessment.storm_rows(groups=groups, cfg=design)
    for row in rows:
        row["cells_simulated"] = row["cells"] + sum(
            contract["identity"]["extension_m"]
            // row["dx"]
            * contract["identity"]["forcing"]["shape"][0]
            for contract, _, _, path in experiments
            if path.name == row["experiment_id"]
        )
    metrics.csv_write(file=destination / "hydrograph_fidelity.csv", rows=rows)
    metrics.csv_write(file=destination / "reference_checks.csv", rows=checks)
    frontier = statistics.frontiers(rows=rows, objectives=design["reference_limits"])
    metrics.csv_write(file=destination / "pareto.csv", rows=frontier)
    budgets.feasible_cost(
        rows=frontier,
        target=destination,
        tolerances=advisor["assessment_percentages"],
        spacings={row["dx"] for row in rows},
    )
