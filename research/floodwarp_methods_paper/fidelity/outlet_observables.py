"""Apply the paper's observable-specific reference tests to the original catchment."""

from collections import defaultdict
import json

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment, budgets, statistics


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
