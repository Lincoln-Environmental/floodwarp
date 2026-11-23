"""Summarise the completed native-grid pair even while the finer references are running."""

from pathlib import Path
import json
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    extended_analysis,
    phase_cost,
    recession_figures,
)
from research.floodwarp_methods_paper.plan import read_config


def original_windows(*, directory, archived, cases, settings):
    from research.floodwarp_methods_paper.fidelity import recession

    rows = []
    for case in cases:
        items = [
            i
            for i in archived["expanded", case, "sampled"]
            if i["s"]["dimensions"][0] / i["s"]["dx"] == 1024
            and i["s"]["dimensions"][1] / i["s"]["dx"] == 1024
            and i["factor"] == 1
            and i["s"]["method"]
            in ("fv1", "hancock_dense", "hancock_adaptive", "hancock_auto")
        ]
        for item in items:
            d = item["d"]
            passed, checks = recession.settled(
                frames=d["frames"],
                storage=d["final_storage_m3"],
                supply=d["rain_m3"] + d["initial_m3"],
                config=settings,
            )
            for station, check in checks.items():
                rows.append(
                    dict(
                        case=case,
                        method=item["s"]["method"],
                        run_id=item["id"],
                        dx=item["s"]["dx"],
                        station=station,
                        end_h=item["s"]["hours"],
                        both_stations_passed=passed,
                        **check,
                    )
                )
    metrics.csv_write(file=directory / "results/original_million_tails.csv", rows=rows)


def review(*, directory, case, dx, methods, target, settings):
    config = read_config(path=directory.parent / "configs/study.json")
    original = assessment.archive(directory=directory, config=config, partial=True)
    cases = json.loads((directory / "operational_settings.json").read_text())[
        "primary_cases"
    ]
    tail = json.loads((directory / "recession_settings.json").read_text())
    original_windows(directory=directory, archived=original, cases=cases, settings=tail)
    key = ("expanded", case, "sampled")
    pairs = {
        key: [
            item
            for item in original[key]
            if item["s"]["dx"] == dx
            and item["s"]["method"] in methods
            and item["factor"] == 1
        ]
    }
    target.mkdir(parents=True, exist_ok=True)
    groups = extended_analysis.attach(directory=directory, groups=pairs, target=target)
    complete = groups["complete-expanded", case, "sampled"]
    if {i["s"]["method"] for i in complete} != set(methods):
        raise ValueError("Incomplete native-grid forecast comparison")
    phase_cost.collect(
        groups=groups,
        directory=directory,
        target=target,
        cadence=settings["cost_interval_s"],
        edges=settings["wet_fraction_edges"],
        phase_edges=settings["phase_end_hours"],
    )
    (target / "figures").mkdir(exist_ok=True)
    recession_figures.operational(groups=groups, directory=target, cases=[(case, dx)])
    points = metrics.csv_read(target / "cost_by_phase.csv")
    end = complete[0]["s"]["hours"]
    lines = [
        "# Mixed-catchment recession",
        "",
        f"The native {dx:g} m FV1 and tiled Hancock forecasts reached a common {end:g} h endpoint. The outlet and internal section passed the two-hour recession check. Finer reference calculations can still extend the end time for the full comparison.",
        "",
        "| Phase | FV1 hydraulic time (s) | Hancock hydraulic time (s) | Hancock / FV1 | FV1 steps | Hancock steps | Hancock / FV1 per step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in points:
        if r["axis"] == "phase" and r["method"] == "hancock_adaptive":
            lines.append(
                f"| {r['level'].capitalize()} | {float(r['fv1_s']):.2f} | {float(r['method_s']):.2f} | {float(r['ratio']):.2f} | {int(r['fv1_steps']):,} | {int(r['steps']):,} | {float(r['per_step_ratio']):.2f} |"
            )
    lines.extend(
        [
            "",
            "Rain ends at hour 6. Early recession covers hours 6–8, intermediate recession hours 8–12 and late recession all subsequent hours. Each ratio divides the sum of measured runtimes for matched physical intervals.",
            "",
            f"[Rain, discharge, wet fraction and hydraulic time](figures/{case}_cost.pdf) · [Signed section measurements and storage](recession_terminal.csv) · [Paired 15-minute costs](cost_intervals.csv)",
        ]
    )
    (target / "README.md").write_text("\n".join(lines) + "\n")
    runner.write_json(
        path=target / "sources.json", data={i["id"]: i["parent_id"] for i in complete}
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "application_settings.json").read_text())
    target = directory / "mixed_native"
    case = "mixed-708-spatial-10240x10240"
    dx = 10
    methods = ("fv1", "hancock_adaptive")
    review(
        directory=directory,
        case=case,
        dx=dx,
        methods=methods,
        target=target,
        settings=settings,
    )
