"""Write the corrective audit and manuscript text from verified observation tables."""

from collections import Counter, defaultdict
import json
from statistics import median
from research.floodwarp_methods_paper.metrics import csv_read


def number(v, places=3):
    return "unresolved" if v in ("", None) else f"{float(v):.{places}g}"


def core_event(row):
    parts = row["case"].split("-")
    return (
        int(row["cells"]) == 1024**2 and len(parts) == 4 and 701 <= int(parts[1]) <= 708
    )


def operational_discussion(*, study):
    rows = [
        r for r in csv_read(study / "processed/million/policies.csv") if core_event(r)
    ]
    if not rows:
        return ""
    time = [float(r["fastest_ratio"]) for r in rows]
    steps = [float(r["step_ratio"]) for r in rows]
    overhead = [float(r["per_step_ratio"]) for r in rows]
    return (
        f"Across the {len(rows)} primary million-cell events, the faster measured explicit Hancock policy met "
        f"1.5× FV1 in {sum(v <= 1.5 for v in time)} of {len(rows)}. Its median full-event ratio was {median(time):.2f}×. "
        f"The accepted-step ratios ranged from {min(steps):.2f} to {max(steps):.2f}, and the corresponding "
        f"mean cost per accepted step ranged from {min(overhead):.2f} to {max(overhead):.2f} times FV1. "
        "These factors multiply within each event. The normal timestep policy therefore matters even where "
        "tile scheduling removes most of the dry-domain arithmetic. [Per-event costs](../tables/11_execution.md)"
    )


def timestep_attribution(*, events):
    if not events:
        return ""
    accepted = sum(int(row["hancock_steps"]) for row in events)
    rejected = sum(int(row["rejected"]) for row in events)
    first_order = sum(int(row["fv1_steps"]) for row in events)
    ratio, family, steps, retries = max(
        (
            float(row["step_ratio"]),
            row["landform"],
            int(row["hancock_steps"]),
            int(row["rejected"]),
        )
        for row in events
    )
    return (
        f"Across the {len(events)} primary events Hancock accepted {accepted:,} steps and rejected {rejected:,} trials "
        f"({100 * rejected / (accepted + rejected):.2f}% of trials). FV1 accepted {first_order:,} steps. "
        f"The {family} case had the largest accepted-step ratio ({ratio:.2f}×), with {retries:,} retries for "
        f"{steps:,} accepted steps. Rejections explain little of the event-wide step count. "
        "Earlier instrumented wet fronts located the restrictive signal at a reconstructed wet/dry face. "
        "A separate incised outlet case was limited by the cell-centred gravity-wave speed. "
        "The million-cell receipts measure the accepted and rejected counts. Identifying the limiting face or "
        "cell at each step would require instrumenting those runs. "
        "[Per-event timestep records](processed/million/observations.csv) · "
        "[Earlier CFL attribution](../../validation/hancock_long_storm_review/REPORT.md#timestep-attribution)"
    )


def material(*, study):
    p = study / "processed"
    pol = csv_read(p / "million/policies.csv")
    raw = csv_read(p / "million/observations.csv")
    small = [
        r for r in pol if int(r["cells"]) == 96 and r["representation"] == "sampled"
    ]
    large = [
        r
        for r in pol
        if int(r["cells"]) >= 1000000 and r["representation"] == "sampled"
    ]
    million = [r for r in large if core_event(r)]
    validation_path = p / "advisor/validation_strata.csv"
    grouped = csv_read(validation_path) if validation_path.is_file() else []
    held = [
        r
        for r in grouped
        if r["axis"] == "scale" and r["group"] == "at least one million"
    ]
    checks = csv_read(p / "million/reference_checks.csv")
    renewed = csv_read(p / "million/reference_reassessment.csv")
    terrain = csv_read(p / "million/terrain.csv")
    intervals = csv_read(p / "million/graph_profile.csv")
    from research.floodwarp_methods_paper.qualification import stages
    from research.floodwarp_methods_paper.plan import read_config
    from research.floodwarp_methods_paper.qualification.observations import current
    from pathlib import Path
    import os

    config = read_config(
        path=Path(
            os.environ.get("FLOODWARP_PAPER_CONFIG", study / "configs/study.json")
        )
    )
    inventory = []
    pending = 0
    for stage in stages.NUMERICAL:
        specs = list(
            stages.specifications(stage=stage, config=config, root=study.parents[1])
        )
        done = len(current(study=study, stage=stage, config=config))
        pending += len(specs) - done
        inventory.append(f"| {stage} | {done} | {len(specs)} |")
    output = []
    for name, label in (
        ("dense_ratio", "Dense"),
        ("tiles_ratio", "Tiles"),
        ("fastest_ratio", "Fastest measured"),
    ):
        vals = [float(r[name]) for r in small]
        output.append(
            f"| {label} | {len(vals)} | {median(vals):.3f} | {sum(x <= 1.5 for x in vals)} |"
        )
    operational = [
        "| Complete event | Grid cells | Δx (m) | Hours | FV1 (s) | Hancock dense (s) | Tiles (s) | Auto (s) | Fastest explicit / FV1 | Hancock / FV1 steps | Fastest / FV1 per step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(large, key=case_order):
        operational.append(
            "| "
            + " | ".join(
                [
                    r["case"],
                    r["cells"],
                    number(r["dx"]),
                    number(r["hours"]),
                    number(r["fv1_s"], 5),
                    number(r["dense_s"], 5),
                    number(r["tiles_s"], 5),
                    number(r.get("auto_s"), 5),
                    number(r["fastest_ratio"]),
                    number(r["step_ratio"]),
                    number(r.get("per_step_ratio")),
                ]
            )
            + " |"
        )
    event_rows = [
        r
        for r in raw
        if r["stage"] in ("million-storms", "million-auto", "recession-controls")
    ]
    best = [float(r["fastest_ratio"]) for r in million]
    outcome = f"The {len(million)} matched 1,024 × 1,024 comparisons covered {len({r['landform'] for r in million})} landform families. The faster verified explicit Hancock schedule met the 1.5× FV1 target in {sum(v <= 1.5 for v in best)} of {len(best)} complete events. Its median ratio was {number(median(best) if best else None)}×. This retrospective choice gives an attainable measured cost for each event. The automatic runs used a policy frozen before these test observations."
    audit = [
        "# Corrective numerical and GPU audit",
        "",
        "The starting development revision was `360b3a37b5389128e7d4146eee00b1c3ca1a66ee`. All numerical forecasts use FP64. Configuration, implementation, device, input hashes, immutable attempts and output hashes identify every measurement. The historical 96-cell controls and original validation archives retain their own identities.",
        "",
        "## Defects and changes",
        "",
        "| Finding | Cause and correction | Verification |",
        "|---|---|---|",
        "| Hancock was represented by tiled runtime | The equal-cost and manuscript code selected a single execution identifier. Both policies now appear separately and the retrospective feasible comparison carries the exact winning run ID. | Dense/tiled field hashes, budgets and timestep controls. |",
        "| GPU scale inference used 96 cells | Grid size and physical domain were conflated. The prospective operational suite uses 250,000 through 4,194,304 cells. A separate 1 km² case varies hydraulic spacing at fixed physical extent. | Dimensions, integrated forcing and accepted horizon in each receipt. |",
        "| Configuration and qualification were implicit in Pareto rows | Every point now identifies formulation, limiter/flux, policy, spacing and source experiment. Dominance uses a strict improvement on at least one axis within an event. Unresolved points remain in the eligibility table. | Tie, reference-self, CSV-flag and unmatched-domain regression tests. |",
        "| Reference uncertainty and output cadence limited ranking | Selected cases have finer output intervals, successive same-domain refinements and an independent fine-grid full-SWE formulation. The more conservative of the two measured differences determines each observable’s screen. | Signed interval integration, finest-reference exclusions and native 15-minute products. |",
        "| Coarsening could remove hydraulic controls | The GIS comparison now accepts the actual represented bed. Full-domain drainage descriptors are supplemented with metre-grid windows across a prescribed opening. Fixed-bed, sampled-bed and initial-map controls are distinct. | Independent evaluation of continuous window elevations and four-neighbour hydraulic connectivity. |",
        "| Advisor omitted dense Hancock and operational domain sizes | Domain scale and execution mode enter the empirical neighbourhood. Validation groups choose the support width and interval expansion. Lost connectivity can require terrain refinement. | Disjoint realisation IDs, out-of-range refusal, frozen model digest, oracle and common-denominator comparisons. |",
        "| Conditional success obscured abstentions | Reports now include feasible coverage, agreement across all feasible opportunities, false feasibility per decision and both cost and error stratification. | Individual held-out decisions, grouped tables and independent archive verification. |",
        "",
        "The numerical adapter records imposed forcing, transport, state and timings. The separate profile worker measures instrumented kernels and peak allocations. [Forecast adapter](qualification/hydraulics.py) · [GPU profiling](qualification/profiled.py) · [Execution preflight](../../src/floodwarp/solver_advisor/execution.py)",
        "",
        "The reference screens and matched comparisons link decisions to experiment IDs. The terrain and advisor modules join those IDs to the represented DEM and preserve independent partitions. [Reference metrics](qualification/observations.py) · [Pareto dominance](metrics.py) · [Cost comparisons](comparisons.py) · [Control geometry](qualification/terrain.py) · [Training evidence](qualification/advisor_data.py) · [Held-out decisions](qualification/decisions.py)",
        "",
        "## Small-domain runtime correction",
        "",
        "| Hancock execution | Events | Median / FV1 | Events ≤1.5× |",
        "|---|---:|---:|---:|",
        *output,
        "",
        "For 96 cells, compacting the footprint costs extra launches and leaves little hydraulic work to skip. On large sparse rasters, each omitted tile avoids reconstruction, HLL flux, draining, update and accounting work. The separate common-cap measurements isolate this effect. Normal-policy storm times also include differences in accepted timesteps and rejected trials.",
        "",
        "## Operational storm matrix",
        "",
        outcome,
        "",
        *operational,
        "",
        f"The archive contains {len(event_rows)} complete operational forecasts across the scale and fixed-area designs. The largest measured primary grid has {max((int(r['cells']) for r in event_rows), default=0):,} cells. The memory observations are live device-pool allocations. Compilation, graph capture, hydraulic execution, extraction and receipt wall time are separate. For changing coastal stages the frame hydraulic timer surrounds the advance call. Prescribed boundary assignments between calls are included in end-to-end time.",
        "",
        "## Mechanisms, measured cost and Pareto evidence",
        "",
        timestep_attribution(events=million),
        "",
        "The steps-ratio column isolates restrictions from the remaining per-step cost. The scheduling table records wet and padded work fractions. The complete graph replays measure individual kernel categories independently of uninstrumented wall time. Dominated, tied and unresolved configurations are traceable to individual runs. Allocation footprint and runtime each have a distinct objective. [Execution rows](processed/million/policies.csv) · [Large-domain Pareto decisions](processed/million/pareto.csv) · [Kernel attribution](processed/million/graph_profile.csv) · [Equal-budget comparisons](tables/08_cost_accuracy.md)",
        "",
        "## Reference and terrain evidence",
        "",
        "Depth, wet-depth and maximum-depth comparisons use area averages on the common physical support specified by each case. Final inundation IoU uses thresholded common-support depths at 1, 5 and 10 cm. Native-grid component counts and wet-area differences are also recorded. A final-depth metric cannot measure a missing connection by itself. The prospective reference supplement retains 60-second interval discharge in fast cases and exact volume aggregation to 900 seconds.",
        "",
        "| Observable | Reassessed small cases | Originally qualified | Qualified on both finer checks |",
        "|---|---:|---:|---:|",
    ]
    for obj in sorted({r["objective"] for r in renewed}):
        g = [
            r
            for r in renewed
            if r["objective"] == obj and r["representation"] == "sampled"
        ]
        audit.append(
            f"| {obj} | {len(g)} | {sum(r['original_qualified'] == 'True' for r in g)} | {sum(r['qualified'] == 'True' for r in g)} |"
        )
    windows = {r["case"]: r for r in terrain if float(r["diagnostic_dx"]) == 10}
    losses = sum(
        r["native_connected"] == "True" and r["represented_connected"] == "False"
        for r in windows.values()
    )
    audit.extend(
        [
            "",
            f"At the prescribed test stage, {losses} of {len(windows)} independent case windows lost the metre-grid connection on the 10 m represented terrain. The stage is 1 cm above the lowest native through-going spill elevation. Each window covers 80 × 80 m around the known control and retains its physical coordinates. Whole-domain metrics use the full represented raster. [Terrain diagnostics](processed/million/terrain.csv) · [Fixed-bed and initial-water controls](processed/million/resolution_components.csv) · [Reference screens](processed/million/reference_checks.csv)",
            "",
            "## Held-out operational recommendations",
            "",
            "| Rule, at least one million cells | Decisions | Qualified feasible opportunities | Covered (%) | Abstained (%) | False feasible | Cheapest / opportunities (%) | Conditional regret (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in held:
        audit.append(
            "| "
            + " | ".join(
                str(r[k])
                if k
                in (
                    "rule",
                    "decisions",
                    "empirical_feasible_opportunities",
                    "false_feasible",
                )
                else number(r[k])
                for k in (
                    "rule",
                    "decisions",
                    "empirical_feasible_opportunities",
                    "feasible_coverage_pct",
                    "abstention_pct",
                    "false_feasible",
                    "agreement_all_feasible_pct",
                    "median_regret_pct",
                )
            )
            + " |"
        )
    from research.floodwarp_methods_paper.qualification.decisions import (
        previous_validation,
    )

    prior = previous_validation(study=study) / "summary.csv"
    old = next(r for r in csv_read(prior) if r["rule"] == "advisor")
    audit.extend(
        [
            "",
            f"The previously audited evaluation selected a method on {old['selected']} of {old['decisions']} decisions. It recorded {old['successful_selections']} successful selections, {old['abstentions']} abstentions, {float(old['median_regret_pct']):.1f}% median conditional regret and {float(old['empirical_error_interval_coverage_pct']):.1f}% empirical interval coverage. [Original immutable validation]({prior.relative_to(study)}) · [Matched old and revised decisions](processed/advisor/previous_comparison.csv)",
        ]
    )
    if not held:
        audit.extend(["", "Independent operational advisor validation is pending."])
    by_rule = {r["rule"]: r for r in held}
    diagnosis = "The held-out physical cases determine whether a measured improvement in computational regret also improves feasible coverage."
    if by_rule and by_rule["advisor"]["feasible_coverage_pct"]:
        delta = float(by_rule["advisor"]["feasible_coverage_pct"]) - max(
            float(by_rule[k]["feasible_coverage_pct"])
            for k in ("fv1", "hancock_dense", "slope")
        )
        diagnosis = f"On the common million-cell feasible denominator, advisor coverage differed from the strongest simple baseline by {delta:+.1f} percentage points."
    audit.extend(
        [
            "",
            diagnosis
            + " The runtime-only comparison and the objective-specific feasibility comparison answer separate questions. All rows from an independent terrain realisation belong to the same split. Pre-run inputs describe terrain and prescribed forcing. Realised wet fraction is used only to stratify results after validation. [Held-out individual decisions](processed/advisor/heldout.csv) · [By landform, grid, size and wet fraction](processed/advisor/validation_strata.csv) · [Runtime-only decisions](processed/advisor/runtime_only.csv)",
            "",
            "## Engineering recommendation and remaining limits",
            "",
            "Keep adaptive Hancock as the development formulation and explicit dense execution as its default. Tiled execution is competitive on adequately large, spatially restricted storms. The optional single-decision preflight uses the measured work crossover on the reference GPU. Each explicit scheduler remains available. Its selection cost is paid during setup. This study does not qualify automatic switching every timestep.",
            "",
            "The primary event timings cover a single reference GPU and normal timestep policy. Local reference windows do not establish full-catchment hydraulic truth. Unqualified objectives, zero or unresolved peaks, the first-order time split of hydraulic structures, shared outlet closures, and missing fine-grid comparisons constrain release claims. A measured difference smaller than the reference change cannot establish a robust accuracy ranking. The Pareto archive describes empirical reference errors, including close unresolved comparisons. Release qualification requires representative independent forcing and boundary verification.",
            "",
            "[Reproduction instructions](README.md) · [Correction regression tests](../../tests/test_million_qualification.py) · [Actual run provenance](processed/million/archive.csv)",
        ]
    )
    if pending:
        audit[2:2] = [
            f"**Qualification in progress.** {pending} numerical records remain outstanding. Values below cover completed events.",
            "",
        ]
    audit.extend(
        [
            "",
            "## Completion by registered stage",
            "",
            "| Numerical stage | Complete | Registered |",
            "|---|---:|---:|",
            *inventory,
        ]
    )
    from research.floodwarp_methods_paper import evidence

    for rec in evidence.records(study=study, stages=("check",)):
        tested = json.loads((study / rec["directory"] / "result.json").read_text())
        audit.extend(
            [
                "",
                f"The full CUDA regression receipt records **{tested['summary']}**. [Test log]({rec['log']}) · [Committed numerical source proof](manifests/corrective_source_tree.json)",
            ]
        )
    attempts = csv_read(p / "metrics/attempt_catalog.csv")
    broken = [
        r
        for r in attempts
        if r["stage"] in stages.NUMERICAL and r["status"] in ("incomplete", "failed")
    ]
    if broken:
        audit.extend(
            [
                "",
                f"The catalogue retains {len(broken)} interrupted or failed numerical attempts outside all hydraulic and timing comparisons. [Attempt catalogue](processed/metrics/attempt_catalog.csv)",
            ]
        )
        for r in broken:
            audit.append(
                f"- {r['stage']}, {r['case']}, {r['method']} — {r['status']}, attempt `{r['attempt']}`."
            )
    groups = defaultdict(dict)
    for row in intervals:
        cat = groups[row["case"], row["method"]]
        cat[row["category"]] = cat.get(row["category"], 0.0) + float(
            row["kernel_seconds"]
        )
    if groups:
        audit.extend(
            [
                "",
                "## Complete graph replay timings",
                "",
                "| Case | Configuration | Largest kernel category | Device seconds | Share of measured kernel time |",
                "|---|---|---|---:|---:|",
            ]
        )
        for (case, method), group in sorted(groups.items()):
            largest = max(group, key=group.get)
            audit.append(
                f"| {case} | {method} | {largest} | {group[largest]:.3f} | {100 * group[largest] / sum(group.values()):.1f}% |"
            )
    repeats_path = p / "million/repeat_times.csv"
    if repeats_path.exists():
        full = [r for r in csv_read(repeats_path) if int(r["n"]) >= 3]
        if full:
            variability = [float(r["range_pct"]) for r in full]
            audit.extend(
                [
                    "",
                    f"Across {len(full)} complete timing controls with at least three runs, observed hydraulic runtime ranges spanned {min(variability):.2f}% to {max(variability):.2f}% of the within-control median. [Individual timing ranges](processed/million/repeat_times.csv)",
                ]
            )
    auto = [r for r in large if r.get("auto_s")]
    if auto:
        relative = [float(r["auto_regret_pct"]) for r in auto]
        audit.extend(
            [
                "",
                f"The prospective preflight used tiled execution in {sum(r['selected_policy'] == 'tiles' for r in auto)} of {len(auto)} measured grids of at least one million cells. Its measured excess runtime over the fastest explicit observation had a median of {median(relative):+.1f}% and a maximum of {max(relative):+.1f}%. Separate executions include timing variability.",
            ]
        )
    fronts = csv_read(p / "million/pareto.csv")
    if fronts:
        audit.extend(
            [
                "",
                "## Measured configurations by observable",
                "",
                "| Observable | Eligible | Empirical frontier | Frontier under reference sensitivity | Unresolved |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for obj in sorted({r["objective"] for r in fronts}):
            sample = [r for r in fronts if r["objective"] == obj]
            valid = [r for r in sample if r["eligible"] == "True"]
            audit.append(
                f"| {obj} | {len(valid)} | {sum(r['nondominated'] == 'True' for r in valid)} | {sum(r['sensitivity_nondominated'] == 'True' for r in valid)} | {len(sample) - len(valid)} |"
            )
    audit.extend(
        [
            "",
            "Final 1 cm inundation is the common prediction target in the enlarged advisor. The historical paper also retains time-integrated inundation results. Costs are stratified by the Hancock wetted and scheduled fractions at the end of each 15-minute window and use the FV1 time for exactly the same physical window. [Window-level costs](processed/million/wetness_cost.csv) · [Joint solver and resolution validation](processed/advisor/joint_resolution.csv)",
        ]
    )
    feasible = csv_read(p / "pareto/least_cost.csv")
    winners = defaultdict(dict)
    for row in feasible:
        if row.get("cohort") != "operational" or row["representation"] != "sampled":
            continue
        method = row.get("sensitivity_method")
        value = (
            f"{method}, {number(row['sensitivity_dx'])} m"
            if method
            else ("no tested option" if int(row["qualified_options"]) else "unresolved")
        )
        winners[row["case"]][row["objective"]] = value
    if winners:
        audit.extend(
            [
                "",
                "## Cheapest measured method and grid meeting tolerance",
                "",
                "Each entry uses the measured error plus the reported refinement sensitivity. Tolerances are prescribed in the advisor configuration. No tested option means a qualified comparison existed but the conservative criterion was missed.",
                "",
                "| Physical event | Final depth | Final inundation | Outlet hydrograph |",
                "|---|---|---|---|",
            ]
        )
        for case, result in sorted(winners.items()):
            audit.append(
                f"| {case} | {result['depth']} | {result['inundation']} | {result['discharge']} |"
            )
    summary = "\n\n".join(
        [
            "## Million-cell operational controls",
            outcome,
            "The partial-domain rain and widespread inundation cases have different scheduling and timestep costs. [Complete operational records](../tables/11_execution.md) · [Costs by scheduled footprint](../figures/13_execution_cost.pdf) · [Finer reference checks](../tables/12_reference_qualification.md) · [Operational hydrographs](../figures/18_operational_storms.pdf).",
            "## Held-out evidence at operational scale",
            diagnosis
            + " [Scale-specific denominators](../tables/13_advisor_scale.md).",
        ]
    )
    return "\n".join(audit) + "\n", summary


def case_order(row):
    return row["case"], float(row["dx"])
