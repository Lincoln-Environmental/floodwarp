"""Render publication tables from archived observations and processed comparisons."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from research.floodwarp_methods_paper.metrics import csv_read, csv_write, load


LABELS = {
    "fv1": "FV1",
    "low_froude_hll": "Low-Froude FV1",
    "euler_mc": "MUSCL–Euler",
    "hancock_adaptive": "Hancock tiles",
    "hancock_dense": "Hancock dense",
    "adaptive": "FV2 RK2",
    "li": "Local inertia",
}


def display(value):
    if value is None or value == "":
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str) and ("." in value or "e" in value.lower()):
        try:
            return f"{float(value):.4g}"
        except ValueError:
            pass
    return str(value)


def latex(value):
    escape = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "#": r"\#",
        "&": r"\&",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    return "".join(escape.get(c, c) for c in display(value))


def heading(key):
    text = key.replace("_", " ")
    for old, new in (
        ("rmse", "RMSE"),
        ("m3s", "m³/s"),
        ("m2s", "m²/s"),
        ("pct", "%"),
        ("median ", "Median "),
    ):
        text = text.replace(old, new)
    return text


def table_panels(*, columns, rows, caption):
    keys = [
        key
        for key in (
            "rule",
            "family",
            "method",
            "reference",
            "method_a",
            "dx_a",
            "method_b",
            "dx_b",
            "objective",
        )
        if key in columns
    ]
    if len(columns) <= 7:
        panels = [columns]
    else:
        rest = [key for key in columns if key not in keys]
        panels = [["Identifier", *rest[i : i + 4]] for i in range(0, len(rest), 4)]
    tex = [r"\begingroup", r"\small", r"\setlength{\tabcolsep}{2pt}"]
    for i, cols in enumerate(panels):
        width = 0.88 / len(cols)
        descriptor = "".join(r"p{" + f"{width:.4f}" + r"\linewidth}" for k in cols)
        panel = caption if len(panels) == 1 else f"{caption}, panel {i + 1}"
        tex.extend(
            [
                r"\begin{longtable}{" + descriptor + "}",
                r"\caption{" + latex(panel) + r"}\\",
                r"\hline",
                " & ".join(latex(heading(k)) for k in cols) + r" \\",
                r"\hline\endhead",
            ]
        )
        for r in rows:
            identifier = ", ".join(display(r.get(k)).replace("_", " ") for k in keys)
            cells = [identifier if k == "Identifier" else r.get(k) for k in cols]
            tex.append(" & ".join(latex(v) for v in cells) + r" \\")
        tex.append(r"\end{longtable}")
    tex.append(r"\endgroup")
    return tex


def render(*, directory, name, rows, caption):
    if not rows:
        raise ValueError(f"No evidence for {name}")
    csv_write(file=directory / (name + ".csv"), rows=rows)
    columns = list(rows[0])
    head = "| " + " | ".join(heading(k) for k in columns) + " |"
    lines = ["# " + caption, "", head, "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(display(r.get(k)).replace("|", r"\|") for k in columns)
            + " |"
        )
    tex = table_panels(columns=columns, rows=rows, caption=caption)
    (directory / (name + ".md")).write_text("\n".join(lines) + "\n")
    (directory / (name + ".tex")).write_text("\n".join(tex) + "\n")


def group(*, rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return sorted(groups.items())


def middle(*, rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return median(values) if values else None


def execution_table(*, policy):
    return (
        "Complete-event production cost at operational scales",
        [
            dict(
                case=r["case"],
                dx=r["dx"],
                cells=r["cells"],
                fv1_s=r["fv1_s"],
                dense_s=r["dense_s"],
                tiles_s=r["tiles_s"],
                auto_s=r.get("auto_s"),
                fastest_ratio=r["fastest_ratio"],
                work_fraction=r["work_fraction"],
                step_ratio=r["step_ratio"],
                per_step_ratio=r["per_step_ratio"],
                fastest_measured_policy=r["fastest_all_policy"],
                fastest_measured_ratio=r["fastest_all_ratio"],
            )
            for r in policy
            if int(r["cells"]) >= 250000
        ],
    )


def generate(*, study, output=None):
    from research.floodwarp_methods_paper.evidence import archived_publication

    with archived_publication(study=study):
        return from_records(study=study, output=output)


def from_records(*, study, output):
    processed = study / "processed"
    out = study / "tables" if output is None else output
    output = {}
    definitions = json.loads((study / "configs/analytical/methods.json").read_text())
    output["01_methods"] = ("Formulations and governing equations", definitions)
    domains = json.loads(
        (study.parents[1] / "validation/fv1_diffusion_review/domains.json").read_text()
    )
    desc = [
        dict(
            case=d["name"],
            family=d["family"],
            grade=d["slope"],
            roughness=d["n"],
            duration_s=d["duration"],
            length_m=d["length"],
        )
        for d in domains
    ]
    controls = next(
        p
        for rec, r, p in load(study=study, stages=("analytical",))
        if rec["spec"]["kind"] == "archived-controls"
    )
    extras = set()
    for f in controls.glob("*.json"):
        content = json.loads(f.read_text())
        if isinstance(content, list):
            extras.update(
                str(r["case"])
                for r in content
                if isinstance(r, dict) and isinstance(r.get("case"), str)
            )
    for case in sorted(extras - {d["name"] for d in domains}):
        desc.append(
            dict(
                case=case,
                family="independent control",
                grade=None,
                roughness=None,
                duration_s=None,
                length_m=None,
            )
        )
    output["02_exact_domains"] = ("Exact benchmarks and independent controls", desc)
    terrain = json.loads((processed / "metrics/terrain.json").read_text())
    top = []
    for name, value in sorted(terrain.items()):
        if value["model"]["dx_m"] != 10:
            continue
        m = value["model"]
        c = value["comparison"]
        top.append(
            dict(
                terrain=name,
                grade=m["median_slope"],
                relief_m=m["relief_m"],
                drainage_m_per_m2=m["channel_density_m_per_m2"],
                storage_m3=m["depression_storage_m3"],
                aggregation_rmse_m=c["elevation_rmse_m"],
                lost_channel_fraction=c["channel_loss_fraction"],
            )
        )
    output["03_terrain"] = ("Terrain realisations aggregated to ten metres", top)
    exact = csv_read(processed / "metrics/analytical.csv")
    accuracy = []
    for (method, dx), g in group(
        rows=[r for r in exact if r["stage"] == "analytical"], keys=("method", "dx")
    ):
        accuracy.append(
            dict(
                method=LABELS[method],
                dx_m=dx,
                n=len(g),
                median_depth_rmse_m=middle(rows=g, key="depth_rmse_m"),
                median_transport_rmse_m2s=middle(rows=g, key="hydrograph_rmse_m2s"),
                median_extent_error=middle(rows=g, key="inundation_error"),
            )
        )
    output["04_analytical_accuracy"] = (
        "Median exact-solution errors across the nineteen primary domains",
        accuracy,
    )
    formal = csv_read(processed / "convergence/orders.csv")
    conv = []
    for (case, method, metric), g in group(
        rows=formal, keys=("case", "method", "metric")
    ):
        conv.append(
            dict(
                case=case,
                method=method,
                metric=metric,
                design="spatial",
                adjacent_pairs=len(g),
                median_order=middle(rows=g, key="order"),
            )
        )
    temporal = csv_read(processed / "convergence/temporal.csv")
    for (case, method, metric, design), g in group(
        rows=temporal, keys=("case", "method", "metric", "design")
    ):
        conv.append(
            dict(
                case=case,
                method=method,
                metric=metric,
                design=design,
                adjacent_pairs=sum(bool(r["order"]) for r in g),
                median_order=middle(rows=g, key="order"),
            )
        )
    fixed = csv_read(processed / "convergence/fixed_dt_space.csv")
    for (case, method, metric), g in group(
        rows=fixed, keys=("case", "method", "metric")
    ):
        conv.append(
            dict(
                case=case,
                method=method,
                metric=metric,
                design="spatial at fixed dt",
                adjacent_pairs=sum(bool(r["order"]) for r in g),
                median_order=middle(rows=g, key="order"),
            )
        )
    output["05_convergence"] = (
        "Adjacent refinement orders, with space and time analysed independently",
        conv,
    )
    rows = csv_read(processed / "metrics/storms.csv")
    primary = [
        r
        for r in rows
        if r["stage"] == "long-storms"
        and r["reference_setup"] == "False"
        and r["representation"] == "sampled"
    ]
    results = []
    for (land, method), g in group(rows=primary, keys=("landform", "method")):
        d = [r for r in g if r["qualified_depth"] == "True"]
        q = [r for r in g if r["qualified_discharge"] == "True"]
        i = [r for r in g if r["qualified_inundation"] == "True"]
        results.append(
            dict(
                family=land,
                method=LABELS[method],
                events=len(g),
                depth_n=len(d),
                depth_rmse_m=middle(rows=d, key="depth_rmse_m"),
                extent_n=len(i),
                extent_error=middle(rows=i, key="inundation_error"),
                outlet_n=len(q),
                outlet_rmse_m3s=middle(rows=q, key="outlet_rmse_m3s"),
            )
        )
    output["06_storm_accuracy"] = (
        "Median ten-metre storm errors against qualified fine-grid references",
        results,
    )
    perf = []
    for (method,), g in group(rows=primary, keys=("method",)):
        perf.append(
            dict(
                matrix="prospective 10 m storms",
                method=LABELS[method],
                runs=len(g),
                median_hydraulic_s=middle(rows=g, key="runtime_s"),
                median_steps=middle(rows=g, key="steps"),
                median_pool_MiB=middle(rows=g, key="memory_mib"),
            )
        )
    original = csv_read(
        study.parents[1] / "validation/hancock_long_storm_review/event_summary.csv"
    )
    for (cells, method, mode), g in group(
        rows=[r for r in original if float(r["cells"]) >= 250000],
        keys=("cells", "method", "mode"),
    ):
        perf.append(
            dict(
                matrix=f"archival {cells} cells",
                method=method + " " + mode,
                runs=len(g),
                median_hydraulic_s=middle(rows=g, key="hydraulic_s"),
                median_steps=middle(rows=g, key="steps"),
                median_pool_MiB=middle(rows=g, key="pool_mib"),
            )
        )
    output["07_runtime"] = ("Operational GPU cost and live process-pool memory", perf)
    pairs = csv_read(processed / "comparisons/equal_cost.csv")
    cost = []
    for (cohort, a, dx, b, dy, obj), g in group(
        rows=[r for r in pairs if r["representation"] == "sampled"],
        keys=("cohort", "method_a", "dx_a", "method_b", "dx_b", "objective"),
    ):
        valid = [r for r in g if r["qualified"] == "True"]
        cost.append(
            dict(
                cohort=cohort,
                method_a=a,
                dx_a=dx,
                method_b=b,
                dx_b=dy,
                objective=obj,
                matched=len(g),
                qualified=len(valid),
                median_error_a=middle(rows=valid, key="error_a"),
                median_error_b=middle(rows=valid, key="error_b"),
                median_cost_ratio=middle(rows=valid, key="cost_ratio"),
            )
        )
    output["08_cost_accuracy"] = (
        "Measured cost and accuracy on matched physical domains",
        cost,
    )
    transfer = csv_read(processed / "comparisons/transfer.csv")
    archive = csv_read(processed / "comparisons/archival_analytical_transfer.csv")
    transfers = []
    for label, source in (
        ("synthetic numerical reference", transfer),
        ("archival exact", archive),
    ):
        for (obj, bound), g in group(rows=source, keys=("objective", "bound")):
            valid = [r for r in g if r.get("qualified_for_objective", "True") == "True"]
            transfers.append(
                dict(
                    reference=label,
                    objective=obj,
                    bound=bound,
                    forecasts=len(g),
                    reference_qualified=len(valid),
                    fit_reference_qualified=sum(
                        r.get("training_reference_qualified", "True") == "True"
                        for r in g
                    ),
                    median_n_multiplier=middle(rows=g, key="multiplier"),
                    median_score_improvement=middle(rows=valid, key="improvement"),
                    deteriorations=sum(float(r["improvement"]) < 0 for r in valid),
                )
            )
    output["09_calibration"] = (
        "Bounded roughness transfer, positive score changes indicate improvement",
        transfers,
    )
    output["10_advisor"] = (
        "Independent solver selection assessment",
        csv_read(processed / "advisor/summary.csv"),
    )
    large = processed / "million"
    policy = csv_read(large / "policies.csv")
    output["11_execution"] = execution_table(policy=policy)
    refs = csv_read(large / "reference_checks.csv")
    output["12_reference_qualification"] = (
        "Per-observable independent refinement and cross-formulation screens",
        [
            dict(
                objective=k,
                representation=rep,
                cohort=cohort,
                physical_cases=len(g),
                qualified=sum(r["qualified"] == "True" for r in g),
                unresolved=sum(r["qualified"] != "True" for r in g),
            )
            for (k, rep, cohort), g in group(
                rows=refs, keys=("objective", "representation", "cohort")
            )
        ],
    )
    strat = csv_read(processed / "advisor/validation_strata.csv")
    output["13_advisor_scale"] = (
        "Independent held-out feasibility and cost across computational scales",
        [
            {
                k: r[k]
                for k in (
                    "group",
                    "rule",
                    "decisions",
                    "empirical_feasible_opportunities",
                    "feasible_coverage_pct",
                    "abstention_pct",
                    "false_feasible",
                    "agreement_all_feasible_pct",
                    "median_regret_pct",
                    "empirical_error_interval_coverage_pct",
                )
            }
            for r in strat
            if r["axis"] == "scale"
        ],
    )
    updated = study / "fidelity/outlet_results/events.csv"
    if updated.exists():
        keys = ("run_id", "method", "dx_m", "extension_m", "cfl_factor", "end_h", "minute_peak_m3s",
                "fixed_15min_peak_m3s", "moving_15min_peak_m3s", "gauge_m3", "footprint_infiltration_m3",
                "footprint_storage_m3", "steps", "hydraulic_s")
        output["16_free_boundary"] = ("Fixed original outlet, complete 16-hour storm comparisons",
                                      [{k:r[k] for k in keys} for r in csv_read(file=updated)])
        for decision_type, title in (("least_cost", "Lowest measured hydraulic cost meeting the stated tolerance"),
                             ("equal_cost", "Lowest qualified measured error within an FV1 runtime budget")):
            decisions = []
            for kind in ("fidelity", "depth_extent"):
                for row in csv_read(file=updated.parent / f"{decision_type}_{kind}.csv"):
                    if row["representation"] != "0 m downstream extension" or row["station"] not in ("outlet", "domain"):
                        continue
                    metric = row["objective"]
                    if decision_type == "least_cost":
                        context = dict(tolerance=row["tolerance"], qualified=row["assessable"],
                                       feasible=row["feasible"], method=row["method"], dx_m=row["dx"], id=row["id"])
                    else:
                        context = dict(budget_dx_m=row["budget_dx"], budget_s=row["budget_s"],
                                       qualified_in_budget=row["assessable_in_budget"], ranking=row["ranking_status"],
                                       method=row["selected_method"], dx_m=row["selected_dx"], id=row["selected_id"])
                    decisions.append(dict(objective=metric, unit="m" if kind == "depth_extent" and "depth" in metric
                                          else "fraction" if kind == "depth_extent" else "%",
                                          station=row["station"], end_h=row["end_h"], interval_s=row["interval_s"],
                                          reference_id=row["reference_id"], **context, policy=row["execution"],
                                          absolute_error=row["measured_error"], reference_sensitivity=row["reference_sensitivity"],
                                          hydraulic_s=row["runtime_s"], accepted_steps=row["steps"],
                                          hydraulic_s_per_step=row["hydraulic_s_per_step"]))
            name = "19_incised_cost_at_tolerance" if decision_type == "least_cost" else "20_incised_error_at_budget"
            output[name] = (title + " on the original incised domain", decisions)
    transfer_file = study / "fidelity/orton_transfer/transfer.csv"
    if transfer_file.exists():
        measured = csv_read(file=transfer_file)
        keys = ("id", "dx_m", "method", "peak_m3s", "peak_end_s", "rain", "infiltration",
                "boundary_inflow", "boundary_outflow", "storage_m3", "residual",
                "final_area_5cm_m2", "maximum_area_5cm_m2", "final_area_10cm_m2", "maximum_area_10cm_m2")
        output["17_orton_storm"] = ("Orton Bradley 48-hour design-storm transfer on completed grids",
                                   [{k: r[k] for k in keys} for r in measured])
        keys = ("id", "dx_m", "method", "policy", "steps", "retries", "minimum_dt_s", "mean_dt_s",
                "hydraulic_s", "end_to_end_solver_s", "sampled_peak_pool_mib")
        output["18_orton_cost"] = ("Orton Bradley measured 48-hour transfer costs and timestep counts",
                                  [{k: r[k] for k in keys} for r in measured])
    for name, (caption, data) in output.items():
        render(directory=out, name=name, rows=data, caption=caption)
    return {name: len(rows) for name, (_, rows) in output.items()}


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    generate(study=study)
