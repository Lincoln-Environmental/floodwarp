"""Measured domain-scale, execution and terrain figures from archived tabulations."""

from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.figures.generate import (
    save,
    COLOURS,
    LABELS,
    UNITS,
)


def scale_cost(*, rows, out):
    policies = [
        ("dense_ratio", "Dense", "#c46b18"),
        ("tiles_ratio", "Tiles", "#0072b2"),
        ("auto_ratio", "Auto", "#704d9f"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(8.3, 5.5), layout="constrained")
    for ax, (key, title) in zip(
        axs.flat,
        (
            ("cells", "Hydraulic cells"),
            ("wet_fraction", "Mean positive-depth fraction"),
            ("work_fraction", "Mean scheduled fraction (tiles)"),
            ("hancock_steps", "Accepted Hancock timesteps"),
        ),
        strict=True,
    ):
        for field, name, color in policies:
            data = [r for r in rows if r.get(field)]
            ax.scatter(
                [float(r[key]) for r in data],
                [float(r[field]) for r in data],
                label=name,
                color=color,
                s=13,
                alpha=0.7,
            )
        if key in ("cells", "hancock_steps"):
            ax.set_xscale("log")
        ax.set(xlabel=title, ylabel="Hydraulic runtime / FV1")
    axs[0, 0].legend(frameon=False, ncol=3)
    save(fig=fig, folder=out, name="13_execution_cost")
    svg = out / "13_execution_cost.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def value(*, study, out, by_case, fronts):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.1), layout="constrained")
    for ax, obj in zip(axes, ("depth", "discharge", "inundation"), strict=True):
        for method in (
            "hancock_dense",
            "hancock_adaptive",
            "hancock_auto",
            "adaptive",
            "li",
        ):
            pts = []
            for case in sorted(by_case):
                rows = by_case[case]
                for dx in sorted({r["dx"] for r in rows}, key=float):
                    g = {
                        r["method"]: r
                        for r in rows
                        if r["dx"] == dx
                        and r["objective"] == obj
                        and r.get("eligible") == "True"
                    }
                    if "fv1" not in g or method not in g:
                        continue
                    a, b = g["fv1"], g[method]
                    if float(a["error"]) <= 0:
                        continue
                    pts.append(
                        (
                            float(b["runtime_s"]) / float(a["runtime_s"]),
                            1 - float(b["error"]) / float(a["error"]),
                        )
                    )
            if pts:
                ax.scatter(
                    *zip(*pts),
                    s=12,
                    alpha=0.65,
                    color=COLOURS.get(method, "#704d9f"),
                    label=LABELS.get(method, "Hancock auto"),
                )
        ax.axhline(0, color="#444", lw=0.7)
        ax.axvline(1, color="#444", lw=0.7)
        ax.set(xlabel="Runtime / FV1", ylabel="Fractional error reduction", title=obj)
        ax.set_yscale("symlog", linthresh=1)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig=fig, folder=out, name="14_relative_cost_accuracy")
    pairs = metrics.csv_read(study / "processed/comparisons/equal_cost.csv")
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.0), layout="constrained")
    for ax, obj in zip(axs, ("depth", "discharge", "inundation"), strict=True):
        for a, b, m in (
            (1, 1, "o"),
            (5, 5, "s"),
            (10, 10, "D"),
            (1, 5, "^"),
            (5, 10, "v"),
        ):
            sub = [
                r
                for r in pairs
                if r["objective"] == obj
                and r["method_a"] == "fv1"
                and r["method_b"] == "fastest qualified Hancock"
                and float(r["dx_a"]) == a
                and float(r["dx_b"]) == b
                and r["qualified"] == "True"
                and r["representation"] == "sampled"
                and float(r["error_a"]) > 0
            ]
            if sub:
                ax.scatter(
                    [float(r["cost_ratio"]) for r in sub],
                    [float(r["error_b"]) / float(r["error_a"]) for r in sub],
                    label=f"{a} / {b} m",
                    marker=m,
                    s=18,
                    alpha=0.7,
                )
        ax.axvline(1, color=".4", ls=":")
        ax.axhline(1, color=".4", ls=":")
        ax.set(
            xlabel="Fastest qualified Hancock / FV1 time",
            ylabel="Hancock / FV1 error",
            title=obj,
        )
        ax.set_yscale("symlog", linthresh=1.0)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        title="Measured FV1 / Hancock spacings",
        frameon=False,
    )
    save(fig=fig, folder=out, name="15_equal_budget")


def family_frontiers(*, rows, out):
    groups = defaultdict(dict)
    for r in rows:
        groups[r["landform"]].setdefault(r["case"], []).append(r)
    for family, cases in sorted(groups.items()):
        fig, axs = plt.subplots(
            len(cases),
            len(UNITS),
            figsize=(15, 1.9 * len(cases) + 0.8),
            layout="constrained",
            squeeze=False,
        )
        for row, (case, data) in zip(axs, sorted(cases.items()), strict=True):
            for ax, obj in zip(row, UNITS, strict=True):
                measured = [
                    r for r in data if r["eligible"] == "True" and r["objective"] == obj
                ]
                for r in measured:
                    change = r.get("reference_change")
                    if change and float(change) > 0:
                        error = float(r["error"])
                        ax.errorbar(
                            float(r["runtime_s"]),
                            error,
                            yerr=[[min(error, float(change))], [float(change)]],
                            fmt="none",
                            color=COLOURS.get(r["method"], "#704d9f"),
                            alpha=0.45,
                            lw=0.6,
                        )
                    ax.scatter(
                        float(r["runtime_s"]),
                        float(r["error"]),
                        color=COLOURS.get(r["method"], "#704d9f"),
                        marker={1: "^", 5: "s", 10: "o"}.get(float(r["dx"]), "D"),
                        s=15,
                    )
                edge = sorted(
                    (float(r["runtime_s"]), float(r["error"]))
                    for r in measured
                    if r["nondominated"] == "True"
                )
                if edge:
                    ax.step(*zip(*edge), where="post", color="#666", lw=0.8)
                if not measured:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(
                        0.5,
                        0.45,
                        "Unresolved",
                        ha="center",
                        transform=ax.transAxes,
                        fontsize=7,
                    )
                ax.set(xlabel="Hydraulic time (s)", ylabel=UNITS[obj])
            row[0].set_title(case, fontsize=7)
        labels = dict(LABELS, hancock_auto="Hancock auto")
        methods = {row["method"] for data in cases.values() for row in data}
        spacings = {float(row["dx"]) for data in cases.values() for row in data}
        handles = [
            Line2D(
                [],
                [],
                color=COLOURS.get(k, "#704d9f"),
                marker="o",
                ls="",
                label=v,
                markersize=4,
            )
            for k, v in labels.items()
            if k in methods
        ]
        handles.extend(
            Line2D([], [], color=".35", marker=m, ls="", label=f"{d} m", markersize=4)
            for d, m in ((1, "^"), (5, "s"), (10, "o"))
            if d in spacings
        )
        fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
        save(fig=fig, folder=out, name=family)


def terrain_validation(*, study, out):
    p = study / "processed"
    rows = metrics.csv_read(p / "million/terrain.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), layout="constrained")
    keys = (
        "max_local_barrier_lowering_m",
        "storage_change_m3",
        "changed_outlet_fraction",
    )
    names = ("Largest lowering (m)", "Storage change (m³)", "Changed outlet fraction")
    unique = {(r["case"], r["diagnostic_dx"]): r for r in rows}
    for ax, key, title in zip(axes, keys, names, strict=True):
        for case in sorted({k[0] for k in unique}):
            data = sorted(
                (float(d), float(v[key])) for (c, d), v in unique.items() if c == case
            )
            ax.plot(*zip(*data), marker=".", lw=0.7, alpha=0.65)
        ax.set(xlabel="Local control DEM spacing (m)", ylabel=title, xticks=[1, 5, 10])
    save(fig=fig, folder=out, name="16_control_resolution")
    summary = metrics.csv_read(p / "advisor/validation_strata.csv")
    test = [r for r in summary if r["axis"] == "scale"]
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.3), layout="constrained")
    key_names = (
        ("feasible_coverage_pct", "Feasible opportunities covered (%)"),
        ("abstention_pct", "Abstentions / all decisions (%)"),
        ("agreement_all_feasible_pct", "Cheapest choices / opportunities (%)"),
    )
    groups = sorted({r["group"] for r in test})
    for ax, (key, title) in zip(axes, key_names, strict=True):
        for j, rule in enumerate(sorted({r["rule"] for r in test})):
            g = {r["group"]: r for r in test if r["rule"] == rule}
            ax.scatter(
                [i + (j - 2) * 0.08 for i, k in enumerate(groups) if g[k][key] != ""],
                [float(g[k][key]) for k in groups if g[k][key] != ""],
                s=22,
                label=rule,
            )
        ax.set_xticks(range(len(groups)), groups, rotation=28, ha="right", fontsize=7)
        ax.set_ylabel(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig=fig, folder=out, name="17_heldout_scale")


def event_panel(*, records, cases, output, name):
    fig, axes = plt.subplots(
        4,
        len(cases),
        figsize=(3.3 * len(cases), 6.4),
        squeeze=False,
        sharex="col",
        layout="constrained",
    )
    for column, case in enumerate(cases):
        data = [r for r in records if r["case"] == case and int(r["cells"]) == 1024**2]
        for method, color, label in (
            ("fv1", "#0072B2", "FV1"),
            ("hancock_dense", "#D55E00", "Hancock dense"),
            ("hancock_adaptive", "#009E73", "Hancock tiled"),
        ):
            rows = sorted((r for r in data if r["method"] == method), key=end_time)
            if not rows:
                continue
            x = [float(r["end_s"]) / 3600 for r in rows]
            if method == "fv1":
                axes[0, column].stairs(
                    [float(r["rain_mm_h"]) for r in rows],
                    [0, *x],
                    fill=True,
                    color="#56b4e9",
                )
            for axis, key in (
                (1, "outlet_m3s"),
                (2, "wet_fraction"),
                (3, "hydraulic_s"),
            ):
                axes[axis, column].plot(
                    x,
                    [float(r[key]) for r in rows],
                    color=color,
                    label=label,
                    ls="--" if method == "hancock_adaptive" else "-",
                    lw=1.1,
                )
        axes[0, column].set_title(case.split("-")[0].capitalize())
        axes[-1, column].set_xlabel("Model time (h)")
        for ax in axes[:, column]:
            ax.axvline(6, color=".4", ls=":", lw=0.8)
    for row, label in enumerate(
        (
            "Rainfall (mm/h)",
            "Outlet flow (m³/s)",
            "Wet fraction",
            "GPU time (s)\nper 15 min",
        )
    ):
        axes[row, 0].set_ylabel(label)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig=fig, folder=output, name=name)


def end_time(row):
    return float(row["end_s"])


def generate(*, study):
    out = study / "figures"
    rows = metrics.csv_read(study / "processed/million/policies.csv")
    scale_cost(rows=rows, out=out)
    large = metrics.csv_read(study / "processed/million/pareto.csv")
    small = metrics.csv_read(study / "processed/pareto/storms.csv")
    by_case = defaultdict(list)
    for r in small + large:
        if r["representation"] == "sampled":
            by_case[r["case"]].append(r)
    value(study=study, out=out, by_case=by_case, fronts=large)
    family_frontiers(rows=large, out=study / "supplementary/pareto/million")
    short = metrics.csv_read(study / "processed/million/short_interval_pareto.csv")
    family_frontiers(rows=short, out=study / "supplementary/pareto/short_interval")
    terrain_validation(study=study, out=out)
    histories = metrics.csv_read(study / "processed/million/event_intervals.csv")
    complete = sorted(
        {
            r["case"]
            for r in histories
            if int(r["cells"]) == 1024**2 and r["method"] == "hancock_adaptive"
        }
    )
    main = [
        case
        for case in complete
        if len(case.split("-")) == 4 and case.split("-")[1] in ("701", "702", "708")
    ]
    if main:
        event_panel(
            records=histories, cases=main, output=out, name="18_operational_storms"
        )
    for case in complete:
        event_panel(
            records=histories,
            cases=[case],
            output=study / "supplementary/operational_storms",
            name=case,
        )
