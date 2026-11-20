"""Regenerate cost envelopes and figures from recorded hydraulic runs."""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from validation.fv1_riemann import report_json


def read(*, directory, name):
    return json.loads((directory / (name + ".json")).read_text())


def efficiencies(*, directory):
    records = []
    for source in ("exact_cost", "synthetic_cost", "lowland_cost"):
        raw = read(directory=directory, name=source)
        for row in raw:
            value = row.get("depth_rmse")
            if source == "exact_cost":
                value = (
                    row["errors"]["momentum_rmse_m2_s"][1]
                    if row["case"] == "shear"
                    else row["errors"]["depth_rmse_m"]
                )
            records.append(
                dict(
                    source=source,
                    case=row["case"],
                    method="fv1" if row["method"] == "contact_hll" else row["method"],
                    dx=row["dx"],
                    error=value,
                    wall=row.get("median_s", row.get("wall_s")),
                )
            )
    envelopes = []
    for case in sorted({r["case"] for r in records}):
        series = [r for r in records if r["case"] == case]
        tol = (
            0.15
            if case == "shear"
            else 0.04
            if case in ("stoker", "ritter")
            else 0.001
            if series[0]["source"] == "lowland_cost"
            else 0.01
        )
        for dx in (5.0, 2.5):
            budget = next(r["wall"] for r in series if r["dx"] == dx and r["method"] == "fv1")
            for m in sorted({r["method"] for r in series}):
                group = [r for r in series if r["method"] == m]
                eligible = [r for r in group if r["wall"] <= budget]
                accepted = [r for r in group if r["error"] <= tol]
                best = eligible[int(np.argmin([r["error"] for r in eligible]))] if eligible else None
                fastest = accepted[int(np.argmin([r["wall"] for r in accepted]))] if accepted else None
                envelopes.append(
                    dict(
                        case=case,
                        method=m,
                        fv1_budget_dx=dx,
                        budget_s=budget,
                        best=best,
                        tolerance=tol,
                        fastest=fastest,
                    )
                )
    report_json(path=directory / "equal_budget.json", content=envelopes)
    return records


def draw(*, directory, cases, styles):
    records = efficiencies(directory=directory)
    fig, axes = plt.subplots(2, len(cases) // 2, figsize=(14, 7), layout="constrained")
    for ax, case in zip(axes.flat, cases, strict=True):
        for m, color in styles.items():
            series = sorted([r for r in records if r["case"] == case and r["method"] == m], key=grid)
            if series:
                ax.loglog(
                    [1000 * r["wall"] for r in series],
                    [r["error"] for r in series],
                    "o-",
                    label=m,
                    color=color,
                    ms=4,
                )
        ax.set_title(case.replace("_", " "))
        ax.set_xlabel("Hydraulic wall time (ms)")
        ax.set_ylabel("Discharge RMSE (m²/s)" if case == "shear" else "Depth RMSE (m)")
        ax.grid(alpha=0.2)
    axes.flat[-1].legend(fontsize=8)
    fig.savefig(directory / "cost.png", dpi=160)
    plt.close(fig)
    rows = read(directory=directory, name="planes")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key in zip(axes, ("slope", "froude"), strict=True):
        for m, color in styles.items():
            method = "contact_hll" if m == "fv1" else m
            subset = [
                r
                for r in rows
                if r["method"] == method and r["datum"] == 0 and r["start"] == "steady" and r["dx"] == 5
            ]
            ax.scatter(
                [r[key] * (100 if key == "slope" else 1) for r in subset],
                [r["q_rmse"] for r in subset],
                s=34,
                facecolors="none",
                edgecolors=color,
                label=m,
            )
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_xlabel("Bed slope (%)" if key == "slope" else "Froude number")
        ax.set_ylabel("Section discharge RMSE (m²/s)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(directory / "regimes.png", dpi=160)
    plt.close(fig)


def grid(row):
    return row["dx"]


if __name__ == "__main__":
    draw(
        directory=Path("validation/fv2_review/regimes"),
        cases=("stoker", "ritter", "shear", "steep", "wide_wave", "overbank", "embankment", "spatial_rain"),
        styles=dict(fv1="#24588a", minmod="#885aaa", adaptive="#c65915", li="#217452"),
    )
