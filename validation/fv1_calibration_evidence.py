"""Preserve fitted physical metrics, independent-event transfer and measured deployment costs."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import forecast
from validation.fv1_diffusion_figures import load_rows
from validation.fv1_riemann import report_json


def compact(*, row):
    return dict(
        depth_rmse_m=row["depth_rmse"],
        wet_depth_rmse_m=row["wet_depth_rmse"],
        max_depth_rmse_m=row["maximum_depth_rmse"],
        iou_1cm=row["extent"]["0.01"]["iou"],
        arrival_error_s=row["extent"]["0.001"]["arrival_rmse_s"],
        hydro_rmse_m2_s=row["hydro_rmse"],
        rms_peak_error=row["peak_error_rmse"],
        rms_peak_time_error_s=row["timing_error_rmse"],
        rms_integral_error_m2=row["volume_error_rmse"],
        residual_m3=row["residual_m3"],
    )


def csv_rows(*, path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def deployment(*, directory, settings, fits, pairs, spacings):
    if not pairs:
        return []
    cases = {c.name: c for c in design()}
    out = []
    for src, test in pairs:
        for dx in spacings:
            fit = next(
                r
                for r in fits
                if r["event"] == src and r["dx"] == dx and r["bound"] == 0.2 and r["target"] == "joint"
            )
            for policy in ("common", "production"):
                r = forecast(
                    case=cases[test],
                    dx=dx,
                    method="fv1",
                    policy=policy,
                    multiplier=fit["multiplier"],
                    settings=settings,
                    keep_series=True,
                    repeats=3,
                )
                out.append(dict(source=src, **r))
        print("deployment", src, test, flush=True)
    report_json(path=directory / "deployment.json", content=out)
    return out


def cost_all_fits(*, directory, fits, existing, settings):
    cases = {c.name: c for c in design()}
    measured = {(r["case"], r["dx"], r["multiplier"]): r for r in existing}
    for f in fits:
        key = (f["event"], f["dx"], f["multiplier"])
        if key in measured:
            continue
        measured[key] = forecast(
            case=cases[key[0]],
            dx=key[1],
            method="fv1",
            policy="common",
            multiplier=key[2],
            settings=settings,
            keep_series=False,
            repeats=3,
        )
    rows = list(measured.values())
    report_json(path=directory / "selected_cost.json", content=rows)
    return rows


def compare(*, directory, baseline_records, costs, fits, trials, transfers, deployed):
    base = {(r["case"], r["dx"], r["method"], r["policy"]): r for r in baseline_records}
    look = {(r["case"], r["dx"], round(r["multiplier"], 6)): r for r in trials}
    timed = {(r["case"], r["dx"], r["multiplier"]): r for r in costs}
    rows = []
    minimum = []
    for fit in fits:
        c, dx, bound, metric = fit["event"], fit["dx"], fit["bound"], fit["target"]
        r = look[c, dx, fit["multiplier"]]
        t = timed[c, dx, fit["multiplier"]]["median_wall_s"]
        rows.append(
            dict(
                event=c,
                dx=dx,
                target=metric,
                bound=bound,
                method="fv1",
                physical_n=fit["physical_n"],
                model_n=fit["calibrated_n"],
                delta_percent=fit["percent_change"],
                wall_s=t,
                **compact(row=r),
            )
        )
        sub = [
            (abs(k[2] - 1), k[2], v["scores"][metric])
            for k, v in look.items()
            if k[0] == c and k[1] == dx and abs(k[2] - 1) <= bound + 1e-9 and v["scores"][metric] <= 1.0
        ]
        smallest = min(sub) if sub else None
        minimum.append(
            dict(
                event=c,
                dx=dx,
                target=metric,
                bound=bound,
                multiplier_meeting_tolerance=smallest[1] if smallest else None,
            )
        )
    for r in baseline_records:
        if r["policy"] == "common" and r["method"] in (
            "fv1",
            "hancock_mc",
            "hancock_adaptive",
            "mc",
            "adaptive",
            "li",
        ):
            rows.append(
                dict(
                    event=r["case"],
                    dx=r["dx"],
                    target="physical",
                    bound=0.0,
                    method=r["method"],
                    physical_n=r["model_n"] if r["method"] != "li" else min(r["physical_n_range"]),
                    model_n=r["model_n"],
                    delta_percent=0.0,
                    wall_s=r["median_wall_s"],
                    **compact(row=r),
                )
            )
    csv_rows(path=directory / "comparison.csv", rows=rows)
    csv_rows(path=directory / "minimum_change.csv", rows=minimum)
    held = []
    for t in transfers:
        r = look[t["test"], t["dx"], t["multiplier"]]
        obs = compact(row=r)
        unfit = compact(row=base[t["test"], t["dx"], "fv1", "common"])
        held.append(
            dict(
                source=t["source"],
                source_dx=t["source_dx"],
                test=t["test"],
                dx=t["dx"],
                objective=t["target"],
                bound=t["bound"],
                multiplier=t["multiplier"],
                model_n=t["fitted_n"],
                **obs,
                **{"baseline_" + k: v for k, v in unfit.items()},
            )
        )
    csv_rows(path=directory / "heldout.csv", rows=held)
    budgets = []
    for r in deployed:
        for method in ("fv1", "hancock_mc", "hancock_adaptive", "mc", "li"):
            candidates = [
                z
                for (c, dx, m, p), z in base.items()
                if c == r["case"]
                and m == method
                and p == r["policy"]
                and z["median_wall_s"] <= r["median_wall_s"]
            ]
            for key in ("hydro_rmse", "depth_rmse", "maximum_depth_rmse"):
                best = min((z[key], z["median_wall_s"], z["dx"]) for z in candidates) if candidates else None
                budgets.append(
                    dict(
                        source=r["source"],
                        test=r["case"],
                        policy=r["policy"],
                        calibrated_dx=r["dx"],
                        method=method,
                        metric=key,
                        calibrated_error=r[key],
                        budget_s=r["median_wall_s"],
                        model_n=r["model_n"],
                        eligible_dx=best[2] if best else None,
                        method_wall_s=best[1] if best else None,
                        method_error=best[0] if best else None,
                    )
                )
    report_json(path=directory / "deployment_budgets.json", content=budgets)


def chart(*, directory, rows, cases, spacings):
    fig, axes = plt.subplots(2, len(cases), figsize=(3.8 * len(cases), 6), squeeze=False)
    for j, c in enumerate(cases):
        for dx in spacings:
            sub = sorted((r for r in rows if r["case"] == c and r["dx"] == dx), key=roughness)
            for i, key in enumerate(("hydro_rmse", "depth_rmse")):
                ax = axes[i, j]
                ax.plot(
                    [100 * (r["multiplier"] - 1) for r in sub],
                    [r[key] for r in sub],
                    ".-",
                    ms=2,
                    lw=0.8,
                    label=f"{dx:g} m",
                )
                ax.axvline(0.0, ls=":", lw=0.5, color="black")
                ax.set(
                    xlabel="Manning n adjustment (%)",
                    ylabel=("Q RMSE (m²/s)" if i == 0 else "Depth RMSE (m)"),
                )
                ax.set_title(c.replace("_", " "))
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(directory / "roughness_sensitivity.png", dpi=155)
    plt.close(fig)


def roughness(row):
    return row["multiplier"]


if __name__ == "__main__":
    root = Path("validation/fv1_diffusion_review")
    directory = root / "calibration"
    settings = json.loads((directory / "search_cost.json").read_text())["settings"]
    fits = json.loads((directory / "fits.json").read_text())
    pairs = (
        ("front_plain", "front_steep"),
        ("front_plain", "front_gentle"),
        ("front_plain", "routing"),
        ("rain20", "rain50"),
        ("rain20", "rain100"),
        ("routing", "slow_recession"),
        ("damped_train", "damped_transfer"),
    )
    spacings = (1.0, 5.0, 10.0)
    deployed = deployment(directory=directory, settings=settings, fits=fits, pairs=pairs, spacings=spacings)
    trials = json.loads((directory / "trials.json").read_text())
    costs = cost_all_fits(
        directory=directory,
        fits=fits,
        existing=json.loads((directory / "forecast_cost.json").read_text()),
        settings=settings,
    )
    compare(
        directory=directory,
        baseline_records=load_rows(directory=root),
        costs=costs,
        fits=fits,
        trials=trials,
        transfers=json.loads((directory / "transfers.json").read_text()),
        deployed=deployed,
    )
    chart(
        directory=directory, rows=trials, cases=("front_plain", "rain20", "damped_train"), spacings=spacings
    )
