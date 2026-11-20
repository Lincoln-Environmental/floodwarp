"""Bounded numerical-error compensation with frozen analytical controls and held-out events."""

from pathlib import Path
from time import perf_counter
from functools import partial
from dataclasses import asdict
import json
import numpy as np
from scipy.optimize import minimize_scalar
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import forecast, matrix
from validation.fv1_riemann import report_json


def scores(*, row, case, dx, settings):
    """Tolerance-scaled RMS within each objective; the joint fit minimises the worse score."""
    scale = max(
        settings["q_floor"], settings["relative_q"] * max(abs(g["exact_peak"]) for g in row["gauges"])
    )
    flow = np.sqrt(
        np.mean(
            np.square(
                [
                    row["hydro_rmse"] / scale,
                    row["peak_error_rmse"] / scale,
                    row["volume_error_rmse"] / (case.duration * scale),
                    (row["timing_error_rmse"] or 0.0) / (case.duration / settings["samples"]),
                ]
            )
        )
    )
    ext = row["extent"][str(settings["wet"][0])]
    space = np.sqrt(
        np.mean(
            np.square(
                [
                    row["depth_rmse"] / settings["h_tol"],
                    row["maximum_depth_rmse"] / settings["h_tol"],
                    (1 - ext["iou"]) / settings["iou_tol"],
                    ext["extent_rmse_m"] / dx,
                    (ext["arrival_rmse_s"] or 0.0) / (case.duration / settings["samples"]),
                ]
            )
        )
    )
    return dict(flow=float(flow), space=float(space), joint=float(max(space, flow)))


class Trials:
    def __init__(self, *, settings):
        self.settings = settings
        self.cache = {}
        self.seconds = 0.0

    def evaluate(self, *, case, dx, multiplier):
        key = (case.name, dx, round(float(multiplier), 6))
        if key not in self.cache:
            start = perf_counter()
            row = forecast(
                case=case,
                dx=dx,
                method="fv1",
                policy="common",
                multiplier=key[2],
                settings=self.settings,
                keep_series=False,
            )
            row["scores"] = scores(row=row, case=case, dx=dx, settings=self.settings)
            self.cache[key] = row
            self.seconds += perf_counter() - start
        return self.cache[key]

    def objective(self, m, *, case, dx, target):
        return self.evaluate(case=case, dx=dx, multiplier=m)["scores"][target]


def fit(*, trials, case, dx, bound, target, tolerance, near_best):
    objective = partial(trials.objective, case=case, dx=dx, target=target)
    grid = np.linspace(1 - bound, 1 + bound, 9)
    values = [objective(m) for m in grid]
    candidates = list(grid)
    # Bracket sampled minima, include exact endpoints, and always compare n_true.
    for i, value in enumerate(values):
        if (i == 0 or value <= values[i - 1]) and (i == len(values) - 1 or value <= values[i + 1]):
            if max(values) - min(values) < 1e-12:
                break
            lo = grid[max(i - 1, 0)]
            hi = grid[min(i + 1, len(grid) - 1)]
            res = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": tolerance})
            candidates.append(float(res.x))
    candidates.extend(
        k[2]
        for k in trials.cache
        if k[:2] == (case.name, dx) and 1 - bound - 1e-10 <= k[2] <= 1 + bound + 1e-10
    )
    best = min(objective(m) for m in candidates)
    baseline = objective(1.0)
    # Prefer a smaller change delivering >=99% of the achievable improvement.
    acceptable = best + near_best * max(0.0, baseline - best) + 1e-11
    eligible = [(abs(m - 1), m) for m in candidates if objective(m) <= acceptable]
    m = float(min(eligible)[1])
    return dict(
        event=case.name,
        dx=dx,
        target=target,
        bound=bound,
        physical_n=case.n,
        multiplier=round(m, 6),
        calibrated_n=case.n * round(m, 6),
        percent_change=100 * (m - 1),
        optimum_score=best,
        selected_score=objective(m),
        baseline_score=baseline,
    )


def calibrate(*, cases, training, spacings, bounds, targets, settings, output):
    trials = Trials(settings=settings)
    fits = []
    transfers = []
    benchmarks = []
    for name in training:
        case = next(c for c in cases if c.name == name)
        for dx in spacings:
            for bound in bounds:
                for target in targets:
                    fits.append(
                        fit(
                            trials=trials,
                            case=case,
                            dx=dx,
                            bound=bound,
                            target=target,
                            tolerance=settings["search_xtol"],
                            near_best=settings["near_best"],
                        )
                    )
            print("fitted", name, dx, len(trials.cache), flush=True)
        report_json(path=output / "fits.json", content=fits)
    training_seconds = trials.seconds
    for source in fits:
        for case in cases:
            if case.n <= 0:
                continue
            for dx in spacings:
                if source["event"] == case.name and dx == source["dx"]:
                    continue
                if dx != source["dx"] and case.name != source["event"]:
                    continue
                fit_row = trials.evaluate(case=case, dx=dx, multiplier=source["multiplier"])
                base = trials.evaluate(case=case, dx=dx, multiplier=1.0)
                transfers.append(
                    dict(
                        source=source["event"],
                        source_dx=source["dx"],
                        target=source["target"],
                        bound=source["bound"],
                        test=case.name,
                        dx=dx,
                        multiplier=source["multiplier"],
                        fitted_n=source["multiplier"] * case.n,
                        scores=fit_row["scores"],
                        baseline=base["scores"],
                    )
                )
        if source["bound"] == bounds[-1] and source["target"] == targets[-1]:
            print("transferred", source["event"], source["dx"], len(trials.cache), flush=True)
    # Once transfer choices are frozen, also estimate the upper bound from refitting each held-out event.
    for case in cases:
        if case.n <= 0 or case.name in training:
            continue
        for dx in spacings:
            for bound in bounds:
                for target in targets:
                    fits.append(
                        fit(
                            trials=trials,
                            case=case,
                            dx=dx,
                            bound=bound,
                            target=target,
                            tolerance=settings["search_xtol"],
                            near_best=settings["near_best"],
                        )
                    )
        print("independent single-event optima", case.name, len(trials.cache), flush=True)
        report_json(path=output / "fits.json", content=fits)
    # Compare repeated warm forecast costs of the selected joint fit, excluding the search itself.
    for source in fits:
        if source["target"] == "joint":
            case = next(c for c in cases if c.name == source["event"])
            benchmarks.append(
                dict(
                    bound=source["bound"],
                    **forecast(
                        case=case,
                        dx=source["dx"],
                        method="fv1",
                        policy="common",
                        multiplier=source["multiplier"],
                        settings=settings,
                        repeats=settings["repeats"],
                    ),
                )
            )
    report_json(path=output / "transfers.json", content=transfers)
    report_json(path=output / "forecast_cost.json", content=benchmarks)
    report_json(path=output / "trials.json", content=list(trials.cache.values()))
    report_json(
        path=output / "search_cost.json",
        content=dict(
            fitting_wall_s=training_seconds,
            fitting_and_transfer_wall_s=trials.seconds,
            distinct_forecasts=len(trials.cache),
            training=training,
            settings=settings,
        ),
    )


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    settings = json.loads((directory / "matrix/configuration.json").read_text())["settings"]
    settings.update(
        q_floor=0.0001,
        relative_q=0.05,
        h_tol=0.005,
        iou_tol=0.05,
        search_xtol=0.0005,
        near_best=0.01,
        repeats=3,
    )
    cases = design()
    # A partial matrix can be resumed before fitting. Always retain the physical-value controls.
    matrix_dir = directory / "matrix"
    manifest = json.loads((matrix_dir / "configuration.json").read_text())
    missing = [c for c in cases if not (matrix_dir / (c.name + ".json")).exists()]
    if missing:
        comparison_methods = (
            "fv1",
            "low_froude_hll",
            "euler_minmod",
            "euler_mc",
            "hancock_minmod",
            "hancock_mc",
            "hancock_adaptive",
            "hancock_low_froude",
            "minmod",
            "mc",
            "adaptive",
            "li",
        )
        matrix(
            cases=missing,
            spacings=(10.0, 5.0, 1.0),
            methods=comparison_methods,
            policies=("common", "production"),
            settings=settings,
            output=matrix_dir,
            repeats=3,
        )
        manifest["cases"] = [asdict(c) for c in cases]
        report_json(path=matrix_dir / "configuration.json", content=manifest)
    training = ("front_plain", "rain20", "routing", "damped_train")
    spacings = (10.0, 5.0, 1.0)
    bounds = (0.05, 0.10, 0.20)
    targets = ("flow", "space", "joint")
    calibrate(
        cases=cases,
        training=training,
        spacings=spacings,
        bounds=bounds,
        targets=targets,
        settings=settings,
        output=directory / "calibration",
    )
