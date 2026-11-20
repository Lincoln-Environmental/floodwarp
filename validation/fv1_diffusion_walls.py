"""Bounded-domain sensitivity on the observation window of the strongest damped event."""

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import initialise, sample, rms
from validation.fv1_riemann import report_json


def compare(*, case, dx, method, settings, extra):
    outcomes = []
    for length, width in ((case.length, case.width), (case.length + extra, case.width + extra)):
        c = replace(case, length=length, width=width)
        solver, states, _, times, x = initialise(
            case=c, dx=dx, method=method, policy="common", multiplier=1.0, settings=settings
        )
        q, hydro, _ = sample(case=c, solver=solver, states=states, times=times, x=x, dx=dx, li=method == "li")
        outcomes.append((q[:, :, abs(x) <= case.length / 6], hydro))
    a, b = outcomes
    return dict(
        dx=dx,
        method=method,
        original_length=case.length,
        original_width=case.width,
        expanded_length=case.length + extra,
        expanded_width=case.width + extra,
        window_half_width_m=case.length / 6,
        depth_rmse_m=rms(a[0][:, 0] - b[0][:, 0]),
        q_rmse_m2_s=rms(a[1] - b[1]),
    )


def transfer_sensitivity(*, case, dx, settings, multiplier, extensions):
    rows = []
    for extension in extensions:
        c = replace(case, length=case.length + extension, width=case.width + extension)
        for m in (1.0, multiplier):
            solver, states, exact, times, x = initialise(
                case=c, dx=dx, method="fv1", policy="common", multiplier=m, settings=settings
            )
            q, hydro, _ = sample(case=c, solver=solver, states=states, times=times, x=x, dx=dx, li=False)
            support = np.abs(x) <= case.length / 6
            rows.append(
                dict(
                    dx=dx,
                    extension_m=extension,
                    multiplier=m,
                    h_rmse_m=rms(q[:, 0, support] - states[:, 0, support]),
                    q_rmse_m2_s=rms(hydro - exact),
                )
            )
    return rows


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    settings = json.loads((directory / "matrix/configuration.json").read_text())["settings"]
    case = next(c for c in design() if c.name == "damped_transfer")
    extra = 80.0
    methods = ("fv1", "hancock_mc", "mc", "li")
    spacings = (10.0, 5.0, 1.0)
    rows = [
        compare(case=case, dx=dx, method=m, settings=settings, extra=extra)
        for m in methods
        for dx in spacings
    ]
    report_json(path=directory / "wall_sensitivity.json", content=rows)
    fits = json.loads((directory / "calibration/fits.json").read_text())
    transferred = []
    for dx in spacings:
        selected = next(
            f
            for f in fits
            if f["event"] == "damped_train" and f["dx"] == dx and f["target"] == "joint" and f["bound"] == 0.2
        )
        transferred.extend(
            transfer_sensitivity(
                case=case,
                dx=dx,
                settings=settings,
                multiplier=selected["multiplier"],
                extensions=(0.0, extra, 2 * extra),
            )
        )
    report_json(path=directory / "damped_transfer_support.json", content=transferred)
