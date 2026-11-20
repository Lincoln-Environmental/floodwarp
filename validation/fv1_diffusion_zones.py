"""Restricted calibration in two prescribed, hydraulically separated roughness zones."""

from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar
from validation.fv1_diffusion_calibrate import Trials, fit
from validation.fv1_diffusion_domains import design, prepare
from validation.fv1_diffusion_matrix import initialise
from floodwarp.full_swe_solver import FullSWESolver
from validation.fv1_riemann import report_json


def paired_check(*, cases, dx, multipliers, settings):
    """Direct two-bench CUDA integration with an inactive levee between the benches."""
    z = []
    state = []
    n = []
    separate = []
    for case, m in zip(cases, multipliers, strict=True):
        d, *_ = prepare(
            case=case,
            dx=dx,
            gravity=settings["gravity"],
            samples=settings["samples"],
            order=settings["order"],
        )
        z.append(d["z"])
        state.append(d["initial"])
        n.append(d["n"] * m)
        s, *_ = initialise(case=case, dx=dx, method="fv1", policy="common", multiplier=m, settings=settings)
        separate.append(s)
    ny, nx = z[0].shape
    bed = np.vstack((z[0], np.zeros((1, nx)), z[1]))
    h = np.vstack((state[0], np.zeros((1, nx, 3)), state[1]))
    active = np.ones_like(bed, bool)
    active[ny] = False
    paired = FullSWESolver(
        ny=2 * ny + 1,
        nx=nx,
        dx=dx,
        dy=dx,
        bed_elevation=bed,
        active_mask=active,
        manning_n=np.vstack((n[0], np.zeros((1, nx)), n[1])),
        device=settings["device"],
        gravity=settings["gravity"],
        maximum_dt=settings["common_cap_per_m"] * dx,
        execution_mode="dense",
    )
    paired.set_water_depth(depth=h[..., 0])
    paired.set_momentum(hu=h[..., 1], hv=h[..., 2])
    for s in [paired, *separate]:
        s.run_device_loop(duration=cases[0].duration)
    out = paired.get_conservative_state()
    maximum = max(
        np.max(abs(out[:ny] - separate[0].get_conservative_state())),
        np.max(abs(out[-ny:] - separate[1].get_conservative_state())),
    )
    if maximum > 2e-12:
        raise ArithmeticError("Two-zone assembly and individual zones differ")
    return float(maximum)


def experiment(*, base, dxs, bounds, settings, output):
    base = replace(base, length=400.0)
    channel = replace(base, name="channel", n=0.025, speed=1.3)
    held = tuple(
        replace(c, name=c.name + "_held", speed=c.speed * 0.6, initial_tip=-5.0) for c in (base, channel)
    )
    trials = Trials(settings=settings)
    rows = []
    for dx in dxs:
        for bound in bounds:

            def joint(m):
                return max(trials.objective(m, case=c, dx=dx, target="joint") for c in (base, channel))

            opt = minimize_scalar(
                joint,
                bounds=(1 - bound, 1 + bound),
                method="bounded",
                options={"xatol": settings["search_xtol"]},
            )
            single = min((joint(m), abs(m - 1), m) for m in [1.0, 1 - bound, 1 + bound, opt.x])[-1]
            individual = [
                fit(
                    trials=trials,
                    case=c,
                    dx=dx,
                    bound=bound,
                    target="joint",
                    tolerance=settings["search_xtol"],
                    near_best=settings["near_best"],
                )["multiplier"]
                for c in (base, channel)
            ]
            for label, ms in (
                ("physical", [1.0, 1.0]),
                ("uniform", [single, single]),
                ("two_zones", individual),
            ):
                for split, cs in (("training", (base, channel)), ("transfer", held)):
                    obs = [trials.evaluate(case=c, dx=dx, multiplier=m) for c, m in zip(cs, ms, strict=True)]
                    parity = paired_check(cases=cs, dx=dx, multipliers=ms, settings=settings)
                    rows.append(
                        dict(
                            dx=dx,
                            bound=bound,
                            model=label,
                            split=split,
                            multipliers=ms,
                            joint=max(r["scores"]["joint"] for r in obs),
                            zones=obs,
                            assembly_error=parity,
                        )
                    )
        print("paired zones", dx, flush=True)
    report_json(path=output, content=rows)


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    settings = json.loads((directory / "calibration/search_cost.json").read_text())["settings"]
    base = next(c for c in design() if c.name == "front_plain")
    experiment(
        base=base,
        dxs=(10.0, 5.0, 1.0),
        bounds=(0.05, 0.10, 0.20),
        settings=settings,
        output=directory / "calibration/zones.json",
    )
