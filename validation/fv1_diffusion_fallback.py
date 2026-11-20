"""Count accepted Hancock predictor downgrades and their absolute transport by depth."""

from pathlib import Path
import json
import numpy as np
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import initialise
from validation.fv1_riemann import report_json


def count(*, case, dx, settings, edges, method):
    s, *_ = initialise(case=case, dx=dx, method=method, policy="common", multiplier=1.0, settings=settings)
    dt = dx * settings["common_cap_per_m"]
    bins = np.zeros((len(edges) + 1, 4))
    past = s.get_cumulative_face_volumes()
    steps = 0
    while s.time < case.duration:
        h = s.get_water_depth()
        run = s.run_device_loop(duration=min(dt, case.duration - s.time))
        if run["steps"] != 1:
            raise ArithmeticError("Diagnostic requires one accepted update")
        flags = s.predictor_fallback_cells()
        volume = s.get_cumulative_face_volumes()
        band = np.digitize(h, edges)
        for b in range(len(bins)):
            sel = (h > s.dry_depth) & (band == b)
            bins[b, 0] += sel.sum()
            bins[b, 1] += (sel & flags).sum()
        for axis, name in ((1, "qx"), (0, "qy")):
            dv = np.diff(np.stack((past[name], volume[name])), axis=0)[0]
            pads = [(0, 0), (0, 0)]
            pads[axis] = (1, 1)
            bands = np.pad(band, pads, mode="edge")
            failed = np.pad(flags, pads, mode="edge")
            idx = np.arange(dv.shape[axis])
            donor = np.where(dv > 0, np.take(bands, idx, axis=axis), np.take(bands, idx + 1, axis=axis))
            downgraded = np.where(
                dv > 0, np.take(failed, idx, axis=axis), np.take(failed, idx + 1, axis=axis)
            )
            for b in range(len(bins)):
                bins[b, 2] += np.abs(dv[donor == b]).sum()
                bins[b, 3] += np.abs(dv[(donor == b) & downgraded]).sum()
        past = volume
        steps += 1
    return dict(
        case=case.name,
        dx=dx,
        method=method,
        steps=steps,
        edges_m=edges,
        columns=[
            "wet_cell_steps",
            "predictor_downgrade_cell_steps",
            "absolute_face_m3",
            "downgrade_absolute_face_m3",
        ],
        bins=bins.tolist(),
    )


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    settings = json.loads((directory / "matrix/configuration.json").read_text())["settings"]
    edges = (0.001, 0.01, 0.1, 1.0)
    cases = ("front_plain", "front_steep", "gravity_pulse")
    methods = ("hancock_mc", "hancock_adaptive")
    spacings = (1.0, 5.0)
    rows = [
        count(case=c, dx=dx, settings=settings, edges=edges, method=m)
        for c in design()
        if c.name in cases
        for m in methods
        for dx in spacings
    ]
    report_json(path=directory / "fallback.json", content=rows)
