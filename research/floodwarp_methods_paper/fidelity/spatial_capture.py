"""Record supported spatial fields during the measured recession, outside the hydraulic timer."""

from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification.hydraulics import average
from validation.fv2_local_inertia import read_state


class Recorder:
    def __init__(self, *, directory, studied, interval):
        self.directory = directory
        self.studied = studied
        self.interval = interval
        self.models = WeakKeyDictionary()

    def register(self, *, model, spec):
        if spec["case"] not in self.studied or spec["method"] == "li":
            return
        support = self.studied[spec["case"]]
        factor = round(support / spec["dx"])
        if not np.isclose(factor * spec["dx"], support):
            raise ValueError("Non-nested transient spatial support")
        s = model.solver
        meta = {
            k: spec[k]
            for k in (
                "case",
                "dx",
                "dem",
                "initial",
                "method",
                "initial_grid",
                "controls",
            )
        }
        meta.update(
            cfl=s.cfl,
            support=support,
            interval=self.interval,
            source=runner.checksum(Path(__file__)),
            terrain=model.identity["fields"]["bed"],
            physical_fields=model.identity["fields"],
            forcing_sha256=runner.digest(model.identity["forcing"]),
        )
        folder = self.directory / "transient" / runner.digest(meta)[:20]
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "spec.json").exists():
            runner.write_json(path=folder / "spec.json", data=meta)
        self.models[s] = (factor, folder)

    def snapshot(self, *, solver, li):
        if li or solver not in self.models:
            return
        t = round(solver.time)
        if abs(t - solver.time) > 1e-5 or t % self.interval:
            return
        factor, folder = self.models[solver]
        dest = folder / f"{t}.npz"
        if dest.exists():
            return
        state = read_state(solver=solver, li=False)
        fields = {
            name: average(a=state[..., k], factor=factor)
            for k, name in enumerate(("h", "hu", "hv"))
        }
        temp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **fields)
        temp.replace(dest)
