"""Write historical checkpoints or compare their continuations in a second checkout."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from floodwarp.inputs import terrain_from_array
from floodwarp.model import advance, create_model, load_checkpoint, save_checkpoint


def verify_restart(*, directory, write_baseline, device, methods, shape, seed, middle, end):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=seed)
    terrain = terrain_from_array(elevation=rng.uniform(-.02, .02, shape), dx=2., dy=3.,
                                 row_order="south_to_north", vertical_datum="local")
    depth = rng.uniform(-.04, .12, shape).clip(min=0.)
    options = dict(terrain=terrain, initial_depth=depth, rainfall=.00012, infiltration=.00002,
                   manning_n=rng.uniform(.02, .08, shape), explicit_maximum_dt=.03, device=device)
    rows = {}
    for method in methods:
        checkpoint = directory / (method + ".npz")
        numerical = dict(solver_method=method)
        if method == "fv1_hancock" and not write_baseline:
            numerical["limiter"] = "mc"
        model = create_model(**options, **numerical)
        if write_baseline:
            advance(model=model, end_time=middle)
            save_checkpoint(model=model, path=checkpoint)
        else:
            load_checkpoint(model=model, path=checkpoint)
        advance(model=model, end_time=end)
        solver = model.solver
        fields = dict(state=solver.get_conservative_state(), maximum=solver.get_max_water_depth(),
                      soil=solver.cumulative_infiltration_m(), **solver.get_cumulative_face_volumes())
        rows[method] = dict(steps=solver.steps, method=solver.numerical_configuration(),
                            sha256={key: hashlib.sha256(value.tobytes()).hexdigest()
                                    for key, value in fields.items()})
    record = directory / "continuations.json"
    if write_baseline:
        record.write_text(json.dumps(rows, indent=2) + "\n")
    elif rows != json.loads(record.read_text()):
        raise AssertionError("The restored trajectory differs from the original checkout.")
    return rows


if __name__ == "__main__":
    directory = Path(os.environ["FLOODWARP_RESTART_DIRECTORY"])
    write_baseline = os.environ.get("FLOODWARP_WRITE_BASELINE") == "1"
    device = "cuda:0"
    methods = ("fv1", "fv2", "fv1_hancock")
    shape = (12, 19)
    seed = 90321
    verify_restart(directory=directory, write_baseline=write_baseline, device=device,
                   methods=methods, shape=shape, seed=seed, middle=.4, end=1.2)
