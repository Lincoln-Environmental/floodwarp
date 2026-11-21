"""Independent fixed-timestep spatial refinement on smooth exact solutions."""

from collections import defaultdict
import json
from unittest.mock import patch

from research.floodwarp_methods_paper import runner


STAGE = "fixed-dt-space"


def cases(*, stage, config, root):
    design = json.loads(
        (
            root / "research/floodwarp_methods_paper/configs/analytical/fixed_dt.json"
        ).read_text()
    )
    for case in design["cases"]:
        for method in config["analytical"]["methods"]:
            for dx in config["analytical"]["formal_spacings"]:
                for dt in design["fixed_seconds"]:
                    yield dict(
                        case=case,
                        method=method,
                        dx=dx,
                        policy="common",
                        kind="exact",
                        fixed_dt_s=dt,
                    )


def calculate(*, stage, spec, study, root, config, context, directory):
    import numpy as np
    from research.floodwarp_methods_paper.exact import evaluate

    dt = spec["fixed_dt_s"]
    settings = dict(
        config, analytical=dict(config["analytical"], common_cap_per_m=dt / spec["dx"])
    )
    result = evaluate(
        stage="convergence",
        spec=spec,
        study=study,
        root=root,
        config=settings,
        context=context,
        directory=directory,
    )
    with np.load(directory / "fields.npz") as data:
        np.testing.assert_allclose(data["accepted_dt_s"], dt, rtol=0, atol=1e-13)
    return result


def source(*, root, study, stage):
    from research.floodwarp_methods_paper.evidence import implementation

    return runner.digest(
        dict(
            numerical_base=implementation(study.resolve(), "convergence"),
            control=runner.checksum(study / "fixed_dt.py"),
            config=runner.checksum(study / "configs/analytical/fixed_dt.json"),
        )
    )


def execute(*, root, study, config, system, flags, selector):
    with (
        patch.object(runner, "SIMULATIONS", (*runner.SIMULATIONS, STAGE)),
        patch.object(runner, "specs", selector),
        patch.object(runner, "execute", calculate),
        patch.object(runner, "source_identity", source),
    ):
        runner.run_stage(
            stage=STAGE,
            opts=flags,
            root=root,
            study=study,
            config=config,
            system=system,
        )


def analyse(*, study):
    import numpy as np
    from research.floodwarp_methods_paper.metrics import csv_write, load

    groups = defaultdict(dict)
    for rec, row, path in load(study=study, stages=(STAGE,)):
        s = rec["spec"]
        groups[s["case"], s["method"], s["dx"]][s["fixed_dt_s"]] = (rec, row, path)
    result = []
    for (case, method, dx), pair in sorted(groups.items()):
        fine, coarse = sorted(pair)
        (rec, a, pa), (second, b, pb) = pair[fine], pair[coarse]
        with np.load(pa / "fields.npz") as fa, np.load(pb / "fields.npz") as fb:
            mask = np.abs(fa["x_m"]) <= a["interior_radius_m"]
            delta = (fa["model_centreline"] - fb["model_centreline"])[:, :, mask]
            d = float(np.sqrt(np.mean(delta[:, 0] ** 2)))
            m = float(np.sqrt(np.mean(delta[:, 1] ** 2 + delta[:, 2] ** 2)))
            q = float(np.sqrt(np.mean((fa["model_gauges"] - fb["model_gauges"]) ** 2)))
        for metric, error, change in (
            ("depth", a["depth_rmse"], d),
            ("momentum", np.hypot(a["momentum_x_rmse"], a["momentum_y_rmse"]), m),
            ("face discharge", a["hydro_rmse"], q),
        ):
            result.append(
                dict(
                    case=case,
                    method=method,
                    dx=dx,
                    metric=metric,
                    unit="m" if metric == "depth" else "m2/s",
                    fixed_dt_s=fine,
                    larger_dt_s=coarse,
                    error=float(error),
                    field_change_on_halving_dt=change,
                    temporal_change_to_error=change / error if error else None,
                    order=None,
                    experiment_id=rec["experiment_id"],
                    halving_control=second["experiment_id"],
                )
            )
    index = {(r["case"], r["method"], r["metric"], r["dx"]): r for r in result}
    for r in result:
        previous = index.get((r["case"], r["method"], r["metric"], 2 * r["dx"]))
        if previous and min(previous["error"], r["error"]) > 0:
            r["order"] = float(np.log2(previous["error"] / r["error"]))
    csv_write(file=study / "processed/convergence/fixed_dt_space.csv", rows=result)
    return len(result)
