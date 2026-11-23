"""Independent refinements for the million-cell soil event with a captured recession."""

import json
from pathlib import Path
from unittest.mock import patch
from research.floodwarp_methods_paper import events, runner
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics, matrix


def run(*, directory, case, refinements, device, support):
    study = directory.parent
    root = study.parents[1]
    config = read_config(path=study / "configs/study.json")
    template = json.loads(
        (root / "validation/hancock_long_storm_review/cases.json").read_text()
    )
    base = next(
        s
        for s in matrix.cases(stage="million-storms", config=config, root=root)
        if s["case"] == case and s["method"] == "fv1"
    )
    options = matrix.design(study=study)
    hardware = runner.machine(root=root)
    source = runner.source_identity(root=root, study=study, stage="long-storms")
    normal = events.numerical
    ids = []
    for dx, method, factor in refinements:
        spec = dict(base, dx=dx, dem=dx, method=method, support=support)
        identity = dict(
            spec=spec,
            source=source,
            code=[
                runner.checksum(p)
                for p in (
                    Path(__file__),
                    study / "qualification/hydraulics.py",
                    study / "qualification/matrix.py",
                )
            ],
            config=config,
            options=options,
            template=template,
            factor=factor,
            device=[hardware["gpu"], hardware["warp"], device],
        )
        ident = runner.digest(identity)[:20]
        folder = directory / "storm_runs" / ident
        folder.mkdir(parents=True, exist_ok=True)
        result = folder / "result.json"
        if not result.exists():

            def numerical(*, method):
                key, mode, opt = normal(method=method)
                return key, mode, dict(opt, cfl=0.4 * factor)

            with patch.object(events, "numerical", numerical):
                d = hydraulics.run_event(
                    spec=spec,
                    config=config,
                    template=template,
                    options=options,
                    directory=folder,
                    device=device,
                )
            tmp = folder / "pending.json"
            runner.write_json(
                path=tmp,
                data=dict(
                    data=d,
                    spec=spec,
                    inputs=identity,
                    hardware=hardware,
                    cohort="expanded",
                    factor=factor,
                ),
            )
            tmp.replace(result)
            print(case, dx, method, factor, d["steps"], d["hydraulic_s"], flush=True)
        ids.append(ident)
    runner.write_json(path=directory / "storm_runs/rolling_index.json", data=ids)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    case = "rolling-704-soil-5120x5120"
    refinements = (
        (4, "adaptive", 1.0),
        (3.2, "adaptive", 1.0),
        (3.2, "hancock_dense", 1.0),
        (3.2, "adaptive", 0.5),
    )
    device = "cuda:0"
    support = 80
    run(
        directory=directory,
        case=case,
        refinements=refinements,
        device=device,
        support=support,
    )
