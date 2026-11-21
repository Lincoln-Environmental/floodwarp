"""Follow the non-exporting sparse plain and test downstream activation on the same DEM."""

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification import automatic, launch, matrix


def cases(*, stage, config, root):
    for spec in matrix.cases(stage="million-storms", config=config, root=root):
        if spec["seed"] != 701 or spec["method"] != "fv1":
            continue
        for rain, duration in (("spatial", 48), ("moderate", 24)):
            for method in ("fv1", "hancock_dense", "hancock_adaptive", "hancock_auto"):
                yield dict(spec, case=f"plain-701-{rain}-10240x10240-{duration}h", storm=rain,
                           hours=duration, method=method)


def calculate(*, stage, spec, study, root, config, context, directory):
    execute = automatic.calculate if spec["method"] == "hancock_auto" else launch.calculate_original
    return execute(stage=stage, spec=spec, study=study, root=root, config=config, context=context, directory=directory)


def source(*, root, study, stage):
    base = automatic.source(root=root, study=study, stage=stage)
    return runner.digest(dict(base=base, adapter=runner.checksum(study/"qualification/recession.py")))

