"""Independent same-area reference refinements for the reserved operational events."""

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification import launch, matrix


def cases(*, stage, config, root):
    for spec in matrix.cases(stage="million-storms", config=config, root=root):
        if spec["method"] != "fv1" or spec["dx"] != 10 or spec["split"] != "test":
            continue
        if spec["seed"] == 708:
            for spacing, method in ((8, "adaptive"), (5, "adaptive"), (5, "hancock_dense")):
                yield dict(spec, dx=spacing, dem=spacing, method=method)
        elif spec["seed"] == 715:
            # Both 5 m formulations already run in the operational matrix.
            # A separate 4 m bed resolves the reference sensitivity of the 5 m forecast.
            for method in ("adaptive", "hancock_dense"):
                yield dict(spec, dx=4, dem=4, method=method)


def source(*, root, study, stage):
    return runner.digest(dict(base=launch.identity_original(root=root, study=study, stage=stage),
                              adapter=runner.checksum(study/"qualification/heldout.py")))

