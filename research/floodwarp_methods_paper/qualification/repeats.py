"""Independent timing repetitions of two complete, matched large-domain controls."""

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification import launch, matrix


def cases(*, stage, config, root):
    for spec in matrix.cases(stage="million-storms", config=config, root=root):
        if spec["seed"] in (701,702) and spec["method"] in ("fv1","hancock_dense","hancock_adaptive"):
            for repeat in (2,3):
                yield dict(spec, repeat=repeat)


def source(*, root, study, stage):
    return runner.digest(dict(base=launch.identity_original(root=root, study=study, stage=stage),
                              adapter=runner.checksum(study/"qualification/repeats.py")))

