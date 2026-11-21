"""Entry points for scaled hydraulics and strictly offline report reconstruction."""
from unittest.mock import patch
from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.evidence import source_identity as numerical_identity
from research.floodwarp_methods_paper.qualification import launch, refinements, automatic, heldout, repeats

NUMERICAL=("million-terrain","million-pilots","million-storms","million-auto","large-references","large-refinements","heldout-references","million-profiles","million-repeats")
OFFLINE=("execution-policy","pareto")


def terrain_identity(*,root,study,stage):
    return runner.digest(dict(base=launch.identity_original(root=root,study=study,stage=stage),
                              diagnostics={p.name:runner.checksum(p) for p in (root/"src/floodwarp/terrain_advisor").glob("*.py")}))


def numerical(*,root,study,config,stage,arguments):
    if stage=="million-auto":
        with patch.object(launch,"cases",automatic.cases),patch.object(launch,"calculate",automatic.calculate),patch.object(launch,"identity_original",launch.identity,create=True),patch.object(launch,"identity",automatic.source):
            launch.run(root=root,study=study,config=config,stage=stage,arguments=arguments)
    elif stage in ("heldout-references", "million-repeats"):
        module = heldout if stage == "heldout-references" else repeats
        with patch.object(launch,"cases",module.cases),patch.object(launch,"identity_original",launch.identity,create=True),patch.object(launch,"identity",module.source):
            launch.run(root=root,study=study,config=config,stage=stage,arguments=arguments)
    elif stage=="large-refinements":
        with patch.object(launch,"cases",refinements.expanded),patch.object(launch,"identity_original",launch.identity,create=True),patch.object(launch,"identity",refinements.source):
            launch.run(root=root,study=study,config=config,stage=stage,arguments=arguments)
    elif stage=="million-terrain":
        with patch.object(launch,"identity_original",launch.identity,create=True),patch.object(launch,"identity",terrain_identity):
            launch.run(root=root,study=study,config=config,stage=stage,arguments=arguments)
    else:
        launch.run(root=root,study=study,config=config,stage=stage,arguments=arguments)


def source(*,root,study,stage):
    with patch.object(runner,"source_identity_base",numerical_identity,create=True):
        if stage=="million-terrain":
            with patch.object(launch,"identity_original",launch.identity,create=True):
                return terrain_identity(root=root,study=study,stage=stage)
        if stage=="million-auto":
            with patch.object(launch,"identity_original",launch.identity,create=True):
                return automatic.source(root=root,study=study,stage=stage)
        if stage in ("heldout-references", "million-repeats"):
            module = heldout if stage == "heldout-references" else repeats
            with patch.object(launch,"identity_original",launch.identity,create=True):
                return module.source(root=root,study=study,stage=stage)
        if stage=="large-refinements":
            with patch.object(launch,"identity_original",launch.identity,create=True):
                return refinements.source(root=root,study=study,stage=stage)
        return launch.identity(root=root,study=study,stage=stage)



def specifications(*, stage, config, root):
    planner = {"large-refinements": refinements.expanded, "heldout-references": heldout.cases, "million-repeats": repeats.cases, "million-auto": automatic.cases}.get(stage, launch.cases)
    yield from planner(stage=stage, config=config, root=root)
