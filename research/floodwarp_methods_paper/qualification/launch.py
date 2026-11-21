"""Manual numerical stages for the corrective study, using immutable study receipts."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.controller import grid_filter,select_specs,device_slot
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification.matrix import cases,design


def identity(*,root,study,stage):
    paths=["qualification/hydraulics.py","qualification/matrix.py","qualification/design.json","qualification/launch.py"]
    if stage=="million-terrain":
        paths+=["qualification/terrain.py"]
    return runner.digest(dict(
        original=runner.source_identity_base(root=root,study=study,stage="long-storms"),
        adapter={p:runner.checksum(study/p) for p in paths},
    ))


def calculate(*,stage,spec,study,root,config,context,directory):
    from research.floodwarp_methods_paper.qualification.hydraulics import run_event
    options=design(study=study)
    template=json.loads((root/"validation/hancock_long_storm_review/cases.json").read_text())
    if stage=="million-terrain":
        from research.floodwarp_methods_paper.qualification.terrain import measure
        return measure(spec=spec,config=config,template=template,options=options,directory=directory)
    return run_event(spec=spec,config=config,template=template,options=options,directory=directory,device=context["device"],
                     profile=stage=="million-profiles")


def run(*,root,study,config,stage,arguments):
    arguments,grid=grid_filter(arguments=arguments)
    flags=runner.options(args=arguments)
    system={} if flags["dry"] else runner.machine(root=root)
    with (device_slot(study=study,stage="resolution",device=flags["device"],dry=flags["dry"] or stage=="million-terrain"),
          patch.object(runner,"source_identity_base",runner.source_identity,create=True),
          patch.object(runner,"SIMULATIONS",(*runner.SIMULATIONS,stage)),
          patch.object(runner,"specs",select_specs(grid=grid,source=cases)),
          patch.object(runner,"source_identity",identity),
          patch.object(runner,"execute",calculate)):
        runner.run_stage(stage=stage,opts=flags,root=root,study=study,config=config,system=system)


if __name__ == "__main__":
    study=Path(__file__).resolve().parents[1]
    root=study.parents[1]
    import os
    config_path=Path(os.environ.get("FLOODWARP_PAPER_CONFIG",study/"configs/study.json"))
    run(root=root,study=study,config=read_config(path=config_path),stage=sys.argv[1],arguments=sys.argv[2:])

