"""Hold the hydraulic initial state fixed between alternate DEM representations."""

from unittest.mock import patch

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.plan import specs


STAGE = "terrain-initial-controls"


def cases(*, stage, config, root):
    for s in specs(stage="resolution", config=config, root=root):
        if s["initial"] == "dry" or s["dx"] == s["dem"]:
            continue
        yield dict(s, initial_geometry="sampled", purpose="isolated bed representation")


def fixed_initial(*, design, template, family, seed, initial, dx, dem):
    from research.floodwarp_methods_paper.terrain.generators.generate import fields

    (z, _, _, x, y), parameters = fields(
        design=design,
        template=template,
        family=family,
        seed=seed,
        initial=initial,
        dx=dx,
        dem=dem,
    )
    (_, h, hu, _, _), _ = fields(
        design=design,
        template=template,
        family=family,
        seed=seed,
        initial=initial,
        dx=dx,
        dem=dx,
    )
    parameters["initial_geometry_dx_m"] = dx
    return (z, h, hu, x, y), parameters


def calculate(*, stage, spec, study, root, config, context, directory):
    from research.floodwarp_methods_paper import events

    with patch.object(events, "fields", fixed_initial):
        return events.evaluate(
            stage="resolution",
            spec=spec,
            study=study,
            root=root,
            config=config,
            context=context,
            directory=directory,
        )


def source(*, root, study, stage):
    from research.floodwarp_methods_paper.evidence import implementation

    return runner.digest(
        dict(
            numerical_base=implementation(study.resolve(), "resolution"),
            control=runner.checksum(study / "terrain_controls.py"),
            amendment=runner.checksum(study / "supplementary/GEOMETRY_CONTROL.md"),
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
