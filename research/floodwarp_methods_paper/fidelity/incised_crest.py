"""Post hoc check of the late incised-catchment crest on the fixed 10 m represented bed."""

from functools import partial
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np

from research.floodwarp_methods_paper import events, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    continuation,
    recession,
)
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics, matrix


def fixed_fields(*, parent, dx, dem, **kwargs):
    """Subdivide the represented bed, initial state and rainfall-mask coordinates together."""
    values, parameters = parent(dx=dem, dem=dem, **kwargs)
    if dx == dem:
        return values, parameters
    factor = round(dem / dx)
    if factor < 1 or factor * dx != dem:
        raise ValueError("Expected an integer subdivision of the fixed bed")
    return tuple(
        np.repeat(np.repeat(a, factor, axis=0), factor, axis=1) for a in values
    ), parameters


def verify_geometry(*, study_config, template, spec, options, spacings):
    parent_dx = spec["dx"]

    def inputs(*, spacing):
        values, _ = hydraulics.represented(
            options=options,
            settings=dict(spec, initial_grid="parent"),
            design=study_config["synthetic"],
            template=template,
            family=spec["landform"],
            seed=spec["seed"],
            initial=spec["initial"],
            dx=spacing,
            dem=parent_dx,
        )
        return values

    parent = inputs(spacing=parent_dx)
    for spacing in spacings:
        if spacing == parent_dx:
            continue
        child = inputs(spacing=spacing)
        factor = round(parent_dx / spacing)
        for coarse, fine in zip(parent, child, strict=True):
            for y in range(factor):
                for x in range(factor):
                    np.testing.assert_array_equal(coarse, fine[y::factor, x::factor])


def prefixes(*, directory, settings, design):
    study = directory.parent
    root = study.parents[1]
    cfg = read_config(path=study / "configs/study.json")
    template = json.loads(
        (root / "validation/hancock_long_storm_review/cases.json").read_text()
    )
    options = matrix.design(study=study)
    base = next(
        s
        for s in matrix.cases(stage="million-storms", config=cfg, root=root)
        if s["case"] == design["case"] and s["method"] == "fv1"
    )
    verify_geometry(
        study_config=cfg,
        template=template,
        spec=base,
        options=options,
        spacings=sorted({dx for dx, _, _ in design["controls"]}),
    )
    source = runner.source_identity(root=root, study=study, stage="long-storms")
    hardware = runner.machine(root=root)
    nominal = events.numerical
    folder = directory / "incised_crest"
    items = []
    for dx, method, factor in design["controls"]:
        spec = dict(
            base, dx=dx, dem=design["bed_m"], initial_grid="parent", method=method
        )
        identity = dict(
            spec=spec,
            factor=factor,
            solver=source,
            design=design,
            code={
                p.name: runner.checksum(p)
                for p in (
                    Path(__file__),
                    study / "qualification/hydraulics.py",
                    study / "qualification/matrix.py",
                )
            },
            settings=settings,
            cfg=cfg,
            template=template,
            options=options,
            gpu=hardware["gpu"],
            warp=hardware["warp"],
        )
        ident = runner.digest(identity)[:24]
        dest = folder / ident
        path = dest / "result.json"
        if not path.is_file():
            dest.mkdir(parents=True, exist_ok=True)

            def numerical(*, method):
                name, mode, controls = nominal(method=method)
                return name, mode, dict(controls, cfl=0.4 * factor)

            with patch.object(events, "numerical", numerical):
                data = hydraulics.run_event(
                    spec=spec,
                    config=cfg,
                    template=template,
                    options=options,
                    directory=dest,
                    device=settings["device"],
                )
            recession.atomic_json(
                path=path, payload=dict(inputs=identity, hardware=hardware, data=data)
            )
        else:
            data = json.loads(path.read_text())["data"]
        items.append(
            assessment.item(
                spec=spec,
                data=data,
                ident=ident,
                path=dest,
                cohort="incised-crest",
                factor=factor,
            )
        )
        print(
            "Incised fixed-bed control",
            dx,
            method,
            factor,
            data["hours_completed"],
            "hours",
            flush=True,
        )
    return items


def experiment(*, directory, settings, design):
    config = read_config(path=directory.parent / "configs/study.json")
    full = assessment.archive(directory=directory, config=config, partial=True)
    native = [
        i
        for i in full["expanded", design["case"], "sampled"]
        if i["s"]["dx"] == design["bed_m"]
        and i["factor"] == 1
        and i["s"]["method"] in ("fv1", "hancock_adaptive")
    ]
    if len(native) != 2:
        raise ValueError("Missing native incised pair")

    def controls():
        return prefixes(directory=directory, settings=settings, design=design)

    check = continuation.capture(
        directory=directory,
        settings=settings,
        stage="incised-reference",
        experiment=controls,
    )
    for i in check:
        if i["s"]["dx"] == design["bed_m"]:
            assessment.matched_reference_inputs(base=native[0], control=i)
    items = [*native, *check]
    key = ("incised-crest", design["case"], "10m bed")

    def selected(**kwargs):
        return {key: items}

    with patch.object(assessment, "archive", selected):
        continuation.extend(directory=directory, settings=settings, cohorts=(key[0],))
    from research.floodwarp_methods_paper.fidelity import extended_analysis

    dest = directory / "incised_crest"
    completed = extended_analysis.attach(
        directory=directory, groups={key: items}, target=dest
    )
    if len(completed["complete-incised-crest", design["case"], "10m bed"]) != len(
        items
    ):
        raise ValueError("Missing late-crest control at the common end")
    from research.floodwarp_methods_paper.fidelity import crest_report

    crest_report.generate(directory=directory, design=design)


def run(*, directory, settings, design):
    geometry = partial(fixed_fields, parent=hydraulics.represented)
    with patch.object(hydraulics, "represented", geometry):
        experiment(directory=directory, settings=settings, design=design)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "recession_settings.json").read_text())
    design = json.loads((directory / "incised_settings.json").read_text())
    run(directory=directory, settings=settings, design=design)
