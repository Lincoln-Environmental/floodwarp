"""Operational and fixed-area comparisons launched only from aligned native-metre inputs."""

import fnmatch
import getopt
import json
from pathlib import Path
import sys
from time import time_ns, perf_counter
import traceback
from unittest.mock import patch

from floodwarp.solver_advisor.execution import configure_hancock_execution
from research.floodwarp_methods_paper import events, metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    continuation,
    extended_analysis,
    native_event,
    recession,
)
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics, matrix
from research.floodwarp_methods_paper.terrain.generators import native_inputs


def specifications(*, study, config, design):
    root = study.parents[1]
    for base in matrix.cases(stage="million-terrain", config=config, root=root):
        seed = base["seed"]
        if seed in design["primary_seeds"]:
            methods = design["primary_methods"] + design["extra_methods"].get(
                str(seed), []
            )
            experiment = "operational"
        elif seed == design["matched_km_seed"]:
            methods = design["km_methods"]
            experiment = "fixed_physical_domain"
        elif seed in design["scale_seeds"]:
            methods = design["scale_methods"]
            experiment = "cell_count"
        else:
            continue
        for name in methods:
            factors = [1.0]
            if (
                name == "adaptive"
                and experiment == "fixed_physical_domain"
                and base["dx"] == 1
            ):
                factors.append(design["reference_cfl_factor"])
            for factor in factors:
                yield dict(
                    base,
                    method="hancock_dense" if name == "hancock_auto" else name,
                    requested_method=name,
                    factor=factor,
                    initial_grid="native_1m_block_mean",
                    terrain_aggregation="native_1m_block_mean",
                    experiment_class=experiment,
                )

    length, width = design["reference_dimensions_m"]
    for domain in design["reference_domains"]:
        name = (
            f"{domain['landform']}-{domain['seed']}-{domain['storm']}-{length}x{width}"
        )
        for spacing, methods in design["reference_grid_methods"].items():
            dx = int(spacing)
            for method in methods:
                factors = [1.0]
                if dx == 1 and method == "adaptive":
                    factors.append(design["reference_cfl_factor"])
                for factor in factors:
                    yield dict(
                        domain,
                        case=name,
                        dimensions=[length, width],
                        dx=dx,
                        dem=dx,
                        hours=8,
                        interval=design["reference_interval_s"],
                        controls=True,
                        support=10,
                        coastal_stage=domain["landform"] == "coastal",
                        multiplier=1.0,
                        method=method,
                        requested_method=method,
                        factor=factor,
                        initial_grid="native_1m_block_mean",
                        terrain_aggregation="native_1m_block_mean",
                        experiment_class="independent_reference",
                    )


def in_suite(*, spec, suite):
    if suite == "references":
        return spec["experiment_class"] in (
            "independent_reference",
            "fixed_physical_domain",
        )
    if suite == "heldout_references":
        return (
            spec["experiment_class"] == "independent_reference"
            and spec["split"] == "test"
        )
    return suite is None or spec["experiment_class"] == suite


def prepare(*, spec, bundle, receipt, **kwargs):
    normal = events.numerical

    def numerical(*, method):
        name, mode, options = normal(method=method)
        if name != "li":
            options = dict(options, cfl=0.4 * spec["factor"])
        return name, mode, options

    with patch.object(events, "numerical", numerical):
        model, expected, forcing = native_event.create(
            spec=spec, bundle=bundle, receipt=receipt, **kwargs
        )
    if spec["requested_method"] == "hancock_auto":
        forcing["execution_selection"] = configure_hancock_execution(
            solver=model.solver, execution_mode="auto"
        )
        forcing["numerical"] = model.solver.numerical_configuration()
        model.identity["requested_method"] = spec["requested_method"]
    return model, expected, forcing


def all_items(*, items, cohort):
    """Time every requested execution policy through the common recession."""
    return items


def prior_prefix(*, directory, spec, receipt, solver, inputs, machine, settings):
    if (
        spec.get("timing_repeat")
        or spec["factor"] != 1
        or spec["requested_method"] != spec["method"]
    ):
        return None
    for path in (directory / "mean_storms").glob("*/result.json"):
        saved = json.loads(path.read_text())
        previous = saved["inputs"]
        if (
            any(spec.get(k) != v for k, v in previous["spec"].items())
            or previous["terrain"] != receipt
            or previous["solver"] != solver
            or previous["physical"] != inputs
            or previous["recession"] != settings
            or previous["factory"] != runner.checksum(Path(native_event.__file__))
            or previous["hydraulic_driver"]
            != runner.checksum(Path(hydraulics.__file__))
            or previous["capture"] != runner.checksum(Path(continuation.__file__))
            or previous["hardware"] != [machine["gpu"], machine["warp"]]
        ):
            continue
        if path.parent.name != runner.digest(previous)[:24]:
            raise ValueError("Invalid archived native experiment identity")
        return assessment.item(
            spec=spec,
            data=saved["data"],
            ident=path.parent.name,
            path=path.parent,
            cohort="native-metre",
        )
    return None


def run_group(*, directory, specs, config, design, settings, machine):
    root = directory.parents[2]
    base = specs[0]
    inputs = dict(
        cfg=config,
        template=json.loads(
            (root / "validation/hancock_long_storm_review/cases.json").read_text()
        ),
        options=matrix.design(study=directory.parent),
    )
    bundle, receipt = native_inputs.generate(
        spec=base,
        inputs=inputs,
        cache_root=root / ".fidelity-restarts/native_inputs",
        strip_rows=design["strip_rows"],
        channel_core_m=design["channel_core_min_depth_m"],
        include_native=base["experiment_class"]
        in ("fixed_physical_domain", "independent_reference"),
    )
    output = directory / "native_matrix" / base["case"]
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        str(p.relative_to(root)): runner.checksum(p)
        for p in (
            Path(__file__),
            Path(native_event.__file__),
            Path(hydraulics.__file__),
            Path(continuation.__file__),
            Path(recession.__file__),
            Path(configure_hancock_execution.__code__.co_filename),
            root / "src/floodwarp/solver_advisor/execution_policy.json",
        )
    }
    physical_sha = runner.digest(receipt)
    runner.write_json(path=output / (physical_sha + ".json"), data=receipt)
    baseline = runner.source_identity(
        root=root, study=directory.parent, stage="long-storms"
    )
    cohort = (
        "native-metre"
        if not base.get("timing_repeat")
        else "native-repeat-" + str(base["timing_repeat"])
    )
    key = cohort, base["case"], "sampled"

    def run_prefixes():
        result = []
        for spec in specs:
            prior = prior_prefix(
                directory=directory,
                spec=spec,
                receipt=receipt,
                solver=baseline,
                inputs=inputs,
                machine=machine,
                settings=settings,
            )
            if prior is not None:
                result.append(prior)
                continue
            definition = dict(
                spec=spec,
                physical_sha=physical_sha,
                sources=sources,
                solver=baseline,
                settings=settings,
                inputs=inputs,
                hardware=[machine[k] for k in ("warp", "gpu", "python")],
                device=settings["device"],
            )
            ident = runner.digest(definition)[:24]
            folder = output / ident
            path = folder / "result.json"
            folder.mkdir(exist_ok=True)
            if not path.exists():
                print(
                    "Starting",
                    spec["case"],
                    spec["dx"],
                    spec["requested_method"],
                    spec["factor"],
                    flush=True,
                )
                attempt = folder / f"attempt_{time_ns()}"
                attempt.mkdir()
                runner.write_json(
                    path=attempt / "started.json",
                    data=dict(inputs=definition, hardware=machine),
                )
                clock = perf_counter()
                try:
                    with runner.log_output(path=attempt / "solver.log"):
                        data = hydraulics.run_event(
                            spec=spec,
                            config=config,
                            template=inputs["template"],
                            options=inputs["options"],
                            directory=folder,
                            device=settings["device"],
                        )
                    recession.atomic_json(
                        path=path,
                        payload=dict(inputs=definition, hardware=machine, data=data),
                    )
                    runner.write_json(
                        path=attempt / "complete.json",
                        data=dict(
                            seconds=perf_counter() - clock,
                            result_sha256=runner.checksum(path),
                            spatial_sha256=runner.checksum(folder / "spatial.npz"),
                        ),
                    )
                except Exception:
                    runner.write_json(
                        path=attempt / "failed.json",
                        data=dict(
                            seconds=perf_counter() - clock,
                            traceback=traceback.format_exc(),
                        ),
                    )
                    raise
            else:
                saved = json.loads(path.read_text())
                if saved["inputs"] != definition:
                    raise ValueError("Mismatched native prefix identity")
                data = saved["data"]
            result.append(
                assessment.item(
                    spec=spec,
                    data=data,
                    ident=ident,
                    path=folder,
                    cohort=cohort,
                    factor=spec["factor"],
                )
            )
        return result

    def create(**kwargs):
        return prepare(bundle=bundle, receipt=receipt, **kwargs)

    with (
        patch.object(events, "create_event", create),
        patch.object(recession, "candidates", all_items),
    ):
        items = continuation.capture(
            directory=directory,
            settings=settings,
            stage=cohort,
            experiment=run_prefixes,
        )
        groups = {key: items}

        def selected(**kwargs):
            return groups

        with patch.object(assessment, "archive", selected):
            continuation.extend(
                directory=directory, settings=settings, cohorts=(cohort,)
            )
        group_id = runner.digest(sorted(i["id"] for i in items))[:24]
        target = output / group_id
        target.mkdir(exist_ok=True)
        finished = extended_analysis.attach(
            directory=directory, groups=groups, target=target
        )
        rows = []
        parents = {item["id"]: item for item in items}
        for item in finished["complete-" + cohort, base["case"], "sampled"]:
            t, q = assessment.curve(it=item, station="outlet")
            parent = parents[item.get("parent_id", item["id"])]
            d, s = item["d"], parent["s"]
            rows.append(
                dict(
                    case=s["case"],
                    class_name=s["experiment_class"],
                    landform=s["landform"],
                    method=s["requested_method"],
                    dx_m=s["dx"],
                    factor=s["factor"],
                    id=item["id"],
                    parent_id=parent["id"],
                    prefix_dir=str(parent["path"].relative_to(directory)),
                    interval_s=s["interval"],
                    support_m=s["support"],
                    execution=d.get("forcing", {})
                    .get("execution_selection", {})
                    .get("selected", parent["d"]["execution"]),
                    end_h=t[-1] / 3600,
                    rain_m3=d["rain_m3"],
                    net_outlet_m3=float(q @ (t[1:] - t[:-1])),
                    peak_m3s=float(q.max()),
                    storage_m3=d["final_storage_m3"],
                    residual_m3=d["residual_m3"],
                    hydraulic_s=d["hydraulic_s"],
                    steps=d["steps"],
                    native_sha=physical_sha,
                )
            )
        metrics.csv_write(file=target / "complete_events.csv", rows=rows)
        print(
            f"Completed {len(rows)} configurations of {base['case']} through {rows[0]['end_h']:g} h",
            flush=True,
        )


def run(*, directory, arguments):
    flags, rest = getopt.getopt(
        arguments,
        "",
        ["dry-run", "resume", "force", "case=", "method=", "dx=", "dem=", "suite="],
    )
    if rest:
        raise ValueError("Unknown arguments " + str(rest))
    opts = dict(flags)
    if "--resume" in opts and "--force" in opts:
        raise ValueError("Choose resume or an independent timing repeat")
    if (
        "--dx" in opts
        and "--dem" in opts
        and float(opts["--dx"]) != float(opts["--dem"])
    ):
        raise ValueError(
            "Practical runs require the same bed and hydraulic grid spacing"
        )
    repeat = time_ns() if "--force" in opts else None
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    config = read_config(path=directory.parent / "configs/study.json")
    settings = json.loads((directory / "recession_settings.json").read_text())
    groups = {}
    for s in specifications(study=directory.parent, config=config, design=design):
        if (
            not fnmatch.fnmatchcase(s["case"], opts.get("--case", "*"))
            or opts.get("--method", s["requested_method"]) != s["requested_method"]
            or ("--dx" in opts and s["dx"] != float(opts["--dx"]))
            or ("--dem" in opts and s["dem"] != float(opts["--dem"]))
            or not in_suite(spec=s, suite=opts.get("--suite"))
        ):
            continue
        groups.setdefault(s["case"], []).append(
            dict(s, timing_repeat=repeat) if repeat else s
        )
    if not groups:
        raise ValueError("No native storm matches the requested selection")
    if "--dry-run" in opts:
        for items in groups.values():
            for s in items:
                print(
                    s["case"],
                    s["dx"],
                    s["requested_method"],
                    "CFL factor",
                    s["factor"],
                    s["experiment_class"],
                )
        return
    machine = runner.machine(root=directory.parents[2])
    for specs in groups.values():
        run_group(
            directory=directory,
            specs=specs,
            config=config,
            design=design,
            settings=settings,
            machine=machine,
        )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    arguments = sys.argv[1:]
    run(directory=directory, arguments=arguments)
