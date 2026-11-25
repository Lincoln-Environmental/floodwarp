"""Read-only hydraulic archive analysis, publication outputs and integrity gates."""

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.floodwarp_methods_paper import (
    advisor,
    comparisons,
    metrics,
    statistics,
    reference_audit,
)
from research.floodwarp_methods_paper import (
    catalog,
    evidence,
    terrain_controls,
    geometry,
)
from research.floodwarp_methods_paper import fixed_dt
from research.floodwarp_methods_paper.plan import specs
from research.floodwarp_methods_paper.runner import (
    archived,
    checksum,
    digest,
    verify_artifacts,
)


DATA_STAGES = (
    "analytical",
    "convergence",
    "synthetic",
    "long-storms",
    "resolution",
    "calibration",
    "performance",
    terrain_controls.STAGE,
    fixed_dt.STAGE,
)


def completeness(*, study, config, stages):
    counts = {}
    for stage in stages:
        planner = {
            terrain_controls.STAGE: terrain_controls.cases,
            fixed_dt.STAGE: fixed_dt.cases,
        }.get(stage, specs)
        expected = {
            digest(dict(stage=stage, settings=s, config=config))[:24]
            for s in planner(stage=stage, root=study.parents[1], config=config)
        }
        complete = {
            r["experiment_id"] for r in evidence.records(study=study, stages=(stage,))
        }
        missing = expected - complete
        if missing:
            raise ValueError(
                f"{stage} requires {len(missing)} further experiments. First missing ID {min(missing)}."
            )
        counts[stage] = len(expected)
    return counts


def hashes(*, study, paths):
    return {
        str(p.relative_to(study)): checksum(p) for p in sorted(paths) if p.is_file()
    }


def verify(*, study, config):
    counts = completeness(study=study, config=config, stages=DATA_STAGES)
    gates = comparisons.exact_check(study=study)
    for r in archived(study=study, stages=DATA_STAGES):
        verify_artifacts(record=r, study=study)
        if not (study / r["log"]).is_file():
            raise FileNotFoundError(r["log"])
    groups = {}
    forcing_keys = (
        "terrain_sha256",
        "rain_mask_sha256",
        "infiltration_sha256",
        "initial_depth_sha256",
        "initial_hu_sha256",
        "boundary_x_sha256",
        "boundary_y_sha256",
        "rain_times_s",
        "rain_rates_ms",
        "stage_m",
        "roughness",
    )
    nr = 0
    parity = 0
    for rec, row, folder in metrics.load(
        study=study,
        stages=(
            "long-storms",
            "resolution",
            "calibration",
            "performance",
            terrain_controls.STAGE,
        ),
    ):
        if "samples" not in row:
            continue
        s = rec["spec"]
        f = row["forcing"]
        samples = row["samples"]
        times = np.array(f["rain_times_s"])
        rates = np.array(f["rain_rates_ms"])
        rain_duration = np.diff(times)[(rates[:-1] + rates[1:]) > 0].sum()
        if (
            rain_duration < 21600
            or samples[-1]["t_s"] < rain_duration + 7200
            or not row["balance_pass"]
        ):
            raise ArithmeticError(
                "Storm duration or water budget failed " + rec["experiment_id"]
            )
        with np.load(folder / "accepted_dt.npz") as a:
            dt = a["dt_s"]
            if (
                not np.isfinite(dt).all()
                or (dt <= 0).any()
                or len(dt) != row["steps"]
                or abs(dt.sum() - samples[-1]["t_s"]) > 1e-6
            ):
                raise ArithmeticError("Accepted history failed " + rec["experiment_id"])
        h = digest([f[k] for k in forcing_keys])
        key = (s["case"], s["dx"], s["dem"], s["multiplier"], s.get("initial_geometry"))
        if key in groups and h != groups[key]:
            raise ArithmeticError("Unmatched forcing " + str(key))
        groups[key] = h
        nr += 1
    matched = {}
    for rec, row, folder in metrics.load(
        study=study, stages=("long-storms", "resolution", terrain_controls.STAGE)
    ):
        s = rec["spec"]
        if s["method"] in ("hancock_dense", "hancock_adaptive"):
            key = s["case"], s["dx"], s["dem"], s.get("initial_geometry")
            matched.setdefault(key, {})[s["method"]] = (row, folder)
    for key, pair in matched.items():
        if len(pair) != 2:
            raise ValueError("Missing dense or tiled control " + str(key))
        (a, pa), (b, pb) = pair.values()
        for k in (
            "accepted_history_sha256",
            "rain_m3",
            "infiltration_m3",
            "final_m3",
            "outflow_m3",
            "inflow_m3",
        ):
            if a[k] != b[k]:
                raise ArithmeticError("Parity failed " + str((key, k)))
        with np.load(pa / a["fields_file"]) as fa, np.load(pb / b["fields_file"]) as fb:
            for field in fa.files:
                np.testing.assert_array_equal(fa[field], fb[field])
        parity += 1
    with sqlite3.connect(study / "processed/metrics/study.sqlite") as db:
        for path in (study / "processed").rglob("*.csv"):
            rows = metrics.csv_read(path)
            if not rows:
                continue
            name = "_".join(path.relative_to(study / "processed").with_suffix("").parts)
            info = db.execute('PRAGMA table_info("' + name + '")').fetchall()
            columns = [v[1] for v in info]
            numeric = [v[2] == "REAL" for v in info]
            values = db.execute('SELECT * FROM "' + name + '" ORDER BY rowid')
            for row, stored in zip(rows, values, strict=True):
                expected = tuple(
                    None if row[k] == "" else float(row[k]) if num else row[k]
                    for k, num in zip(columns, numeric, strict=True)
                )
                if expected != stored:
                    raise ValueError("SQL value differs for " + name)
    from research.floodwarp_methods_paper.tables.generate import generate

    with TemporaryDirectory() as temporary:
        render = Path(temporary)
        rendered = generate(study=study, output=render)
        for item in render.iterdir():
            if checksum(item) != checksum(study / "tables" / item.name):
                raise ValueError(
                    "Published table differs from source evidence " + item.name
                )
    table_count, figure_count = verify_publication_inventory(study=study, rendered=rendered)
    from research.floodwarp_methods_paper.fidelity import catalogue
    from research.floodwarp_methods_paper.fidelity.orton_transfer import verify_index

    native_results = study / "fidelity/native_results"
    native_sql = catalogue.verify_database(target=native_results) if native_results.is_dir() else None

    return dict(
        completed=counts,
        analytical=gates,
        complete_hydrographs=nr,
        matched_input_groups=len(groups),
        exact_scheduler_pairs=parity,
        main_tables=table_count,
        main_figures=figure_count,
        native_database=native_sql,
        verified_real_terrain_transfers=verify_index(directory=study / "fidelity/orton_transfer"),
    )


def verify_publication_inventory(*, study, rendered):
    table_ids = set(range(1, 14))
    figure_ids = set(range(1, 19))
    if (study / "fidelity/outlet_results/events.csv").is_file():
        table_ids.update((16, 19, 20))
        figure_ids.add(19)
    native_results = study / "fidelity/native_results"
    for path in native_results.glob("least_cost_*.csv"):
        if any(r["cohort"] == "complete-native-metre" and int(r["assessable"]) > 0
               and not r["objective"].endswith("_1cm") for r in metrics.csv_read(file=path)):
            table_ids.update((21, 22))
            break
    if (study / "fidelity/orton_transfer/transfer.csv").is_file():
        table_ids.update((17, 18))
        figure_ids.add(20)
    for folder, suffixes, expected in (("tables", ("csv", "md", "tex"), table_ids),
                                       ("figures", ("pdf", "svg", "png"), figure_ids)):
        stems = {p.stem for p in (study / folder).glob(f"[0-9]*.{suffixes[0]}")}
        if len(stems) != len(expected) or {int(name.split("_", 1)[0]) for name in stems} != expected:
            raise ValueError(f"Incomplete or unexpected publication inventory in {folder}")
        if folder == "tables" and stems != set(rendered):
            raise ValueError("Archived table inventory differs from regeneration")
        for ext in suffixes[1:]:
            if {p.stem for p in (study / folder).glob(f"[0-9]*.{ext}")} != stems:
                raise ValueError(f"Incomplete {ext} products in {folder}")
    return len(table_ids), len(figure_ids)


def evaluate(*, stage, spec, study, root, config, context, directory):
    processed = study / "processed"
    terrain = processed / "metrics/terrain.json"
    paths = []
    if stage == "terrain-features":
        completeness(study=study, config=config, stages=DATA_STAGES)
        gates = comparisons.exact_check(study=study)
        catalog.collect(study=study)
        rows = metrics.build(study=study, config=config)
        from research.floodwarp_methods_paper.qualification import reporting

        expanded = reporting.aggregate(study=study, config=config)
        comparisons.pairwise(study=study, config=config)
        comparisons.historical(root=root, study=study)
        reference_audit.independent(study=study)
        reference_audit.assess(study=study)
        reference_audit.integrity_by_family(study=study)
        geometry.compare(study=study)
        fixed_dt.analyse(study=study)
        statistics.temporal_orders(study=study)
        statistics.operating(study=study)
        statistics.timing_replacements(study=study)
        statistics.transfer_groups(study=study, settings=config["advisor"])
        features = advisor.terrain_features(study=study, config=config, output=terrain)
        result = dict(
            terrain_grids=len(features), records=rows, gates=gates, expanded=expanded
        )
        paths = [
            *(p for p in processed.rglob("*.csv") if p.parent.name != "advisor"),
            terrain,
            *(study / "terrain").rglob("*.tif"),
        ]
    elif stage == "advisor-train":
        result = advisor.train(
            study=study,
            config=config,
            features=json.loads(terrain.read_text()),
            context=context,
            directory=directory,
        )
        paths = [processed / "advisor/model.json"]
    elif stage == "advisor-validate":
        result = advisor.validate(
            study=study,
            config=config,
            features=json.loads(terrain.read_text()),
            directory=directory,
        )
        statistics.advisor_errors(study=study, settings=config["advisor"])
        paths = [
            *(processed / "advisor").glob("*.csv"),
            statistics.database(study=study),
        ]
    elif stage == "tables":
        from research.floodwarp_methods_paper.tables.generate import generate

        result = generate(study=study)
        paths = (study / "tables").glob("[0-9]*.*")
    elif stage == "figures":
        from research.floodwarp_methods_paper.figures.generate import generate

        result = generate(study=study)
        paths = [
            *(study / "figures").glob("[0-9]*.*"),
            *(study / "supplementary/pareto").rglob("*"),
            *(study / "supplementary/operational_storms").rglob("*"),
            *(study / "supplementary/native_grid_extent").glob("*"),
            *(study / "supplementary/incised").glob("hour_eight_water.*"),
            *(study / "supplementary/incised").glob("simultaneous_extent.*"),
        ]
    elif stage == "manuscript":
        from research.floodwarp_methods_paper.manuscript.generate import generate

        result = generate(study=study)
        paths = [
            *list((study / "manuscript").glob("*.md")),
            study / "RESULTS.md",
            study / "DISCUSSION.md",
            study / "LIMITATIONS.md",
            study / "CORRECTIVE_AUDIT.md",
        ]
    elif stage == "verify":
        from research.floodwarp_methods_paper.qualification import reporting

        with evidence.archived_publication(study=study):
            result = verify(study=study, config=config)
            result["expanded"] = reporting.verification(study=study, config=config)
        result["cohort"] = "archived paper experiments under their recorded numerical revisions"
    else:
        raise ValueError(stage)
    result["output_sha256"] = hashes(study=study, paths=paths)
    return result
