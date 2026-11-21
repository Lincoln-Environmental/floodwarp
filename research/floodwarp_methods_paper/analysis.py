"""Read-only hydraulic archive analysis, publication outputs and integrity gates."""

import json
import sqlite3

import numpy as np

from research.floodwarp_methods_paper import advisor, comparisons, metrics, statistics
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
)


def completeness(*, study, config, stages):
    counts = {}
    for stage in stages:
        expected = {
            digest(dict(stage=stage, settings=s, config=config))[:24]
            for s in specs(stage=stage, root=study.parents[1], config=config)
        }
        complete = {r["experiment_id"] for r in archived(study=study, stages=(stage,))}
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
        study=study, stages=("long-storms", "resolution", "calibration", "performance")
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
        key = (s["case"], s["dx"], s["dem"], s["multiplier"])
        if key in groups and h != groups[key]:
            raise ArithmeticError("Unmatched forcing " + str(key))
        groups[key] = h
        nr += 1
    matched = {}
    for rec, row, folder in metrics.load(
        study=study, stages=("long-storms", "resolution")
    ):
        s = rec["spec"]
        if s["method"] in ("hancock_dense", "hancock_adaptive"):
            key = s["case"], s["dx"], s["dem"]
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
            n = db.execute('SELECT COUNT(*) FROM "' + name + '"').fetchone()[0]
            if n != len(rows):
                raise ValueError("SQL count differs for " + name)
    table_count = 0
    for path in (study / "tables").glob("[0-9]*.csv"):
        if not all(path.with_suffix(ext).is_file() for ext in (".md", ".tex")):
            raise ValueError("Incomplete table " + path.stem)
        table_count += 1
    figs = list((study / "figures").glob("[0-9]*.pdf"))
    if (
        table_count != 10
        or len(figs) != 12
        or not all(p.with_suffix(e).is_file() for p in figs for e in (".svg", ".png"))
    ):
        raise ValueError("Publication output count mismatch")
    return dict(
        completed=counts,
        analytical=gates,
        complete_hydrographs=nr,
        matched_input_groups=len(groups),
        exact_scheduler_pairs=parity,
        main_tables=table_count,
        main_figures=len(figs),
    )


def evaluate(*, stage, spec, study, root, config, context, directory):
    processed = study / "processed"
    terrain = processed / "metrics/terrain.json"
    paths = []
    if stage == "terrain-features":
        completeness(study=study, config=config, stages=DATA_STAGES)
        gates = comparisons.exact_check(study=study)
        rows = metrics.build(study=study, config=config)
        comparisons.pairwise(study=study, config=config)
        comparisons.historical(root=root, study=study)
        statistics.temporal_orders(study=study)
        statistics.operating(study=study)
        statistics.transfer_groups(study=study, settings=config["advisor"])
        features = advisor.terrain_features(study=study, config=config, output=terrain)
        result = dict(terrain_grids=len(features), records=rows, gates=gates)
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
        ]
    elif stage == "manuscript":
        from research.floodwarp_methods_paper.manuscript.generate import generate

        result = generate(study=study)
        paths = [
            *list((study / "manuscript").glob("*.md")),
            study / "RESULTS.md",
            study / "DISCUSSION.md",
        ]
    elif stage == "verify":
        result = verify(study=study, config=config)
    else:
        raise ValueError(stage)
    result["output_sha256"] = hashes(study=study, paths=paths)
    return result
