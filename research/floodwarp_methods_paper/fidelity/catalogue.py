"""Typed SQLite mirror of the observed result tables with verifiable source and row counts."""

import csv
from itertools import chain
from pathlib import Path
import sqlite3
import tempfile
from research.floodwarp_methods_paper import runner


def cell(*, value, name):
    if value == "":
        return None
    if name.endswith(("_id", "sha256", "revision")) or name == "id":
        return value
    if value == "True":
        return 1
    if value == "False":
        return 0
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def database(*, target):
    with tempfile.NamedTemporaryFile(dir=target, suffix=".sqlite", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE csv_sources (file TEXT PRIMARY KEY, sha256 TEXT, rows INTEGER)"
            )
            for file in sorted(target.glob("*.csv")):
                with file.open(newline="") as stream:
                    reader = csv.reader(stream)
                    names = next(reader, [])
                    table = '"' + file.stem + '"'
                    if not names:
                        db.execute(
                            "INSERT INTO csv_sources VALUES (?,?,?)",
                            (file.name, runner.checksum(file), 0),
                        )
                        continue
                    columns = ",".join(
                        '"' + name.replace('"', '""') + '"' for name in names
                    )
                    db.execute("CREATE TABLE " + table + " (" + columns + ")")
                    rows = (
                        tuple(
                            cell(value=value, name=name)
                            for name, value in zip(names, row, strict=True)
                        )
                        for row in reader
                    )
                    marks = ",".join("?" for _ in names)
                    db.executemany(
                        "INSERT INTO " + table + " VALUES (" + marks + ")", rows
                    )
                    count = db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                    db.execute(
                        "INSERT INTO csv_sources VALUES (?,?,?)",
                        (file.name, runner.checksum(file), count),
                    )
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Invalid fidelity database")
        path.replace(target / "fidelity.sqlite")
    finally:
        path.unlink(missing_ok=True)


def manifest(*, directory, package):
    root = directory.parents[2]
    raw = (
        "long_pulses",
        "sharp_pulses",
        "oblique_runs",
        "pulse_runs",
        "storm_runs",
        "extended_runs",
        "bed_runs",
        "policy_runs",
        "small_storms",
        "mixed_native",
        "endpoint_fields",
        "storage_tail",
        "incised_crest",
        "mean_storms",
        "mean_windows",
        "native_matrix",
        "native_terrain",
        "native_results",
        "native_cpu_control",
        "native_outlet",
        "native_tail",
        "free_plane",
        "free_reach",
        "verification",
        "extension_timings",
    )
    sources = chain(
        directory.glob("*.py"),
        directory.glob("*.md"),
        directory.glob("*settings.json"),
        (
            directory / "design.json",
            package,
            directory.parent / "terrain/generators/native_inputs.py",
            directory.parent / "terrain/generators/mean_dem.py",
            root / "scripts/run_hydrograph_fidelity.sh",
        ),
        (
            file
            for name in raw
            for file in (directory / name).rglob("*")
            if file.is_file()
        ),
        (directory / "results").glob("*"),
        (directory / "figures").glob("*"),
    )
    return {
        str(file.relative_to(root)): runner.checksum(file)
        for file in sorted(set(sources))
        if file.is_file()
    }
