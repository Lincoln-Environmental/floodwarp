"""Typed SQLite mirror of the observed result tables with verifiable source and row counts."""

import csv
from itertools import chain, zip_longest
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


def verify_database(*, target):
    """Compare every stored observation and its source receipt with the CSV archive."""
    files = {p.name: p for p in target.glob("*.csv")}
    path = target / "fidelity.sqlite"
    count = 0
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Invalid fidelity SQLite database")
        receipts = {name: (sha, rows) for name, sha, rows in
                    db.execute("SELECT file,sha256,rows FROM csv_sources")}
        if files.keys() != receipts.keys():
            raise ValueError("SQLite source inventory differs from the CSV archive")
        tables = {"csv_sources"}
        for name, file in files.items():
            sha, row_count = receipts[name]
            if runner.checksum(file) != sha:
                raise ValueError(f"SQLite source checksum differs for {name}")
            with file.open(newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                if not header:
                    if row_count:
                        raise ValueError(f"Missing CSV header for {name}")
                    continue
                tables.add(file.stem)
                table = '"' + file.stem.replace('"', '""') + '"'
                query = db.execute("SELECT * FROM " + table + " ORDER BY rowid")
                if [column[0] for column in query.description] != header:
                    raise ValueError(f"SQLite columns differ for {name}")
                rows = 0
                for fields, stored in zip_longest(reader, query):
                    if fields is None or stored is None:
                        raise ValueError(f"SQLite row count differs for {name}")
                    expected = tuple(cell(value=v, name=k) for k, v in zip(header, fields, strict=True))
                    rows += 1
                    if stored != expected:
                        raise ValueError(f"SQLite observation differs for {name} at data row {rows}")
                if rows != row_count:
                    raise ValueError(f"SQLite receipt row count differs for {name}")
                count += rows
        actual = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if actual != tables:
            raise ValueError("Unexpected table in fidelity SQLite database")
    return dict(source_files=len(files), observations=count)


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
        "native_features",
        "native_publication",
        "native_advisor",
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
