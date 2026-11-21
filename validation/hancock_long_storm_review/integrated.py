"""Remeasure the integrated source operator and check every physical output against its predecessor."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from validation.hancock_long_storm_review.storms import event, experiment_cases


def identity(*, root):
    paths = sorted((root / "src/floodwarp").glob("*.py"))
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def compare(*, reference, measured, previous_arrays, measured_arrays, times):
    for key, value in reference.items():
        if key in times or key == "samples":
            continue
        if value != measured[key]:
            raise ArithmeticError(f"{reference['name']} differs in {key}")
    for old, new in zip(reference["samples"], measured["samples"], strict=True):
        for key, value in old.items():
            if key not in times and value != new[key]:
                raise ArithmeticError(
                    f"{reference['name']} differs in {key} at {old['t_s']} s"
                )
    if previous_arrays.exists():
        with np.load(previous_arrays) as old, np.load(measured_arrays) as new:
            if old.files != new.files:
                raise ArithmeticError("Field set changed")
            for key in old.files:
                np.testing.assert_array_equal(old[key], new[key], err_msg=key)


def run(*, review, fields, suite, device, times, root):
    config = json.loads((review / "cases.json").read_text())
    directory = review / "integrated_events"
    directory.mkdir(exist_ok=True)
    archive = fields.parent / "integrated_fields"
    archive.mkdir(exist_ok=True)
    stamp = directory / "source.json"
    source = identity(root=root)
    if stamp.exists():
        if json.loads(stamp.read_text())["sources"] != source["sources"]:
            raise ValueError(
                "Integrated results have a different numerical source identity"
            )
    else:
        stamp.write_text(json.dumps(source, indent=2) + "\n")
    originals = {
        (
            r["landform"],
            r["storm"],
            r["method"],
            r["mode"],
            r["dx"],
            r["terrain_dx"],
            tuple(r["dimensions_m"]),
        ): r
        for f in (review / "events").glob(suite + "_*.json")
        if (r := json.loads(f.read_text()))["group"] == suite
    }
    passed = []
    for case in experiment_cases(config=config, suite=suite):
        if case["method"] not in ("hancock", "fv2"):
            continue
        key = (
            case["landform"],
            case["storm"],
            case["method"],
            case["mode"],
            case["dx"],
            case["terrain_dx"],
            tuple(case["dimensions"]),
        )
        old = originals[key]
        event(
            config=config,
            directory=directory,
            fields=archive,
            device=device,
            label_name=suite,
            **case,
        )
        name = old["name"]
        fresh = json.loads((directory / (name + ".json")).read_text())
        compare(
            reference=old,
            measured=fresh,
            previous_arrays=fields / (name + ".npz"),
            measured_arrays=archive / (name + ".npz"),
            times=times,
        )
        passed.append(name)
    (directory / (suite + ".verified")).write_text("\n".join(passed) + "\n")


if __name__ == "__main__":
    review = Path(__file__).resolve().parent
    root = review.parents[1]
    fields = root / "outputs/hancock_long_storm_review/fields"
    suites = os.environ.get(
        "FLOODWARP_INTEGRATED_SUITES",
        "primary,moving,large,terrain_large,refinement,directional",
    ).split(",")
    device = "cuda:0"
    times = {
        "end_to_end_s",
        "extraction_s",
        "field_write_s",
        "process_pool_mib",
        "hydraulic_s",
        "loop_s",
        "capture_s",
        "preload_s",
        "hydraulic_wall_s",
    }
    for suite in suites:
        run(
            review=review,
            fields=fields,
            suite=suite,
            device=device,
            times=times,
            root=root,
        )
