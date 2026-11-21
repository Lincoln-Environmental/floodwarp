"""Check complete-event coverage, hydraulic budgets, parity and source provenance."""

from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from validation.hancock_long_storm_review.landforms import physical_fields
from validation.hancock_long_storm_review.storms import experiment_cases


def checksum(*, path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hydraulic_inputs(*, config, records):
    seen = {}
    for r in records:
        length, width = r["dimensions_m"]
        key = f"{r['landform']}_{length}x{width}_{r['dx']}m_{r['terrain_dx']}dem"
        if key in seen:
            continue
        z, h, hu, _, _ = physical_fields(
            landform=r["landform"],
            spec=config["landforms"][r["landform"]],
            seed=config["seed"],
            length=length,
            width=width,
            dx=r["dx"],
            terrain_dx=r["terrain_dx"],
        )
        seen[key] = dict(
            dimensions_m=[length, width],
            shape=z.shape,
            spacing_m=r["dx"],
            terrain_spacing_m=r["terrain_dx"],
            landform=r["landform"],
            sha256={
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in (("bed_f64", z), ("h0_f64", h), ("hu0_f64", hu))
            },
        )
    return seen


def verify(*, review, fields, counts, baseline, methods, modules, pytest_summary):
    files = sorted((review / "events").glob("*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if Counter(r["group"] for r in rows) != counts:
        raise ValueError("Incomplete set of full events")
    config = json.loads((review / "cases.json").read_text())
    inputs = hydraulic_inputs(config=config, records=rows)
    (review / "input_checksums.json").write_text(json.dumps(inputs, indent=2) + "\n")
    index = []
    for f, r in zip(files, rows, strict=True):
        samples = r["samples"]
        if samples[-1]["t_s"] != r["hours"] * 3600 or samples[-1]["t_s"] < 28800:
            raise ArithmeticError("Incomplete recession in " + r["name"])
        if not r["balance_pass"] or any(
            s["steps"] <= 0 or s["dt_min_s"] <= 0 for s in samples
        ):
            raise ArithmeticError("Mass or timestep failure in " + r["name"])
        end_rain = next(s["rain_m3"] for s in samples if s["t_s"] == 21600)
        np.testing.assert_allclose(end_rain, r["rain_m3"], rtol=4e-12, atol=1e-10)
        np.testing.assert_allclose(
            r["rain_m3"], r["expected_rain_m3"], rtol=1e-10, atol=1e-8
        )
        archive = fields / (r["name"] + ".npz")
        if r["group"] not in ("large", "terrain_large") and not archive.exists():
            raise FileNotFoundError(archive)
        index.append(
            dict(
                name=r["name"],
                terrain=f"{r['landform']}_{r['dimensions_m'][0]}x{r['dimensions_m'][1]}_{r['dx']}m_{r['terrain_dx']}dem",
                forcing=r["storm"],
                event_sha256=checksum(path=f),
                fields_sha256=checksum(path=archive) if archive.exists() else None,
            )
        )
    parity = json.loads((review / "event_parity.json").read_text())
    if len(parity) != 96 or not all(r["matched"] for r in parity):
        raise ArithmeticError("Primary Hancock parity failed")
    moving = [r for r in rows if r["group"] == "moving" and r["method"] == "hancock"]
    if len(moving) != 3 or any(
        r["final_sha256"] != moving[0]["final_sha256"] for r in moving
    ):
        raise ArithmeticError("Moving storm differs between Hancock schedules")
    for case in json.loads((review / "structure_event_cuda.json").read_text()):
        peer = next(
            r
            for r in json.loads((review / "structure_event_cpu.json").read_text())
            if r["mode"] == case["mode"] and r["cap_s"] == case["cap_s"]
        )
        np.testing.assert_allclose(
            [s["depths_m"] for s in case["history"]],
            [s["depths_m"] for s in peer["history"]],
            rtol=1e-13,
            atol=1e-15,
        )
    primary = [r for r in rows if r["group"] == "primary"]
    tiles = [r for r in primary if r["mode"] == "tiles"]
    if len({(r["landform"], r["storm"]) for r in tiles}) != 48:
        raise ValueError("Missing landform or storm")
    for r in tiles:
        actual = {
            (c["method"], c["mode"])
            for c in primary
            if (c["landform"], c["storm"]) == (r["landform"], r["storm"])
        }
        if not set(methods) <= actual:
            raise ValueError("Missing matched solver or scheduler")
    large = json.loads((review / "large_parity.json").read_text())
    if len(large) != 20 or not all(r["matched"] for r in large):
        raise ArithmeticError("Large-event schedule comparison failed")
    profiles = {f.stem for f in (review / "graph_profiles").glob("*.json")}
    if profiles != {r["name"] for r in primary}:
        raise ValueError("Incomplete graph kernel profiles")
    if len(json.loads((review / "storm_kernel_profiles.json").read_text())) != 6:
        raise ValueError("Incomplete uncaptured storm profiles")
    dt = json.loads((review / "accepted_dt.json").read_text())
    hashes = Counter((r["steps"], r["dt_sha256"]) for r in dt)
    if len(hashes) != 6 or any(count != 3 for count in hashes.values()):
        raise ArithmeticError("Dense, fixed and tiled accepted histories differ")
    receipts = {}
    for suite in (
        "primary",
        "large",
        "terrain_large",
        "refinement",
        "moving",
        "directional",
    ):
        receipt = review / "integrated_events" / (suite + ".verified")
        entries = set(receipt.read_text().splitlines())
        expected = sum(
            c["method"] in ("hancock", "fv2")
            for c in experiment_cases(config=config, suite=suite)
        )
        if len(entries) != expected:
            raise ValueError("Incomplete source parity checks for " + suite)
        receipts[suite] = dict(
            count=len(entries), receipt_sha256=checksum(path=receipt)
        )
        for name in entries:
            if not (review / "integrated_events" / (name + ".json")).is_file():
                raise ValueError("Missing integrated run " + name)
    (review / "event_checksums.json").write_text(json.dumps(index, indent=2) + "\n")
    paths = [*review.glob("*.py"), review / "cases.json", *modules]
    device = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    info = dict(
        git_base=baseline,
        checked_out_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        qualifications="Local working-tree implementation on the recorded development baseline",
        python=platform.python_version(),
        platform=platform.platform(),
        device=device,
        dependencies={
            name: metadata.version(name)
            for name in ("numpy", "scipy", "warp-lang", "matplotlib", "pytest")
        },
        source_sha256={str(p.relative_to(Path.cwd())): checksum(path=p) for p in paths},
        baseline_results_sha256=checksum(path=review / "baseline.json"),
        event_group_counts=counts,
        integrated_operator_checks=receipts,
        scheduler_implementation="29140772eb63c52bb5d8d765d0e7531c51a1d7ac",
        integrated_operator_implementation="e63c936",
        graph_kernel_profiles=len(profiles),
        large_dense_parity_pairs=len(large),
        primary_dense_parity_pairs=len(parity),
        pytest=pytest_summary,
        precision="FP64",
        seed=json.loads((review / "cases.json").read_text())["seed"],
    )
    (review / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).resolve().parent
    fields = Path("outputs/hancock_long_storm_review/fields")
    counts = dict(
        primary=226,
        reference=96,
        refinement=40,
        large=32,
        moving=4,
        directional=48,
        terrain_large=8,
        resolution_check=3,
        fine_check=3,
        remaining_refinements=14,
    )
    baseline = "018032761146a915bd6089f858a97500715e3758"
    methods = (
        ("hancock", "tiles"),
        ("hancock", "dense"),
        ("hancock", "sparse"),
        ("fv1", "sparse"),
    )
    modules = [p.resolve() for p in Path("src/floodwarp").glob("*.py")]
    pytest_summary_path = review / "test_results.json"
    pytest_summary = json.loads(pytest_summary_path.read_text())
    verify(
        review=review,
        fields=fields,
        counts=counts,
        baseline=baseline,
        methods=methods,
        modules=modules,
        pytest_summary=pytest_summary,
    )
