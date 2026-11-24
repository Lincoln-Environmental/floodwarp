"""Repeat the temporally unresolved sharp channel pulse at a finer reporting interval."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    hydrograph,
    pulse,
    pulse_recession,
    pulse_reference,
    statistics,
)


def archive_config(*, directory):
    folder = directory / "long_pulses"
    ident = (folder / "current.txt").read_text().strip()
    return json.loads((folder / ident / "inputs.json").read_text())["cfg"]


def reference_resolution(*, directory, settings):
    """Measure support and averaging bias against the independent analytical flux."""
    parent = archive_config(directory=directory)
    profile = settings["profile"]
    folder = directory / "long_pulses"
    archive = folder / (folder / "current.txt").read_text().strip()
    original = sorted(archive.glob(profile + "_*.npz"))[0]
    with np.load(original) as saved:
        old_edges, original_q, stations = (
            saved["edges"],
            saved["truth"],
            saved["stations"],
        )
    dt = settings["report_interval_s"]
    ratio = round(parent["interval"] / dt)
    if ratio < 2 or ratio * dt != parent["interval"]:
        raise ValueError(
            "The refinement must subdivide each original observation interval"
        )
    edges = np.arange(0, parent["duration"] + dt, dt)
    np.testing.assert_array_equal(edges[::ratio], old_edges)
    design = json.loads((directory / "design.json").read_text())
    rows = []
    for j, station in enumerate(stations):
        q = pulse_reference.section_flux(
            edges=edges, station=station, config=parent, profile=profile
        )
        coarse = q.reshape(-1, ratio).mean(axis=1)
        np.testing.assert_allclose(
            coarse,
            original_q[:, j],
            atol=settings["reference_atol"],
            rtol=settings["reference_rtol"],
        )
        for times, values in ((old_edges, coarse), (edges, q)):
            summary = hydrograph.profile(t=times, q=values, cfg=design)
            interval = float(np.diff(times).max())
            rows.append(
                dict(
                    source_file=str(original.relative_to(directory)),
                    source_sha256=runner.checksum(original),
                    profile=profile,
                    distance_m=station - parent["centre"],
                    interval_s=interval,
                    analytical_peak_m3s=summary["peak"],
                    central80_s=summary["width80"],
                    support_intervals=summary["width80"] / interval,
                    passes_width_screen=summary["width80"]
                    >= design["width_bins"] * interval,
                    peak_bias_over_fine_pct=100 * (summary["peak"] / q.max() - 1),
                )
            )
    metrics.csv_write(
        file=directory / "results/sharp_analytical_cadence.csv", rows=rows
    )


def run(*, directory, settings):
    parent = archive_config(directory=directory)
    name = settings["profile"]
    dt = settings["report_interval_s"]
    if not 0 < dt < parent["interval"] or parent["interval"] % dt:
        raise ValueError(
            "Fine reporting intervals must nest within original interval edges"
        )
    cfg = dict(
        parent,
        profiles={name: parent["profiles"][name]},
        interval=dt,
        refinement_source=runner.checksum(Path(__file__)),
        extended_truth_source=runner.checksum(Path(pulse_recession.__file__)),
    )
    with patch.object(pulse, "state", pulse_recession.state):
        pulse.run(cfg=cfg, output=directory / "sharp_pulses", root=directory.parents[2])


def review(*, directory, settings):
    design = json.loads((directory / "design.json").read_text())
    parent = archive_config(directory=directory)
    folder = directory / "sharp_pulses"
    ident = (folder / "current.txt").read_text().strip()
    archive = folder / ident
    config = json.loads((archive / "inputs.json").read_text())["cfg"]
    if config["interval"] != settings["report_interval_s"]:
        raise ValueError("Archive has the wrong sharp-pulse reporting interval")
    expected = {
        f"{settings['profile']}_{dx}_{cap}_{method}.npz"
        for dx in parent["dx"]
        for cap in parent["cap_courant"]
        for method in parent["methods"]
    }
    if expected != {f.name for f in archive.glob("*.npz")}:
        raise ValueError("Incomplete sharp pulse series")
    target = directory / "results"
    pulse_reference.verify(
        directory=directory,
        absolute_tolerance=settings["reference_atol"],
        relative_tolerance=settings["reference_rtol"],
        archive_name="sharp_pulses",
        result_name="sharp_reference_checks.csv",
    )
    rows = assessment.pulse_rows(
        directory=directory, cfg=design, archive_name="sharp_pulses"
    )
    cadence = []
    for f in sorted(archive.glob("*.npz")):
        with np.load(f) as data:
            for j, station in enumerate(data["stations"]):
                for name in ("flow", "truth"):
                    q = data[name][:, j]
                    if q[-1] > design["tail_fraction"] * np.max(q):
                        raise ValueError(
                            f"Sharp pulse still discharging at end of {f.name}"
                        )
                    observed = statistics.cadence_observation(
                        edges=data["edges"],
                        flow=q,
                        seconds=parent["interval"],
                        config=design,
                    )
                    cadence.append(
                        dict(
                            archive=ident,
                            run=f.stem,
                            station_m=station,
                            source=name,
                            **observed,
                        )
                    )
    for row in rows:
        row["cohort"] = "sharp-refined"
        row["experiment_id"] = ident + "/" + row["experiment_id"]
        if not row["width_resolved"]:
            raise ValueError("The reference sharp hydrograph is still undersampled")
    metrics.csv_write(file=target / "sharp_pulses.csv", rows=rows)
    metrics.csv_write(file=target / "sharp_reporting_intervals.csv", rows=cadence)
    return rows


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "sharp_settings.json").read_text())
    stage = sys.argv[1] if len(sys.argv) == 2 else "sharp-pulses"
    if stage == "sharp-reference":
        reference_resolution(directory=directory, settings=settings)
    elif stage == "sharp-pulses":
        reference_resolution(directory=directory, settings=settings)
        run(directory=directory, settings=settings)
        review(directory=directory, settings=settings)
    else:
        raise SystemExit(f"Unknown sharp-pulse stage {stage}")
