"""Rebuild the exact pulse comparison independently of the storm experiments."""

import json
from pathlib import Path
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment, figures


def distance_effects(*, rows, target):
    """Compare the same simulated pulse at successive downstream sections."""
    runs = {}
    for row in rows:
        runs.setdefault(row["experiment_id"], []).append(row)
    changes = []
    for ident, sections in runs.items():
        sections.sort(key=propagation_distance)
        for first, last in zip(sections[:-1], sections[1:], strict=True):
            record = dict(
                experiment_id=ident,
                case=last["case"],
                method=last["method"],
                dx=last["dx"],
                cap_courant=last["cap_courant"],
                upstream_m=first["distance_m"],
                downstream_m=last["distance_m"],
            )
            for metric in (
                "volume_bias_pct",
                "peak_bias_pct",
                "width80_bias_pct",
                "centroid_delay_pct",
                "shape_tv_pct",
            ):
                ok = bool(first["qualified_" + metric] and last["qualified_" + metric])
                record["qualified_" + metric] = ok
                record[metric + "_change_points"] = (
                    last[metric] - first[metric] if ok else None
                )
            changes.append(record)
    metrics.csv_write(file=target / "pulse_distance.csv", rows=changes)


def propagation_distance(row):
    return row["distance_m"]


def publish(*, directory, spacings):
    design = json.loads((directory / "design.json").read_text())
    rows = assessment.pulse_rows(directory=directory, cfg=design)
    metrics.csv_write(file=directory / "results/pulses.csv", rows=rows)
    distance_effects(rows=rows, target=directory / "results")
    for dx in spacings:
        figures.channel(directory=directory, dx=dx)
    print(len(rows), "measured section comparisons")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    spacings = (1, 5, 10)
    publish(directory=directory, spacings=spacings)
