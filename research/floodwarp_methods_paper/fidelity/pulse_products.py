"""Rebuild the exact pulse comparison independently of the storm experiments."""

import json
from pathlib import Path
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment, figures


def publish(*, directory, spacings):
    design = json.loads((directory / "design.json").read_text())
    rows = assessment.pulse_rows(directory=directory, cfg=design)
    metrics.csv_write(file=directory / "results/pulses.csv", rows=rows)
    for dx in spacings:
        figures.channel(directory=directory, dx=dx)
    print(len(rows), "measured section comparisons")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    spacings = (1, 5, 10)
    publish(directory=directory, spacings=spacings)
