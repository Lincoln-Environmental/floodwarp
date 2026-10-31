"""Probe larger static allocations separately from timing and profiling."""

from pathlib import Path
import json

from validation.static_fv1_memory import measure


def measure_saved_controls(*, report_path, names, device, block_sizes, duration, output):
    report = json.loads(report_path.read_text())
    measure(names=names, shape=tuple(report['shape']), device=device,
        options=report['options'], mesh_options=report['mesh_options'],
        block_sizes=block_sizes, duration=duration, output=output)


if __name__ == "__main__":
    report_path = Path('validation/static_reconstruction_review/large_controls.json')
    names = ('lake', 'front', 'ponding', 'channel')
    device = 'cuda:0'
    block_sizes = (4, 8)
    duration = .2
    output = Path('validation/static_reconstruction_review/memory.json')
    measure_saved_controls(report_path=report_path, names=names, device=device,
        block_sizes=block_sizes, duration=duration, output=output)
