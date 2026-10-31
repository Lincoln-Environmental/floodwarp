"""Profile larger static controls separately from their throughput timings."""

from pathlib import Path
import json

from validation.static_fv1_profile import profile


def profile_saved_controls(*, report_path, names, device, block_size, duration, output):
    report = json.loads(report_path.read_text())
    profile(names=names, shape=tuple(report['shape']), device=device,
        options=report['options'], mesh_options=report['mesh_options'],
        block_size=block_size, duration=duration, output=output)


if __name__ == "__main__":
    report_path = Path('validation/static_reconstruction_review/large_controls.json')
    names = ('lake', 'front', 'channel')
    device = 'cuda:0'
    block_size = 4
    duration = .4
    output = Path('validation/static_reconstruction_review/profile.json')
    profile_saved_controls(report_path=report_path, names=names, device=device,
        block_size=block_size, duration=duration, output=output)
