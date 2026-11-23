"""Render publication figures using retained measurements without launching any solver."""

from pathlib import Path
from research.floodwarp_methods_paper.fidelity import assessment, figures
from research.floodwarp_methods_paper.plan import read_config


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    config = read_config(path=directory.parent / "configs/study.json")
    groups = assessment.archive(directory=directory, config=config)
    from research.floodwarp_methods_paper.fidelity.extended_analysis import attach

    from research.floodwarp_methods_paper.fidelity import policy_recession

    completed = attach(directory=directory, groups=groups)
    policy_recession.attach(directory=directory, groups=groups, completed=completed)
    groups.update(completed)
    figures.plots(directory=directory, groups=groups)
