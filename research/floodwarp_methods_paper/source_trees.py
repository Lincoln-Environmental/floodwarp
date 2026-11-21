"""Verify measured numerical source fingerprints directly against a committed Git tree."""

from io import BytesIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import tarfile

from research.floodwarp_methods_paper.runner import source_identity, write_json


def verify_tree(*, root, revision, stages):
    sha = subprocess.check_output(
        ["git", "rev-parse", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    study = Path("research/floodwarp_methods_paper")
    directories = (
        Path("src/floodwarp"),
        Path("validation"),
        Path("validation/hancock_long_storm_review"),
        study,
        study / "terrain/generators",
    )
    special = {
        study / "PROTOCOL.md",
        Path("validation/fv1_diffusion_review/domains.json"),
        Path("validation/hancock_long_storm_review/cases.json"),
    }
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", sha], cwd=root, text=True
    ).splitlines()
    paths = [
        path
        for path in map(Path, tree)
        if (path.parent in directories and path.suffix == ".py") or path in special
    ]
    data = subprocess.check_output(
        ["git", "archive", "--format=tar", sha, "--", *map(str, paths)], cwd=root
    )
    with TemporaryDirectory() as work:
        directory = Path(work)
        with tarfile.open(fileobj=BytesIO(data), mode="r:") as archive:
            archive.extractall(directory, filter="data")
        fingerprints = {
            stage: source_identity(root=directory, study=directory / study, stage=stage)
            for stage in stages
        }
    return dict(commit=sha, source_sha256=fingerprints)


if __name__ == "__main__":
    study = Path(__file__).resolve().parent
    root = study.parents[1]
    revision = "894a375762c9cbcbf24a4b612532ca2628ea7a01"
    stages = (
        "analytical",
        "convergence",
        "synthetic",
        "long-storms",
        "resolution",
        "calibration",
        "performance",
    )
    output = study / "manifests/numerical_source_tree.json"
    verified = verify_tree(root=root, revision=revision, stages=stages)
    write_json(path=output, data=verified)
