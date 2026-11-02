"""Build distributions from a clean committed source export with revision stamp."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def git_output(*, root, arguments):
    return subprocess.run(args=['git', '-C', str(root), *arguments], capture_output=True,
        text=True, check=True).stdout.strip()


def build_candidate(*, root, output):
    if git_output(root=root, arguments=['status', '--porcelain', '--untracked-files=all']):
        raise ValueError('Build from a clean candidate worktree, including tracked release documentation')
    commit = git_output(root=root, arguments=['rev-parse', 'HEAD'])
    if output.exists():
        raise ValueError('Choose a fresh artifact directory; do not overwrite previous release artifacts')
    output.mkdir(parents=True)
    archive = output/'candidate-source.tar'
    subprocess.run(args=['git', '-C', str(root), 'archive', '--format=tar', '-o', str(archive), commit], check=True)
    source = output/'source'
    source.mkdir()
    with tarfile.open(name=archive) as bundle:
        bundle.extractall(path=source, filter='data')
    stamp = dict(commit=commit, source='git archive of committed candidate', tracked_changes=False)
    (source/'src/floodwarp/_build_info.json').write_text(json.dumps(stamp, indent=2)+'\n')
    subprocess.run(args=[sys.executable, '-m', 'build', '--outdir', str(output/'dist'), str(source)], check=True)
    artifacts = []
    for path in sorted((output/'dist').iterdir()):
        artifacts.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
    record = dict(commit=commit, artifacts=artifacts, stamp=stamp)
    (output/'build.json').write_text(json.dumps(record, indent=2)+'\n')
    (output/'SHA256SUMS').write_text(''.join(item['sha256']+'  '+Path(item['path']).name+'\n' for item in artifacts))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    root = Path.cwd()
    output = root/'outputs/release/v0.1.3'
    build_candidate(root=root, output=output)
