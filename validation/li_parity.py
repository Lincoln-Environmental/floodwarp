"""Export specified commits without touching either repository, then replay."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def export_and_replay(*, repository, commit, destination, module_root, namespace, fixtures, harness, output, default_output):
    destination.mkdir(parents=True, exist_ok=False)
    archive = subprocess.run(args=['git', '-C', str(repository), 'archive', commit], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(path=destination, filter='data')
    shutil.copyfile(src=harness, dst=destination / module_root / 'li_replay.py')
    environment = os.environ.copy()
    environment.update(REPLAY_NAMESPACE=namespace, REPLAY_FIXTURES=str(fixtures.resolve()), REPLAY_OUTPUT=str(output.resolve()),
                       DARCY_PRECISION='fp64', FLOODWARP_PRECISION='fp64')
    # Package discovery follows each export's native source layout. No installed
    # FloodWarp or source-checkout import is used by either subprocess.
    environment.pop('PYTHONPATH', None)
    environment['REPLAY_DEFAULTS'] = '0'
    with (destination / 'replay.log').open('w') as log:
        subprocess.run(args=[sys.executable, '-m', 'li_replay'], cwd=destination / module_root,
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    environment.update(REPLAY_DEFAULTS='1', REPLAY_OUTPUT=str(default_output.resolve()))
    with (destination / 'default_replay.log').open('w') as log:
        subprocess.run(args=[sys.executable, '-m', 'li_replay'], cwd=destination / module_root,
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    source_repository = Path('../Overlander').resolve()
    target_repository = Path('.').resolve()
    source_commit = 'ba503b8bde51d3f212199e81efc7038d9fffecc8'
    target_commit = '77c794a95db7aad8d79fa0574420a13e79273cb8'
    root = Path('outputs/li_review_v1/baselines').resolve()
    fixtures = Path('validation/fixtures').resolve()
    harness = Path('validation/li_replay.py').resolve()
    root.mkdir(parents=True, exist_ok=False)
    starting_head = subprocess.run(args=['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    (root / 'starting_head.json').write_text(json.dumps(dict(actual_starting_head=starting_head,
        source_baseline=source_commit, target_baseline=target_commit), indent=2) + '\n')
    export_and_replay(repository=source_repository, commit=source_commit, destination=root / 'source',
        module_root=Path('.'), namespace='overland_flow', fixtures=fixtures, harness=harness, output=root / 'source_results', default_output=root / 'source_defaults')
    export_and_replay(repository=target_repository, commit=target_commit, destination=root / 'target',
        module_root=Path('src'), namespace='floodwarp', fixtures=fixtures, harness=harness, output=root / 'target_results', default_output=root / 'target_defaults')
