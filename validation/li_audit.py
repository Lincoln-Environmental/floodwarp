"""Preservation audit for the versioned LI review; no hydraulic execution."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import warp as wp

from validation.li_assess import write_json


def audit(*, repository, source, baseline, source_baseline, protected, replay_root, controls_root, output, logs):
    names = subprocess.run(args=['git', '-C', str(repository), 'ls-tree', '-r', '--name-only', baseline, '--', *protected],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    preserved = {}
    for name in names:
        expected = subprocess.run(args=['git', '-C', str(repository), 'show', baseline + ':' + name], capture_output=True, check=True).stdout
        actual = (repository / name).read_bytes()
        if actual != expected:
            raise AssertionError('Frozen artifact changed: ' + name)
        preserved[name] = hashlib.sha256(actual).hexdigest()
    source_head = subprocess.run(args=['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    assert source_head == source_baseline
    subprocess.run(args=['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--'], check=True, capture_output=True)
    files = dict(source_manifest=replay_root / 'source_results/manifest.json',
                 target_manifest=replay_root / 'target_results/manifest.json',
                 source_default_manifest=replay_root / 'source_defaults/manifest.json',
                 target_default_manifest=replay_root / 'target_defaults/manifest.json',
                 attribution_manifest=controls_root / 'attribution_v2/manifest.json',
                 public_default_manifest=controls_root / 'shipped_defaults_v2/manifest.json')
    for label, path in files.items():
        shutil.copyfile(src=path, dst=output / (label + '.json'))
    for label, path in logs.items():
        shutil.copyfile(src=path, dst=output / label)
    parity = json.loads((output / 'parity.json').read_text())
    default_parity = json.loads((output / 'default_parity.json').read_text())
    assert parity['all_bitwise'] and default_parity['all_bitwise']
    write_json(path=output / 'provenance.json', value=dict(starting_target_head=baseline, source_head=source_head,
        source_tracked_worktree_unchanged=True, python=platform.python_version(), numpy=np.__version__, warp=wp.__version__,
        environment='Existing darcywarp; no upgrades; CPU FP64 extraction replay; separate retained CPU/CUDA suite',
        original_artifact_sha256=preserved, preserved_file_count=len(preserved),
        evidence_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file() and path.name != 'provenance.json'}))


if __name__ == "__main__":
    repository = Path('.').resolve()
    source = Path('../Overlander').resolve()
    baseline = '77c794a95db7aad8d79fa0574420a13e79273cb8'
    source_baseline = 'ba503b8bde51d3f212199e81efc7038d9fffecc8'
    protected = ('src', 'validation/fixtures', 'validation/ACCEPTANCE.md', 'validation/REPORT.md',
                 'validation/results.json', 'validation/sensitivity.json', 'LICENSE', 'docs/extraction.json', 'docs/numerical_integrity.json')
    replay_root = Path('outputs/li_review_v1/baselines')
    controls_root = Path('outputs/li_review_v1')
    output = Path('validation/li_review_v1')
    logs = {'retained_tests.txt': Path('/tmp/floodwarp_li_review_v1/tests.log'),
            'final_tests.txt': Path('/tmp/floodwarp_li_review_v1/final_tests.log')}
    audit(repository=repository, source=source, baseline=baseline, source_baseline=source_baseline, protected=protected,
          replay_root=replay_root, controls_root=controls_root, output=output, logs=logs)
