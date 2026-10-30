"""Bind v2 engineering evidence to preserved strict, runtime and v1 artifacts."""
import hashlib
from pathlib import Path
import shutil
import subprocess

from validation.li_assess import write_json


def audit(*, baseline, protected, output, test_log, implementation):
    names = subprocess.run(args=['git', 'ls-tree', '-r', '--name-only', baseline, '--', *protected],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    preserved = {}
    for name in names:
        expected = subprocess.run(args=['git', 'show', baseline + ':' + name], capture_output=True, check=True).stdout
        actual = Path(name).read_bytes()
        if actual != expected:
            raise AssertionError('Protected artifact changed: ' + name)
        preserved[name] = hashlib.sha256(actual).hexdigest()
    log = test_log.read_text()
    if '1337 passed, 24 skipped' not in log:
        raise AssertionError('Expected completed retained regression log')
    shutil.copyfile(src=test_log, dst=output / 'tests.txt')
    evidence = [path for path in sorted(output.iterdir()) if path.is_file() and path.name != 'provenance.json']
    write_json(path=output / 'provenance.json', value=dict(starting_head=baseline,
        protected_file_count=len(preserved), protected_sha256=preserved,
        evidence_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence},
        implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation},
        execution='Existing pinned darcywarp; checksum-verified retained replay arrays; no hydraulic reruns or solver modifications',
        test_result='1337 passed, 24 skipped; existing opt-in FV1 CUDA controls enabled'))


if __name__ == "__main__":
    baseline = '1dd02e4985441cb4ce76a8630c3e96d1447a7829'
    protected = ('src', 'tests', 'validation/fixtures', 'validation/li_review_v1',
                 'validation/ACCEPTANCE.md', 'validation/REPORT.md', 'validation/results.json',
                 'validation/sensitivity.json', 'LICENSE', 'docs/extraction.json', 'docs/numerical_integrity.json')
    output = Path('validation/li_review_v2')
    test_log = Path('/tmp/floodwarp_li_v2_tests.log')
    implementation = (Path('validation/li_engineering_v2.py'), Path('validation/li_engineering_audit.py'), Path('tests/test_li_engineering_v2.py'))
    audit(baseline=baseline, protected=protected, output=output, test_log=test_log, implementation=implementation)
