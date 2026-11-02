"""Recompute retained event metrics and check current native source identities."""
import gzip
import hashlib
import json
from pathlib import Path


def audit_retained_metrics(*, repository: Path, output: Path) -> dict:
    evidence = repository/'validation/sparse_modified_event_review'
    summary = json.loads((evidence/'summary.json').read_text())
    compressed = (evidence/'complete_record.json.gz').read_bytes()
    raw = gzip.decompress(compressed)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != summary['complete_record_sha256']:
        raise ValueError('Retained complete-record checksum does not match its summary')
    record = json.loads(raw)
    sources = {}
    for name, expected in record['signature']['sources'].items():
        actual = hashlib.sha256((repository/'src/floodwarp'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Current native runtime differs from retained evidence: {name}')
        sources[name] = actual
    stats = [row['stats'] for row in record['rows']]
    steps = sum(row['steps'] for row in stats)
    hydraulic = sum(row['loop_seconds'] for row in stats)
    if steps != record['steps'] or abs(hydraulic-record['hydraulic_seconds']) > 1.e-9:
        raise ValueError('Retained interval counters differ from complete-event totals')
    result = dict(scope='Retained completed 72-hour event; no new performance measurement',
        measured_source_commit='d34acab3c33f3be92bd94d49f8d23d859422da1b',
        record_path='validation/sparse_modified_event_review/complete_record.json.gz',
        record_uncompressed_sha256=digest,
        record_compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        native_sources_sha256=sources, current_native_sources_match=True,
        actual_final_time_s=record['time_s'], accepted_steps=steps,
        hydraulic_seconds=hydraulic, recorded_component_seconds=record['process_seconds'],
        dt_min_s=min(row['dt_min_s'] for row in stats),
        dt_mean_s=record['time_s']/steps,
        dt_max_s=max(row['dt_max_s'] for row in stats))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == "__main__":
    repository = Path.cwd()
    output = repository/'validation/run_metrics_review/retained_metrics.json'
    result = audit_retained_metrics(repository=repository, output=output)
    print(json.dumps(result, indent=2))
