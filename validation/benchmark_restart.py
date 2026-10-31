"""Durable benchmark progress with immutable, checksummed solver generations."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np


def digest_file(*, path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def payload_digest(*,value):
    payload=dict(value)
    payload.pop('payload_sha256',None)
    return hashlib.sha256(json.dumps(obj=payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def publish_json(*, path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                prefix=path.name+'.', suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target=path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sample_targets(*, duration: float, interval: float) -> np.ndarray:
    """Include the requested final time exactly, with no overshoot or duplicate."""
    if not np.isfinite([duration, interval]).all() or min(duration, interval) <= 0.:
        raise ValueError('Positive finite duration and sample interval required')
    interior = np.arange(1, int(np.floor(duration/interval))+1, dtype=np.float64)*interval
    return np.r_[interior[interior < duration], float(duration)]


def read_progress(*, directory: Path, signature):
    path = directory/'progress.json'
    if not path.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Existing benchmark lacks a validated progress manifest: '+str(directory))
        return None
    try:
        progress = json.loads(path.read_text())
        if progress['schema'] != 1 or progress['signature'] != signature:
            raise ValueError('Benchmark checkpoint configuration/source signature mismatch')
        if progress['checkpoint'] is not None and progress['payload_sha256']!=payload_digest(value=progress):
            raise ValueError('Benchmark progress/sample checksum mismatch')
        if progress['checkpoint'] is None:
            if progress['time_s']!=0. or progress['ledger'] or progress['statistics']:
                raise ValueError('Missing benchmark solver checkpoint')
        else:
            checkpoint = directory/progress['checkpoint']
            if checkpoint.parent != directory or digest_file(path=checkpoint) != progress['checkpoint_sha256']:
                raise ValueError('Benchmark checkpoint checksum/path mismatch')
        time = float(progress['time_s'])
        ledger = progress['ledger']
        statistics = progress['statistics']
        if (not np.isfinite(time) or time < 0. or
                any(not np.isfinite(row['seconds']) for row in ledger) or
                any(b['seconds'] <= a['seconds'] for a,b in zip(ledger,ledger[1:])) or
                (ledger and ledger[-1]['seconds'] != time) or
                len(ledger) != len(statistics)):
            raise ValueError('Benchmark checkpoint time/sample records are inconsistent')
        return progress
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise ValueError('Invalid benchmark progress: '+str(path)) from error


def publish_progress(*, directory: Path, signature, time_s, ledger, statistics,
                     timings, save_state, suffix: str):
    """The manifest swap is the sole commit point; interrupted writes are orphans.

    Keep earlier committed generations so a failed state or manifest publication
    never overwrites the last valid restart. No automatic fallback hides damage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/('checkpoint-'+uuid.uuid4().hex+suffix)
    save_state(path=checkpoint)
    progress = dict(schema=1, signature=signature, time_s=float(time_s),
        checkpoint=checkpoint.name, checkpoint_sha256=digest_file(path=checkpoint),
        ledger=ledger, statistics=statistics, timings=timings)
    progress['payload_sha256']=payload_digest(value=progress)
    publish_json(path=directory/'progress.json', value=progress)
    return progress


def read_completed(*, directory: Path, signature, duration):
    path = directory/'record.json'
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        if record['signature'] != signature or record['time_s'] != duration:
            raise ValueError('Completed benchmark configuration/source/time mismatch: '+str(path))
        if payload_digest(value=record)!=record['payload_sha256']:
            raise ValueError('Completed benchmark record checksum mismatch: '+str(path))
        if digest_file(path=directory/'fields.npz') != record['fields_sha256']:
            raise ValueError('Completed benchmark output checksum mismatch: '+str(path))
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError('Invalid completed benchmark: '+str(path)) from error
    return record


def publish_record(*,directory: Path,record):
    """Bind completed scalar samples, timings and provenance to one checksum."""
    record.pop('payload_sha256',None)
    record['payload_sha256']=payload_digest(value=record)
    publish_json(path=directory/'record.json',value=record)
