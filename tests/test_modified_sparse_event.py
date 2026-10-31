"""Full-event reporting preserves cumulative timing and complete state checks."""

import hashlib
import json

import numpy as np
import pytest

from validation.sparse_fv1_modified_assess import checkpoint_differences, outlet_series, read_checkpoint


def write_snapshot(*, path, arrays, checksums):
    metadata = dict(configuration=dict(ny=1, nx=2), time=60., steps=60, sha256=checksums)
    np.savez(file=path, metadata=json.dumps(metadata), **arrays)


def test_outlet_interval_means_use_exact_cumulative_samples():
    record = dict(ledger=[dict(seconds=1., outflow_m3=100., inflow_m3=0.),
        dict(seconds=60., outflow_m3=80., inflow_m3=20.),
        dict(seconds=120., outflow_m3=220., inflow_m3=40.)])
    volume, discharge = outlet_series(record=record, times=np.array([60., 120.]), interval=60.)
    np.testing.assert_array_equal(actual=volume, desired=[60., 180.])
    np.testing.assert_array_equal(actual=discharge, desired=[1., 2.])


def test_outlet_comparison_rejects_missing_exact_sample():
    record = dict(ledger=[dict(seconds=59., outflow_m3=60., inflow_m3=0.)])
    with pytest.raises(ValueError, match='Missing exact'):
        outlet_series(record=record, times=np.array([60.]), interval=60.)


def test_checkpoint_comparison_includes_history_and_checksums(tmp_path):
    arrays = dict(q=np.array([[.01, 0., 0.], [0., 0., 0.]]), soil=np.array([.25, .5]))
    checksums = {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in arrays.items()}
    path = tmp_path / 'reference.npz'
    write_snapshot(path=path, arrays=arrays, checksums=checksums)
    current = {name: values.copy() for name, values in arrays.items()}
    current['soil'][0] += .125
    metadata, differences = checkpoint_differences(current=current, reference_path=path)
    assert metadata['time'] == 60.
    assert differences['q'] == 0.
    assert differences['soil'] == .125
    arrays['soil'][0] += .125
    write_snapshot(path=path, arrays=arrays, checksums=checksums)
    with pytest.raises(ValueError, match='checksum differs: soil'):
        read_checkpoint(path=path)
