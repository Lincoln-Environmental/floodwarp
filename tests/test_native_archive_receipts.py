import gzip
import json

import pytest

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity import native_archive


def test_completed_attempt_authenticates_fields_and_result(tmp_path):
    result = tmp_path / "result.json"
    spatial = tmp_path / "spatial.npz"
    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    result.write_text('{"steps":2}')
    spatial.write_bytes(b"fields")
    receipt = dict(result_sha256=runner.checksum(path=result), spatial_sha256=runner.checksum(path=spatial))
    (attempt / "complete.json").write_text(json.dumps(receipt))
    assert native_archive.prefix_evidence(folder=tmp_path) == dict(receipt, attempt=attempt.name)
    spatial.write_bytes(b"altered")
    with pytest.raises(ValueError, match="authenticates"):
        native_archive.prefix_evidence(folder=tmp_path)
    spatial.write_bytes(b"fields")
    result.write_text('{"steps":3}')
    with pytest.raises(ValueError, match="authenticates"):
        native_archive.prefix_evidence(folder=tmp_path)


def test_frozen_horizon_detects_changed_gauge_with_same_final_storage(tmp_path):
    ident, parent, end = "series", "prefix", 900
    folder = tmp_path / ident
    horizons = folder / "horizons"
    horizons.mkdir(parents=True)
    saved = dict(id=ident, parent_id=parent, end_s=end, storage_m3=20, frames=[dict(section_m3s=2.0)])
    path = horizons / "900.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(saved, stream)
    cursor = folder / "series.json"
    cursor.write_text(json.dumps(saved))
    expected = dict(horizon_sha256=runner.checksum(path=path))
    assert native_archive.horizon_evidence(root=tmp_path, ident=ident, parent=parent, end_s=end) == expected
    cursor.write_text(json.dumps(dict(saved, frames=[dict(section_m3s=1.5)])))
    with pytest.raises(ValueError, match="frozen measurements"):
        native_archive.horizon_evidence(root=tmp_path, ident=ident, parent=parent, end_s=end)
    cursor.write_text(json.dumps(dict(saved, end_s=1800)))
    assert native_archive.horizon_evidence(root=tmp_path, ident=ident, parent=parent, end_s=end) == expected
    with pytest.raises(ValueError, match="wrong identity"):
        native_archive.horizon_evidence(root=tmp_path, ident=ident, parent="other", end_s=end)
