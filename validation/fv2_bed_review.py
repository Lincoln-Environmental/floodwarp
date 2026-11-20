"""Replay the archived hydraulic blockage and check face ordering independently."""

import json
from pathlib import Path

import numpy as np

from floodwarp.fv2_reference import traces
from floodwarp.full_swe_reference import hll_flux
from validation.fv1_riemann import report_json


def slopes(*, z, h, axis, dry, ordered):
    def mc(*, field):
        a = field - np.roll(field, 1, axis=axis)
        b = np.roll(field, -1, axis=axis) - field
        return np.where(
            a * b > 0, np.sign(a) * np.minimum.reduce((2 * abs(a), 2 * abs(b), abs(a + b) / 2)), 0.0
        )

    sz = mc(field=z)
    se = mc(field=z + h)
    if ordered:
        bound = np.minimum(abs(z - np.roll(z, 1, axis=axis)), abs(np.roll(z, -1, axis=axis) - z))
        weight = np.divide(bound, abs(sz), out=np.ones_like(z), where=abs(sz) > bound)
        sz = np.clip(sz, -bound, bound)
        se = se * weight
    ok = (
        (h > dry)
        & (np.roll(h, 1, axis=axis) > dry)
        & (np.roll(h, -1, axis=axis) > dry)
        & (abs(se - sz) < 2 * h)
    )
    end = [slice(None)] * 2
    end[axis] = [0, -1]
    ok[tuple(end)] = False
    return np.where(ok, sz, 0), np.where(ok, se, 0)


def audit_face(*, fixture, ordered):
    bed = np.array(fixture["elevation_m"])
    state = np.array(fixture["state"])
    h = state[..., 0]
    dry = fixture["dry"]
    sz, se = slopes(z=bed, h=h, axis=1, dry=dry, ordered=ordered)
    row, col = np.array(bed.shape) // 2
    dl, dr = (
        h[row, col] + (se[row, col] - sz[row, col]) / 2,
        h[row, col + 1] - (se[row, col + 1] - sz[row, col + 1]) / 2,
    )
    zl = bed[row, col] + sz[row, col] / 2
    zr = bed[row, col + 1] - sz[row, col + 1] / 2
    ul = state[row, col, 1:] / h[row, col]
    ur = state[row, col + 1, 1:] / h[row, col + 1]
    # The cell-trace velocities are reconstructed separately from depth.
    v = np.divide(state[..., 1:], h[..., None], out=np.zeros_like(state[..., 1:]), where=h[..., None] > dry)
    va = v[:, 1:-1] - v[:, :-2]
    vb = v[:, 2:] - v[:, 1:-1]
    dv = np.where(
        va * vb > 0, np.sign(va) * np.minimum.reduce((2 * abs(va), 2 * abs(vb), abs(va + vb) / 2)), 0
    )
    ul = ul + (2 * h[row, col] - dl) / (2 * h[row, col]) * dv[row, col - 1]
    ur = ur - (2 * h[row, col + 1] - dr) / (2 * h[row, col + 1]) * dv[row, col]
    exposed_l = max(0.0, dl - max(0.0, zr - zl))
    exposed_r = max(0.0, dr - max(0.0, zl - zr))
    left = np.r_[exposed_l, exposed_l * ul]
    right = np.r_[exposed_r, exposed_r * ur]
    f = hll_flux(
        left=left[None, :],
        right=right[None, :],
        gravity=fixture["gravity"],
        dry_depth=dry,
        flux_scheme="contact_hll",
    )[0]
    if ordered:
        minus, plus, eta, bed_slope = traces(q=state, z=bed, dry=dry, axis=1, limiter="mc")
        np.testing.assert_allclose(
            [plus[row, col, 0], minus[row, col + 1, 0], eta[row, col], bed_slope[row, col]],
            [dl, dr, se[row, col], sz[row, col]],
            rtol=1e-12,
            atol=1e-12,
        )
    return dict(
        centre_elevations_m=[bed[row, col], bed[row, col + 1]],
        face_elevations_m=[zl, zr],
        reconstructed_h_m=[dl, dr],
        exposed_h_m=[exposed_l, exposed_r],
        cell_velocity_m_s=float(np.linalg.norm(v[row, col])),
        mass_flux_m2_s=float(f[0]),
        normal_momentum_flux_m3_s2=float(f[1]),
        transverse_momentum_flux_m3_s2=float(f[2]),
    )


def count_reversals(*, checkpoint, ordered):
    with np.load(checkpoint) as a:
        cfg = json.loads(str(a["metadata"]))["configuration"]
        shape = cfg["ny"], cfg["nx"]
        z = a["bed"].reshape(shape)
        h = a["q"].reshape(*shape, 3)[..., 0]
        active = a["active"].reshape(shape).astype(bool)
    result = {}
    for axis, name in ((0, "north"), (1, "east")):
        sz, _ = slopes(z=z, h=h, axis=axis, dry=cfg["dry_depth"], ordered=ordered)
        lower = np.arange(z.shape[axis] - 1)
        higher = lower + 1
        dz = np.diff(z, axis=axis)
        diff = dz - (np.take(sz, lower, axis=axis) + np.take(sz, higher, axis=axis)) / 2
        valid = np.take(active, lower, axis=axis) & np.take(active, higher, axis=axis)
        reversed_face = valid & (dz * diff < 0)
        result[name] = dict(
            active_interior_faces=int(valid.sum()),
            reversed=int(reversed_face.sum()),
            max_adverse_jump_m=float(np.max(np.where(reversed_face, abs(diff), 0.0))),
        )
    return result


def verify(*, fixture_file, output):
    fixture = json.loads(fixture_file.read_text())
    audit = {
        key: audit_face(fixture=fixture, ordered=mode)
        for key, mode in (("superseded", False), ("corrected", True))
    }
    assert audit["superseded"]["exposed_h_m"][0] == 0 and audit["superseded"]["mass_flux_m2_s"] == 0
    assert audit["corrected"]["exposed_h_m"][0] > 0 and audit["corrected"]["mass_flux_m2_s"] > 0
    result = dict(stencil=str(fixture_file), instantaneous_flux_on_identical_archived_state=audit)
    path = Path(fixture["original_checkpoint"])
    if path.exists():
        result["whole_snapshot_face_order"] = {
            key: count_reversals(checkpoint=path, ordered=mode)
            for key, mode in (("superseded", False), ("corrected", True))
        }
        for r in result["whole_snapshot_face_order"]["corrected"].values():
            assert r["reversed"] == 0
    report_json(path=output, content=result)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    verify(fixture_file=directory / "orton_false_step_stencil.json", output=directory / "bed_face_audit.json")
