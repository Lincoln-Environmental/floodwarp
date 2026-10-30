"""Array-based D4 contact classification, retaining historical insertion order."""

import numpy as np


def open_faces(*, cells, sides, shape, face_x, face_y):
    j, i = cells // shape[1], cells % shape[1]
    result = np.ones(len(cells), dtype=bool)
    for side in range(4):
        selected = sides == side
        if side < 2:
            result[selected] = face_x[j[selected], i[selected] + side]
        else:
            result[selected] = face_y[j[selected] + side - 2, i[selected]]
    return result


def native_contact_groups(*, owner, bed, state_cell, shape, open_sides, face_x, face_y, north):
    """Return sorted record buffers and key groups in first-encounter order.

    Discover only owner-changing edges. Native horizontal edges precede native
    vertical edges; outer faces follow in the supplied open_sides order. Masked
    exterior reversal preserves the original pre-reversal spatial line key.
    """
    width = bed.shape[1]
    j, i = np.nonzero((owner[:, :-1] != owner[:, 1:]) & ((owner[:, :-1] >= 0) | (owner[:, 1:] >= 0)))
    east = j * width + i
    j, i = np.nonzero((owner[:-1] != owner[1:]) & ((owner[:-1] >= 0) | (owner[1:] >= 0)))
    above = j * width + i
    a = np.r_[east, above]
    b = np.r_[east + 1, above + width]
    side = np.r_[np.full(len(east), 1, np.int64), np.full(len(above), 3, np.int64)]
    line = np.where(side == 1, a % width, a // width)
    owners, z = owner.ravel(), bed.ravel()
    cells = np.asarray(state_cell, dtype=np.int64)
    sa, sb = owners[a].astype(np.int64), owners[b].astype(np.int64)
    exterior = (sa < 0) | (sb < 0)
    reverse = sa < 0
    active_a = np.where(reverse, b, a)
    sa = np.where(reverse, sb, sa)
    side = np.where(reverse, np.array([1, 0, 3, 2])[side], side)
    sb = np.where(exterior, -1, sb)
    sill = np.where(exterior, z[active_a], np.where(z[a] >= z[b], z[a], z[b]))
    a, b = active_a, np.where(exterior, -1, b)
    valid = ~exterior | ((side == 3) & north.ravel()[a] if north is not None else np.isin(side, open_sides))
    needs_face = ((sb < 0) & (north is None)) | ((sb >= 0) & (cells[sa] != cells[np.maximum(sb, 0)]))
    valid &= ~needs_face | open_faces(cells=cells[sa], sides=side, shape=shape, face_x=face_x, face_y=face_y)
    records = [[a[valid]], [b[valid]], [sa[valid]], [sb[valid]], [side[valid]], [line[valid]], [sill[valid]]]
    for direction in open_sides:
        if direction == 0:
            ids = np.arange(0, bed.size, width)
        elif direction == 1:
            ids = np.arange(width - 1, bed.size, width)
        elif direction == 2:
            ids = np.arange(width)
        else:
            ids = np.arange(bed.size - width, bed.size)
        ids = ids[owners[ids] >= 0]
        if north is not None:
            ids = ids[north.ravel()[ids]] if direction == 3 else ids[:0]
        sides = np.full(len(ids), direction, np.int64)
        ids = ids[open_faces(cells=cells[owners[ids]], sides=sides, shape=shape, face_x=face_x, face_y=face_y)]
        length = len(ids)
        for target, values in zip(records, (ids, np.full(length, -1, np.int64), owners[ids].astype(np.int64),
            np.full(length, -1, np.int64), np.full(length, direction, np.int64), np.full(length, -1, np.int64), z[ids]), strict=True):
            target.append(values)
    a, b, sa, sb, side, line, sill = [np.concatenate(parts) for parts in records]
    if not len(a):
        return np.empty((0, 4), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), a, b, sill
    order = np.lexsort((a, line, side, sb, sa))
    keys = np.column_stack((sa[order], sb[order], side[order], line[order]))
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
    stops = np.r_[starts[1:], len(order)]
    first = np.minimum.reduceat(order, starts)
    groups = np.argsort(first)
    return keys[starts[groups]], starts[groups], stops[groups], a[order], b[order], sill[order]
