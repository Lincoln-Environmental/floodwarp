"""Compiled CPU preprocessing; strict arithmetic, no hydraulic approximation."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def wet_labels(bed, stage):
    ny, nx = bed.shape
    z = bed.ravel()
    wet = z < stage
    parent = np.arange(z.size)
    # Preserve the original union order, root selection and root-number labels.
    for direction in range(2):
        for j in range(ny if direction == 0 else ny - 1):
            for i in range(nx - 1 if direction == 0 else nx):
                a = j * nx + i
                b = a + (1 if direction == 0 else nx)
                if wet[a] and wet[b]:
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    while parent[b] != b:
                        parent[b] = parent[parent[b]]
                        b = parent[b]
                    parent[b] = a
    for a in range(z.size):
        if wet[a]:
            b = a
            while parent[b] != b:
                b = parent[b]
            parent[a] = b
    roots = np.unique(parent[wet])
    result = np.full(z.size, -1, np.int32)
    for a in range(z.size):
        if wet[a]:
            result[a] = np.searchsorted(roots, parent[a])
    return result.reshape(bed.shape)


@njit(cache=True, nogil=True)
def valley_labels(bed):
    ny, nx = bed.shape
    z = bed.ravel()
    order = np.argsort(z)
    owners = np.full(z.size, -1, np.int32)
    visited = np.zeros(z.size, np.bool_)
    queue = np.empty(z.size, np.int64)
    next_owner = 0
    start = 0
    while start < z.size:
        stop = start + 1
        while stop < z.size and z[order[stop]] == z[order[start]]:
            stop += 1
        # scipy.ndimage.label enumerates equal-height plateaus in raster order.
        seeds = np.sort(order[start:stop])
        for seed in seeds:
            if visited[seed]:
                continue
            queue[0] = seed
            visited[seed] = True
            head, count = 0, 1
            owner = z.size
            while head < count:
                node = queue[head]
                head += 1
                j, i = node // nx, node % nx
                for direction in range(4):
                    other = node
                    valid = False
                    if direction == 0 and i > 0:
                        other, valid = node - 1, True
                    elif direction == 1 and i + 1 < nx:
                        other, valid = node + 1, True
                    elif direction == 2 and j > 0:
                        other, valid = node - nx, True
                    elif direction == 3 and j + 1 < ny:
                        other, valid = node + nx, True
                    if valid:
                        if z[other] == z[seed] and not visited[other]:
                            visited[other] = True
                            queue[count] = other
                            count += 1
                        elif owners[other] >= 0:
                            owner = min(owner, owners[other])
            if owner == z.size:
                owner = next_owner
                next_owner += 1
            for index in range(count):
                owners[queue[index]] = owner
        start = stop
    return owners.reshape(bed.shape)


@njit(cache=True, nogil=True)
def chord_error(table, start, end):
    worst = 0.
    for row in range(start, end + 1):
        for endpoint in range(2):
            fraction = (table[row, endpoint] - table[start, 0]) / (table[end, 1] - table[start, 0])
            for column in range(2, table.shape[1], 2):
                actual = table[row, column + endpoint]
                predicted = table[start, column] + fraction * (table[end, column + 1] - table[start, column])
                worst = max(worst, abs(predicted - actual) / actual)
    return worst


@njit(cache=True, nogil=True)
def coefficient_chords(table, tolerance, critical, merge_jumps):
    rows = np.empty_like(table)
    start, count = 0, 0
    maximum_error = 0.
    while start < len(table):
        end = start
        for candidate in range(start + 1, len(table)):
            if table[candidate - 1, 1] != table[candidate, 0]:
                break
            mandatory = False
            for level in critical:
                if table[candidate, 0] == level:
                    mandatory = True
            if mandatory:
                break
            if not merge_jumps:
                continuous = True
                for column in range(2, table.shape[1], 2):
                    if abs(table[candidate - 1, column + 1] - table[candidate, column]) > 1.e-12 * abs(table[candidate, column]):
                        continuous = False
                if not continuous:
                    break
            if chord_error(table, start, candidate) > tolerance:
                break
            end = candidate
        rows[count, 0], rows[count, 1] = table[start, 0], table[end, 1]
        for column in range(2, table.shape[1], 2):
            rows[count, column], rows[count, column + 1] = table[start, column], table[end, column + 1]
        maximum_error = max(maximum_error, chord_error(table, start, end))
        count += 1
        start = end + 1
    return rows[:count].copy(), maximum_error


@njit(cache=True, nogil=True)
def scalar_nodes(table):
    levels, values = np.empty(3 * len(table)), np.empty(3 * len(table))
    count = 0
    for row in table:
        low, high, left, right = row
        if count and low > levels[count - 1]:
            levels[count], values[count] = low, values[count - 1]
            count += 1
        if not count or low != levels[count - 1] or left != values[count - 1]:
            levels[count], values[count] = low, left
            count += 1
        levels[count], values[count] = high, right
        count += 1
    return levels[:count].copy(), values[:count].copy()


@njit(cache=True, nogil=True)
def encode_scalar(original, column, tolerance, critical, compress):
    table = np.empty((len(original), 4))
    table[:, :2] = original[:, :2]
    table[:, 2:] = original[:, column:column + 2]
    error = 0.
    if compress:
        table, error = coefficient_chords(table, tolerance, critical, True)
    maximum_rounding = 0.
    for row in range(len(table)):
        for j in range(2, 4):
            value = table[row, j]
            packed = np.float32(value)
            if float(packed) < value:
                packed = np.nextafter(packed, np.float32(np.inf))
            if not np.isfinite(packed) or packed <= 0:
                raise ValueError('coefficient outside positive finite FP32 range')
            relative = (float(packed) - value) / value
            maximum_rounding = max(maximum_rounding, relative)
            if relative < 0. or maximum_rounding > np.finfo(np.float32).eps:
                raise ValueError('coefficient rounding exceeds the declared FP32 error bound')
            table[row, j] = float(packed)
    levels, values = scalar_nodes(table)
    old_nodes = len(levels)
    if len(values) and np.all(values == values[0]):
        levels, values = np.array([0.]), values[:1]
    return levels, values.astype(np.float32), error, maximum_rounding, old_nodes


@njit(cache=True, nogil=True)
def encode_port(original, critical, tolerance, compress):
    dl, dv, de, dr, dn = encode_scalar(original, 2, tolerance, critical, compress)
    rl, rv, re, rr, rn = encode_scalar(original, 4, tolerance, critical, compress)
    return ((dl, dv), (rl, rv)), max(de, re), max(dr, rr), dn + rn
