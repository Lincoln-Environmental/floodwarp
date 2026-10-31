"""Read-only occupancy audit of saved FV1 states; never a production mask."""

from pathlib import Path
import json

import numpy as np


def tile_mask(*, cells, tile_size):
    ny, nx = cells.shape
    padded = np.pad(array=cells, pad_width=((0, (-ny) % tile_size), (0, (-nx) % tile_size)))
    return padded.reshape(padded.shape[0] // tile_size, tile_size,
                          padded.shape[1] // tile_size, tile_size).any(axis=(1, 3))


def dependency_tiles(*, seeds):
    """Conservative full neighbouring-tile closure, including diagonal receivers."""
    padded = np.pad(array=seeds, pad_width=1)
    result = np.zeros_like(seeds)
    for row in range(3):
        for column in range(3):
            result |= padded[row:row + seeds.shape[0], column:column + seeds.shape[1]]
    return result


def occupancy(*, q, active, tile_size, boundary_seeds=None):
    seeds = active & np.any(q != 0., axis=2)
    if boundary_seeds is not None:
        seeds |= boundary_seeds
    tiles = tile_mask(cells=seeds, tile_size=tile_size)
    closure = dependency_tiles(seeds=tiles)
    cells = closure.repeat(tile_size, axis=0).repeat(tile_size, axis=1)[:active.shape[0], :active.shape[1]]
    # Each west/south face is owned by the cell immediately east/north;
    # the last column/row also owns the exterior east/north face.
    faces_x = int(cells.sum() + cells[:, -1].sum())
    faces_y = int(cells.sum() + cells[-1, :].sum())
    return dict(seed_tiles=int(tiles.sum()), work_tiles=int(closure.sum()),
                total_tiles=int(tiles.size), evaluated_cells=int(cells.sum()),
                evaluated_faces_x=faces_x, evaluated_faces_y=faces_y)


def checkpoint_time(*, path):
    with np.load(file=path, allow_pickle=False) as saved:
        return float(json.loads(str(saved['metadata']))['time'])


def audit(*, directory, progress_path, tile_sizes, output):
    paths = sorted(directory.glob('checkpoint-*.npz'))
    progress = json.loads(progress_path.read_text())
    timings = progress['statistics']
    rows = []
    for path in paths:
        with np.load(file=path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            config = meta['configuration']
            shape = (config['ny'], config['nx'])
            q = saved['q'].reshape(*shape, 3)
            active = saved['active'].reshape(shape).astype(bool)
            dry = config['dry_depth']
            rows.append(dict(time_s=meta['time'], steps=meta['steps'], checkpoint=str(path),
                physical_cells=int(active.sum()), permanently_inactive_cells=int((~active).sum()),
                exactly_dry_cells=int(np.count_nonzero(active & (q[..., 0] == 0.))),
                shallow_nonzero_cells=int(np.count_nonzero(active & (q[..., 0] > 0.) & (q[..., 0] <= dry))),
                momentum_cells=int(np.count_nonzero(active & np.any(q[..., 1:] != 0., axis=2))),
                tiles={str(size): occupancy(q=q, active=active, tile_size=size) for size in tile_sizes}))
    rows.sort(key=state_time)
    previous = 0.
    for row in rows:
        interval = [timing for timing in timings if previous < timing['time_s'] <= row['time_s']]
        row['preceding_hour_seconds'] = sum(timing['loop_seconds'] for timing in interval)
        row['preceding_hour_steps'] = sum(timing['steps'] for timing in interval)
        previous = row['time_s']
    total_cost = sum(row['preceding_hour_seconds'] for row in rows)
    weighted = {str(size): sum(row['preceding_hour_seconds'] *
        row['tiles'][str(size)]['work_tiles'] / row['tiles'][str(size)]['total_tiles']
        for row in rows) / total_cost for size in tile_sizes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(scope='Hourly endpoint diagnostics, weighted by measured preceding-hour cost; '
        'not a per-step occupancy integral or a predictive production mask. Boundary activation is measured in replay.',
        cost_weighted_work_tile_fraction=weighted, states=rows), indent=2) + '\n')


def state_time(row):
    return row['time_s']


if __name__ == "__main__":
    directory = Path('outputs/qualification_optimized/purau_event_72h/fv1_0')
    progress_path = directory / 'progress.json'
    tile_sizes = (8, 16, 32)
    output = Path('validation/sparse_review/occupancy.json')
    audit(directory=directory, progress_path=progress_path, tile_sizes=tile_sizes, output=output)
