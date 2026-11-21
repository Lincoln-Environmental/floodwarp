"""Transient Hancock trial schedule. The accepted-state ledger stays dense."""

import numpy as np
import warp as wp

# One horizontal warp per row. A tile halo covers the two-cell sensor stencil.
TILE_WIDTH = 32
TILE_HEIGHT = 4
TILE_AREA = TILE_WIDTH * TILE_HEIGHT


@wp.struct
class TileGrid:
    tiles: wp.array(dtype=wp.int32)
    count: wp.array(dtype=wp.int32)
    columns: int
    width: int
    height: int


def scheduled_kernel(function):
    """Register both work layouts before Warp preloads or captures a graph."""
    kernel = wp.kernel(function)
    dense = wp.overload(kernel, {"work": wp.array(dtype=wp.int32)})
    dense.tile_kernel = wp.overload(kernel, {"work": TileGrid})
    return dense


@wp.func
def scheduled_index(thread: int, work: TileGrid):
    slot = thread / TILE_AREA
    if slot >= work.count[0]:
        return int(-1)
    tile = work.tiles[slot]
    x = tile % work.columns * TILE_WIDTH + thread % TILE_WIDTH
    y = tile / work.columns * TILE_HEIGHT + thread % TILE_AREA / TILE_WIDTH
    if x >= work.width or y >= work.height:
        return int(-1)
    return y * work.width + x


@wp.func
def scheduled_index(thread: int, work: wp.array(dtype=wp.int32)):  # noqa: F811 (Warp overload)
    if work.shape[0] > 0:
        return work[thread]
    return thread


@wp.kernel
def mark_trial(
    q: wp.array(dtype=wp.vec3d),
    active: wp.array(dtype=wp.int32),
    rain: wp.array(dtype=wp.float64),
    multiplier: wp.array(dtype=wp.float64),
    bx: wp.array(dtype=wp.int32),
    by: wp.array(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float64),
    nx: int,
    columns: int,
    seed: wp.array(dtype=wp.int32),
):
    k = wp.tid()
    if active[k] == 0:
        return
    row = k / nx
    x = k + row
    raining = rain[k] != wp.float64(0.0) or (
        multiplier[k] != wp.float64(0.0)
        and (ctrl[9] != wp.float64(0.0) or ctrl[11] != wp.float64(0.0))
    )
    if (
        not (q[k] == wp.vec3d(wp.float64(0.0)))
        or raining
        or bx[x] == 3
        or bx[x + 1] == 3
        or by[k] == 3
        or by[k + nx] == 3
    ):
        wp.atomic_max(seed, row / TILE_HEIGHT * columns + (k % nx) / TILE_WIDTH, 1)


@wp.kernel
def compact_trial(
    seed: wp.array(dtype=wp.int32),
    previous: wp.array(dtype=wp.int32),
    current: wp.array(dtype=wp.int32),
    work: TileGrid,
    rows: int,
):
    tile = wp.tid()
    x = tile % work.columns
    y = tile / work.columns
    needed = int(0)
    for j in range(wp.max(0, y - 1), wp.min(rows, y + 2)):
        for i in range(wp.max(0, x - 1), wp.min(work.columns, x + 2)):
            needed |= seed[j * work.columns + i]
    current[tile] = needed
    # Re-evaluate released tiles once, including faces, fallback flags and theta.
    if needed != 0 or previous[tile] != 0:
        slot = wp.atomic_add(work.count, 0, 1)
        work.tiles[slot] = tile


@wp.kernel
def commit_tiles(
    current: wp.array(dtype=wp.int32),
    previous: wp.array(dtype=wp.int32),
    ctrl: wp.array(dtype=wp.float64),
):
    if ctrl[2] <= ctrl[17] and ctrl[7] == wp.float64(0.0):
        previous[wp.tid()] = current[wp.tid()]


class HancockTiles:
    def __init__(self, *, nx, ny, device):
        # Include the last east/north face even for exactly tile-aligned domains.
        columns = nx // TILE_WIDTH + 1
        self.rows = ny // TILE_HEIGHT + 1
        self.size = columns * self.rows
        self.threads = self.size * TILE_AREA
        self.seed = wp.zeros(self.size, dtype=wp.int32, device=device)
        self.previous = wp.ones_like(self.seed)
        self.current = wp.zeros_like(self.seed)
        self.tiles = wp.empty_like(self.seed)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.grids = []
        for width, height in ((nx, ny), (nx + 1, ny), (nx, ny + 1)):
            grid = TileGrid()
            grid.width, grid.height, grid.columns = width, height, columns
            grid.tiles, grid.count = self.tiles, self.count
            self.grids.append(grid)

    def enqueue(self, *, solver, distributed_forcing):
        self.count.zero_()
        f = solver._fields
        grid = self.grids[0]
        if distributed_forcing:
            # A validation subclass may add a residual anywhere on the grid.
            self.seed.fill_(1)
        else:
            self.seed.zero_()
            solver._launch(
                kernel=mark_trial,
                count=solver.nx * solver.ny,
                inputs=[
                    solver.q,
                    f["active"],
                    f["rain"],
                    f["multiplier"],
                    f["types_x"],
                    f["types_y"],
                    solver.ctrl,
                    solver.nx,
                    grid.columns,
                    self.seed,
                ],
            )
        solver._launch(
            kernel=compact_trial,
            count=self.size,
            inputs=[self.seed, self.previous, self.current, grid, self.rows],
        )

    def diagnostics(self):
        grid = self.grids[0]
        count = int(self.count.numpy()[0])
        tiles = self.tiles.numpy()[:count]

        widths = np.minimum(TILE_WIDTH, grid.width - tiles % grid.columns * TILE_WIDTH)
        heights = np.minimum(
            TILE_HEIGHT, grid.height - tiles // grid.columns * TILE_HEIGHT
        )
        return dict(
            execution="hancock_tiles",
            storage="dense",
            work_tiles=count,
            total_tiles=self.size,
            evaluated_cells=int((widths * heights).sum()),
            scheduling_bytes=sum(
                a.capacity
                for a in (
                    self.seed,
                    self.current,
                    self.previous,
                    self.tiles,
                    self.count,
                )
            ),
        )
