"""Exact-zero execution scheduling with persistent dense uniform FV1 fields."""

import numpy as np
import warp as wp

from floodwarp import sparse_fv1_kernels as kernels


class SparseFV1Execution:
    """Disposable scheduling cache, independent of physical restart state."""

    def __init__(self, *, solver, tile_size, dense_fraction):
        if tile_size not in (8, 16, 32):
            raise ValueError('FV1 tile_size must be 8, 16 or 32')
        if not np.isfinite(dense_fraction) or not 0. < dense_fraction <= 1.:
            raise ValueError('FV1 dense_fraction must be in (0, 1]')
        self.tile_size = tile_size
        self.tiles_x = (solver.nx + tile_size - 1) // tile_size
        self.tiles_y = (solver.ny + tile_size - 1) // tile_size
        self.tile_count = self.tiles_x * self.tiles_y
        self.dense_cutoff = max(1, int(np.ceil(dense_fraction * self.tile_count)))
        self.stride = min(self.tile_count * tile_size * tile_size, 65536)
        self.flags = wp.zeros(self.tile_count, dtype=wp.int32, device=solver.device)
        self.work = wp.empty(self.tile_count, dtype=wp.int32, device=solver.device)
        self.count = wp.empty(1, dtype=wp.int32, device=solver.device)
        self.invalidate()

    def invalidate(self):
        # A fresh/restored dense checkpoint may contain scratch anywhere.
        # The first clear must therefore use dense indexing, without a list.
        self.count.fill_(self.tile_count)

    def enqueue(self, *, solver):
        f = solver._fields
        common = [solver.nx, solver.ny]
        tail = [self.work, self.count, self.tile_size, self.tiles_x, self.dense_cutoff, self.stride]
        wp.launch(kernels.clear_previous, dim=self.stride,
            inputs=[f['flux_x'], f['flux_y'], f['pressure_x'], f['pressure_y'], f['theta'],
                    self.work, self.count, *common, self.tile_size, self.tiles_x,
                    self.dense_cutoff, self.stride], device=solver.device)
        self.flags.zero_()
        self.count.zero_()
        wp.launch(kernels.mark_tiles, dim=solver.nx * solver.ny,
            inputs=[solver.q, f['active'], f['bed'], f['types_x'], f['types_y'],
                    f['stages_x'], f['stages_y'], self.flags, solver.nx,
                    self.tile_size, self.tiles_x], device=solver.device)
        wp.launch(kernels.compact_tiles, dim=self.tile_count,
            inputs=[self.flags, self.work, self.count, self.tiles_x, self.tiles_y], device=solver.device)
        for axis, index in (('x', 0), ('y', 1)):
            wp.launch(kernels.sparse_faces, dim=self.stride,
                inputs=[solver.q, f['bed'], f['active'], f['types_' + axis], f['stages_' + axis],
                        f['flux_' + axis], f['pressure_' + axis], *common, index, solver.gravity,
                        solver.dry_depth, solver._stage_momentum_mode, solver.ctrl, solver._flux_mode,
                        *tail], device=solver.device)
        wp.launch(kernels.sparse_draining, dim=self.stride,
            inputs=[solver.q, f['flux_x'], f['flux_y'], f['theta'], solver.ctrl, solver.nx,
                    solver.dx, solver.dy, solver.ny, *tail], device=solver.device)
        for axis, index, width in (('x', 0, solver.dy), ('y', 1, solver.dx)):
            wp.launch(kernels.sparse_limit_faces, dim=self.stride,
                inputs=[f['flux_' + axis], f['theta'], f['active'], f['volume_' + axis],
                        solver.ctrl, *common, index, width, *tail], device=solver.device)
        wp.launch(kernels.sparse_update, dim=self.stride,
            inputs=[solver.q, f['flux_x'], f['flux_y'], f['pressure_x'], f['pressure_y'],
                    f['active'], f['manning'], f['maximum'], f['inflow'], f['outflow'], solver.ctrl,
                    *common, solver.dx, solver.dy, solver.gravity, solver.dry_depth, *tail], device=solver.device)

    def diagnostics(self, *, solver):
        """Explicit, optional host readback; never called by the step loop."""
        count = int(self.count.numpy()[0])
        fallback = count >= self.dense_cutoff
        cells = solver.nx * solver.ny
        fx = solver.ny * (solver.nx + 1)
        fy = (solver.ny + 1) * solver.nx
        if not fallback:
            tiles = self.work.numpy()[:count]
            rows = (tiles // self.tiles_x) * self.tile_size
            columns = (tiles % self.tiles_x) * self.tile_size
            heights = np.minimum(self.tile_size, solver.ny - rows)
            widths = np.minimum(self.tile_size, solver.nx - columns)
            cells = int(np.sum(heights * widths))
            fx = int(cells + np.sum(heights[columns + widths == solver.nx]))
            fy = int(cells + np.sum(widths[rows + heights == solver.ny]))
        return dict(execution='sparse_uniform', tile_size=self.tile_size,
                    seed_tiles=int(np.count_nonzero(self.flags.numpy())), work_tiles=count,
                    total_tiles=self.tile_count, dense_fallback=fallback,
                    evaluated_cells=cells, evaluated_faces_x=fx, evaluated_faces_y=fy,
                    scheduling_bytes=sum(array.capacity for array in (self.flags, self.work, self.count)),
                    storage='dense', resolution='uniform')
