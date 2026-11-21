"""Validation-only per-cell and per-face compaction on the historical control."""

import json
from pathlib import Path
from unittest.mock import patch

import warp as wp

from floodwarp import fv2_kernels, hancock_tiles
from floodwarp.fv2_solver import HancockSolver
from validation.fv2_regime_profile import record


@wp.kernel
def cell_seeds(
    q: wp.array(dtype=wp.vec3d),
    active: wp.array(dtype=int),
    rain: wp.array(dtype=wp.float64),
    multiplier: wp.array(dtype=wp.float64),
    bx: wp.array(dtype=int),
    by: wp.array(dtype=int),
    ctrl: wp.array(dtype=wp.float64),
    nx: int,
    seed: wp.array(dtype=int),
):
    k = wp.tid()
    xf = k + k / nx
    supply = rain[k] != wp.float64(0.0) or (
        multiplier[k] != wp.float64(0.0)
        and (ctrl[9] != wp.float64(0.0) or ctrl[11] != wp.float64(0.0))
    )
    seed[k] = int(
        active[k] != 0
        and (
            not (q[k] == wp.vec3d(wp.float64(0.0)))
            or supply
            or bx[xf] == 3
            or bx[xf + 1] == 3
            or by[k] == 3
            or by[k + nx] == 3
        )
    )


@wp.kernel
def expand(
    source: wp.array(dtype=int), dest: wp.array(dtype=int), stride: int, width: int
):
    k = wp.tid()
    position = (k / stride) % width
    value = int(0)
    for offset in range(wp.max(-3, -position), wp.min(4, width - position)):
        value |= source[k + offset * stride]
    dest[k] = value


@wp.kernel
def pack(
    mask: wp.array(dtype=int),
    previous: wp.array(dtype=int),
    nx: int,
    ny: int,
    axis: int,
    work: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    k = wp.tid()
    if axis == 0:
        if mask[k] != 0 or previous[k] != 0:
            work[wp.atomic_add(counts, axis, 1)] = k
        return
    pair = fv2_kernels.indices(k, axis - 1, nx, ny)
    a = wp.max(0, pair[0])
    b = wp.max(0, pair[1])
    if (pair[0] >= 0 and (mask[a] != 0 or previous[a] != 0)) or (
        pair[1] >= 0 and (mask[b] != 0 or previous[b] != 0)
    ):
        work[wp.atomic_add(counts, axis, 1)] = k


class CompactHancock(HancockSolver):
    def configure_execution(self, *, mode, **options):
        super().configure_execution(mode="sparse", **options)
        self._work_indices = [
            wp.empty(n, dtype=int, device=self.device)
            for n in (
                self.nx * self.ny,
                (self.nx + 1) * self.ny,
                self.nx * (self.ny + 1),
            )
        ]
        self.seed, self.horizontal, self.current, self.previous = [
            wp.empty_like(self._work_indices[0]) for _ in range(4)
        ]
        self.previous.fill_(1)
        self.counts = wp.zeros(3, dtype=int, device=self.device)

    def _load_extra_step_modules(self):
        super()._load_extra_step_modules()
        wp.load_module(module=pack.module, device=self.device)

    def _launch(self, kernel, count, inputs):
        if kernel is fv2_kernels.finish:
            super()._launch(
                kernel=hancock_tiles.commit_tiles,
                count=self.nx * self.ny,
                inputs=[self.current, self.previous, self.ctrl],
            )
        super()._launch(kernel=kernel, count=count, inputs=inputs)
        if kernel is not fv2_kernels.predictor:
            return
        n = self.nx * self.ny
        f = self._fields
        self.counts.zero_()
        super()._launch(
            kernel=cell_seeds,
            count=n,
            inputs=[
                self.q,
                f["active"],
                f["rain"],
                f["multiplier"],
                f["types_x"],
                f["types_y"],
                self.ctrl,
                self.nx,
                self.seed,
            ],
        )
        super()._launch(
            kernel=expand, count=n, inputs=[self.seed, self.horizontal, 1, self.nx]
        )
        super()._launch(
            kernel=expand,
            count=n,
            inputs=[self.horizontal, self.current, self.nx, self.ny],
        )
        for axis, work in enumerate(self._work_indices):
            work.fill_(-1)
            super()._launch(
                kernel=pack,
                count=work.size,
                inputs=[
                    self.current,
                    self.previous,
                    self.nx,
                    self.ny,
                    axis,
                    work,
                    self.counts,
                ],
            )

    def execution_diagnostics(self):
        arrays = [
            self.seed,
            self.horizontal,
            self.current,
            self.previous,
            self.counts,
            *self._work_indices,
        ]
        cells,x_faces,y_faces=self.counts.numpy().tolist()
        return dict(scheduling_bytes=sum(a.capacity for a in arrays),evaluated_cells=cells,
                    evaluated_x_faces=x_faces,evaluated_y_faces=y_faces,execution="compact_cells_and_faces")


def compare(*, review, cases, options):
    baseline = json.loads((review / "final_profile.json").read_text())
    rows = []
    with patch("floodwarp.fv2_solver.HancockSolver", CompactHancock):
        for case in cases:
            row = record(
                case=case, method="hancock_adaptive", mode="compact", **options
            )
            dense = next(
                r for r in baseline if r["case"] == case and r["mode"] == "dense"
            )
            if row["field_sha256"] != dense["field_sha256"]:
                raise ValueError("Compact candidate differs from dense Hancock")
            rows.append(row)
    (review / "compact.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    cases = ("dry_patch", "stoker")
    options = dict(
        device="cuda:0",
        dx=1.0,
        length=1024.0,
        width=640.0,
        seconds=2.0,
        cap=1 / 32,
        repeats=7,
    )
    compare(review=review, cases=cases, options=options)
