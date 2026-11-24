"""Read back the actual raster files used by the practical storm driver."""

import json
from pathlib import Path

import numpy as np
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.terrain.generators import native_inputs


def read_grids(*, folder, terrain):
    rows = []
    for name in terrain["native_sha256"]:
        five = np.load(folder / f"{name}_5m.npy", mmap_mode="r")
        ten = np.load(folder / f"{name}_10m.npy", mmap_mode="r")
        if five.shape != tuple(2 * n for n in ten.shape):
            raise ValueError("5 m children and 10 m footprints are misaligned")
        cells = five.reshape(ten.shape[0], 2, ten.shape[1], 2)
        np.testing.assert_array_equal(ten, cells.mean(axis=(1, 3)))
        if 1 in terrain["inputs"]["grids_m"]:
            native = np.load(folder / f"{name}_1m.npy", mmap_mode="r")
            cells_1m = native.reshape(five.shape[0], 5, five.shape[1], 5)
            np.testing.assert_array_equal(five, cells_1m.mean(axis=(1, 3)))
        rows.append(
            dict(
                field=name,
                native_m=terrain["inputs"]["native_m"],
                native_sha256=terrain["native_sha256"][name],
                parent_cells=ten.size,
                parents_with_different_children=int(
                    np.count_nonzero(np.ptp(cells, axis=(1, 3)))
                ),
                max_difference_direct_1m_to_10m=terrain["max_direct_10m_difference"][
                    name
                ],
                area_integral_5m=float(five.sum() * 25),
                area_integral_10m=float(ten.sum() * 100),
                five_sha256=terrain["files"][f"{name}_5m.npy"],
                ten_sha256=terrain["files"][f"{name}_10m.npy"],
            )
        )
    return rows


def audit(*, directory):
    terrain = json.loads((directory / "mean_storms/terrain.json").read_text())
    ident = runner.digest(terrain["inputs"])
    folder = directory.parents[2] / ".fidelity-restarts/native_inputs" / ident
    native_inputs.verify(folder=folder, identity=terrain["inputs"])
    rows = read_grids(folder=folder, terrain=terrain)
    metrics.csv_write(file=directory / "mean_storms/grid_audit.csv", rows=rows)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    audit(directory=directory)
