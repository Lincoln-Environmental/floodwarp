"""Locate native storm depth discrepancies and verify the actual downstream faces in saved states."""

import json
from pathlib import Path

import numpy as np
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import native_products
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def run(*, directory, strip_m):
    entries = [
        it
        for it in native_products.prefixes(directory=directory).values()
        if it["path"].parent == directory / "mean_storms"
    ]
    wanted = {it["d"]["hashes"]["state"]: it for it in entries}
    domain = []
    bands = []
    folder = directory / "native_outlet"
    folder.mkdir(exist_ok=True)
    for p in (directory.parents[2] / ".fidelity-restarts/prefixes").glob("*/state.npz"):
        with np.load(p) as state:
            manifest = json.loads(state["_direct_manifest"].item())
            state_sha = manifest["sha256"].get("q")
            if state_sha not in wanted:
                continue
            it = wanted.pop(state_sha)
            spec, d = it["s"], it["d"]
            nx, ny = (side // spec["dx"] for side in spec["dimensions"])
            dx = spec["dx"]
            width = strip_m // dx
            meta = json.loads(state["metadata"].item())
            if (
                meta["time"] != spec["hours"] * 3600
                or meta["steps"] != d["steps"]
                or strip_m % dx
            ):
                raise ValueError("Wrong snapshot time or grid alignment")
            q = state["q"].reshape(ny, nx, 3)
            faces = state["volume_x"].reshape(ny, nx + 1)
            if array_sha(q) != state_sha or array_sha(faces) != d["hashes"]["volume_x"]:
                raise ValueError("Checkpoint differs from accepted storm arrays")
            types = state["types_x"].reshape(ny, nx + 1)
            if not np.all(types[:, -1] == 2):
                raise ValueError("Physical outlet contains a non-FREE face")
            np.testing.assert_allclose(
                faces[:, -1].sum(),
                d["outflow_m3"] - d["inflow_m3"],
                rtol=1e-12,
                atol=1e-8,
            )
            strip = q[:, -width:, 0]
            meta_row = dict(
                method=spec["method"],
                dx_m=dx,
                case=spec["case"],
                hours=spec["hours"],
                id=it["id"],
                strip_width_m=strip_m,
            )
            domain.append(
                dict(
                    meta_row,
                    strip_storage_m3=float(strip.sum() * dx**2),
                    interior_storage_m3=float(q[:, :-width, 0].sum() * dx**2),
                    strip_maximum_final_depth_m=float(strip.max()),
                    boundary_cells_deeper_5cm=int(
                        np.count_nonzero(q[:, -1, 0] >= 0.05)
                    ),
                    net_outlet_m3=float(faces[:, -1].sum()),
                    net_into_eastern_strip_m3=float(faces[:, -1 - width].sum()),
                )
            )
            for start in range(0, ny, width):
                wet = strip[start : start + width]
                hu = q[start : start + width, -width:, 1]
                bands.append(
                    dict(
                        meta_row,
                        south_m=start * dx,
                        north_m=(start + width) * dx,
                        mean_depth_m=float(wet.mean()),
                        mean_hu_m2s=float(hu.mean()),
                        storage_m3=float(wet.sum() * dx**2),
                        net_east_face_m3=float(faces[start : start + width, -1].sum()),
                    )
                )
    if wanted:
        raise ValueError("Missing verified native prefix checkpoints")
    metrics.csv_write(file=folder / "strip_storage.csv", rows=domain)
    metrics.csv_write(file=folder / "strip_by_northing.csv", rows=bands)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    strip_m = 40
    run(directory=directory, strip_m=strip_m)
