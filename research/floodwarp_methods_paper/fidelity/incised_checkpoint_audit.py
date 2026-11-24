"""Audit the incised experiment directly against native FP64 state and accepted face volumes."""

import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def locate(*, directory):
    found = {}
    for path in directory.glob("*/state.npz"):
        with np.load(path) as saved:
            manifest = json.loads(str(saved["_direct_manifest"]))
        found[manifest["sha256"]["q"]] = path
    return found


def compare(*, folder, checkpoints, design):
    runs = []
    maps = {}
    inputs = {}
    support = None
    times = []
    for path in sorted(folder.glob("*/result.json")):
        record = json.loads(path.read_text())
        identity = record["inputs"]
        s, d = identity["spec"], record["data"]
        if identity["design"] != design or identity["factor"] != 1:
            continue
        key = (s["dx"], s["method"])
        ck = checkpoints[d["hashes"]["state"]]
        dx = s["dx"]
        ny, nx = d["forcing"]["shape"]
        if support is not None and support != s["support"]:
            raise ValueError("Depth differences require the same physical support")
        support = s["support"]
        factor = support // dx
        with np.load(ck) as a, np.load(path.with_name("spatial.npz")) as spatial:
            meta = json.loads(str(a["metadata"]))
            if (
                meta["time"] != design["diagnostic_start_h"] * 3600
                or meta["steps"] != d["steps"]
            ):
                raise ValueError("Wrong checkpoint time or step count")
            for field, raw in (
                ("state", "q"),
                ("maximum", "maximum"),
                ("volume_x", "volume_x"),
                ("volume_y", "volume_y"),
                ("soil", "soil"),
                ("rain_sum", "rain_sum"),
            ):
                if array_sha(a[raw]) != d["hashes"][field]:
                    raise ValueError("State or transport hash mismatch: " + field)
            h = a["q"][:, 0].reshape(ny, nx)

            def blocks(values):
                return values.reshape(ny // factor, factor, nx // factor, factor).mean(
                    axis=(1, 3)
                )

            mapped = blocks(h)
            np.testing.assert_allclose(mapped, spatial["depth"], rtol=0, atol=0)
            maps[key] = mapped
            record_area = {}
            for cm in (1, 5, 10):
                wet = h >= cm / 100
                np.testing.assert_array_equal(
                    blocks(wet), spatial[f"wet_fraction_{cm}cm"]
                )
                record_area[f"area_{cm}cm_km2"] = np.count_nonzero(wet) * dx**2 / 1e6
            input_hash = {}
            for name in ("bed", "manning", "active", "multiplier", "infil"):
                field = a[name].reshape(ny, nx)
                if dx != design["bed_m"]:
                    parent_ratio = design["bed_m"] // dx
                    parent = field[::parent_ratio, ::parent_ratio]
                    for y in range(parent_ratio):
                        for x in range(parent_ratio):
                            np.testing.assert_array_equal(
                                parent, field[y::parent_ratio, x::parent_ratio]
                            )
                    field = parent
                input_hash[name] = array_sha(field)
                if name == "bed":
                    drops = -np.diff(field, axis=1)
            inputs[key] = input_hash
            bx, by = a["types_x"].reshape(ny, nx + 1), a["types_y"].reshape(ny + 1, nx)
            if not (
                np.all(bx[:, -1] == 2)
                and np.all(bx[:, 0] == 1)
                and np.all(bx[:, 1:-1] == 0)
                and np.all(by[[0, -1], :] == 1)
                and np.all(by[1:-1, :] == 0)
            ):
                raise ValueError("Unexpected boundary face geometry")
            vx, vy = (
                a["volume_x"].reshape(ny, nx + 1),
                a["volume_y"].reshape(ny + 1, nx),
            )
            closed = max(
                float(np.max(abs(vx[:, 0]))), float(np.max(abs(vy[[0, -1], :])))
            )
            if closed != 0:
                raise ValueError("Measured transport through a closed boundary")
            station = vx[:, [nx // 2, -1]].sum(axis=0)
        frames = d["frames"]
        ends = np.array([f["t_s"] for f in frames])
        dt = np.diff(np.r_[0, ends])
        flows = np.array([[f["section_m3s"], f["outlet_m3s"]] for f in frames])
        cumulative = np.cumsum(dt[:, None] * flows, axis=0)
        np.testing.assert_allclose(cumulative[-1], station, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(
            station[-1], d["outflow_m3"] - d["inflow_m3"], rtol=1e-12, atol=1e-8
        )
        np.testing.assert_allclose(h.sum() * dx**2, d["final_storage_m3"], rtol=1e-12)
        for j, t in enumerate(ends):
            times.append(
                dict(
                    run_id=path.parent.name,
                    grid_m=dx,
                    method=s["method"],
                    t_s=t,
                    section_m3s=flows[j, 0],
                    outlet_m3s=flows[j, 1],
                    section_volume_m3=cumulative[j, 0],
                    outlet_volume_m3=cumulative[j, 1],
                )
            )
        runs.append(
            dict(
                run_id=path.parent.name,
                checkpoint_id=ck.parent.name,
                grid_m=dx,
                method=s["method"],
                checked_steps=d["steps"],
                end_s=meta["time"],
                q_sha256=d["hashes"]["state"],
                outlet_m3=station[-1],
                section_m3=station[0],
                closed_face_volume_m3=closed,
                storage_m3=h.sum() * dx**2,
                median_parent_eastward_drop_m=float(np.median(drops)),
                **record_area,
            )
        )
    fine, coarse = sorted({dx for dx, _ in maps})
    for fields in inputs.values():
        if fields != inputs[coarse, "adaptive"]:
            raise ValueError(
                "Bed, roughness, mask or infiltration differs between grids"
            )
    comparisons = []
    common = np.maximum.reduce(list(maps.values()))
    for label, left, right in (
        ("FV2 grid refinement", (fine, "adaptive"), (coarse, "adaptive")),
        ("Hancock versus FV2", (fine, "hancock_adaptive"), (fine, "adaptive")),
    ):
        delta = maps[left] - maps[right]
        for cm in (0, 5, 10):
            mask = common >= cm / 100
            values = delta[mask]
            comparisons.append(
                dict(
                    comparison=label,
                    left_grid_m=left[0],
                    right_grid_m=right[0],
                    common_support_m=support,
                    any_run_depth_cm=cm,
                    blocks=int(mask.sum()),
                    signed_mean_difference_m=values.mean(),
                    rmse_m=np.sqrt(np.mean(values**2)),
                    abs_p95_m=np.percentile(abs(values), 95),
                    min_difference_m=values.min(),
                    max_difference_m=values.max(),
                )
            )
    metrics.csv_write(file=folder / "checkpoint_audit.csv", rows=runs)
    metrics.csv_write(
        file=folder / "checkpoint_spatial_differences.csv", rows=comparisons
    )
    metrics.csv_write(file=folder / "checkpoint_station_histories.csv", rows=times)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    root = directory.parents[2] / ".fidelity-restarts/prefixes"
    design = json.loads((directory / "incised_settings.json").read_text())
    compare(
        folder=directory / "incised_crest",
        checkpoints=locate(directory=root),
        design=design,
    )
