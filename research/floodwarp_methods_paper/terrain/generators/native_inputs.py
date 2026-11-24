"""Build storm inputs from native metre cells before aggregating any hydraulic field."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification.terrain import window_bed
from research.floodwarp_methods_paper.terrain.generators.generate import parameters
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def native_strip(*, x, y, dimensions, params, family, storm, options, channel_core_m):
    length, width = dimensions
    u, v = x / length, y / width
    bed = window_bed(
        xb=x,
        yb=y,
        length=length,
        width=width,
        seed=params["phase_seed"],
        grade=params["grade"],
        incision=params["incision_m"],
        landform=family,
        options=options,
    )
    depth = np.zeros_like(bed)
    momentum = np.zeros_like(bed)
    if params["initial"] == "channel":
        phase = np.random.default_rng(seed=params["phase_seed"]).uniform(
            low=-0.3, high=0.3
        )
        centre = 0.47 + 0.10 * np.sin(2 * np.pi * u + phase)
        valley = np.exp(-(((v - centre) / 0.13) ** 2))
        depth = 0.035 * np.maximum(0.0, (valley - 0.65) / 0.35)
        momentum = np.where(depth > channel_core_m, 0.12 * depth, 0.0)
    elif params["initial"] == "pond":
        depth = 0.025 * np.maximum(
            0.0, 1.0 - ((u - 0.67) / 0.15) ** 2 - ((v - 0.43) / 0.18) ** 2
        )
    elif params["initial"] != "dry":
        raise ValueError("Unknown initial state " + params["initial"])
    rain = (
        np.maximum(0.0, 1 - ((u - 0.30) / 0.23) ** 2 - ((v - 0.70) / 0.20) ** 2)
        if storm == "spatial"
        else np.ones_like(bed)
    )
    soil = ((6 + 22 * u + 6 * v) if storm == "soil" else np.full_like(bed, 2.0)) / 3.6e6
    return dict(bed=bed, depth=depth, hu=momentum, rain=rain, infiltration=soil)


def verify(*, folder, identity):
    info = json.loads((folder / "manifest.json").read_text())
    if identity != info["inputs"] or any(
        runner.checksum(folder / f) != sha for f, sha in info["files"].items()
    ):
        raise ValueError("Native input bundle failed identity or checksum verification")
    return info


def generate(*, spec, inputs, cache_root, strip_rows, include_native, channel_core_m):
    length, width = spec["dimensions"]
    if min(length, width, strip_rows) <= 0 or any(
        n % 10 for n in (length, width, strip_rows)
    ):
        raise ValueError("Complete aligned native and coarse footprints required")
    params = parameters(
        design=inputs["cfg"]["synthetic"],
        template=inputs["template"],
        family=spec["landform"],
        seed=spec["seed"],
        initial=spec["initial"],
    )
    options = dict(inputs["options"])
    if not spec["controls"]:
        options["control_bank_height_m"] = 0.0
    grids = [1, 5, 10] if include_native else [5, 10]
    identity = dict(
        native_m=1,
        grids_m=grids,
        aggregation="native_1m_block_mean",
        dimensions_m=[length, width],
        landform=spec["landform"],
        storm=spec["storm"],
        parameters=params,
        moving_channel_min_depth_m=channel_core_m,
        options=options,
        sources={
            p.name: runner.checksum(p)
            for p in (
                Path(__file__),
                Path(window_bed.__code__.co_filename),
                Path(mean_cells.__code__.co_filename),
                Path(parameters.__code__.co_filename),
            )
        },
    )
    folder = cache_root / runner.digest(identity)
    if folder.is_dir():
        return folder, verify(folder=folder, identity=identity)
    cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_root) as stage:
        staging = Path(stage) / "bundle"
        staging.mkdir()
        rasters, digests, errors = {}, {}, {}
        x = np.arange(length, dtype=np.float64)[None, :] + 0.5
        for start in range(0, width, strip_rows):
            stop = min(width, start + strip_rows)
            y = np.arange(start, stop, dtype=np.float64)[:, None] + 0.5
            fields = native_strip(
                x=x,
                y=y,
                dimensions=(length, width),
                params=params,
                family=spec["landform"],
                storm=spec["storm"],
                options=options,
                channel_core_m=channel_core_m,
            )
            for name, values in fields.items():
                digests.setdefault(name, hashlib.sha256()).update(
                    np.ascontiguousarray(values).data
                )
                five = mean_cells(values=values, factor=5)
                ten = mean_cells(values=five, factor=2)
                direct = mean_cells(values=values, factor=10)
                np.testing.assert_allclose(ten, direct, rtol=2e-15, atol=1e-12)
                errors[name] = max(
                    errors.get(name, 0.0), float(np.max(np.abs(direct - ten)))
                )
                levels = {1: values, 5: five, 10: ten}
                for dx in grids:
                    file = f"{name}_{dx}m.npy"
                    if file not in rasters:
                        rasters[file] = np.lib.format.open_memmap(
                            staging / file,
                            mode="w+",
                            dtype=np.float64,
                            shape=(width // dx, length // dx),
                        )
                    rasters[file][start // dx : stop // dx] = levels[dx]
        for raster in rasters.values():
            raster.flush()
        info = dict(
            inputs=identity,
            native_cells=length * width,
            native_sha256={k: v.hexdigest() for k, v in digests.items()},
            max_direct_10m_difference=errors,
            files={f: runner.checksum(staging / f) for f in rasters},
        )
        runner.write_json(path=staging / "manifest.json", data=info)
        try:
            staging.rename(folder)
        except OSError:
            if not folder.is_dir():
                raise
            info = verify(folder=folder, identity=identity)
    return folder, info
