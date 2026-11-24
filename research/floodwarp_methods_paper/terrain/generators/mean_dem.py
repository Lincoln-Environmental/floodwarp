"""Aligned, equal-area 1 m -> 5 m -> 10 m elevation aggregation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.qualification.terrain import window_bed
from research.floodwarp_methods_paper.terrain.generators.generate import parameters


def mean_cells(*, values, factor):
    if values.ndim != 2 or not isinstance(factor, int) or factor < 1:
        raise ValueError("Expected a raster and a positive integer aggregation factor")
    ny, nx = values.shape
    if ny % factor or nx % factor or not np.isfinite(values).all():
        raise ValueError(
            "Area means require complete aligned blocks of finite elevations"
        )
    return values.reshape(ny // factor, factor, nx // factor, factor).mean(
        axis=(1, 3), dtype=np.float64
    )


def generate(*, spec, inputs, cache_root, strip_rows):
    length, width = spec["dimensions"]
    if length % 10 or width % 10 or strip_rows < 10 or strip_rows % 10:
        raise ValueError("Domain and row strips must align with each 10 m footprint")
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
    identity = dict(
        method="mean_of_four_5m_cells_from_native_1m",
        dimensions_m=[length, width],
        parameters=params,
        landform=spec["landform"],
        options=options,
        code={
            p.name: runner.checksum(p)
            for p in (
                Path(__file__),
                Path(window_bed.__code__.co_filename),
                Path(parameters.__code__.co_filename),
            )
        },
    )
    folder = cache_root / runner.digest(identity)
    manifest = folder / "mean_dem.json"
    if manifest.is_file():
        desc = json.loads(manifest.read_text())
        if desc["inputs"] != identity or any(
            runner.checksum(folder / name) != value
            for name, value in desc["files"].items()
        ):
            raise ValueError(
                "Cached area-mean DEM failed identity or checksum verification"
            )
        return folder, desc
    folder.mkdir(parents=True, exist_ok=True)
    levels = {
        dx: np.lib.format.open_memmap(
            folder / f"bed_{dx}m.pending.npy",
            mode="w+",
            shape=(width // dx, length // dx),
            dtype=np.float64,
        )
        for dx in (5, 10)
    }
    x = np.arange(length, dtype=np.float64)[None, :] + 0.5
    source = hashlib.sha256()
    worst = 0.0
    for start in range(0, width, strip_rows):
        end = min(start + strip_rows, width)
        y = np.arange(start, end, dtype=np.float64)[:, None] + 0.5
        native = window_bed(
            xb=x,
            yb=y,
            length=length,
            width=width,
            seed=params["phase_seed"],
            grade=params["grade"],
            incision=params["incision_m"],
            landform=spec["landform"],
            options=options,
        )
        source.update(np.ascontiguousarray(native).data)
        five = mean_cells(values=native, factor=5)
        ten = mean_cells(values=five, factor=2)
        direct = mean_cells(values=native, factor=10)
        np.testing.assert_allclose(ten, direct, rtol=2e-15, atol=1e-12)
        worst = max(worst, float(np.max(abs(ten - direct))))
        levels[5][start // 5 : end // 5] = five
        levels[10][start // 10 : end // 10] = ten
    for dx, array in levels.items():
        array.flush()
        (folder / f"bed_{dx}m.pending.npy").replace(folder / f"bed_{dx}m.npy")
    desc = dict(
        inputs=identity,
        native_1m_cells=width * length,
        native_1m_sha256=source.hexdigest(),
        max_difference_direct_1_to_10_m=worst,
        files={
            f"bed_{dx}m.npy": runner.checksum(folder / f"bed_{dx}m.npy")
            for dx in levels
        },
    )
    tmp = folder / "mean_dem.pending.json"
    runner.write_json(path=tmp, data=desc)
    tmp.replace(manifest)
    return folder, desc
