"""Pre-event descriptors and hydraulic-control maps from the grids used for each native forecast."""

from pathlib import Path
import json
import numpy as np
from affine import Affine
from rasterio.crs import CRS

from floodwarp.solver_advisor.model import descriptors
from floodwarp.terrain_advisor.diagnostics import analyse_array, terrain
from floodwarp.terrain_advisor.raster import write_maps
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_matrix
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification.terrain import connection
from research.floodwarp_methods_paper.terrain.generators import native_inputs
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def spill_level(*, bed):
    heights = np.unique(bed)
    lo, hi = 0, heights.size - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if connection(elevation=bed, level=heights[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(heights[lo])


def window(*, receipt, options):
    info = receipt["inputs"]
    params = info["parameters"]
    length, width = info["dimensions_m"]
    phase = np.random.default_rng(params["phase_seed"]).uniform(-0.3, 0.3)
    centre = width * (0.47 + 0.10 * np.sin(2 * np.pi * 0.875 + phase))
    size = options["control_window_m"]
    align = 10
    west = int((0.875 * length - size / 2) // align) * align
    south = int((centre - size / 2) // align) * align
    if (
        size % align
        or min(west, south) < 0
        or west + size > length
        or south + size > width
    ):
        raise ValueError("Control window is not aligned within the physical terrain")
    z = native_inputs.native_strip(
        x=west + np.arange(size)[None, :] + 0.5,
        y=south + np.arange(size)[:, None] + 0.5,
        dimensions=info["dimensions_m"],
        params=params,
        family=info["landform"],
        storm=info["storm"],
        options=info["options"],
        channel_core_m=info["moving_channel_min_depth_m"],
    )["bed"]
    return west, south, z


def forcing(*, spec, template, folder, cfg):
    dx = spec["dx"]
    rate = np.asarray(template["storms_mm_h"][spec["storm"]])
    hours = np.asarray(template["hours"])
    rain = float(np.trapezoid(rate, hours))
    first_moment = float(
        np.sum(
            np.diff(hours)
            * (
                hours[:-1] * (2 * rate[:-1] + rate[1:])
                + hours[1:] * (rate[:-1] + 2 * rate[1:])
            )
            / 6
        )
    )

    def mean(name):
        return float(np.load(folder / f"{name}_{dx}m.npy", mmap_mode="r").mean())

    mean_mask = mean("rain")
    return dict(
        forecast_duration_h=spec["hours"],
        rain_duration_h=float(hours[-1]),
        rain_centroid_h=first_moment / rain,
        spatial_mean_multiplier=mean_mask,
        rain_mm=rain * mean_mask,
        peak_mm_h=float(rate.max()),
        initial_mean_m=mean("depth"),
        infiltration_mean_mm_h=3.6e6 * mean("infiltration"),
        manning_n=cfg["storms"]["manning_n"] * spec["multiplier"],
        downstream_stage_m=0.015 if spec["coastal_stage"] else None,
    )


def measure(*, directory, spec, design, config, options):
    inputs, folder, receipt = native_matrix.physical_inputs(
        directory=directory, base=spec, config=config, design=design
    )
    dx = spec["dx"]
    output = directory / "native_features" / spec["case"] / f"{dx}m"
    identity = dict(
        physical=runner.digest(receipt),
        dx=dx,
        options=options,
        spec={
            k: spec[k]
            for k in (
                "landform",
                "seed",
                "storm",
                "initial",
                "hours",
                "split",
                "dimensions",
                "controls",
                "coastal_stage",
                "multiplier",
            )
        },
        storms=inputs["template"]["storms_mm_h"][spec["storm"]],
        rain_hours=inputs["template"]["hours"],
        manning=config["storms"]["manning_n"],
        source=runner.checksum(Path(__file__)),
        analysis=runner.checksum(Path(analyse_array.__code__.co_filename)),
        model=runner.checksum(Path(descriptors.__code__.co_filename)),
        writer=runner.checksum(Path(write_maps.__code__.co_filename)),
    )
    target = output / runner.digest(identity)[:24]
    record = target / "pre_run.json"
    if record.exists():
        result = json.loads(record.read_text())
        if result["identity"] != identity or any(
            runner.checksum(target / file) != sha
            for file, sha in result["maps"].items()
        ):
            raise ValueError("Pre-run descriptor archive failed verification")
        return result
    target.mkdir(exist_ok=True, parents=True)
    west, south, native = window(receipt=receipt, options=options)
    size = native.shape[0]
    coarse = mean_cells(values=native, factor=5) if dx > 1 else native
    if dx == 10:
        coarse = mean_cells(values=coarse, factor=2)
    full = np.load(folder / f"bed_{dx}m.npy", mmap_mode="r")
    np.testing.assert_array_equal(
        coarse,
        full[south // dx : (south + size) // dx, west // dx : (west + size) // dx],
    )
    domain, _ = terrain(
        z=full[::-1],
        dx=dx,
        areas=np.full_like(full, dx**2),
        channel_area=max(
            options["minimum_channel_area_m2"],
            options["domain_channel_fraction"] * full.size * dx**2,
        ),
        window_m=options["relief_window_m"],
    )
    stats, maps, represented = analyse_array(
        elevation=native[::-1],
        native_resolution=1,
        resolution=dx,
        represented_elevation=coarse[::-1],
        window_m=options["relief_window_m"],
    )
    spill = spill_level(bed=native)
    stage = spill + options["spill_head_m"]
    comparison = dict(
        stats["comparison"],
        native_connected=connection(elevation=native, level=stage),
        represented_connected=connection(elevation=coarse, level=stage),
        probe_stage_m=stage,
        native_spill_m=spill,
        represented_spill_m=spill_level(bed=coarse),
        opening_width_cells=2
        * receipt["inputs"]["options"]["family_control_halfwidth_m"]
        / dx,
        max_local_barrier_lowering_m=float(maps["barrier_lowering"].max()),
    )
    crs = CRS.from_wkt(options["local_crs"])
    transform = Affine.translation(west, south + size) @ Affine.scale(1, -1)
    write_maps(
        maps={k: maps[k] for k in options["native_maps"]},
        folder=target / "native",
        transform=transform,
        crs=crs,
    )
    write_maps(
        maps={k: represented[k] for k in options["represented_maps"]},
        folder=target / "represented",
        transform=transform @ Affine.scale(dx),
        crs=crs,
    )
    physical = forcing(
        spec=spec, template=inputs["template"], folder=folder, cfg=config
    )
    result = dict(
        identity=identity,
        case=spec["case"],
        dx_m=dx,
        terrain_id=f"{spec['landform']}-{spec['seed']}",
        split=spec["split"],
        landform=spec["landform"],
        storm=spec["storm"],
        bed_sha256=array_sha(full),
        control_window_m=[west, south, size, size],
        represented_domain=domain,
        control=comparison,
        forcing=physical,
        features=descriptors(
            terrain=dict(native={"dx_m": 1}, model=domain, comparison=comparison),
            hydraulic=physical,
        ),
        maps={
            str(p.relative_to(target)): runner.checksum(p)
            for p in sorted(target.rglob("*.tif"))
        },
    )
    runner.write_json(path=record, data=result)
    return result


def run(*, directory, design, config, options):
    registered = {}
    for s in native_matrix.specifications(
        study=directory.parent, config=config, design=design
    ):
        registered.setdefault((s["case"], s["dx"]), s)
    rows = []
    for spec in registered.values():
        record = measure(
            directory=directory,
            spec=spec,
            design=design,
            config=config,
            options=options,
        )
        rows.append(
            dict(
                case=spec["case"],
                dx_m=spec["dx"],
                identity=runner.digest(record["identity"])[:24],
                terrain_id=record["terrain_id"],
                split=record["split"],
                **record["features"],
            )
        )
        print("Verified terrain descriptors", spec["case"], spec["dx"], flush=True)
    metrics.csv_write(file=directory / "native_features/observations.csv", rows=rows)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    options = json.loads((directory / "native_feature_settings.json").read_text())
    config = read_config(path=directory.parent / "configs/study.json")
    run(directory=directory, design=design, config=config, options=options)
