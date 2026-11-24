"""Matched CPU reach replay, changing only the representation of the survey bed."""

import json
from pathlib import Path
import numpy as np

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.qualification import hydraulics
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def parent_fields(*, record, spacing):
    inputs = record["inputs"]
    spec = dict(inputs["spec"], dx=spacing, dem=spacing, initial_grid="parent")
    return hydraulics.represented(
        options=inputs["options"],
        settings=spec,
        design=inputs["cfg"]["synthetic"],
        template=inputs["template"],
        family=spec["landform"],
        seed=spec["seed"],
        initial=spec["initial"],
        dx=spacing,
        dem=spacing,
    )[0]


def subdivide(*, a, ratio):
    return np.repeat(np.repeat(a, ratio, axis=0), ratio, axis=1)


def limited_slope(*, a, axis):
    back = np.diff(a, axis=axis, prepend=np.take(a, [0], axis=axis))
    ahead = np.diff(a, axis=axis, append=np.take(a, [-1], axis=axis))
    return np.where(
        back * ahead > 0, np.sign(back) * np.minimum(abs(back), abs(ahead)), 0.0
    )


def parent_linear(*, z, ratio):
    sy, sx = (limited_slope(a=z, axis=k) for k in (0, 1))
    fine = subdivide(a=z, ratio=ratio)
    offsets = (np.arange(ratio) + 0.5) / ratio - 0.5
    for y, dy in enumerate(offsets):
        for x, dx in enumerate(offsets):
            fine[y::ratio, x::ratio] += dy * sy + dx * sx
    np.testing.assert_allclose(
        fine.reshape(z.shape[0], ratio, z.shape[1], ratio).mean(axis=(1, 3)),
        z,
        atol=1e-12,
        rtol=0,
    )
    return fine


def run(*, directory, settings):
    folder = directory / "incised_crest"
    prefix = next(
        z
        for p in folder.glob("*/result.json")
        if (z := json.loads(p.read_text()))["data"]["numerical"]["dx"]
        == settings["parent_m"]
        and z["inputs"]["spec"]["method"] == "adaptive"
        and z["inputs"]["factor"] == 1
    )
    parent = parent_fields(record=prefix, spacing=settings["parent_m"])
    continuous = parent_fields(record=prefix, spacing=settings["child_m"])[0]
    z, h, hu, u, v = parent
    source = prefix["data"]["forcing"]
    if any(
        array_sha(field) != source[k]
        for field, k in (
            (z, "terrain_sha256"),
            (h, "initial_depth_sha256"),
            (hu, "initial_hu_sha256"),
        )
    ):
        raise ValueError("Parent bed or initial water differs from the archived storm")
    ratio = settings["parent_m"] // settings["child_m"]
    fine_inputs = tuple(subdivide(a=a, ratio=ratio) for a in parent[1:])
    variant_beds = {
        "parent": z,
        "replicated": subdivide(a=z, ratio=ratio),
        "linear": parent_linear(z=z, ratio=ratio),
        "sampled": continuous,
    }
    from research.floodwarp_methods_paper.fidelity.mean_storms import reference_inputs
    from research.floodwarp_methods_paper import runner

    area_design = json.loads((directory / "mean_settings.json").read_text())
    _, _, mean_folder, definition = reference_inputs(
        directory=directory, design=area_design
    )
    runner.write_json(path=folder / "mean_dem_inputs.json", data=definition)
    for grid in (settings["parent_m"], settings["child_m"]):
        variant_beds[f"area_mean_{grid}m"] = np.load(
            mean_folder / f"bed_{grid}m.npy", mmap_mode="r"
        )
    config = prefix["inputs"]["spec"]
    centre_x = settings["mid_reach_x_m"]
    centre_y = (int(np.argmax(h[:, centre_x // settings["parent_m"]]))) * settings[
        "parent_m"
    ]
    rows = []
    for margin in settings["margins_m"]:
        for variant, bed in variant_beds.items():
            dx = (
                settings["parent_m"]
                if variant in ("parent", f"area_mean_{settings['parent_m']}m")
                else settings["child_m"]
            )
            half = settings["section_half_width_m"]
            bounds = (
                centre_x - margin,
                centre_x + margin,
                centre_y - half - margin,
                centre_y + half + margin,
            )
            x0, x1, y0, y1 = (v // dx for v in bounds)
            cut = np.s_[y0:y1, x0:x1]
            dh, dhu, uu, vv = (
                a[cut]
                for a in (parent[1:] if dx == settings["parent_m"] else fine_inputs)
            )
            dem = bed[cut].copy()
            terrain = terrain_from_array(
                elevation=dem,
                dx=dx,
                dy=dx,
                row_order="south_to_north",
                vertical_datum="local",
            )
            mask = np.maximum(
                0.0, 1 - ((uu - 0.30) / 0.23) ** 2 - ((vv - 0.70) / 0.20) ** 2
            )
            model = create_model(
                terrain=terrain,
                solver_method="fv2",
                limiter="adaptive",
                execution_mode="dense",
                device="cpu",
                explicit_maximum_dt=settings["maximum_dt_s"],
                gravity=prefix["data"]["numerical"]["gravity"],
                manning_n=source["roughness"],
                initial_depth=dh,
                rainfall_series=(source["rain_times_s"], source["rain_rates_ms"]),
                infiltration=2 / 3.6e6,
            )
            model.solver.set_momentum(hu=dhu, hv=np.zeros_like(dhu))
            model.solver.set_rainfall_spatial_multiplier(multiplier=mask)
            info = model.solver.run_device_loop(duration=settings["duration_s"])
            ny, nx = dem.shape
            section = (
                model.solver._fields["volume_x"]
                .numpy()
                .reshape(ny, nx + 1)[margin // dx : (margin + 2 * half) // dx, nx // 2]
            )
            rows.append(
                dict(
                    representation=variant,
                    dx_m=dx,
                    margin_m=margin,
                    physical_section_width_m=2 * half,
                    section_x_m=centre_x,
                    section_y_m=centre_y,
                    seconds=model.solver.time,
                    hydraulic_steps=info["steps"],
                    interval_mean_section_m3s=float(section.sum() / model.solver.time),
                    case=config["case"],
                )
            )
    metrics.csv_write(file=folder / "reach_replay.csv", rows=rows)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "reach_settings.json").read_text())
    run(directory=directory, settings=settings)
