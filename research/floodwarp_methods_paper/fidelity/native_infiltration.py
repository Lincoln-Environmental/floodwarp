"""Partition measured runoff differences at a shared time using cumulative infiltration."""

from pathlib import Path
import json
import numpy as np

from research.floodwarp_methods_paper import metrics, runner


def losses(*, soil, rain, h, multiplier, dx, seconds, capacity):
    if np.any(soil < 0) or np.any(soil > capacity * seconds + 1e-9):
        raise ValueError("Cumulative infiltration lies outside the prescribed capacity")
    fields = {
        "rain_footprint": multiplier > 0,
        "outside_rain_footprint": multiplier == 0,
    }
    if np.any(multiplier < 0):
        raise ValueError("Negative rainfall multiplier")
    result = {}
    for name, mask in fields.items():
        infil = float(soil[mask].sum() * dx**2)
        result[name] = dict(
            infiltration_m3=infil,
            storage_m3=float(h[mask].sum() * dx**2),
            rain_m3=float(rain[mask].sum() * dx**2),
            area_m2=int(mask.sum()) * dx**2,
            equivalent_area_hours_km2_h=infil / (capacity * 3600 * 1e6),
        )
    return result


def run(*, directory):
    finished = metrics.csv_read(file=directory / "mean_storms/complete_events.csv")
    measurements = []
    common = {}
    for row in finished:
        end = float(row["end_h"]) * 3600
        ident = row["series_id"]
        dx = int(row["dx_m"])
        meta = json.loads(
            (directory / "extended_runs" / ident / "series.json").read_text()
        )
        prefix = json.loads(
            (directory / "mean_storms" / meta["parent_id"] / "result.json").read_text()
        )["data"]
        state = directory.parents[2] / ".fidelity-restarts" / ident / f"{int(end)}.npz"
        with np.load(state) as saved:
            if json.loads(saved["metadata"].item())["time"] != end:
                raise ValueError("Mismatched final time")
            if np.any(saved["f0"]) or np.any(saved["conductivity"]):
                raise ValueError("Variable soil capacity requires a separate area-hours analysis")
            capacity = saved["infil"]
            rate = float(capacity.flat[0])
            if rate <= 0 or not np.all(capacity == rate):
                raise ValueError(
                    "Equivalent area requires a common positive constant infiltration capacity"
                )
            soil = saved["soil"].reshape(-1)
            rain = saved["rain_sum"].reshape(-1)
            h = saved["q"][:, 0]
            multiplier = saved["multiplier"].reshape(-1)
            regions = losses(
                soil=soil,
                rain=rain,
                h=h,
                multiplier=multiplier,
                dx=dx,
                seconds=end,
                capacity=rate,
            )
        for region, values in regions.items():
            measurements.append(
                dict(
                    id=ident,
                    method=row["method"],
                    dx_m=dx,
                    end_h=end / 3600,
                    region=region,
                    capacity_mm_h=rate * 3.6e6,
                    **values,
                )
            )
        for field, key in (
            ("infiltration_m3", "infiltration_m3"),
            ("storage_m3", "storage_m3"),
            ("rain_m3", "rain_m3"),
        ):
            observed = sum(r[field] for r in regions.values())
            expected = float(row[key]) if key in row else prefix[key]
            np.testing.assert_allclose(observed, expected, rtol=2e-11, atol=1e-7)
        common[dx, row["method"]] = row, prefix, regions
    differences = []
    for (dx, method), (row, prefix, regions) in common.items():
        if method == "fv1":
            continue
        fv, first, fregions = common[dx, "fv1"]
        for name in (
            "rain_mask_sha256",
            "infiltration_sha256",
            "initial_depth_sha256",
            "terrain_sha256",
            "roughness",
        ):
            if first["forcing"][name] != prefix["forcing"][name]:
                raise ValueError("Unequal physical forcing: " + name)
        if row["end_h"] != fv["end_h"]:
            raise ValueError("Unequal simulation windows")
        export = float(fv["net_outlet_m3"]) - float(row["net_outlet_m3"])
        reduction = float(row["infiltration_m3"]) - float(fv["infiltration_m3"])
        storage = float(row["storage_m3"]) - float(fv["storage_m3"])
        np.testing.assert_allclose(export, reduction + storage, rtol=1e-10, atol=1e-6)
        differences.append(
            dict(
                dx_m=dx,
                end_h=row["end_h"],
                method=method,
                fv1_id=fv["series_id"],
                comparison_id=row["series_id"],
                extra_fv1_export_m3=export,
                less_fv1_infiltration_m3=reduction,
                less_fv1_final_storage_m3=storage,
                less_fv1_infiltration_in_rain_footprint_m3=regions["rain_footprint"][
                    "infiltration_m3"
                ]
                - fregions["rain_footprint"]["infiltration_m3"],
                less_fv1_infiltration_outside_rain_footprint_m3=regions[
                    "outside_rain_footprint"
                ]["infiltration_m3"]
                - fregions["outside_rain_footprint"]["infiltration_m3"],
            )
        )
    output = directory / "native_outlet"
    metrics.csv_write(file=output / "runoff_partition.csv", rows=differences)
    metrics.csv_write(
        file=output / "infiltration_by_rain_footprint.csv", rows=measurements
    )
    runner.write_json(
        path=output / "infiltration_sources.json",
        data=dict(
            matched_receipt=runner.checksum(
                directory / "mean_storms/complete_events.csv"
            ),
            code=runner.checksum(Path(__file__)),
            definition="Cumulative soil depth integrated over cells with positive or zero storm rainfall multiplier. Capacity-equivalent area-hours describe infiltration opportunity.",
        ),
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
