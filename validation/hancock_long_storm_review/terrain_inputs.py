"""Export south-first common-domain DEMs, initial conditions and terrain metrics."""

import json
from pathlib import Path

import numpy as np

from validation.hancock_long_storm_review.landforms import physical_fields


def export(*, review, dimensions, spacings, seed, landforms):
    folder = review / "terrain"
    folder.mkdir(exist_ok=True)
    metrics = []
    for land, spec in landforms.items():
        for dx in spacings:
            z, h, hu, _, _ = physical_fields(
                landform=land,
                spec=spec,
                seed=seed,
                length=dimensions[0],
                width=dimensions[1],
                dx=dx,
                terrain_dx=dx,
            )
            dy_slope, dx_slope = np.gradient(z, dx, dx)
            slopes = np.hypot(dx_slope, dy_slope)
            record = dict(
                landform=land,
                spacing_m=dx,
                nx=z.shape[1],
                ny=z.shape[0],
                nominal_grade=spec[0],
                slope_p99=float(np.quantile(slopes, 0.99)),
                slope_max=float(slopes.max()),
                bed_min_m=z.min(),
                bed_max_m=z.max(),
                initial_volume_m3=float(h.sum() * dx**2),
            )
            if land == "floodplain":
                # Lowest transverse crossing in the ridge's nearest raster column.
                column = round(0.70 * dimensions[0] / dx - 0.5)
                record["embankment_column_east_m"] = (column + 0.5) * dx
                row = int(z[:, column].argmin())
                record["lowest_crossing_north_m"] = (row + 0.5) * dx
                record["lowest_crossing_bed_m"] = z[row, column]
            np.savez_compressed(
                folder / f"{land}_{dx}m.npz",
                elevation_m=z,
                initial_depth_m=h,
                initial_hu_m2s=hu,
                dx_m=dx,
                dy_m=dx,
                x0_m=0.0,
                y0_m=0.0,
            )
            metrics.append(record)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    spacings = (1, 5, 10)
    export(
        review=review,
        dimensions=config["dimensions_m"],
        spacings=spacings,
        seed=config["seed"],
        landforms=config["landforms"],
    )
