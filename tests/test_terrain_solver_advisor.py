"""Controlled topography, georeferencing and evidence-based abstention."""

from pathlib import Path

import numpy as np
import pytest

from floodwarp.terrain_advisor.diagnostics import analyse_array
from floodwarp.solver_advisor.model import descriptors, scales, recommend


def test_closed_pit_volume_and_planar_grid_identity():
    z = np.ones((9, 11))
    z[3:6, 4:7] = 0.0
    same, maps, _ = analyse_array(
        elevation=z, native_resolution=1, resolution=1, channel_area_m2=4
    )
    assert same["native"]["depression_storage_m3"] == 9.0
    assert same["comparison"]["elevation_rmse_m"] == 0
    assert same["comparison"]["changed_outlet_fraction"] == 0
    assert maps["fill_depth"][4, 5] == 1
    plane = np.broadcast_to(np.arange(11.0) * 0.01, (9, 11)).copy()
    coarse, _, _ = analyse_array(elevation=plane, native_resolution=1, resolution=2.5)
    assert coarse["model"]["area_m2"] == plane.size
    np.testing.assert_allclose(coarse["native"]["mean_slope"], 0.01)
    assert coarse["native"]["depression_storage_m3"] == 0


def test_projected_raster_metadata_and_nodata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from floodwarp.terrain_advisor import analyse_dem
    from rasterio.transform import from_origin

    data = np.arange(120, dtype=float).reshape(10, 12)
    data[3, 3] = -9999.0
    dem = tmp_path / "terrain.tif"
    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        count=1,
        dtype="float64",
        height=10,
        width=12,
        transform=from_origin(10.0, 50.0, 1.0, 1.0),
        crs="EPSG:2193",
        nodata=-9999.0,
    ) as out:
        out.write(data, 1)
    result = analyse_dem(dem=dem, resolution=2.0, output=tmp_path / "result")
    assert result["native"]["cells"] == 119
    with rasterio.open(tmp_path / "result/model/elevation.tif") as out:
        assert (
            out.crs.to_epsg() == 2193 and out.res == (2.0, 2.0) and out.units == ("m",)
        )


def test_advisor_empirical_feasibility_and_unsupported_scale():
    base = dict(
        mean_slope=0.01, area_m2=100.0, cells=100, resolution=1.0, native_resolution=1.0
    )
    rows = []
    for i in range(3):
        for method, t, error in (("fv1", 1.0, 0.03), ("hancock_adaptive", 2.0, 0.002)):
            rows.append(
                dict(
                    features=dict(base, mean_slope=0.005 * (i + 1)),
                    terrain_id=str(i),
                    method=method,
                    errors={"depth": error},
                    qualified={"depth": True},
                    memory_mib=10.0,
                    runtime_s=t,
                    run_id=f"{method}{i}",
                )
            )
    model = dict(
        training=rows,
        scales=scales(rows=rows),
        k=3,
        quantile=0.9,
        distance_limit=1.0,
        tolerances={"depth": 0.01},
        revision="test",
        hardware="test",
    )
    got = recommend(model=model, features=base, objective="depth")
    assert got["method"] == "hancock_adaptive"
    assert (
        recommend(model=model, features=base, objective="depth", tolerance=0.001)[
            "status"
        ]
        == "no_feasible_candidate"
    )
    assert (
        recommend(model=model, features=dict(base, area_m2=1000.0), objective="depth")[
            "status"
        ]
        == "insufficient_evidence"
    )
