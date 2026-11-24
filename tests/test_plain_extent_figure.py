import json

import numpy as np
import pytest

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.figures.generate import plain_extent_field


def test_reporting_cell_preserves_quarter_wet_hydraulic_area(tmp_path):
    path = tmp_path / "native_matrix/plain/run"
    path.mkdir(parents=True)
    spec = dict(dimensions=[20, 10], support=10, dx=5, interval=60)
    raw = dict(inputs=dict(spec=spec, physical_sha="inputs"), data=dict(forcing=dict(terrain_sha256="bed")))
    (path / "result.json").write_text(json.dumps(raw))
    # Four 5 m cells in each reporting cell. One and two exceed 10 cm.
    np.savez(path / "spatial.npz", wet_fraction_10cm=[[.25, .5]])
    row = dict(case="plain", source_id="run", end_h="8", section_x_m="10", interval_s="60",
               dx_m="5", model_grid_wet_area_10cm_m2="75")
    source = dict(case="plain", end_s=28800, spatial_sha256=runner.checksum(path=path / "spatial.npz"),
                  result_sha256=runner.checksum(path=path / "result.json"))
    wet, area, grid, physical = plain_extent_field(directory=tmp_path, row=row, source=source, threshold_cm=10)
    np.testing.assert_array_equal(wet, [[.25, .5]])
    assert (area, grid[:4], physical) == (75, (20, 10, 10, 10), "inputs")
    with pytest.raises(ValueError, match="wet area"):
        plain_extent_field(directory=tmp_path, row=dict(row, model_grid_wet_area_10cm_m2=200),
                           source=source, threshold_cm=10)
    np.savez(path / "spatial.npz", wet_fraction_10cm=[[.5, .25]])
    with pytest.raises(ValueError, match="completed forecast"):
        plain_extent_field(directory=tmp_path, row=row, source=source, threshold_cm=10)
