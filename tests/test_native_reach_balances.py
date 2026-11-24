import numpy as np
import pytest

from research.floodwarp_methods_paper.fidelity.reach_balances import integrate, verify_totals


def test_partition_closes_with_signed_lateral_and_reverse_section_transport():
    # The internal face carries 3 m³ west. Both reaches exchange surface water
    # across the north and south boundaries and start with different storage.
    initial = np.array([[2., 5.]])
    rain = np.array([[1., 2.]])
    soil = np.array([[.5, 1.]])
    depth = np.array([[4., 4.]])
    vx = np.array([[1., -3., 1.]])
    vy = np.array([[1., 2.], [-1., 2.]])
    rows = integrate(initial=initial, depth=depth, rain=rain, soil=soil, volume_x=vx, volume_y=vy,
                     split=1, cell_area=4.)
    assert [row["residual_m3"] for row in rows] == [0., 4.]
    event = dict(initial_m3=28, rain_m3=12, storage_m3=32, infiltration_m3=6,
                 net_section_m3=-3, net_outlet_m3=1)
    with pytest.raises(ValueError, match="downstream"):
        verify_totals(reaches=rows, event=event)
    depth[0, 1] += 1
    rows = integrate(initial=initial, depth=depth, rain=rain, soil=soil, volume_x=vx, volume_y=vy,
                     split=1, cell_area=4.)
    verify_totals(reaches=rows, event=dict(event, storage_m3=36))
    with pytest.raises(AssertionError):
        verify_totals(reaches=rows, event=dict(event, storage_m3=36, net_section_m3=3))


def test_partition_rejects_a_gauge_placed_on_the_outer_edge():
    h = np.ones((2, 2))
    with pytest.raises(ValueError, match="geometry"):
        integrate(initial=h, depth=h, rain=h, soil=h, volume_x=np.zeros((2, 3)),
                  volume_y=np.zeros((3, 2)), split=2, cell_area=1.)


def test_archive_join_checks_the_measured_endpoint_and_both_reaches(tmp_path):
    from research.floodwarp_methods_paper import metrics, runner
    from research.floodwarp_methods_paper.fidelity.native_archive import regional_water

    checker = tmp_path / "reach_balances.py"
    checker.write_text("test checker")
    endpoint = tmp_path / "endpoint_fields/a/3600.npz"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_bytes(b"endpoint identity")
    event = dict(case="test", method="fv1", factor="1.0", dx_m="1", end_h="1", interval_s="60",
                 section_x_m="1", source_id="a")
    identity = dict(event=event, analysis_sha=runner.checksum(path=checker),
                    endpoint_sha=runner.checksum(path=endpoint), hydraulic_source="solver", physical_sha="inputs")
    folder = tmp_path / "reach_balances" / runner.digest(identity)[:24]
    table = folder / "water.csv"
    h = np.ones((1, 2))
    rows = integrate(initial=h, depth=h, rain=h, soil=h, volume_x=np.zeros((1, 3)), volume_y=np.zeros((2, 2)),
                     split=1, cell_area=1.)
    metrics.csv_write(file=table, rows=rows)
    runner.write_json(path=folder / "receipt.json", data=dict(identity=identity, water_sha=runner.checksum(path=table)))
    source = dict(**event, solver_source="solver", horizon="complete-native-metre", initial_m3=2, rain_m3=2,
                  storage_m3=2, infiltration_m3=2, net_section_m3=0, net_outlet_m3=0)
    assert len(regional_water(directory=tmp_path, completed=[source])) == 2
    with pytest.raises(ValueError, match="archived event"):
        regional_water(directory=tmp_path, completed=[dict(source, section_x_m=2)])
    endpoint.write_bytes(b"different state at the same time")
    with pytest.raises(ValueError, match="archived event"):
        regional_water(directory=tmp_path, completed=[source])
