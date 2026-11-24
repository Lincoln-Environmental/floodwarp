from pathlib import Path

import pytest

from research.floodwarp_methods_paper.fidelity import fv1_clock, fv1_clock_report


def test_accepted_flux_tied_peak_and_water_ledger():
    frames = [dict(t_s=end, outlet_m3s=q, section_m3s=q, steps=n)
              for end, q, n in ((30, 3, 2), (75, 3, 3), (120, -1, 4))]
    data = dict(frames=frames, budget=dict(boundary_outflow=225, boundary_inflow=45,
                storage_m3=20, infiltration=0, rain=200, residual_m3=0),
                prescribed=200, folder=Path("fresh"))
    config = dict(rain_end_s=120, window_s=7200)
    values = fv1_clock.observe(forecast=data, stopping=config)
    assert values["net_export_m3"] == 180
    assert values["steps"] == 9
    assert values["peak_m3s"] == 3
    assert values["peak_interval_end_s"] is None
    assert not values["both_stations_passed"]
    data["budget"]["boundary_inflow"] = 0
    with pytest.raises(AssertionError):
        fv1_clock.observe(forecast=data, stopping=config)


def test_signed_time_refinement_on_a_common_settled_comparison():
    contract = dict(normal="n", reduced="r", end_s=100, normal_cfl=0.4, reduced_cfl=0.2)
    first = dict(id="n", end_s="100", dx_m="10", case="plain", device="cpu", method="fv1",
                 interval_s="20", both_stations_passed="True", peak_m3s="4", net_export_m3="200",
                 storage_m3="5", infiltration_m3="2", rain_m3="207", residual_m3="0",
                 steps="60", peak_interval_end_s="60")
    second = dict(first, id="r", peak_m3s="3", peak_interval_end_s="80", steps="102",
                  net_export_m3="198", storage_m3="7")
    rows = [dict(first, end_s="80"), first, second]
    result = fv1_clock_report.end_comparison(observations=rows, source=contract)
    assert result["pct_change_peak_m3s"] == -25
    assert result["change_steps"] == 42
    assert result["change_net_export_m3"] == -2
    assert result["peak_interval_shift_s"] == 20
    second["both_stations_passed"] = "False"
    with pytest.raises(ValueError, match="Unsettled"):
        fv1_clock_report.end_comparison(observations=rows, source=contract)


def test_hancock_clock_selects_hancock_and_never_pairs_with_fv1():
    assert fv1_clock.method_settings(method="hancock_dense") == ("hancock", "Hancock", "fv1_hancock")
    with pytest.raises(ValueError, match="Unsupported"):
        fv1_clock.method_settings(method="li")
    first = dict(id="normal", case="plain", dx_m="10", device="cpu", end_s="100", interval_s="20",
                 method="hancock_dense", both_stations_passed="True")
    second = dict(first, id="reduced", method="fv1")
    contract = dict(normal="normal", reduced="reduced", end_s=100, normal_cfl=.4, reduced_cfl=.2)
    with pytest.raises(ValueError, match="method"):
        fv1_clock_report.end_comparison(observations=[first, second], source=contract)
