import numpy as np
import pytest

from research.floodwarp_methods_paper.fidelity.embankment_transport import profiles, summarize


def test_barrier_lowering_and_flow_reversal_at_a_fixed_physical_section():
    # Each 5 m raster cell is the area mean of 25 native 1 m cells.
    fine = np.zeros((20, 30))
    fine[:, 13] = .2
    fine[5:10, 13] = 0
    coarse = fine.reshape(4, 5, 6, 5).mean(axis=(1, 3))
    volume = np.zeros((4, 7))
    volume[:, 3] = [2, -4, 3, 5]
    window = [5, 5, 20, 5]
    data = profiles(native_bed=fine, bed=coarse, volume=volume, dx=5, window=window, face=15)
    row = summarize(data=data, dx=5, window=window)
    assert row["net_in_band_m3"] == -4
    assert row["net_outside_band_m3"] == 10
    assert row["net_section_m3"] == 6
    assert row["outside_median_crest_shift_cm"] == pytest.approx(-16.)
    with pytest.raises(ValueError, match="coincide"):
        profiles(native_bed=fine, bed=coarse, volume=volume, dx=5, window=window, face=13)
    with pytest.raises(ValueError, match="Invalid"):
        summarize(data=dict(data, native_crest_m=data["native_crest_m"][:-1]), dx=5, window=window)


def test_transect_keeps_individual_native_northings_in_the_crest_comparison():
    fine = np.zeros((10, 10))
    fine[:5, 4] = [.05, .10, .15, .20, .25]
    bed = fine.reshape(2, 5, 2, 5).mean(axis=(1, 3))
    window = [0, 5, 10, 5]
    data = profiles(native_bed=fine, bed=bed, volume=np.zeros((2, 3)), dx=5, window=window, face=5)
    row = summarize(data=data, dx=5, window=window)
    assert row["outside_median_crest_shift_cm"] == pytest.approx(-12)
    assert row["outside_min_crest_shift_cm"] == pytest.approx(-22)
    assert row["outside_max_crest_shift_cm"] == pytest.approx(-2)
