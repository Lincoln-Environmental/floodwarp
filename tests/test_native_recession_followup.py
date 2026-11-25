import pytest

from research.floodwarp_methods_paper.fidelity.recession_followup import transport


def test_later_larger_crest_and_reversal_use_accepted_interval_transport():
    frames = [dict(t_s=t, outlet_m3s=q) for t, q in ((10, 2), (15, 1), (20, -2), (30, 3))]
    settings = dict(minimum_volume_m3=1, minimum_peak_m3s=0.1)
    row = transport(frames=frames, station="outlet", cut=15, settings=settings)
    assert row["larger_crest_during_followup"]
    assert row["complete_peak_m3s"] == 3
    assert row["followup_outward_m3"] == 30
    assert row["followup_inward_m3"] == 10
    assert row["followup_net_m3"] == 20
    assert row["followup_pct_of_complete_outward"] == pytest.approx(3000/55)
    with pytest.raises(ValueError, match="observed accepted-transport"):
        transport(frames=frames, station="outlet", cut=14, settings=settings)


def test_negligible_outflow_and_unobserved_followup_stay_unassessed():
    frames = [dict(t_s=10, outlet_m3s=-1), dict(t_s=20, outlet_m3s=-2)]
    opts = dict(frames=frames, station="outlet", settings=dict(minimum_volume_m3=1, minimum_peak_m3s=0.01))
    early = transport(cut=10, **opts)
    assert early["followup_outward_m3"] == 0 and early["followup_inward_m3"] == 20
    assert early["followup_pct_of_complete_outward"] is None and early["larger_crest_during_followup"] is None
    final = transport(cut=20, **opts)
    assert final["followup_outward_m3"] is None and final["followup_peak_m3s"] is None
