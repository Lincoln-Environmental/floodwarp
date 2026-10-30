"""Practical criteria retain consequential errors and distinguish sparse coverage."""
import numpy as np

from validation.li_engineering_v2 import depth_assessment, extent_assessment, discharge_assessment


def test_false_inundation_is_included_in_meaningful_depth_assessment():
    result = depth_assessment(actual=np.full((3, 2, 2), .2), reference=np.zeros((3, 2, 2)), weights=np.ones(3),
        criteria=dict(meaningful_depth_m=.05, absolute_depth_m=.01, relative_depth=.05, percentile=.95))
    assert result['covered']
    assert not result['passed']


def test_depth_absence_is_not_an_agreement_pass():
    result = depth_assessment(actual=np.zeros((2, 2, 2)), reference=np.zeros((2, 2, 2)), weights=np.ones(2),
        criteria=dict(meaningful_depth_m=.05, absolute_depth_m=.01, relative_depth=.05, percentile=.95))
    assert not result['covered']
    assert result['passed'] is None


def test_extent_is_sensitive_to_persistence_without_single_sample_veto():
    reference = np.full((20, 10, 10), .1)
    actual = reference.copy()
    actual[0, :2, :] = 0.
    criteria = dict(extent_band_m=.01, instant_extent_fraction=.1, extent_fraction=.05, duration_fraction=.1)
    brief = extent_assessment(actual=actual, reference=reference, weights=np.ones(20), threshold=.05, criteria=criteria)
    assert brief['passed']
    actual[:4, :2, :] = 0.
    sustained = extent_assessment(actual=actual, reference=reference, weights=np.ones(20), threshold=.05, criteria=criteria)
    assert not sustained['passed']
    # Overall area-time error is only 4%; persistence catches the affected patch.
    assert sustained['resolved_area_time_fraction'] < .05


def test_extent_ambiguity_requires_both_depths_near_threshold():
    criteria = dict(extent_band_m=.01, instant_extent_fraction=.1, extent_fraction=.05, duration_fraction=.1)
    near = extent_assessment(actual=np.full((2, 1, 1), .051), reference=np.full((2, 1, 1), .049), weights=np.ones(2), threshold=.05, criteria=criteria)
    assert near['passed']
    distant = extent_assessment(actual=np.full((2, 1, 1), .2), reference=np.full((2, 1, 1), .049), weights=np.ones(2), threshold=.05, criteria=criteria)
    assert not distant['passed']


def test_zero_reference_flow_cannot_hide_actual_export():
    result = discharge_assessment(actual=np.ones(3), reference=np.zeros(3), times=np.array([60., 120., 180.]),
        criteria=dict(timing_fraction=.05, q_nrmse=.05, peak_fraction=.1))
    assert not result['passed']
