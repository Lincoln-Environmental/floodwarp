"""Guard the engineering review's threshold semantics and empty denominators."""
import numpy as np

from validation.li_assess import depth_metrics, wet_metrics


def test_threshold_ties_false_wet_and_false_dry():
    actual = np.array([[.001, .001001, .000999, .01]])
    reference = np.array([[.001001, .001, .001001, .01]])
    result = wet_metrics(actual=actual, reference=reference, threshold=.001, area=25., bands=(.00001,))
    assert result['false_wet'] == 1
    assert result['false_dry'] == 2
    assert result['disagreement_m2'] == 75.
    assert result['iou'] == .25
    assert result['precision'] == .5
    assert result['recall'] == 1 / 3
    assert result['both_within_1e-05m_cells'] == 3


def test_empty_extent_is_explicit_and_not_precision_evidence():
    result = wet_metrics(actual=np.zeros((2, 2)), reference=np.zeros((2, 2)), threshold=.1, area=100., bands=(.001,))
    assert result['empty_union']
    assert result['iou'] == 1.
    assert result['precision'] is None
    assert result['recall'] is None


def test_uncertainty_band_requires_both_depths_near_threshold():
    result = wet_metrics(actual=np.array([.02]), reference=np.array([.0099]), threshold=.01, area=1., bands=(.001,))
    assert result['both_within_0.001m_cells'] == 0
    assert result['disagreement_cells'] == 1


def test_depth_errors_include_dry_cells_and_are_not_thresholded():
    result = depth_metrics(actual=np.array([.0001, .0002]), reference=np.zeros(2), percentiles=(95,))
    np.testing.assert_allclose(actual=result['mae_m'], desired=.00015)
    assert result['relative_l2'] is None
    assert result['max_m'] == .0002
