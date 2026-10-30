"""Practical hydraulic assessment of preserved LI replay fields; no solver run."""
import hashlib
import json
from pathlib import Path

import numpy as np

from validation.li_assess import wet_metrics, write_csv, write_json


def weighted_percentile(*, values, weights, quantile):
    order = np.argsort(values)
    ordered_weights = weights[order]
    return float(values[order[np.searchsorted(np.cumsum(ordered_weights), quantile * ordered_weights.sum(), side='left')]])


def depth_assessment(*, actual, reference, weights, criteria):
    mask = np.maximum(actual, reference) >= criteria['meaningful_depth_m']
    error = np.abs(actual - reference)
    volume_weights = np.broadcast_to(weights[:, None, None], actual.shape)[mask]
    if not volume_weights.size:
        return dict(covered=False, passed=None, max_error_all_depths_m=float(error.max()))
    errors, expected = error[mask], reference[mask]
    rmse = float(np.sqrt(np.average(errors**2, weights=volume_weights)))
    norm = float(np.sum(expected**2 * volume_weights))
    relative = float(np.sqrt(np.sum(errors**2 * volume_weights) / norm)) if norm else None
    scaled = errors / (criteria['absolute_depth_m'] + criteria['relative_depth'] * expected)
    p95 = weighted_percentile(values=scaled, weights=volume_weights, quantile=criteria['percentile'])
    return dict(covered=True, rmse_m=rmse, relative_l2=relative, scaled_p95=p95,
        fraction_above_local_budget=float(np.average(scaled > 1., weights=volume_weights)),
        max_error_all_depths_m=float(error.max()),
        passed=bool((rmse <= criteria['absolute_depth_m'] or (relative is not None and relative <= criteria['relative_depth'])) and p95 <= 1.))


def extent_assessment(*, actual, reference, weights, threshold, criteria):
    wa, wr = actual > threshold, reference > threshold
    union = np.count_nonzero(wa | wr, axis=(1, 2))
    mismatch = wa != wr
    ambiguous = (np.abs(actual - threshold) <= criteria['extent_band_m']) & (np.abs(reference - threshold) <= criteria['extent_band_m'])
    resolved = np.count_nonzero(mismatch & ~ambiguous, axis=(1, 2))
    total = float(np.sum(union * weights))
    fractions = np.divide(resolved, union, out=np.zeros_like(weights), where=union > 0)
    duration = float(weights.sum())
    persistent = float(weights[fractions > criteria['instant_extent_fraction']].sum() / duration)
    fraction = float(np.sum(resolved * weights) / total) if total else None
    return dict(threshold_m=threshold, covered=bool(total), resolved_area_time_fraction=fraction,
        duration_fraction_above_instant_budget=persistent,
        passed=bool(fraction <= criteria['extent_fraction'] and persistent <= criteria['duration_fraction']) if total else None)


def discharge_assessment(*, actual, reference, times, criteria):
    peak = float(reference.max())
    actual_peak = float(actual.max())
    nrmse = float(np.sqrt(np.mean((actual - reference)**2)) / peak) if peak else None
    peak_error = abs(actual_peak - peak) / peak if peak else None
    timing = abs(float(times[np.argmax(actual)] - times[np.argmax(reference)])) if peak else None
    timing_budget = max(float(np.max(np.diff(np.r_[0., times]))), criteria['timing_fraction'] * float(times[-1]))
    passed = nrmse <= criteria['q_nrmse'] and peak_error <= criteria['peak_fraction'] and timing <= timing_budget if peak else actual_peak == 0.
    return dict(nrmse=nrmse, peak_relative_error=peak_error, peak_time_difference_s=timing,
                peak_time_budget_s=timing_budget, passed=bool(passed))



def default_sensitivity(*, roots, criteria):
    results = []
    for row in json.loads((roots['shipped_defaults'] / 'manifest.json').read_text()):
        matched_label = row['case'] + '__physical_candidate__None'
        with np.load(file=roots['shipped_defaults'] / (row['label'] + '.npz')) as archive:
            actual = {key: archive[key].copy() for key in archive.files}
        with np.load(file=roots['baseline'] / (matched_label + '.npz')) as archive:
            reference = {key: archive[key].copy() for key in archive.files}
        assert np.array_equal(actual['times'], reference['times'])
        # Common face width cancels from normalized discharge diagnostics.
        actual_q = np.maximum(np.concatenate((-actual['qx'][:, :, 0], actual['qx'][:, :, -1], -actual['qy'][:, 0, :], actual['qy'][:, -1, :]), axis=1), 0.).sum(axis=1)
        reference_q = np.maximum(np.concatenate((-reference['qx'][:, :, 0], reference['qx'][:, :, -1], -reference['qy'][:, 0, :], reference['qy'][:, -1, :]), axis=1), 0.).sum(axis=1)
        weights = np.diff(np.r_[0., actual['times']])
        depth = depth_assessment(actual=actual['depth'], reference=reference['depth'], weights=weights, criteria=criteria)
        discharge = discharge_assessment(actual=actual_q, reference=reference_q, times=actual['times'], criteria=criteria)
        export_index = row['ledger_keys'].index('boundary_outflow')
        results.append(dict(case=row['case'], boundary='dry_exterior_head', comparison='public defaults versus fixture-cap same-closure control',
            depth=depth, discharge=discharge, export_difference_m3=float(actual['ledger'][-1, export_index] - reference['ledger'][-1, export_index])))
    return results

def assess(*, roots, fixtures, previous, output, criteria, thresholds):
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((previous / 'engineering_results.json').read_text())['rows']
    original_map = {(row['group'], row['label']): row for row in original}
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    for case in manifest['cases']:
        for name, expected in case['files'].items():
            assert hashlib.sha256((fixtures / case['id'] / name).read_bytes()).hexdigest() == expected
    for name in ('parity.json', 'default_parity.json'):
        assert json.loads((previous / name).read_text())['all_bitwise']
    results, wet_rows = [], []
    for group, root in roots.items():
        for row in json.loads((root / 'manifest.json').read_text()):
            identity = (group, row['label'])
            if identity not in original_map:
                continue
            old = original_map[identity]
            with np.load(file=root / (row['label'] + '.npz')) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            for key, value in data.items():
                assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == row['hashes'][key]
            with np.load(file=fixtures / row['case'] / 'depth_series.npz') as archive:
                reference = archive['depth'].copy()
                assert np.array_equal(data['times'], archive['times_s'])
            with np.load(file=fixtures / row['case'] / 'arrays.npz') as archive:
                expected_q = np.interp(data['times'], archive['times'], archive['qout'])
            scenario = json.loads((fixtures / row['case'] / 'scenario.json').read_text())
            weights = np.diff(np.r_[0., data['times']])
            depth = depth_assessment(actual=data['depth'], reference=reference, weights=weights, criteria=criteria)
            extents = []
            for threshold in thresholds:
                for time, actual, expected in zip(data['times'], data['depth'], reference, strict=True):
                    wet_rows.append(dict(group=group, label=row['label'], kind='simultaneous', time_s=float(time),
                        **wet_metrics(actual=actual, reference=expected, threshold=threshold, area=scenario['spacing']**2, bands=(criteria['extent_band_m'],))))
                wet_rows.append(dict(group=group, label=row['label'], kind='sampled_envelope', time_s=float(data['times'][-1]),
                    **wet_metrics(actual=data['depth'].max(axis=0), reference=reference.max(axis=0), threshold=threshold,
                                  area=scenario['spacing']**2, bands=(criteria['extent_band_m'],))))
                if threshold in criteria['extent_thresholds_m']:
                    extents.append(extent_assessment(actual=data['depth'], reference=reference, weights=weights, threshold=threshold, criteria=criteria))
            q = np.maximum(np.concatenate((-data['qx'][:, :, 0], data['qx'][:, :, -1], -data['qy'][:, 0, :], data['qy'][:, -1, :]), axis=1), 0.).sum(axis=1) * scenario['spacing']
            discharge = discharge_assessment(actual=q, reference=expected_q, times=data['times'], criteria=criteria)
            matched = row['numerical']['free_outflow_closure'] == 'local_or_prescribed_slope'
            volume_budget = criteria['volume_fraction'] * old['supplied_m3']
            budgets = dict(export_pass=abs(old['export_difference_m3']) <= volume_budget,
                storage_pass=abs(old['storage_difference_m3']) <= volume_budget, tolerance_m3=volume_budget,
                export_difference_m3=old['export_difference_m3'], storage_difference_m3=old['storage_difference_m3'],
                export_fraction_supplied=old['export_difference_fraction_supplied'])
            concerns = []
            if depth['passed'] is False:
                concerns.append('meaningful_depth')
            for extent in extents:
                if extent['passed'] is False:
                    concerns.append('extent_' + str(extent['threshold_m']))
            if not discharge['passed']:
                concerns.append('hydrograph')
            if not all((budgets['export_pass'], budgets['storage_pass'])):
                concerns.append('event_allocation')
            review = depth['max_error_all_depths_m'] > criteria['local_review_m']
            infiltration = old['infiltration_difference_m3']
            reference_infil = old['reference_budget']['infiltration']
            physical_infil = old['budget']['infiltration']
            recharge_relative = abs(infiltration) / reference_infil if reference_infil else None
            recharge_flag = recharge_relative > criteria['recharge_fraction'] if reference_infil else physical_infil != 0.
            status = 'non_equivalent_boundary' if not matched else ('integrity_failure' if not old['integrity'] else ('outside_screening_budgets' if concerns else ('local_review_required' if review else 'within_screening_budgets')))
            results.append(dict(group=group, label=row['label'], status=status, equivalent_boundary=matched,
                integrity=old['integrity'], hydraulic_concerns=concerns if matched else [], unmatched_diagnostics=concerns if not matched else [],
                depth=depth, extent=extents, discharge=discharge, water_allocation=budgets, local_review_required=review,
                recharge=dict(infiltration_difference_m3=infiltration, relative_to_reference=recharge_relative,
                    relative_to_actual=abs(infiltration) / physical_infil if physical_infil else None,
                    relative_to_rainfall=abs(infiltration) / old['reference_budget']['rain'] if old['reference_budget']['rain'] else None,
                    agreement_budget_exceeded=bool(recharge_flag), independently_qualified=False)))
    write_json(path=output / 'results.json', value=dict(criteria=criteria, rows=results, default_timestep_sensitivity=default_sensitivity(roots=roots, criteria=criteria)))
    write_csv(path=output / 'wet_thresholds.csv', rows=wet_rows)


if __name__ == "__main__":
    fixtures = Path('validation/fixtures')
    previous = Path('validation/li_review_v1')
    output = Path('validation/li_review_v2')
    roots = dict(baseline=Path('outputs/li_review_v1/baselines/target_results'),
                 shipped_defaults=Path('outputs/li_review_v1/shipped_defaults_v2'))
    thresholds = (.0001, .001, .005, .01, .05, .1)
    criteria = dict(meaningful_depth_m=.05, absolute_depth_m=.01, relative_depth=.05, percentile=.95,
        local_review_m=.05, extent_thresholds_m=(.05, .1), extent_band_m=.01,
        extent_fraction=.05, instant_extent_fraction=.1, duration_fraction=.1,
        q_nrmse=.05, peak_fraction=.1, timing_fraction=.05, volume_fraction=.05, recharge_fraction=.1)
    assess(roots=roots, fixtures=fixtures, previous=previous, output=output, criteria=criteria, thresholds=thresholds)
